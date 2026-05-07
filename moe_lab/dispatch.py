"""Forward-only expert dispatch with explicit, per-invocation routing state."""
from dataclasses import dataclass
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DispatchContext:
    send_splits: tuple
    recv_splits: tuple
    expert_order: torch.Tensor
    route_slots: torch.Tensor
    weights: torch.Tensor
    num_tokens: int
    top_k: int
    owner: object
    expert_counts: tuple | None = None


@dataclass(frozen=True)
class Dispatched:
    tokens: torch.Tensor
    local_expert_ids: torch.Tensor
    context: DispatchContext


class AllToAllDispatcher:
    """Contiguous, evenly sharded experts; -1 marks a dropped route.

    All ranks must call in the same order with matching dtype, hidden width,
    expert count and process group. Validate distributed configuration before
    entry: a rank-local exception cannot safely recover an ongoing collective.
    """

    def __init__(self, num_experts, *, optimized=True, group=None, known_output_size=False,
                 combine_mode='atomic', host_expert_counts=False):
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if num_experts <= 0 or num_experts % self.world:
            raise ValueError('Experts must be positive and evenly divisible by world size')
        self.num_experts = num_experts
        self.local_experts = num_experts // self.world
        self.optimized = optimized
        self.known_output_size = known_output_size
        if combine_mode not in ('atomic', 'ordered'):
            raise ValueError('Unknown combine mode')
        self.combine_mode = combine_mode
        self.host_expert_counts = host_expert_counts
        self.collectives = []
        self._owner = object()

    def _exchange(self, output, source, output_splits=None, input_splits=None, label=''):
        dist.all_to_all_single(output, source, output_split_sizes=output_splits,
                               input_split_sizes=input_splits, group=self.group)
        self.collectives.append({'label': label, 'input_bytes': source.numel()*source.element_size(),
                                 'output_bytes': output.numel()*output.element_size()})

    @torch.no_grad()
    def dispatch(self, x, expert_ids, weights):
        if x.ndim != 2 or not x.is_floating_point() or not x.is_contiguous():
            raise ValueError('Tokens must be a contiguous floating matrix')
        if expert_ids.ndim != 2 or expert_ids.shape[0] != x.shape[0] or expert_ids.shape[1] == 0:
            raise ValueError('Routes must have shape [tokens, positive top_k]')
        if expert_ids.dtype != torch.int64 or weights.shape != expert_ids.shape:
            raise ValueError('Routes require int64 IDs and matching weights')
        if weights.dtype != torch.float32 or weights.device != x.device or expert_ids.device != x.device:
            raise ValueError('Use same-device routes and FP32 weights')
        if x.requires_grad or weights.requires_grad:
            raise ValueError('Forward-only interface')
        if not torch.isfinite(weights).all() or ((expert_ids < -1) | (expert_ids >= self.num_experts)).any():
            raise ValueError('Invalid expert ID or nonfinite weight')
        flat = expert_ids.reshape(-1)
        slots = torch.nonzero(flat >= 0).flatten()
        order = torch.argsort(flat[slots], stable=True)
        slots = slots[order]
        counts = torch.bincount(flat[slots], minlength=self.num_experts)
        send = counts.view(self.world, self.local_experts).sum(1)
        if not self.optimized:
            redundant_recv = torch.empty_like(send)
            self._exchange(redundant_recv, send, label='dispatch_rank_counts')
        received_counts = torch.empty_like(counts)
        self._exchange(received_counts, counts, label='dispatch_expert_counts')
        if self.host_expert_counts:
            host_counts = received_counts.tolist()
            recv_splits = tuple(sum(host_counts[r*self.local_experts:(r+1)*self.local_experts])
                                for r in range(self.world))
            expert_counts = tuple(sum(host_counts[r*self.local_experts+e] for r in range(self.world))
                                  for e in range(self.local_experts))
            recv = None
        else:
            recv = received_counts.view(self.world, self.local_experts).sum(1)
            recv_splits = tuple(recv.tolist())
            expert_counts = None
        if not self.optimized:
            agree = tuple(redundant_recv.tolist()) == recv_splits if self.host_expert_counts else torch.equal(redundant_recv, recv)
            if not agree:
                raise RuntimeError('Rank and expert counts disagree')
        send_splits = tuple(send.tolist())
        packed = x.index_select(0, slots // expert_ids.shape[1]).contiguous()
        incoming = x.new_empty((sum(recv_splits), x.shape[1]))
        self._exchange(incoming, packed, list(recv_splits), list(send_splits), 'dispatch_tokens')
        # Transport is source-rank major; each source sorts by global expert ID.
        labels = torch.arange(self.local_experts, device=x.device).repeat(self.world)
        size_hint = {'output_size': sum(recv_splits)} if self.known_output_size else {}
        labels = torch.repeat_interleave(labels, received_counts, **size_hint)
        expert_order = torch.argsort(labels, stable=True)
        context = DispatchContext(send_splits, recv_splits, expert_order, slots,
                                  weights.clone(), x.shape[0], expert_ids.shape[1], self._owner, expert_counts)
        return Dispatched(incoming[expert_order], labels[expert_order], context)

    @torch.no_grad()
    def combine(self, expert_output, context):
        if context.owner is not self._owner:
            raise ValueError('Context belongs to another dispatcher')
        if expert_output.ndim != 2 or expert_output.shape[0] != sum(context.recv_splits):
            raise ValueError('Expert output must preserve dispatched row count')
        if expert_output.requires_grad or expert_output.device != context.weights.device:
            raise ValueError('Use forward-only output on the routing device')
        # Invert expert-major grouping before reversing the transport.
        rank_order = torch.empty_like(expert_output)
        rank_order[context.expert_order] = expert_output
        if not self.optimized:
            counts = torch.tensor(context.recv_splits, dtype=torch.int64, device=expert_output.device)
            returned_counts = torch.empty_like(counts)
            self._exchange(returned_counts, counts, label='combine_rank_counts')
            if tuple(returned_counts.tolist()) != context.send_splits:
                raise RuntimeError('Return counts disagree')
        returned = expert_output.new_empty((sum(context.send_splits), expert_output.shape[1]))
        self._exchange(returned, rank_order.contiguous(), list(context.send_splits),
                       list(context.recv_splits), 'combine_tokens')
        weighted = returned.float() * context.weights.reshape(-1)[context.route_slots, None]
        if self.combine_mode == 'ordered':
            routes = torch.zeros((context.num_tokens * context.top_k, expert_output.shape[1]),
                                 device=returned.device, dtype=torch.float32)
            routes.index_copy_(0, context.route_slots, weighted)
            output = routes.view(context.num_tokens, context.top_k, expert_output.shape[1]).sum(1)
        else:
            output = torch.zeros((context.num_tokens, expert_output.shape[1]), device=returned.device, dtype=torch.float32)
            output.index_add_(0, context.route_slots // context.top_k, weighted)
        return output.to(expert_output.dtype)
