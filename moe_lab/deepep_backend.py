"""BF16 adapter with synchronous, event, and hook completion for the pinned DeepEP v1.2.1 low-latency API."""
from dataclasses import dataclass
import torch
import torch.distributed as dist
from moe_lab.dispatch import Dispatched


@dataclass(frozen=True)
class DeepEPContext:
    handle: tuple
    counts: tuple
    packed_shape: tuple
    expert_ids: torch.Tensor
    weights: torch.Tensor
    owner: object


@dataclass(frozen=True)
class PackedDispatched:
    packed_tokens: torch.Tensor
    gpu_counts: torch.Tensor
    context: DeepEPContext


class DeepEPLowLatencyDispatcher:
    """One outstanding dispatch per adapter; combine must precede the next dispatch."""

    def __init__(self, num_experts, hidden, *, max_tokens=128, group=None,
                 completion='sync', packing='zeros', known_output_size=False, allow_nvlink=True):
        from deep_ep import Buffer
        self.group = dist.group.WORLD if group is None else group
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        if num_experts <= 0 or num_experts % self.world:
            raise ValueError('Experts must be evenly divisible by world size')
        if hidden not in (2048, 2560, 4096, 5120, 6144, 7168, 8192):
            raise ValueError('Unsupported pinned low-latency hidden dimension')
        self.num_experts = num_experts
        self.hidden = hidden
        self.max_tokens = max_tokens
        if completion not in ('sync', 'event', 'hook'):
            raise ValueError('Unknown completion mode')
        if packing not in ('zeros', 'empty', 'zero_copy'):
            raise ValueError('Unknown combine packing mode')
        self.completion = completion
        self.packing = packing
        self.known_output_size = known_output_size
        self.local_experts = num_experts // self.world
        self._owner = object()
        self._pending = None
        self._reserved_output = None
        size = Buffer.get_low_latency_rdma_size_hint(max_tokens, hidden, self.world, num_experts)
        self.buffer = Buffer(self.group, num_rdma_bytes=size, low_latency_mode=True,
                             num_qps_per_rank=self.local_experts, explicitly_destroy=True,
                             allow_nvlink_for_low_latency_mode=allow_nvlink)

    @torch.no_grad()
    def dispatch_packed(self, x, expert_ids, weights):
        if self._pending is not None:
            raise RuntimeError('Combine the outstanding dispatch before reusing the buffer')
        if x.device.type != 'cuda' or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError('Use contiguous CUDA BF16 input')
        if x.ndim != 2 or x.shape[1] != self.hidden or x.shape[0] > self.max_tokens:
            raise ValueError('Invalid shape or token capacity')
        if expert_ids.dtype != torch.int64 or expert_ids.device != x.device:
            raise ValueError('Use same-device int64 routes')
        if weights.dtype != torch.float32 or weights.device != x.device or weights.shape != expert_ids.shape:
            raise ValueError('Use same-device FP32 routing weights')
        if expert_ids.ndim != 2 or expert_ids.shape[0] != x.shape[0] or expert_ids.shape[1] < 1:
            raise ValueError('Invalid routing shape')
        if x.requires_grad or weights.requires_grad:
            raise ValueError('Forward-only interface')
        if not torch.isfinite(weights).all() or ((expert_ids < -1) | (expert_ids >= self.num_experts)).any():
            raise ValueError('Invalid expert ID or nonfinite weight')
        # Native routing expects at most one route to each expert per token.
        ordered = expert_ids.sort(dim=1).values
        if ((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)).any():
            raise ValueError('Duplicate routes are not accepted by this native adapter')
        packed, counts, handle, event, hook = self.buffer.low_latency_dispatch(
            x, expert_ids, self.max_tokens, self.num_experts, use_fp8=False,
            async_finish=self.completion == 'event', return_recv_hook=self.completion == 'hook')
        if self.completion == 'event':
            event.current_stream_wait()
        elif self.completion == 'hook':
            hook()
        count_values = tuple(int(v) for v in counts.tolist())
        context = DeepEPContext(handle, count_values, tuple(packed.shape),
                                expert_ids.clone(), weights.clone(), self._owner)
        self._pending = context
        return PackedDispatched(packed, counts, context)

    @torch.no_grad()
    def dispatch(self, x, expert_ids, weights):
        batch = self.dispatch_packed(x, expert_ids, weights)
        counts = batch.context.counts
        tokens = torch.cat([batch.packed_tokens[e,:count] for e,count in enumerate(counts)])
        size_hint = {'output_size':sum(counts)} if self.known_output_size else {}
        labels = torch.repeat_interleave(torch.arange(self.local_experts,device=x.device),
                                         batch.gpu_counts.long(), **size_hint)
        return Dispatched(tokens, labels, batch.context)

    @torch.no_grad()
    def combine(self, expert_output, context):
        if context.owner is not self._owner or context is not self._pending:
            raise ValueError('Invalid or already-consumed dispatch context')
        if expert_output.dtype != torch.bfloat16 or expert_output.shape != (sum(context.counts), self.hidden):
            raise ValueError('Expert output must preserve BF16 row count and hidden dimension')
        if expert_output.device != context.weights.device or expert_output.requires_grad:
            raise ValueError('Use same-device forward-only output')
        if self.packing == 'zero_copy':
            packed = self.buffer.get_next_low_latency_combine_buffer(context.handle)
        elif self.packing == 'empty':
            packed = expert_output.new_empty(context.packed_shape)
        else:
            packed = expert_output.new_zeros(context.packed_shape)
        offset = 0
        for expert, count in enumerate(context.counts):
            packed[expert, :count] = expert_output[offset:offset + count]
            offset += count
        return self._finish_combine(packed, context, self.packing == 'zero_copy')

    def registered_output(self, context):
        if context.owner is not self._owner or context is not self._pending:
            raise ValueError('Invalid or already-consumed dispatch context')
        if self.packing != 'zero_copy':
            raise ValueError('Registered output requires zero-copy packing')
        if self._reserved_output is None:
            self._reserved_output = self.buffer.get_next_low_latency_combine_buffer(context.handle)
        return self._reserved_output

    @torch.no_grad()
    def combine_registered(self, registered, context):
        if context.owner is not self._owner or context is not self._pending:
            raise ValueError('Invalid or already-consumed dispatch context')
        if registered is not self._reserved_output or registered is None:
            raise ValueError('Use the registered output reserved for this dispatch')
        return self._finish_combine(registered, context, True)

    def _finish_combine(self, packed, context, zero_copy):
        output, event, hook = self.buffer.low_latency_combine(
            packed, context.expert_ids, context.weights, context.handle,
            zero_copy=zero_copy, async_finish=self.completion == 'event',
            return_recv_hook=self.completion == 'hook')
        if self.completion == 'event':
            event.current_stream_wait()
        elif self.completion == 'hook':
            hook()
        self._pending = None
        self._reserved_output = None
        return output

    def close(self):
        if self._pending is not None:
            raise RuntimeError('An outstanding dispatch must be combined before destruction')
        self.buffer.destroy()
