"""Compare native DeepEP and PyTorch all-to-all by token identity and expert output."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import itertools
import json
import os
import traceback
from pathlib import Path
import torch
import torch.distributed as dist
from moe_lab.dispatch import AllToAllDispatcher, Dispatched
from moe_lab.deepep_backend import DeepEPLowLatencyDispatcher
from moe_lab.expert_mlp import create_weights as expert_weights, evaluate_experts as experts, evaluate_experts_into


def inputs(rank, world, hidden, scenario, *, seed=811, tokens_base=16):
    gen = torch.Generator().manual_seed(seed + rank)
    tokens, experts, topk = tokens_base + rank * 3, world * 4, 2
    if scenario == 'empty_rank' and rank == 0 or scenario == 'all_empty':
        tokens = 0
    x = torch.randn(tokens, hidden, generator=gen).to(torch.bfloat16)
    x[:, 0] = rank
    identity = torch.arange(tokens)
    x[:, 1] = identity // 128
    x[:, 2] = identity % 128
    ids = torch.stack([torch.randperm(experts, generator=gen)[:topk] for _ in range(tokens)]) if tokens else torch.empty((0, topk), dtype=torch.int64)
    if scenario == 'skewed':
        ids[:] = torch.tensor([0, 1])
    if scenario == 'dropped':
        ids[:, 1] = -1
    if scenario == 'all_dropped':
        ids[:] = -1
    weights = torch.rand(tokens, topk, generator=gen)
    weights /= weights.sum(dim=1, keepdim=True)
    return x, ids, weights


def oracle(x, ids, scores, weights):
    first, second, bias = [t.double() for t in weights]
    out = torch.zeros(x.shape, dtype=torch.float64)
    for expert in range(first.shape[0]):
        token, slot = torch.where(ids == expert)
        values = torch.relu(x[token].double() @ first[expert]) @ second[expert] + bias[expert]
        out.index_add_(0, token, values * scores[token, slot, None].double())
    return out


def aligned_rows(batch, local_expert):
    rows = batch.tokens[batch.local_expert_ids == local_expert]
    return rows[row_order(rows)]


def row_order(rows):
    return (rows[:, 0].long() * 32768 + rows[:, 1].long() * 128 + rows[:, 2].long()).argsort()


def rdma_counters():
    counters = {}
    for device in sorted(Path('/sys/class/infiniband').glob('mlx5_*')):
        values = {}
        for name in ['port_xmit_data', 'port_rcv_data', 'port_xmit_packets', 'port_rcv_packets']:
            path = device / 'ports/1/counters' / name
            if path.exists():
                values[name] = int(path.read_text())
        counters[device.name] = values
    return counters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hidden', type=int, default=2048)
    parser.add_argument('--scenarios', nargs='+', default=['balanced', 'skewed', 'dropped'])
    parser.add_argument('--rdma-only', action='store_true')
    parser.add_argument('--packed-experts', action='store_true')
    parser.add_argument('--registered-experts', action='store_true')
    parser.add_argument('--grouped-experts', action='store_true')
    parser.add_argument('--completion', choices=['sync', 'event', 'hook'], default='sync')
    parser.add_argument('--packing', choices=['zeros', 'empty', 'zero_copy'], default='zeros')
    parser.add_argument('--known-output-size', action='store_true')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--seed', type=int, default=811)
    parser.add_argument('--tokens', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--stream', choices=['default', 'non_default'], default='default')
    parser.add_argument('--normalized-max-tolerance', type=float, default=.01)
    args = parser.parse_args()
    args.registered_experts = args.registered_experts or args.packed_experts
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(1)
    torch.cuda.set_device(local)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    original_stream = torch.cuda.current_stream()
    if args.stream == 'non_default':
        stream = torch.cuda.Stream()
        stream.wait_stream(original_stream)
        torch.cuda.set_stream(stream)
    native = None
    records = []
    try:
        print(f"rank={rank} initializing native DeepEP", flush=True)
        native = DeepEPLowLatencyDispatcher(world * 4, args.hidden, max_tokens=args.max_tokens,
                                            completion=args.completion, packing=args.packing,
                                            known_output_size=args.known_output_size, allow_nvlink=not args.rdma_only)
        cpu_weights = expert_weights(world, args.hidden)
        gpu_weights = [t.cuda() for t in cpu_weights]
        counters_before = rdma_counters() if rank == 0 else None
        for iteration, scenario in itertools.product(range(args.iterations), args.scenarios):
            print(f"rank={rank} iteration={iteration} scenario={scenario} starting", flush=True)
            cpu_x, cpu_ids, cpu_scores = inputs(rank, world, args.hidden, scenario,
                                               seed=args.seed + iteration * 97, tokens_base=args.tokens)
            expected = oracle(cpu_x, cpu_ids, cpu_scores, cpu_weights).cuda()
            x, ids, scores = [t.cuda() for t in (cpu_x, cpu_ids, cpu_scores)]
            reference = AllToAllDispatcher(world * 4, host_expert_counts=args.grouped_experts,
                                          known_output_size=args.known_output_size)
            ref_batch = reference.dispatch(x, ids, scores)
            if args.packed_experts:
                packed_batch = native.dispatch_packed(x, ids, scores)
                context = packed_batch.context
                compact = torch.cat([packed_batch.packed_tokens[e,:count] for e,count in enumerate(context.counts)])
                labels = torch.repeat_interleave(torch.arange(4,device=x.device),packed_batch.gpu_counts.long(),output_size=sum(context.counts))
                native_batch = Dispatched(compact,labels,context)
            else:
                native_batch = native.dispatch(x, ids, scores)
            for expert in range(4):
                torch.testing.assert_close(aligned_rows(native_batch, expert), aligned_rows(ref_batch, expert), atol=0, rtol=0)
            ref_expert = experts(ref_batch, gpu_weights, rank, contiguous=args.grouped_experts)
            if args.registered_experts:
                registered = native.registered_output(native_batch.context)
                evaluate_experts_into(packed_batch if args.packed_experts else native_batch, gpu_weights, rank, registered)
                native_expert = torch.cat([registered[e,:count] for e,count in enumerate(native_batch.context.counts)])
            else:
                native_expert = experts(native_batch, gpu_weights, rank, contiguous=args.grouped_experts)
            for expert in range(4):
                ref_mask, native_mask = ref_batch.local_expert_ids == expert, native_batch.local_expert_ids == expert
                ref_order = row_order(ref_batch.tokens[ref_mask])
                native_order = row_order(native_batch.tokens[native_mask])
                torch.testing.assert_close(native_expert[native_mask][native_order], ref_expert[ref_mask][ref_order], atol=0.02, rtol=0.02)
            ref_out = reference.combine(ref_expert, ref_batch.context)
            native_out = native.combine_registered(registered, native_batch.context) if args.registered_experts else native.combine(native_expert, native_batch.context)
            torch.testing.assert_close(native_out, ref_out, atol=0.03, rtol=0.03)
            torch.testing.assert_close(native_out.double(), expected, atol=0.06, rtol=0.06)
            def errors_against(target):
                actual, target = native_out.double(), target.double()
                absolute = float((actual-target).abs().max()) if target.numel() else 0.
                scale = float(target.abs().max()) if target.numel() else 0.
                return {'max_abs_error':absolute, 'reference_abs_max':scale,
                        'normalized_max_error':absolute/max(scale,1e-30),
                        'relative_l2_error':float(torch.linalg.vector_norm(actual-target) /
                                                  torch.linalg.vector_norm(target).clamp_min(1e-30))}
            backend_errors, oracle_errors = errors_against(ref_out), errors_against(expected)
            if backend_errors['normalized_max_error'] > args.normalized_max_tolerance:
                raise AssertionError(f'Normalized backend error exceeded {args.normalized_max_tolerance}: {backend_errors}')
            records.append({'rank': rank, 'scenario': scenario, 'iteration': iteration,
                            'seed': args.seed + iteration * 97, 'passed': True,
                            'dispatch_exact': True,
                            'backend_errors':backend_errors, 'fp64_errors':oracle_errors,
                            'backend_max_abs_error': float((native_out.float() - ref_out.float()).abs().max()) if x.numel() else 0.,
                            'fp64_max_abs_error': float((native_out.double() - expected).abs().max()) if x.numel() else 0.,
                            'backend_atol': 0.03, 'backend_rtol': 0.03, 'fp64_atol': 0.06, 'fp64_rtol': 0.06})
            checkpoint = args.output.parent / 'checkpoints' / f'{args.output.stem}-rank{rank}.json'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps({'rank':rank, 'world_size':world, 'hidden':args.hidden,
                                              'updated_utc':datetime.now(timezone.utc).isoformat(),
                                              'scope':'Completed rank-local cases; the final aggregate is written after all ranks finish',
                                              'tokens_base':args.tokens, 'capacity':args.max_tokens,
                                              'completion':args.completion, 'packed_experts':args.packed_experts,
                                              'cases':records}, indent=2)+'\n')
            temporary.replace(checkpoint)
        dist.barrier()
        torch.cuda.synchronize()
        counters_after = rdma_counters() if rank == 0 else None
        combined = [None] * world
        dist.all_gather_object(combined, records)
        if rank == 0:
            root = Path(__file__).resolve().parents[1]
            payload = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                       'world_size': world, 'hidden': args.hidden, 'fp8_dispatch': False,
                       'native_build':json.loads((root/'results/deepep-build-manifest.json').read_text()) if (root/'results/deepep-build-manifest.json').exists() else None,
                       'completion': args.completion, 'packing': args.packing,
                       'known_output_size': args.known_output_size, 'stream': args.stream,
                       'allow_nvlink': not args.rdma_only, 'max_tokens':args.max_tokens,
                       'tokens_base':args.tokens, 'iterations':args.iterations,
                       'normalized_max_tolerance':args.normalized_max_tolerance,
                       'normalized_max_definition':'max(abs(actual-reference)) / max(abs(reference)); all-zero agreement is zero',
                       'grouped_experts':args.grouped_experts, 'registered_experts':args.registered_experts, 'packed_experts':args.packed_experts,
                       'nccl_ib_disabled': os.environ.get('NCCL_IB_DISABLE') == '1',
                       'rdma_counters_before':counters_before, 'rdma_counters_after':counters_after,
                       'reference': 'CPU FP64 two-layer expert MLP with ReLU',
                       'cases': [record for items in combined for record in items],
                       'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                         for folder in ['scripts', 'moe_lab'] for p in sorted((root / folder).glob('*.py'))}}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2) + '\n')
            temporary.replace(args.output)
            print(f"PASS: {len(payload['cases'])} rank-local native DeepEP precision cases")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if native is not None and native._pending is None:
            native.close()
        torch.cuda.current_stream().synchronize()
        torch.cuda.set_stream(original_stream)
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
