"""Attribute host synchronization and padding work outside performance measurements."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity, record_function
from moe_lab.dispatch import AllToAllDispatcher
from moe_lab.deepep_backend import DeepEPLowLatencyDispatcher
from scripts.benchmark_pipeline import make_inputs
from scripts.validate_deepep import expert_weights, experts
from moe_lab.expert_mlp import evaluate_experts_into


def execute(dispatcher, x, ids, weights, matrices, rank, *, contiguous=False, packed=False):
    with record_function('moe/dispatch'):
        batch = dispatcher.dispatch_packed(x, ids, weights) if packed else dispatcher.dispatch(x, ids, weights)
    with record_function('moe/expert'):
        if packed:
            output = dispatcher.registered_output(batch.context)
            evaluate_experts_into(batch, matrices, rank, output)
        else:
            output = experts(batch, matrices, rank, contiguous=contiguous)
    with record_function('moe/combine'):
        return dispatcher.combine_registered(output, batch.context) if packed else dispatcher.combine(output, batch.context)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-prefix', type=Path, required=True)
    parser.add_argument('--packing', choices=['zeros', 'empty', 'zero_copy'], default='zeros')
    parser.add_argument('--known-output-size', action='store_true')
    parser.add_argument('--grouped-experts', action='store_true')
    parser.add_argument('--packed-experts', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    native = DeepEPLowLatencyDispatcher(world*4, 7168, max_tokens=1024,
                                        packing=args.packing, known_output_size=args.known_output_size)
    try:
        x, ids, weights = make_inputs(rank, world, 16, 7168, 'balanced')
        matrices = [w.cuda() for w in expert_weights(world, 7168)]
        methods = {'reference': AllToAllDispatcher(world*4, known_output_size=args.known_output_size,
                                                  host_expert_counts=args.grouped_experts),
                   'native': native}
        reports = []
        for name, dispatcher in methods.items():
            for _ in range(5):
                execute(dispatcher, x, ids, weights, matrices, rank, contiguous=args.grouped_experts, packed=args.packed_experts and name == 'native')
            dist.barrier()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(5):
                    execute(dispatcher, x, ids, weights, matrices, rank, contiguous=args.grouped_experts, packed=args.packed_experts and name == 'native')
                torch.cuda.synchronize()
            args.trace_prefix.parent.mkdir(parents=True, exist_ok=True)
            trace = f'{args.trace_prefix}-{name}-rank{rank}.json'
            prof.export_chrome_trace(trace)
            events = [{'name': event.key, 'calls': event.count,
                       'self_cpu_us': event.self_cpu_time_total,
                       'cpu_us': event.cpu_time_total,
                       'self_device_us': event.self_device_time_total,
                       'device_us': event.device_time_total} for event in prof.key_averages()]
            reports.append({'method': name, 'rank': rank, 'iterations': 5,
                            'events': sorted(events, key=lambda e:e['self_cpu_us'], reverse=True)})
        gathered = [None] * world
        dist.all_gather_object(gathered, reports)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'scope': 'Profiler attribution, not benchmark latency',
                                              'hidden':7168, 'tokens_rank_zero':16, 'capacity':1024,
                                              'packing':args.packing, 'known_output_size':args.known_output_size,
                                              'grouped_experts':args.grouped_experts,
                                              'packed_experts':args.packed_experts,
                                              'reports':[r for rs in gathered for r in rs]}, indent=2)+'\n')
            print('COMPLETE: CPU/CUDA stage attribution', flush=True)
    finally:
        native.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
