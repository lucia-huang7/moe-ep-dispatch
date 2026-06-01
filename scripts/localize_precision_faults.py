"""Validate stage localization with explicit, synthetic MoE faults."""
import argparse
from datetime import timedelta
import json
import hashlib
import os
from pathlib import Path
import traceback
import torch
import torch.distributed as dist
from moe_lab.dispatch import AllToAllDispatcher
from moe_lab.deepep_backend import DeepEPLowLatencyDispatcher
from moe_lab.diagnostics import tensor_comparison, first_failed_stage
from scripts.validate_deepep import inputs, expert_weights, experts, oracle, aligned_rows, row_order


EXPECTED = {'clean': None, 'token_value': 'dispatch', 'expert_weights': 'expert',
            'combine_order': 'combine', 'routing_weights': 'combine', 'nan_expert': 'expert'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(seconds=90))
    native, records = None, []
    try:
        native = DeepEPLowLatencyDispatcher(world * 4, 2048)
        cpu_weights = expert_weights(world, 2048)
        gpu_weights = [w.cuda() for w in cpu_weights]
        for fault, expected_stage in EXPECTED.items():
            print(f'rank={rank} fault={fault}', flush=True)
            cx, ci, cw = inputs(rank, world, 2048, 'balanced')
            expected = oracle(cx, ci, cw, cpu_weights).cuda()
            x, ids, weights = [t.cuda() for t in (cx, ci, cw)]
            reference = AllToAllDispatcher(world * 4)
            rb = reference.dispatch(x, ids, weights)
            nb = native.dispatch(x, ids, weights)
            if fault == 'token_value' and nb.tokens.numel():
                nb.tokens[0, 3] += 8
            stages = {'dispatch': [], 'expert': [], 'combine': []}
            for expert in range(4):
                stages['dispatch'].append(tensor_comparison(
                    aligned_rows(nb, expert), aligned_rows(rb, expert)))
            ro = experts(rb, gpu_weights, rank)
            wrong_weights = [w.flip(0) for w in gpu_weights] if fault == 'expert_weights' else gpu_weights
            no = experts(nb, wrong_weights, rank)
            if fault == 'nan_expert' and no.numel():
                no[0, 3] = float('nan')
            for expert in range(4):
                rm, nm = rb.local_expert_ids == expert, nb.local_expert_ids == expert
                rr, nr = row_order(rb.tokens[rm]), row_order(nb.tokens[nm])
                stages['expert'].append(tensor_comparison(no[nm][nr], ro[rm][rr], atol=.02, rtol=.02))
            if fault == 'combine_order':
                no = no.roll(1, dims=0)
            if fault == 'routing_weights':
                nb.context.weights.zero_()
            ref_out = reference.combine(ro, rb.context)
            native_out = native.combine(no, nb.context)
            stages['combine'].append(tensor_comparison(native_out, ref_out, atol=.03, rtol=.03))
            observed = first_failed_stage(stages)
            local = {'rank': rank, 'fault': fault, 'synthetic_fault': fault != 'clean',
                     'expected_first_failure': expected_stage, 'observed_first_failure': observed,
                     'localization_passed': observed == expected_stage, 'stages': stages,
                     'fp64_oracle': tensor_comparison(native_out.double(), expected, atol=.06, rtol=.06)}
            gathered = [None] * world
            dist.all_gather_object(gathered, local)
            if rank == 0:
                records.extend(gathered)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix('.partial.json')
                temporary.write_text(json.dumps({'scope': 'Controlled synthetic faults, not an upstream bug reproduction',
                                                'native_build':json.loads(Path('results/deepep-build-manifest.json').read_text()),
                                                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                'cases': records}, indent=2, allow_nan=False) + '\n')
                temporary.replace(args.output)
        if rank == 0:
            assert all(c['localization_passed'] for c in records)
            print(f'PASS: {len(records)} rank-local precision-localization controls', flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if native is not None and native._pending is None:
            native.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
