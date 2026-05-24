"""Reject foreign or incorrectly-strided zero-copy views before buffer mutation."""
import argparse
from datetime import timedelta
import json
import hashlib
import os
from pathlib import Path
import torch
import torch.distributed as dist
from moe_lab.deepep_backend import DeepEPLowLatencyDispatcher
from moe_lab.dispatch import AllToAllDispatcher
from scripts.validate_deepep import inputs, expert_weights, experts, oracle


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
    native = DeepEPLowLatencyDispatcher(world*4, 2048, packing='zero_copy', completion='event')
    try:
        cx, ci, cw = inputs(rank, world, 2048, 'balanced')
        cpu_weights = expert_weights(world, 2048)
        matrices = [w.cuda() for w in cpu_weights]
        x, ids, weights = [t.cuda() for t in (cx, ci, cw)]
        expected = oracle(cx, ci, cw, cpu_weights).cuda()
        reference = AllToAllDispatcher(world*4)
        rb = reference.dispatch(x, ids, weights)
        nb = native.dispatch(x, ids, weights)
        packed = native.buffer.get_next_low_latency_combine_buffer(nb.context.handle)
        assert not packed.is_contiguous(), 'This regression requires a padded registered view'
        a,b,c = packed.shape
        wrong_stride = packed.as_strided(packed.shape, (packed.stride(0), c, 1))
        foreign = torch.empty_strided(packed.shape, packed.stride(), dtype=packed.dtype, device=packed.device)
        cases = [('foreign_strided_view', foreign, True, 'x.data_ptr()'),
                 ('wrong_stride', wrong_stride, True, 'x.stride(2)'),
                 ('wrong_dtype', packed.float(), True, 'x.scalar_type()'),
                 ('ordinary_noncontiguous', packed, False, 'x.is_contiguous()')]
        records = []
        for name, candidate, zero_copy, guard in cases:
            try:
                native.buffer.low_latency_combine(candidate, nb.context.expert_ids, nb.context.weights,
                                                  nb.context.handle, zero_copy=zero_copy)
            except RuntimeError as error:
                assert guard in str(error), str(error)
                records.append({'rank':rank, 'case':name, 'rejected':True, 'expected_guard':guard})
            else:
                raise AssertionError(f'{name} was accepted')
        # Rejection must not advance the internal alternating-buffer index.
        reference_out = reference.combine(experts(rb, matrices, rank), rb.context)
        actual = native.combine(experts(nb, matrices, rank), nb.context)
        torch.testing.assert_close(actual, reference_out, atol=.03, rtol=.03)
        torch.testing.assert_close(actual.double(), expected, atol=.06, rtol=.06)
        records.append({'rank':rank, 'case':'valid_after_rejections', 'passed':True,
                        'registered_stride':list(packed.stride()), 'shape':list(packed.shape)})
        # Preserve the upstream convention: populate registered storage, then pass
        # a contiguous placeholder whose values are not read in zero-copy mode.
        nb = native.dispatch(x, ids, weights)
        values = experts(nb, matrices, rank)
        registered = native.buffer.get_next_low_latency_combine_buffer(nb.context.handle)
        offset = 0
        for expert, count in enumerate(nb.context.counts):
            registered[expert, :count] = values[offset:offset+count]
            offset += count
        placeholder = values.new_empty(nb.context.packed_shape)
        legacy, _, _ = native.buffer.low_latency_combine(placeholder, nb.context.expert_ids,
                                                        nb.context.weights, nb.context.handle,
                                                        zero_copy=True)
        native._pending = None  # The raw compatibility call consumed this context.
        torch.testing.assert_close(legacy.double(), expected, atol=.06, rtol=.06)
        records.append({'rank':rank, 'case':'legacy_contiguous_placeholder', 'passed':True})
        gathered = [None] * world
        dist.all_gather_object(gathered, records)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'native_build':json.loads(Path('results/deepep-build-manifest.json').read_text()), 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases':[r for rs in gathered for r in rs]},indent=2)+'\n')
            print(f'PASS: {len(records)*world} zero-copy contract controls', flush=True)
    finally:
        if native._pending is None:
            native.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
