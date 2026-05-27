"""Validate inclusive native token capacity and recover after overflow rejection."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from moe_lab.deepep_backend import DeepEPLowLatencyDispatcher
from moe_lab.expert_mlp import evaluate_experts_into
from scripts.validate_deepep import inputs, expert_weights, oracle


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh output path')
    rank,world=int(os.environ['RANK']),int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=90))
    capacity,hidden=128,2048
    native=DeepEPLowLatencyDispatcher(world*4,hidden,max_tokens=capacity,completion='event',
                                      packing='zero_copy',allow_nvlink=False)
    records=[]
    try:
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            cx,ci,cw=inputs(rank,world,hidden,'balanced',tokens_base=capacity-rank*3)
            assert len(cx)==capacity
            x,ids,weights=[t.cuda() for t in (cx,ci,cw)]
            try:
                native.dispatch_packed(torch.cat([x,x[:1]]),torch.cat([ids,ids[:1]]),torch.cat([weights,weights[:1]]))
            except ValueError:
                assert native._pending is None
                records.append({'rank':rank,'case':'capacity_plus_one_rejected','tokens':capacity+1,'passed':True})
            else:raise AssertionError('Capacity overflow was accepted')
            cpu_matrices=expert_weights(world,hidden)
            matrices=[w.cuda() for w in cpu_matrices]
            for scenario in ('balanced','skewed','all_dropped','all_empty'):
                cx,ci,cw=inputs(rank,world,hidden,scenario,tokens_base=capacity-rank*3)
                counts=(capacity-1,capacity) if len(cx) else (0,)
                for count in counts:
                    tx,ti,tw=cx[:count],ci[:count],cw[:count]
                    expected=oracle(tx,ti,tw,cpu_matrices).cuda()
                    batch=native.dispatch_packed(tx.cuda(),ti.cuda(),tw.cuda())
                    registered=native.registered_output(batch.context)
                    evaluate_experts_into(batch,matrices,rank,registered)
                    actual=native.combine_registered(registered,batch.context)
                    torch.testing.assert_close(actual.double(),expected,atol=.06,rtol=.06)
                    records.append({'rank':rank,'case':scenario,'tokens':len(tx),'capacity':capacity,'passed':True})
        stream.synchronize()
        gathered=[None]*world
        dist.all_gather_object(gathered,records)
        if rank==0:
            root=Path(__file__).resolve().parents[1]
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps({'world_size':world,'capacity':capacity,'capacity_is_inclusive':True,
                                              'native_build':json.loads((root/'results/deepep-build-manifest.json').read_text()),
                                              'adapter_sha256':hashlib.sha256((root/'moe_lab/deepep_backend.py').read_bytes()).hexdigest(),
                                              'cases':[r for rs in gathered for r in rs]},indent=2)+'\n')
            print(f'PASS: {len(records)*world} capacity boundary controls',flush=True)
    finally:
        if native._pending is None:native.close()
        dist.destroy_process_group()


if __name__=='__main__':main()
