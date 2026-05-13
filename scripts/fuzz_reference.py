"""Randomize routing, feature widths and context snapshots against a direct oracle."""
import argparse
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from moe_lab.dispatch import AllToAllDispatcher
from moe_lab.reference import dense_reference, run_experts


def make_case(seed, rank, world):
    config = torch.Generator().manual_seed(719+seed)
    local_experts = (1,2,4)[seed%3]
    hidden, outdim = (1,7,33,64)[seed%4], (1,5,13)[seed%3]
    topk = (1,2,5)[seed%3]
    dtype = (torch.float32,torch.float16,torch.bfloat16)[seed%3]
    gen = torch.Generator().manual_seed(1009+seed*101+rank)
    tokens = int(torch.randint(0,25,(),generator=gen))
    if seed%17 == 0 or seed%7 == rank:
        tokens=0
    x=torch.randn(tokens,hidden,generator=gen).to(dtype)
    ids=torch.randint(world*local_experts,(tokens,topk),generator=gen)
    if seed%11 == 0: ids.fill_(0)
    ids[torch.rand(tokens,topk,generator=gen)<.2]=-1
    if seed%23 == 0: ids.fill_(-1)
    weights=torch.rand(tokens,topk,generator=gen)
    weights/=weights.sum(1,keepdim=True)
    matrix=torch.randn(world*local_experts,hidden,outdim,generator=config).to(dtype)/max(hidden,1)
    bias=torch.randn(world*local_experts,outdim,generator=config).to(dtype)
    return x,ids,weights,matrix,bias,local_experts


def worker(rank,world,rendezvous,temporary,seeds):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=world,timeout=timedelta(seconds=120))
    records=[]
    try:
        for seed in range(seeds):
            x,ids,weights,matrix,bias,local_experts=make_case(seed,rank,world)
            expected=dense_reference(x,ids,weights,matrix,bias)
            expected_tokens,expected_labels=[],[]
            for local in range(local_experts):
                for source in range(world):
                    sx,si,*_=make_case(seed,source,world)
                    for token in range(len(sx)):
                        for slot in range(si.shape[1]):
                            if int(si[token,slot]) == rank*local_experts+local:
                                expected_tokens.append(sx[token])
                                expected_labels.append(local)
            rows=torch.stack(expected_tokens) if expected_tokens else x.new_empty((0,x.shape[1]))
            outputs=[]
            for optimized in (False,True):
                dispatcher=AllToAllDispatcher(world*local_experts,optimized=optimized,
                                               known_output_size=seed%3==0,host_expert_counts=seed%2==0,
                                               combine_mode='ordered' if seed%2 else 'atomic')
                mutable_weights=weights.clone()
                batch=dispatcher.dispatch(x,ids,mutable_weights)
                mutable_weights.fill_(float('nan'))
                torch.testing.assert_close(batch.tokens,rows,atol=0,rtol=0)
                assert batch.local_expert_ids.tolist()==expected_labels
                actual=dispatcher.combine(run_experts(batch,matrix,bias,rank,local_experts),batch.context)
                tol={torch.float32:2e-6,torch.float16:8e-3,torch.bfloat16:6e-2}[x.dtype]
                torch.testing.assert_close(actual.double(),expected,atol=tol,rtol=tol)
                assert len(dispatcher.collectives)==(3 if optimized else 5)
                outputs.append(actual)
                records.append({'rank':rank,'seed':seed,'tokens':len(x),'hidden':x.shape[1],
                                'outdim':matrix.shape[2],'top_k':ids.shape[1],'local_experts':local_experts,
                                'dtype':str(x.dtype),'optimized':optimized,'passed':True,
                                'caller_weight_mutation_ignored':True,'dispatch_exact':True,
                                'max_abs_error':float((actual.double()-expected).abs().max()) if actual.numel() else 0.})
            torch.testing.assert_close(outputs[0],outputs[1],atol=0,rtol=0)
        Path(temporary,f'rank-{rank}.json').write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--world-size',type=int,default=3)
    parser.add_argument('--seeds',type=int,default=200)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Choose a fresh output path')
    with tempfile.TemporaryDirectory() as temporary:
        mp.spawn(worker,args=(args.world_size,'file://'+temporary+'/rendezvous',temporary,args.seeds),nprocs=args.world_size,join=True)
        records=[r for rank in range(args.world_size) for r in json.loads(Path(temporary,f'rank-{rank}.json').read_text())]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    payload={'scope':'CPU/Gloo randomized routing and context-snapshot validation','torch':torch.__version__,
             'world_size':args.world_size,'seeds':args.seeds,'cases':records,
             'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in ('scripts','moe_lab') for p in sorted((root/folder).glob('*.py'))}}
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(f'PASS: {len(records)} randomized rank-local cases',flush=True)


if __name__=='__main__':main()
