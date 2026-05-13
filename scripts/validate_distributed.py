"""Validate Gloo or NCCL collectives against a CPU FP64 MoE oracle."""
import argparse
from datetime import timedelta
import hashlib
import json
import platform
from pathlib import Path
import tempfile
import traceback
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from moe_lab import AllToAllDispatcher
from moe_lab.reference import dense_reference, run_experts


def make_case(rank, world, scenario, dtype):
    generator = torch.Generator().manual_seed(1000 + rank)
    experts, hidden, outdim, topk = world*2, 7, 5, 3
    tokens = 6 + rank
    if scenario == 'empty_rank' and rank == 0 or scenario == 'all_empty':
        tokens = 0
    x = torch.randn(tokens, hidden, generator=generator).to(dtype)
    ids = torch.stack([torch.randperm(experts, generator=generator)[:topk] for _ in range(tokens)]) if tokens else torch.empty((0,topk),dtype=torch.int64)
    if scenario == 'duplicates':
        ids[:] = 1
    elif scenario == 'skewed':
        ids[:] = 0
    elif scenario == 'dropped':
        ids[:, 1] = -1
    elif scenario == 'all_dropped':
        ids[:] = -1
    weights = torch.rand(tokens, topk, generator=generator)
    weights /= weights.sum(1,keepdim=True)
    wg = torch.Generator().manual_seed(17)
    matrices = torch.randn(experts,hidden,outdim,generator=wg).to(dtype)/hidden
    biases = torch.randn(experts,outdim,generator=wg).to(dtype)
    return x,ids,weights,matrices,biases


def worker(rank, world, rendezvous, output, backend, combine_mode, known_output_size):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')
    if backend == 'nccl':
        torch.cuda.set_device(device)
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=world, timeout=timedelta(seconds=90))
    records = []
    try:
        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            for scenario in ['balanced','duplicates','skewed','dropped','all_dropped','empty_rank','all_empty']:
                x,ids,weights,matrices,biases = make_case(rank,world,scenario,dtype)
                oracle = dense_reference(x,ids,weights,matrices,biases).to(device)
                x,ids,weights,matrices,biases = [t.to(device) for t in (x,ids,weights,matrices,biases)]
                methods = []
                for optimized in [False,True]:
                    print(f"rank={rank} dtype={dtype} scenario={scenario} optimized={optimized}", flush=True)
                    dispatcher = AllToAllDispatcher(world*2, optimized=optimized,
                                                     combine_mode=combine_mode, known_output_size=known_output_size)
                    batch = dispatcher.dispatch(x,ids,weights)
                    # Check dispatch independently, including source/token/route order.
                    expected_rows = []
                    expected_labels = []
                    for local_id in range(2):
                        global_id = rank*2+local_id
                        for source_rank in range(world):
                            sx, si, _, _, _ = make_case(source_rank,world,scenario,dtype)
                            for token in range(sx.shape[0]):
                                for route in range(si.shape[1]):
                                    if int(si[token,route]) == global_id:
                                        expected_rows.append(sx[token])
                                        expected_labels.append(local_id)
                    expected_tokens = torch.stack(expected_rows) if expected_rows else x.new_empty((0,x.shape[1]))
                    torch.testing.assert_close(batch.tokens,expected_tokens.to(device),atol=0,rtol=0)
                    assert batch.local_expert_ids.tolist() == expected_labels
                    local = run_experts(batch,matrices,biases,rank,2)
                    result = dispatcher.combine(local,batch.context)
                    tolerance = {torch.float32:2e-6,torch.float16:8e-3,torch.bfloat16:6e-2}[dtype]
                    torch.testing.assert_close(result.double(),oracle,atol=tolerance,rtol=tolerance)
                    count = len(dispatcher.collectives)
                    assert count == (3 if optimized else 5), count
                    methods.append(result)
                    records.append({'rank':rank,'world_size':world,'dtype':str(dtype),'scenario':scenario,
                                    'optimized':optimized,'collectives':count,'passed':True,
                                    'max_abs_error':float((result.double()-oracle).abs().max()) if result.numel() else 0.,
                                    'atol':tolerance,'rtol':tolerance,
                                    'local_collective_payloads':dispatcher.collectives})
                # CUDA index_add_ may sum FP32 routes in different orders.
                # Keep transport rows exact and retain the independent FP64 oracle.
                comparison_tolerance = 0. if backend == 'gloo' or combine_mode == 'ordered' else tolerance
                torch.testing.assert_close(methods[0], methods[1],
                                           atol=comparison_tolerance, rtol=comparison_tolerance)
                delta = float((methods[0].float()-methods[1].float()).abs().max()) if methods[0].numel() else 0.
                for record in records[-2:]:
                    record.update(baseline_optimized_max_abs_error=delta,
                                  baseline_optimized_atol=comparison_tolerance,
                                  baseline_optimized_rtol=comparison_tolerance)
        # Outstanding dispatches retain independent contexts; combine in reverse order.
        x,ids,w,a,b = make_case(rank,world,'balanced',torch.float32)
        x,ids,w,a,b = [t.to(device) for t in (x,ids,w,a,b)]
        dispatcher = AllToAllDispatcher(world*2, combine_mode=combine_mode, known_output_size=known_output_size)
        first=dispatcher.dispatch(x,ids,w);second=dispatcher.dispatch(x+1,ids,w)
        for batch,source in [(second,x+1),(first,x)]:
            result=dispatcher.combine(run_experts(batch,a,b,rank,2),batch.context)
            torch.testing.assert_close(result.double(),dense_reference(*[t.cpu() for t in (source,ids,w,a,b)]).to(device),atol=2e-6,rtol=2e-6)
        # Swapped expert identities must fail the oracle even with identical routing.
        batch=dispatcher.dispatch(x,ids,w)
        wrong=dispatcher.combine(run_experts(batch,a.flip(0),b.flip(0),rank,2),batch.context)
        assert not torch.allclose(wrong.double(),dense_reference(*[t.cpu() for t in (x,ids,w,a,b)]).to(device),atol=2e-6,rtol=2e-6)
        Path(output,f'rank-{rank}.json').write_text(json.dumps({'cases':records,'independent_contexts_passed':True,'wrong_expert_negative_control_detected':True},indent=2))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backend',choices=['gloo','nccl'],default='gloo');parser.add_argument('--world-size',type=int,default=2);parser.add_argument('--output',type=Path,default=Path('results/gloo-validation.json'))
    parser.add_argument('--combine-mode', choices=['atomic', 'ordered'], default='atomic')
    parser.add_argument('--known-output-size', action='store_true')
    args=parser.parse_args()
    if args.world_size < 2:raise ValueError('Use at least two processes')
    if args.output.exists():raise FileExistsError('Choose a fresh result path')
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker,args=(args.world_size,'file://'+tmp+'/rendezvous',tmp,args.backend,args.combine_mode,args.known_output_size),nprocs=args.world_size,join=True)
        ranks=[json.loads(Path(tmp,f'rank-{r}.json').read_text()) for r in range(args.world_size)]
    root=Path(__file__).resolve().parents[1]
    payload={'backend':args.backend,'device':'cuda' if args.backend=='nccl' else 'cpu','torch':torch.__version__,'platform':platform.system(),
             'combine_mode':args.combine_mode,'known_output_size':args.known_output_size,
             'world_size':args.world_size,'cases':[c for r in ranks for c in r['cases']],
             'independent_contexts_passed':all(r['independent_contexts_passed'] for r in ranks),
             'wrong_expert_negative_control_detected':all(r['wrong_expert_negative_control_detected'] for r in ranks),
             'source_hashes':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['moe_lab','scripts'] for p in sorted((root/folder).glob('*.py'))}}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(payload,indent=2)+'\n')
    print(f"PASS: {len(payload['cases'])} rank-local numerical cases; independent contexts and negative controls pass")


if __name__=='__main__':main()
