"""Aggregate fresh-process medians without pooling samples across runs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(root):
    rows, sources = [], {}
    for world in (2, 4, 8):
        paths = [root/f'pipeline-packed-world{world}-rep{rep}.json' for rep in (1, 2, 3)]
        payloads = [json.loads(p.read_text()) for p in paths]
        for path in paths:
            sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        keys = [(r['hidden'], r['tokens_rank_zero'], r['scenario']) for r in payloads[0]['records']]
        if any([(r['hidden'],r['tokens_rank_zero'],r['scenario']) for r in p['records']] != keys for p in payloads):
            raise ValueError('Fresh-process workload grids differ')
        for index, (hidden, tokens, scenario) in enumerate(keys):
            records = [p['records'][index] for p in payloads]
            methods = {}
            for method in payloads[0]['methods']:
                latencies = [r['summary'][method]['median_ms'] for r in records]
                median = statistics.median(latencies)
                methods[method] = {'median_ms':median, 'process_medians_ms':latencies,
                                   'range_min_ms':min(latencies), 'range_max_ms':max(latencies),
                                   'global_input_tokens_per_second':records[0]['total_input_tokens']*1000/median}
            row = {'world_size':world, 'hidden':hidden, 'tokens_rank_zero':tokens, 'scenario':scenario,
                   'total_input_tokens':records[0]['total_input_tokens'], 'experts':records[0]['experts'],
                   'top_k':records[0]['top_k'], 'expert_intermediate':records[0]['expert_intermediate'],
                   'samples_per_method_per_process_launch':records[0]['samples_per_method'], 'methods':methods}
            baseline = methods['baseline_grouped']['median_ms']
            previous = methods['native_grouped']['median_ms']
            packed = methods['native_packed']['median_ms']
            row['three_collective_latency_reduction_percent'] = 100*(1-methods['optimized_grouped']['median_ms']/baseline)
            row['packed_vs_previous_native_latency_reduction_percent'] = 100*(1-packed/previous)
            row['packed_vs_baseline_speedup'] = baseline/packed
            rows.append(row)
    return {'aggregation':'Median of three fresh-process medians; whiskers span these three medians, not a confidence interval',
            'measurement':'Worst-rank synchronized host wall time for complete dispatch, BF16 expert MLP and combine; control barriers excluded',
            'scaling':'Per-rank tokens and experts held fixed as world size increases; total tokens and expert count increase',
            'source_sha256':sources, 'records':rows}


def markdown(payload):
    lines = ['# Pipeline measurements', '', payload['aggregation']+'.', '',
             payload['measurement']+'.', '',
             'Each rank r receives N + 3r input tokens. Top-k is two, with four experts per rank and an expert intermediate width of 128.', '']
    for world in (2, 4, 8):
        lines += [f'<img src="../assets/tables/measurements-{world}gpu.png" alt="{world}-GPU measurements" width="940">', '']
    lines += ['Packed reduction compares the packed-input/registered-output path with the compact native adapter in the same process launches.',
              '', 'Native transport permits NVLink in these runs. RDMA-only transport is measured in a separate run.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=Path('results'))
    parser.add_argument('--output',type=Path,default=Path('results/pipeline-summary.json'))
    parser.add_argument('--table',type=Path,default=Path('docs/MEASUREMENTS.md'))
    args = parser.parse_args()
    payload = summarize(args.results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.table.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2)+'\n')
    args.table.write_text(markdown(payload))


if __name__ == '__main__':
    main()
