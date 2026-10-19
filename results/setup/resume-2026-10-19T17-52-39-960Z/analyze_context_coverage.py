from pathlib import Path
import sys,math
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,read_jsonl,write_json,sha256_file
setup_environment()
from swe_study.audit import Run
with Run('context_coverage_analysis',{'mode':'derive full-history coverage from existing exact 8K tokenization; no truncation or GPU probe'}) as run:
    source=root/'results/tokenize/20261019T212832-1d649e8b/token_stats.json';stats=read_json(source);lengths={};inputs=[source]
    for split,entry in stats['files'].items():
        path=root/entry['output'];inputs.append(path)
        if sha256_file(path)!=entry['output_sha256']:raise ValueError('Retained tokenized data changed')
        lengths[split]=[len(row['input_ids']) for _,row in read_jsonl(path)]
    for row in stats['exclusions']:
        if not row['reason'].startswith('over_context:'):raise ValueError('Other exclusion prevents exact length reconstruction')
        lengths[row['split']].append(int(row['reason'].split(':')[1]))
    if len(lengths['train'])!=439 or len(lengths['validation'])!=50:raise ValueError('Full retained corpus not reconstructed')
    coverage=[]
    for limit in [4096,8192,16384,32768,65536]:
        counts={split:sum(value<=limit for value in values) for split,values in lengths.items()};total=sum(counts.values())
        coverage.append({'context_tokens':limit,**counts,'total':total,'fraction_of_489':total/489,'gpu_feasibility':'measured NF4/LoRA pilot/probes at 8K' if limit==8192 else 'not established by this length-only analysis'})
    ordered=sorted(sum(lengths.values(),[]))
    report={'passed':True,'total_histories':489,'coverage':coverage,'all_history_p50_tokens':ordered[len(ordered)//2],'all_history_p95_tokens':ordered[math.ceil(len(ordered)*.95)-1],'max_tokens':max(ordered),'source_hashes':{str(p.relative_to(root)):sha256_file(p) for p in inputs},'policy_changed':False,'gpu_probes_performed':False}
    write_json(run.path/'coverage.json',report);print(report['coverage'])
