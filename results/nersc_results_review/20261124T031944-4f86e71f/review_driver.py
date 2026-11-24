"""Read-only numerical and coverage audit of the imported NERSC snapshot."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import canonical_hash,read_json,sha256_file,write_json
from swe_study.validation import recorded_trainer_loss
from swe_study.benchmark import study_summary


def close(a,b):return math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-7)


def main():
    source=ROOT/'results/nersc_uploaded/20261124T031613Z'
    with Run('nersc_results_review',{'import_directory':str(source.relative_to(ROOT)),
                                  'scope':'Read imported JSON; no GPU calls or source record changes'}) as run:
        shutil.copyfile(__file__,run.path/'review_driver.py')
        files=sorted(source.rglob('*.json'))
        hashes={str(p.relative_to(source)):sha256_file(p) for p in files}
        checks=[];warnings=[]
        def check(name,passed,**details):
            checks.append({'check':name,'passed':bool(passed),**details})
        training={p.parent.name:read_json(p) for p in (source/'training').glob('*/experiment.json')}
        train_reports=[]
        for key,meta in sorted(training.items()):
            cfg=meta['config'];logs=meta['log_history']
            losses=[x for x in logs if 'loss' in x]
            epochs=[x for x in logs if 'eval_loss' in x]
            check(key+': complete training steps',meta['global_step']==150 and sorted(x['step'] for x in losses)==list(range(1,151)))
            check(key+': finite logged losses',all(math.isfinite(x['loss']) and x['loss']>=0 for x in losses))
            check(key+': finite nonzero adapter gradients',meta['adapter_updated'] and meta['adapter_hash_before']!=meta['adapter_hash_after'] and all(math.isfinite(x) and x>0 for x in meta['gradient_checks']))
            check(key+': batch and epoch accounting',meta['effective_batch_size']==8==cfg['effective_batch_size']==meta['gpu_count']*cfg['gradient_accumulation_steps'] and meta['train_examples']*cfg['num_train_epochs']/8==meta['global_step'])
            check(key+': selected final checkpoint is best',meta['best_eval_loss']==min(x['eval_loss'] for x in epochs) and Path(meta['best_checkpoint']).name=='checkpoint-150')
            check(key+': epoch coverage',[(x['epoch'],x['step']) for x in epochs]==[(1,50),(2,100),(3,150)])
            start=int(re.search(r'checkpoint-(\d+)$',meta['resume_from_checkpoint'])[1]) if meta.get('resume_from_checkpoint') else 0
            remaining=meta['global_step']-start
            check(key+': segment token count',meta['token_counts']['microbatches']==remaining*8)
            check(key+': segment GPU-hour arithmetic',close(meta['gpu_hours'],meta['wall_seconds_including_eval_and_checkpointing']*meta['gpu_count']/3600))
            item={'adapter':key,'epochs':[{'epoch':x['epoch'],'step':x['step'],'eval_loss':x['eval_loss']} for x in epochs],
                  'resumed_at_step':start,'steps_in_final_segment':remaining,
                  'reported_gpu_hours':meta['gpu_hours'],'reported_train_loss':meta['training_result']['train_loss'],
                  'mean_logged_step_loss_full_history':statistics.mean(x['loss'] for x in losses),
                  'mean_logged_step_loss_final_segment':statistics.mean(x['loss'] for x in losses if x['step']>start),
                  'epoch_2_to_3_relative_eval_improvement_percent':100*(epochs[1]['eval_loss']-epochs[2]['eval_loss'])/epochs[1]['eval_loss']}
            train_reports.append(item)
            if start:
                warnings.append(key+': GPU hours, wall time, peak memory and token counters describe the last resumed segment; aggregate previous and failed runs for total costs.')
                warnings.append(key+': Trainer train_loss after resume is not a full-training mean; compare per-step history or held-out evaluation instead.')
        check('identical training source across sizes',len({x['train_source_sha256'] for x in training.values()})==1)
        check('identical validation source across sizes',len({x['validation_source_sha256'] for x in training.values()})==1)
        validations=[];latest={}
        for path in sorted((source/'cloud_validation').glob('*/comparison.json')):
            report=read_json(path);meta=read_json(path.parent/'run.json')
            identity=read_json(path.parent/'evaluation_identity.json')
            selection=read_json(path.parent/'validation_selection.json')
            check(path.parent.name+': selection identity',canonical_hash(selection['examples'])==report['selection_sha256'])
            check(path.parent.name+': comparison identity',all(report[k]==v for k,v in identity.items()))
            base=report['base_validation_loss'];adapter=report['adapter_validation_loss']
            check(path.parent.name+': reduction arithmetic',close(report['adapter_minus_base'],adapter-base) and close(report['relative_loss_reduction_percent'],100*(base-adapter)/base))
            check(path.parent.name+': held-out context bounds',len(selection['examples'])==report['examples']==43 and all(0<x['target_tokens']<=x['processed_tokens']<=32768 for x in selection['examples']))
            experiment=training[Path(report['adapter']).name]
            check(path.parent.name+': training reference loss',report['recorded_training_best_eval_loss']==experiment['best_eval_loss'])
            item={'run':path.parent.name,'model':report['model'],'status':meta['status'],'base_loss':base,'adapter_loss':adapter,
                  'relative_reduction_percent':report['relative_loss_reduction_percent'],
                  'selection_sha256':report['selection_sha256'],'direct_training_match':report['adapter_matches_recorded_loss']}
            history=path.parent/'adapter_history_losses.json'
            if history.exists():
                adapted_hist=read_json(history);base_hist=read_json(path.parent/'base_history_losses.json')
                check(path.parent.name+': paired per-history coverage',len(adapted_hist)==len(base_hist)==43 and
                      all(a['history_index']==b['history_index']==i and a['target_tokens']==b['target_tokens']==selection['examples'][i]['target_tokens'] for i,(a,b) in enumerate(zip(adapted_hist,base_hist))))
                check(path.parent.name+': per-history means',close(statistics.mean(x['loss'] for x in adapted_hist),adapter) and close(statistics.mean(x['loss'] for x in base_hist),base))
                reconstruction=recorded_trainer_loss(adapted_hist,experiment['gpu_count'])
                reference=report['training_metric_reproduction']
                check(path.parent.name+': distributed training metric reconstruction',reference['passed'] and close(reconstruction,reference['reconstructed_training_aggregation_loss']) and close(reconstruction,experiment['best_eval_loss']))
                item.update(reconstructed_training_loss=reconstruction,
                            absolute_training_reconstruction_error=abs(reconstruction-experiment['best_eval_loss']),
                            histories_with_lower_adapter_loss=sum(a['loss']<b['loss'] for a,b in zip(adapted_hist,base_hist)))
            validations.append(item)
            if meta['status']=='passed':latest[report['model']]=(report,path,selection)
        check('three model sizes have passed validation',len(latest)==3)
        check('same retained validation histories across sizes',len({v[0]['selection_sha256'] for v in latest.values()})==1)
        selection=next(iter(latest.values()))[2]
        selection_ids={r['instance_id'] for r in selection['examples']}
        check('validation histories have distinct issue IDs',len(selection_ids)==43)
        data_info={}
        for split,hash_name in [('train','train_source_sha256'),('validation','validation_source_sha256')]:
            path=ROOT/'data/processed/swegym'/(split+'.jsonl')
            data=path.read_bytes();digest=hashlib.sha256(data).hexdigest()
            rows=[json.loads(line) for line in data.splitlines() if line.strip()]
            data_info[split]={'local_sha256':digest,'matches_imported_hash':all(x[hash_name]==digest for x in training.values()),
                              'raw_history_count':len(rows),'instance_ids':sorted({r['instance_id'] for r in rows})}
        data_match=all(x['matches_imported_hash'] for x in data_info.values())
        if data_match:
            train_ids=set(data_info['train']['instance_ids']);val_ids=set(data_info['validation']['instance_ids'])
            check('local data confirms task-disjoint training and validation',not train_ids & val_ids)
            check('local data confirms validation selection/exclusions',selection_ids<=val_ids and len(selection['excluded'])+43==data_info['validation']['raw_history_count'])
        else:warnings.append('Local processed data hashes differ; imported file identities are consistent, but source-data identity was not independently verified from this local copy.')
        state=source/'benchmark_15b_window_v1';protocol=read_json(state/'protocol.json')
        ids=protocol['frozen_development_ids']
        calculated=study_summary(state,'development',ids);stored=read_json(state/'development/summary.json')
        check('benchmark summary matches saved episode records',calculated==stored)
        check('benchmark frozen denominators',len(ids)==5 and len(set(protocol['frozen_final_ids']))==100 and not set(ids)&set(protocol['frozen_final_ids']))
        onefive=latest['models/Qwen2.5-Coder-1.5B-Instruct'][0]
        check('benchmark and validation name the same adapter weights',protocol['adapter_weights_sha256']==onefive['adapter_weights_sha256'])
        prepared=[];episodes=[]
        for task in ids:
            folder=state/'development/tasks'/task
            if (folder/'prepared.json').exists():
                prep=read_json(folder/'prepared.json');prepared.append(task)
                check(task+': preparation accepted',prep['passed'] and prep['gold_passed'])
            for condition in ('base','adapter'):
                if not (folder/(condition+'.json')).exists():continue
                record=read_json(folder/(condition+'.json'))
                check(task+'/'+condition+': paired environment identity',record['prepared_image_identity']==read_json(folder/'prepared.json'))
                patch=record.get('prediction',{}).get('model_patch','')
                episodes.append({'task':task,'condition':condition,'status':record['status'],
                                 'resolved':record.get('resolved'),'agent_error':record.get('agent_error'),
                                 'empty_patch':not patch.strip(),'patch_lines':len(patch.splitlines()),
                                 'placeholder_implementation':'Placeholder for actual WVM implementation' in patch})
        ledgers={c:read_json(state/'development/token_ledgers'/(c+'_task_tokens.json')) for c in ('base','adapter')}
        check('paired ledger identities and bounds',len({v['tasks_sha256'] for v in ledgers.values()})==1 and
              all(v['limit_per_task']==32768 and set(v['completion_tokens_by_task'])==set(ids) and
                  all(0<=x<=32768 for x in v['completion_tokens_by_task'].values()) for v in ledgers.values()))
        if data_match:
            check('local data confirms no training or validation overlap with benchmark tasks',
                  not (set(data_info['train']['instance_ids'])|set(data_info['validation']['instance_ids'])) & (set(ids)|set(protocol['frozen_final_ids'])))
        warnings.extend(['Only 43 of 50 original validation histories fit the recorded 32K context; the seven exclusions are consistent across model sizes.',
                         'Validation was used to select checkpoints; these losses do not establish SWE-bench issue-resolution improvement.',
                         'Raw NERSC adapter/model weights, runtime images, full agent event logs and official grading reports were not included; this audit checks uploaded records and arithmetic.'])
        report={'source':str(source.relative_to(ROOT)),'imported_json_files':len(files),'input_sha256':hashes,
                'all_consistency_checks_passed':all(x['passed'] for x in checks),'checks':checks,
                'training':train_reports,'validations':validations,'local_data':data_info,
                'benchmark':{'prepared_tasks':prepared,'episodes':episodes,'summary':calculated,'ledgers':ledgers},
                'limitations_and_reporting_issues':warnings}
        write_json(run.path/'review.json',report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('input_sha256','checks','local_data')},indent=2))
        print('Checks:',len(checks),'Failures:',[x['check'] for x in checks if not x['passed']])
        print('Report:',run.path/'review.json')


if __name__=='__main__':main()
