"""Record audited NERSC observations without replacing historical host state."""
import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import read_json,write_json,sha256_file


def main():
    review_path=ROOT/'results/nersc_results_review/20261124T032614-36127ffd/review.json'
    review=read_json(review_path)
    imported=ROOT/review['source']
    with Run('nersc_observations_documentation',{'review':str(review_path.relative_to(ROOT)),
                                               'export_as_of':'2026-11-24T03:16:13+00:00'}) as run:
        shutil.copyfile(__file__,run.path/'record_driver.py')
        assert review['all_consistency_checks_passed'] and len(review['checks'])==63
        models=[]
        for entry in review['training']:
            meta_path=imported/'training'/entry['adapter']/'experiment.json'
            meta=read_json(meta_path)
            comparisons=[(p,read_json(p)) for p in (imported/'cloud_validation').glob('*/comparison.json')
                         if read_json(p)['adapter']==meta['config']['output'] and
                         read_json(p.parent/'run.json')['status']=='passed']
            path,comparison=max(comparisons,key=lambda x:x[0].parent.name)
            models.append({'model':meta['config']['model'],'adapter':meta['config']['output'],
                           'training_complete':True,'epochs':meta['config']['num_train_epochs'],
                           'optimizer_steps':meta['global_step'],'training_gpu_count':meta['gpu_count'],
                           'effective_batch_size':meta['effective_batch_size'],'max_training_context':meta['config']['max_length'],
                           'train_histories':meta['train_examples'],'validation_histories':meta['validation_examples'],
                           'base_validation_loss':comparison['base_validation_loss'],
                           'adapter_validation_loss':comparison['adapter_validation_loss'],
                           'relative_loss_reduction_percent':comparison['relative_loss_reduction_percent'],
                           'validation_status':'passed','direct_training_loss_match':comparison['adapter_matches_recorded_loss'],
                           'training_metric_reproduction':comparison.get('training_metric_reproduction'),
                           'final_resume_step':entry['resumed_at_step'],'steps_in_final_segment':entry['steps_in_final_segment'],
                           'reported_final_segment_gpu_hours':entry['reported_gpu_hours'],
                           'total_training_gpu_hours':None,
                           'cost_scope':'Final successful invocation only; not allocation or full-attempt accounting',
                           'training_metadata':str(meta_path.relative_to(ROOT)),
                           'validation_comparison':str(path.relative_to(ROOT)),
                           'adapter_weights_sha256_reported':comparison['adapter_weights_sha256']})
        base_ledger=review['benchmark']['ledgers']['base']['completion_tokens_by_task']
        adapter_ledger=review['benchmark']['ledgers']['adapter']['completion_tokens_by_task']
        record={
            'schema_version':1,'record_type':'nersc_export_snapshot',
            'as_of':'2026-11-24T03:16:13+00:00',
            'documented_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status':'three_models_trained_and_validated_development_partial',
            'source_export':review['source'],'source_git_commit':'2b5d05b',
            'review':str(review_path.relative_to(ROOT)),'review_sha256':sha256_file(review_path),
            'imported_json_files':review['imported_json_files'],'consistency_checks_passed':len(review['checks']),
            'models':models,
            'data':{'raw_train_histories':439,'raw_train_distinct_issues':263,
                    'raw_validation_histories':50,'raw_validation_distinct_issues':29,
                    'retained_train_histories':400,'retained_validation_histories':43,
                    'retained_validation_distinct_issues':25,'validation_histories_excluded_over_context':7,
                    'validation_selection_sha256':'a0e4b56a761399750ffea62da861cb7d7194f4ba5c34bce1d79a90f164b9a81d',
                    'local_processed_data_hashes_match':True,
                    'train_validation_benchmark_issue_sets_disjoint':True,
                    'aggregation':'Mean per history; repeated issues receive proportionally more history weight'},
            'development':{'state':'results/benchmark_15b_window_v1','model':'1.5B','frozen_tasks':5,
                           'episodes_required':10,'prepared_tasks':review['benchmark']['prepared_tasks'],
                           'graded_episodes':sum(x['completed'] for k,x in review['benchmark']['summary'].items() if k in ('base','adapter')),
                           'completed_task_pairs':1,'resolved_task_pairs':0,
                           'complete':False,'valid_comparison':False,
                           'episodes':review['benchmark']['episodes'],
                           'completion_tokens':{'base':base_ledger,'adapter':adapter_ledger},
                           'agent_context_tokens':8192,'context_policy':'history-window-v1',
                           'max_steps':30,'max_seconds_per_episode':1800,
                           'max_completion_tokens_per_episode':32768,
                           'three_b_and_seven_b_agent_results_in_export':False},
            'final_evaluation':{'frozen_tasks':100,'completion_established_by_export':False},
            'local_regression':{'tests_passed':91,'evidence':'results/cloud_evaluation_validation/20261124T023913-fa98fc57/validation.json',
                                'scope':'CPU compatibility and reporting; not a count of real benchmark successes'},
            'installer_validation':'results/evaluation_installer_validation/20261124T023944-c2a2b9d5/validation.json',
            'durable_archive_transfer_verified_by_export':False,
            'active_remote_jobs':'Not established by the exported snapshot',
            'remaining':['Prepare remaining three development task environments',
                         'Complete remaining eight 1.5B development episodes and grade them',
                         'Run separate 3B and 7B paired development studies',
                         'Aggregate complete training attempts and Slurm resource charges',
                         'Validate cleanup isolation before concurrent agent controllers share a daemon',
                         'Complete each model development gate before final 100-task evaluation',
                         'Confirm verified durable preservation beyond scratch staging'],
            'observations':review['limitations_and_reporting_issues'],
            'documentation':'docs/nersc_results_and_challenges.md',
        }
        write_json('manifests/nersc_status.json',record)
        historical=read_json('manifests/implementation_status.json')
        historical['record_type']='historical_workstation_snapshot'
        historical['current_nersc_status']='manifests/nersc_status.json'
        write_json('manifests/implementation_status.json',historical)
        write_json(run.path/'recorded_observations.json',record)
        print(json.dumps({'status':record['status'],'models':len(models),
                          'checks':len(review['checks']),'observations':'manifests/nersc_status.json',
                          'challenges_document':record['documentation']},indent=2))


if __name__=='__main__':main()
