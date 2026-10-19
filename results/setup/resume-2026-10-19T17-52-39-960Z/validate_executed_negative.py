from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json,write_jsonl
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
from swe_study.data import records
from swe_study.evaluation import validate_grader_report
instance='pvlib__pvlib-python-1048'
with Run('executed_grader_negative',{'instance_id':instance,'control':'gold patch with deliberately incorrect zero numerical derivative','benchmark_prediction':False,'agent_access_to_control':False}) as run:
    guard=StorageGuard();guard.check(2*10**9)
    row=next(row for row,_ in records('data/raw/swebench/dev') if row['instance_id']==instance)
    old='+from pvlib.ivtools import sde, sdm, utils  # noqa: F401'
    new='+from pvlib.ivtools import sde, sdm, utils; utils._numdiff = lambda x, f: (x * 0, f * 0)  # deliberate negative control'
    if row['patch'].count(old)!=1:raise ValueError('Reference patch changed; control needs review')
    patch=row['patch'].replace(old,new)
    dataset=run.path/'grading_dataset.json';predictions=run.path/'negative_predictions.jsonl'
    write_json(dataset,[row]);write_jsonl(predictions,[{'instance_id':instance,'model_name_or_path':'executed-negative-control','model_patch':patch}])
    run.command([root/'.envs/eval/bin/python',root/'scripts/swebench_entry.py','--dataset_name',dataset,'--predictions_path',predictions,'--max_workers','1','--run_id',run.id,'--instance_ids',instance],cwd=run.path,guard=guard,timeout=1200)
    reports=[]
    for path in run.path.rglob('*.json'):
        try:report=json.loads(path.read_text())
        except (ValueError,UnicodeError):continue
        if isinstance(report,dict) and 'resolved_ids' in report and 'total_instances' in report:reports.append((path,report))
    if len(reports)!=1:raise ValueError('Expected one official summary')
    summary=validate_grader_report(reports[0][1],{instance})
    if summary['resolved_ids'] or reports[0][1].get('empty_patch_ids'):raise ValueError('Negative control did not execute as an unresolved nonempty prediction')
    paths=list(run.path.rglob('report.json'))
    if len(paths)!=1:raise ValueError('Expected one per-task report')
    individual=read_json(paths[0])[instance];tests=individual['tests_status']['FAIL_TO_PASS']
    if individual.get('infra_failure') or not individual.get('patch_successfully_applied') or len(tests['failure'])!=1 or not tests['failure'][0].endswith('::test__numdiff') or len(tests['success'])!=4:raise ValueError('Unexpected executed negative outcomes')
    write_json(run.path/'negative_validation.json',{'passed':True,'control':'deliberately incorrect derivative in reference patch','instance_id':instance,'empty_prediction':False,'patch_applied':True,'actual_tests_passed':tests['success'],'actual_tests_failed':tests['failure'],'official_summary':str(reports[0][0]),'benchmark_score':False})
