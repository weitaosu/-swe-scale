from __future__ import annotations
import argparse
import json
from .audit import Run
from .common import local_path,write_json


def main():
    parser=argparse.ArgumentParser(description="Audited CS6158 local SWE pipeline")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("preflight");sub.add_parser("checkout");sub.add_parser("select-tasks");sub.add_parser("prepare");sub.add_parser("inventory")
    docker=sub.add_parser("docker-storage");docker.add_argument("--backing-file")
    download=sub.add_parser("download");download.add_argument("--kind",choices=["all","model","dataset"],default="all");download.add_argument("--only")
    tokenize=sub.add_parser("tokenize");tokenize.add_argument("--model",required=True);tokenize.add_argument("--max-length",type=int,default=4096)
    train=sub.add_parser("train");train.add_argument("--config",default="configs/qlora_pilot.yaml");train.add_argument("--resume")
    val=sub.add_parser("validate-checkpoint");val.add_argument("--adapter",required=True)
    serve=sub.add_parser("serve");serve.add_argument("--model",required=True);serve.add_argument("--adapter");serve.add_argument("--port",type=int,default=8615);serve.add_argument("--max-context",type=int,default=8192)
    serve.add_argument("--task-budget-file",default="data/metadata/dev_task_budgets.jsonl");serve.add_argument("--task-token-budget",type=int,default=32768)
    rollout=sub.add_parser("rollout");rollout.add_argument("--model-name",required=True);rollout.add_argument("--python")
    export=sub.add_parser("export");export.add_argument("--input",required=True);export.add_argument("--output",required=True);export.add_argument("--model-name",required=True)
    grade=sub.add_parser("grade");grade.add_argument("--predictions");grade.add_argument("--control",choices=["gold","blank"]);grade.add_argument("--python");grade.add_argument("--instance-id")
    for smoke_parser in [rollout,export,grade]: smoke_parser.add_argument("--smoke",action="store_true",help="Only the first frozen task; does not replace five-task validation")
    args=parser.parse_args()
    with Run(args.command,vars(args)) as run:
        if args.command=="preflight":
            from .preflight import run_preflight
            run_preflight(run)
        elif args.command=="inventory":
            from .inventory import inventory
            inventory(run)
        elif args.command=="docker-storage":
            from .docker_storage import record_storage
            record_storage(run,args.backing_file)
        elif args.command=="download":
            from .assets import download_assets
            download_assets(run,args.kind,args.only)
        elif args.command=="checkout":
            from .assets import checkout_repos
            checkout_repos(run)
        elif args.command=="select-tasks":
            from .data import select_tasks
            select_tasks(run)
        elif args.command=="prepare":
            from .data import prepare
            prepare(run)
        elif args.command=="tokenize":
            from .tokenization import tokenize_files
            tokenize_files(run,args.model,args.max_length)
        elif args.command=="train":
            from .training import train
            train(run,args.config,args.resume)
        elif args.command=="validate-checkpoint":
            from .training import validate_checkpoint
            validate_checkpoint(run,args.adapter)
        elif args.command=="serve":
            from .serving import serve
            serve(run,args.model,args.adapter,args.port,args.max_context,task_file=args.task_budget_file,task_token_budget=args.task_token_budget)
        elif args.command=="rollout":
            from .evaluation import rollout
            rollout(run,args.model_name,args.python,args.smoke)
        elif args.command=="export":
            from .evaluation import export_predictions,development_task_ids
            from .common import read_json
            export_predictions(run,args.input,args.output,args.model_name,development_task_ids(args.smoke))
        elif args.command=="grade":
            from .evaluation import grade
            grade(run,args.predictions,args.control,args.python,args.instance_id,args.smoke)
    print(f"Run log: {run.path}")
