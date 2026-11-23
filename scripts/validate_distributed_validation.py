#!/usr/bin/env python3
"""Verify loss reconstruction against an actual two-process Trainer evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json


def worker(directory):
    import torch
    from peft import LoraConfig,get_peft_model
    from transformers import Qwen2Config,Qwen2ForCausalLM,Trainer,TrainingArguments
    from types import SimpleNamespace
    from swe_study.memory_head import enable_chunked_head_loss
    from swe_study.tokenization import SupervisedCollator
    from swe_study.training import assistant_target_inputs
    from swe_study.validation import recorded_trainer_loss
    torch.set_num_threads(1);torch.manual_seed(6158)
    cfg=Qwen2Config(vocab_size=61,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                    num_attention_heads=2,num_key_value_heads=1,use_cache=False)
    cfg._attn_implementation='eager'
    model=get_peft_model(Qwen2ForCausalLM(cfg),LoraConfig(r=4,lora_alpha=8,lora_dropout=.05,
                        target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.eval();enable_chunked_head_loss(model,2)
    rows=[{'input_ids':list(range(3,3+n)),'attention_mask':[1]*n,
           'labels':[-100,-100]+list(range(5,3+n))} for n in [5+i%7 for i in range(43)]]
    histories=[]
    with torch.no_grad():
        for row in rows:
            count=sum(value!=-100 for value in row['labels'])
            batch={k:torch.tensor([v]) for k,v in row.items()}
            loss=model(**assistant_target_inputs(batch),num_items_in_batch=count).loss
            histories.append({'loss':float(loss),'target_tokens':count})
    class MatchingTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            return super().compute_loss(model,assistant_target_inputs(inputs),return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
    args=TrainingArguments(output_dir=str(directory/'trainer'),use_cpu=True,per_device_eval_batch_size=1,
                           label_names=['labels'],prediction_loss_only=True,remove_unused_columns=False,
                           dataloader_num_workers=0,report_to='none',disable_tqdm=True)
    trainer=MatchingTrainer(model=model,args=args,eval_dataset=rows,data_collator=SupervisedCollator(0),
                            processing_class=SimpleNamespace(pad_token_id=0,model_input_names=['input_ids','attention_mask']))
    metrics=trainer.evaluate()
    expected=recorded_trainer_loss(histories,2)
    canonical=recorded_trainer_loss(histories,1)
    if abs(metrics['eval_loss']-expected)>1e-6:raise AssertionError((metrics,expected,canonical))
    if abs(expected-canonical)<1e-6:raise AssertionError('Fixture did not distinguish aggregations')
    if int(os.environ['RANK'])==0:
        write_json(directory/'distributed_metric.json',{'passed':True,'synthetic_fixture':True,
                   'examples':43,'world_size':2,'canonical_history_mean':canonical,
                   'actual_distributed_trainer_loss':metrics['eval_loss'],'reconstructed_loss':expected})
    torch.distributed.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args();setup_environment()
    if args.worker:
        worker(args.worker);return
    with Run('distributed_validation_regression') as run:
        run.command([sys.executable,'-m','pytest','tests/test_cloud_validation.py','-q','--capture=sys',
                     '-p','no:cacheprovider','--basetemp',ROOT/'.cache/tmp'/('validation-regression-'+run.id)],timeout=300)
        directory=ROOT/'.cache/tmp'/('distributed-validation-'+run.id);directory.mkdir()
        # No GPU is used: this checks Trainer aggregation, not NF4 performance.
        os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
        run.command([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=2',
                     str(ROOT/'scripts/validate_distributed_validation.py'),'--worker',str(directory)],timeout=300)
        report=json.loads((directory/'distributed_metric.json').read_text())
        write_json(run.path/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
