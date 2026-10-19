from __future__ import annotations
import json
from collections import Counter
from .common import canonical_hash, local_path, read_jsonl, sha256_file, write_json, write_jsonl


def template_tools(row):
    tools=row.get("tools") or []
    if row.get("tool_prompt_mode")=="embedded": return None
    return tools or None


def encode_assistant_only(tokenizer,row,max_length):
    messages=row["messages"]; kwargs={"tools":template_tools(row)}
    if kwargs["tools"] is None: kwargs={}
    render=lambda ms,g=False:tokenizer.apply_chat_template(ms,tokenize=False,add_generation_prompt=g,**kwargs)
    text=render(messages)
    spans=[]
    for index,message in enumerate(messages):
        if message["role"]!="assistant": continue
        before=render(messages[:index],True)
        through=render(messages[:index+1])
        if not text.startswith(before) or not text.startswith(through) or len(through)<=len(before):
            raise ValueError("Chat template is not prefix-stable at an assistant boundary; explicit template review required")
        spans.append((len(before),len(through)))
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False)
    ids=encoded["input_ids"]
    if len(ids)>max_length: raise ValueError(f"over_context:{len(ids)}")
    labels=[token if b>a and any(a>=start and b<=end for start,end in spans) else -100 for token,(a,b) in zip(ids,encoded["offset_mapping"])]
    target_count=sum(x!=-100 for x in labels)
    if not target_count: raise ValueError("zero_assistant_targets")
    return {"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels,"instance_id":row.get("instance_id"),
            "trajectory_id":row.get("trajectory_id"),"target_tokens":target_count,"rendered_sha256":canonical_hash(text)}


def tokenize_files(run,model,max_length):
    from transformers import AutoTokenizer
    p=local_path(model)
    if not (p/"tokenizer_config.json").exists(): raise FileNotFoundError(f"Local tokenizer missing: {p}")
    tokenizer=AutoTokenizer.from_pretrained(p,local_files_only=True,use_fast=True)
    if not tokenizer.is_fast: raise ValueError("A fast tokenizer with offset mappings is required for the audited label mask")
    report={"model":model,"max_length":max_length,"template_hash":canonical_hash(tokenizer.chat_template),"files":{},"exclusions":[]}
    for split in ["train","validation"]:
        source=f"data/processed/swegym/{split}.jsonl"
        output=f"data/tokenized/{p.name}-{max_length}/{split}.jsonl"
        rows=[];counts=Counter();lengths=[]
        for index,row in read_jsonl(source):
            try:
                encoded=encode_assistant_only(tokenizer,row,max_length);rows.append(encoded)
                counts["retained"]+=1;counts["processed_tokens"]+=len(encoded["input_ids"]);counts["target_tokens"]+=encoded["target_tokens"];lengths.append(len(encoded["input_ids"]))
                if len(rows)<=3:
                    run.event("label_sample",split=split,instance_id=row["instance_id"],targets=tokenizer.decode([t for t in encoded["labels"] if t!=-100]),processed_tokens=len(encoded["input_ids"]),target_tokens=encoded["target_tokens"])
            except ValueError as e:
                counts["excluded"]+=1;report["exclusions"].append({"split":split,"row":index,"instance_id":row.get("instance_id"),"reason":str(e)})
        if not rows: raise ValueError(f"No usable {split} examples at {max_length} tokens; revise the context policy")
        write_jsonl(output,rows)
        ordered=sorted(lengths)
        counts["p50_length"]=ordered[len(ordered)//2];counts["p95_length"]=ordered[min(len(ordered)-1,int(len(ordered)*0.95))]
        report["files"][split]={"source":source,"source_sha256":sha256_file(local_path(source)),"output":output,"output_sha256":sha256_file(local_path(output)),**counts}
    write_json(run.path/"token_stats.json",report)
    write_json(f"data/tokenized/{p.name}-{max_length}/manifest.json",report)
    print(json.dumps(report,indent=2))
    return report


class SupervisedCollator:
    def __init__(self,pad_token_id): self.pad_token_id=pad_token_id
    def __call__(self,features):
        import torch
        size=max(len(f["input_ids"]) for f in features)
        return {"input_ids":torch.tensor([f["input_ids"]+[self.pad_token_id]*(size-len(f["input_ids"])) for f in features]),
                "attention_mask":torch.tensor([f["attention_mask"]+[0]*(size-len(f["input_ids"])) for f in features]),
                "labels":torch.tensor([f["labels"]+[-100]*(size-len(f["labels"])) for f in features])}
