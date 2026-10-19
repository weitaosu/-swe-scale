from __future__ import annotations
import json
import re
import threading
import time
import uuid


def parse_tool_output(text):
    calls=[]
    pattern=r"<tool_call>\s*(.*?)\s*</tool_call>"
    for chunk in re.findall(pattern,text,re.S):
        value=json.loads(chunk)
        args=value.get("arguments",{})
        if isinstance(args,str): json.loads(args)
        else: args=json.dumps(args,ensure_ascii=False)
        calls.append({"id":"call_"+uuid.uuid4().hex[:16],"type":"function","function":{"name":value["name"],"arguments":args}})
    if calls: return re.sub(pattern,"",text,flags=re.S).strip(),calls
    pattern=r"<function=([A-Za-z0-9_]+)>(.*?)</function>"
    for name,body in re.findall(pattern,text,re.S):
        args={}
        for key,value in re.findall(r"<parameter=([A-Za-z0-9_]+)>(.*?)</parameter>",body,re.S):
            try: args[key]=json.loads(value.strip())
            except json.JSONDecodeError: args[key]=value.strip("\n")
        if not args and body.strip(): raise ValueError("Unsupported textual tool argument serialization")
        calls.append({"id":"call_"+uuid.uuid4().hex[:16],"type":"function","function":{"name":name,"arguments":json.dumps(args,ensure_ascii=False)}})
    return re.sub(pattern,"",text,flags=re.S).strip() if calls else text,calls


class TaskBudget:
    def __init__(self,task_file,limit,ledger_path):
        from .common import read_jsonl
        self.tasks={r["instance_id"]:re.sub(r"\s+"," ",r["problem_statement"]).strip() for _,r in read_jsonl(task_file)}
        if not self.tasks or any(not issue for issue in self.tasks.values()): raise ValueError("Task budget file has no usable problem statements")
        self.limit=limit;self.used={key:0 for key in self.tasks};self.ledger_path=ledger_path
    def identify(self,messages):
        text=re.sub(r"\s+"," "," ".join(m.get("content","") or "" for m in messages if m.get("role")=="user")).strip()
        matches=[key for key,issue in self.tasks.items() if issue in text]
        if len(matches)!=1: raise ValueError("Request must identify exactly one frozen development task; budget cannot be reset by dropping the issue")
        return matches[0]
    def remaining(self,key): return max(0,self.limit-self.used[key])
    def record(self,key,tokens):
        from .common import write_json
        if tokens<0 or tokens>self.remaining(key): raise ValueError("Generated tokens exceed remaining task budget")
        self.used[key]+=tokens
        write_json(self.ledger_path,{"limit_per_task":self.limit,"completion_tokens_by_task":self.used})


def stream_chunks(response):
    choice=response["choices"][0];delta=dict(choice["message"])
    if "tool_calls" in delta: delta["tool_calls"]=[{"index":i,**call} for i,call in enumerate(delta["tool_calls"])]
    base={k:response[k] for k in ["id","created","model"]};base["object"]="chat.completion.chunk"
    yield "data: "+json.dumps({**base,"choices":[{"index":0,"delta":delta,"finish_reason":None}]})+"\n\n"
    yield "data: "+json.dumps({**base,"choices":[{"index":0,"delta":{},"finish_reason":choice["finish_reason"]}],"usage":response["usage"]})+"\n\n"
    yield "data: [DONE]\n\n"


def serve(run,model_path,adapter=None,port=8000,max_context=8192,max_new_tokens=2048,quantization="nf4",task_file=None,task_token_budget=32768):
    import torch
    import uvicorn
    from fastapi import FastAPI,HTTPException
    from fastapi.responses import StreamingResponse
    from .training import load_model
    from .storage import StorageGuard
    budget=TaskBudget(task_file,task_token_budget,run.path/"task_tokens.json") if task_file else None
    model,tokenizer=load_model(model_path,quantization,adapter);model.eval()
    if hasattr(model.config,"use_cache"): model.config.use_cache=True
    lock=threading.Lock();app=FastAPI(title="CS6158 local model server");name="cs6158-"+model_path.replace("\\","/").rstrip("/").split("/")[-1]
    @app.middleware("http")
    async def log_http(request,call_next):
        started=time.perf_counter()
        try:
            response=await call_next(request)
            run.event("http_response",method=request.method,path=request.url.path,status_code=response.status_code,seconds=time.perf_counter()-started)
            return response
        except Exception as exc:
            run.event("http_error",method=request.method,path=request.url.path,error_type=type(exc).__name__,error=str(exc),seconds=time.perf_counter()-started)
            raise
    @app.get("/health")
    def health(): return {"status":"ready","model":name,"adapter":adapter,"quantization":quantization,"max_context":max_context}
    @app.get("/v1/models")
    def models(): return {"object":"list","data":[{"id":name,"object":"model","owned_by":"local"}]}
    @app.post("/v1/chat/completions")
    def completion(body:dict):
        with lock:
            StorageGuard().check()
            request_id=uuid.uuid4().hex
            run.event("completion_request",request_id=request_id,request=body)
            requested_model=body.get("model",name)
            if requested_model not in {name,model_path}: raise HTTPException(404,"Requested model does not match the loaded checkpoint")
            if int(body.get("n",1))!=1: raise HTTPException(400,"Only one completion per request is supported")
            tools=body.get("tools") or None
            messages=body.get("messages") or []
            prompt=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True)
            inputs=tokenizer(prompt,add_special_tokens=False,return_tensors="pt").to("cuda")
            prompt_count=inputs["input_ids"].shape[1]
            try: task_key=budget.identify(messages) if budget else None
            except ValueError as e: raise HTTPException(400,str(e))
            if budget and budget.remaining(task_key)==0: raise HTTPException(429,"Frozen task completion-token budget exhausted")
            output_cap=min(int(body.get("max_tokens",body.get("max_completion_tokens",max_new_tokens))),max_new_tokens)
            if budget: output_cap=min(output_cap,budget.remaining(task_key))
            if output_cap<=0 or prompt_count+output_cap>max_context:
                raise HTTPException(413,"Prompt plus requested output exceeds the frozen context budget; no silent truncation is applied")
            temperature=float(body.get("temperature",0));kwargs={"max_new_tokens":output_cap,"do_sample":temperature>0,"pad_token_id":tokenizer.pad_token_id}
            if temperature>0: kwargs["temperature"]=temperature
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode(): output=model.generate(**inputs,**kwargs)
            torch.cuda.synchronize();duration=time.perf_counter()-start
            generated=output[0,prompt_count:];text=tokenizer.decode(generated,skip_special_tokens=True)
            if budget: budget.record(task_key,len(generated))
            try: content,calls=parse_tool_output(text)
            except (ValueError,KeyError) as e:
                run.event("tool_parse_error",request_id=request_id,error=str(e));content=text;calls=[]
            response={"id":"chatcmpl-"+request_id,"object":"chat.completion","created":int(time.time()),"model":name,
              "choices":[{"index":0,"message":{"role":"assistant","content":content or None,**({"tool_calls":calls} if calls else {})},"finish_reason":"tool_calls" if calls else "length" if len(generated)>=output_cap else "stop"}],
              "usage":{"prompt_tokens":prompt_count,"completion_tokens":len(generated),"total_tokens":prompt_count+len(generated)}}
            run.event("completion",request_id=request_id,task_id=task_key,remaining_task_tokens=budget.remaining(task_key) if budget else None,request=body,response=response,raw_output=text,seconds=duration,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if body.get("stream"):
                return StreamingResponse(stream_chunks(response),media_type="text/event-stream")
            return response
    run.event("server_start",model=model_path,adapter=adapter,port=port,max_context=max_context,quantization=quantization,task_file=task_file,task_token_budget=task_token_budget if budget else None)
    uvicorn.run(app,host="127.0.0.1",port=port,log_level="info")
