#!/usr/bin/env python3
"""CPU checks using real Qwen tokenizers, pinned prompts and mocked SDK errors."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor/SWE-Gym-OpenHands'))
from swe_study.audit import Run
from swe_study.common import setup_environment,write_json
from swe_study.context_window import fit_history


def main():
    setup_environment();os.environ.update(USE_HINT_TEXT='false',USE_INSTANCE_IMAGE='true',RUN_WITH_BROWSING='false')
    with Run('context_window_validation') as run:
        from transformers import AutoTokenizer
        import yaml
        from evaluation.swe_bench.run_infer import get_instruction
        from openhands.utils.prompt import PromptManager
        from openhands.agenthub.codeact_agent.function_calling import get_tools
        from openhands.llm.fn_call_converter import convert_fncall_messages_to_non_fncall_messages
        folder=ROOT/'vendor/swe-bench-tasks/tasks/pvlib__pvlib-python-807'
        task=yaml.safe_load((folder/'task.yaml').read_text());issue=(folder/'problem_statement.md').read_text()
        instance=SimpleNamespace(repo=task['repo'],version=task['version'],problem_statement=issue)
        manager=PromptManager(str(ROOT/'vendor/SWE-Gym-OpenHands/openhands/agenthub/codeact_agent/prompts'))
        instruction=manager.get_example_user_message()+'\n'+get_instruction(instance,SimpleNamespace(agent_class='CodeActAgent'))
        rows=convert_fncall_messages_to_non_fncall_messages(
            [{'role':'system','content':manager.get_system_message()},{'role':'user','content':instruction}],
            get_tools(codeact_enable_browsing=False,codeact_enable_jupyter=False,codeact_enable_llm_editor=False))
        # Convert text blocks exactly as the serving path does.
        from swe_study.data import text_content
        rows=[{**m,'content':text_content(m.get('content'))} for m in rows]
        results={}
        for size in ('1.5B','3B','7B'):
            tokenizer=AutoTokenizer.from_pretrained(ROOT/'models'/('Qwen2.5-Coder-'+size+'-Instruct'),local_files_only=True)
            def count(value):return len(tokenizer.encode(tokenizer.apply_chat_template(value,tools=None,tokenize=False,add_generation_prompt=True),add_special_tokens=False))
            initial=count(rows)
            if initial>6144:raise AssertionError('Protected initial prompt exceeds the frozen input budget')
            history=deepcopy(rows)
            for i in range(20):
                history.extend([{'role':'assistant','content':f'<function=execute_bash>\n<parameter=command>echo {i}</parameter>\n</function>'},
                                {'role':'user','content':'EXECUTION RESULT of [execute_bash]:\n'+('sample log output '+str(i)+'\n')*80}])
            before=deepcopy(history);fitted,audit=fit_history(history,None,tokenizer,6144,issue)
            assert history==before and audit['changed'] and count(fitted)<=6144
            assert any(issue in m['content'] for m in fitted if m.get('role')=='user')
            large=rows+[{'role':'assistant','content':'<function=execute_bash>\n<parameter=command>cat large.txt</parameter>\n</function>'},
                        {'role':'user','content':'EXECUTION RESULT of [execute_bash]:\n'+('large observation\n'*10000)}]
            clipped,clip_audit=fit_history(large,None,tokenizer,6144,issue)
            assert clip_audit['clipped_messages'] and count(clipped)<=6144
            results[size]={'initial_prompt_tokens':initial,'windowed_tokens':count(fitted),
                           'dropped_messages':len(audit['dropped_message_indices']),'clipped_observation_prompt_tokens':count(clipped)}
        # Exercise the actual installed SDK using the exact entry-point override.
        import openai,httpx
        tree=ast.parse((ROOT/'scripts/openhands_entry.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='single_attempt_openai')
        original=openai.OpenAI.__init__;namespace={'original_openai_init':original}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'openhands_entry.py','exec'),namespace)
        calls=[]
        def reject(request):
            calls.append(request)
            return httpx.Response(413,json={'error':{'message':'oversized fixture','type':'invalid_request_error'}},request=request)
        try:
            openai.OpenAI.__init__=namespace['single_attempt_openai']
            with openai.OpenAI(api_key='local-fixture',base_url='http://fixture.invalid/v1',max_retries=8,
                               http_client=httpx.Client(transport=httpx.MockTransport(reject))) as client:
                try:client.chat.completions.create(model='fixture',messages=[{'role':'user','content':'fixture'}])
                except openai.APIStatusError as exc:assert exc.status_code==413
                else:raise AssertionError('Expected mocked 413')
            assert len(calls)==1
        finally:openai.OpenAI.__init__=original
        from openhands.llm.retry_mixin import RetryMixin
        attempts=[]
        @RetryMixin().retry_decorator(num_retries=1,retry_exceptions=(RuntimeError,),retry_min_wait=0,retry_max_wait=0,retry_multiplier=1)
        def fail():attempts.append(1);raise RuntimeError('fixture')
        try:fail()
        except RuntimeError:pass
        assert len(attempts)==1
        report={'passed':True,'real_local_tokenizers':results,'pinned_mock_tool_prompt':True,
                'sdk_requests_on_413':len(calls),'openhands_attempts':len(attempts),'synthetic_histories':True,
                'limitation':'CPU tokenization and mocked API errors; no GPU performance or agent-resolution claim'}
        write_json(run.path/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
