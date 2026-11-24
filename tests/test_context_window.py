from copy import deepcopy
import json
import pytest
from swe_study.context_window import fit_history,NOTICE,CLIP


class ExactTokenizer:
    def encode(self,text,**kwargs):return list(text.encode('utf-8'))
    def decode(self,ids,**kwargs):return bytes(ids).decode('utf-8',errors='replace')
    def apply_chat_template(self,rows,tools=None,**kwargs):
        return json.dumps({'messages':rows,'tools':tools},ensure_ascii=False)


def token_count(rows,tools=None):
    tok=ExactTokenizer()
    return len(tok.encode(tok.apply_chat_template(rows,tools=tools)))


def test_under_budget_preserves_complete_prompt_and_input_objects():
    rows=[{'role':'system','content':'instructions'},{'role':'user','content':'Fix widget'}]
    copied=deepcopy(rows);out,audit=fit_history(rows,None,ExactTokenizer(),1000,'Fix widget')
    assert out==rows==copied and not audit['changed']
    assert audit['prompt_tokens']==token_count(rows)


def test_window_discards_whole_old_tool_exchanges_but_keeps_issue_and_recent_pair():
    rows=[{'role':'system','content':'instructions'},{'role':'user','content':'Fix widget'}]
    for i in range(3):
        rows.extend([{'role':'assistant','content':'inspect','tool_calls':[{'id':str(i),'function':{'name':'read'}}]},
                     {'role':'tool','tool_call_id':str(i),'content':('old output '*80 if i<2 else 'latest result')}])
    copied=deepcopy(rows);out,audit=fit_history(rows,[{'name':'read'}],ExactTokenizer(),700,'Fix widget')
    assert rows==copied and out[1]==rows[1]
    assert out[-2:]==rows[-2:] and NOTICE in out[0]['content']
    assert audit['dropped_message_indices']==[2,3,4,5]
    assert not audit['clipped_messages'] and token_count(out,[{'name':'read'}])<=700


def test_oversized_latest_observation_is_marked_and_logged_without_editing_action():
    rows=[{'role':'system','content':'instructions'},{'role':'user','content':'Fix widget'},
          {'role':'assistant','content':'read large file','tool_calls':[{'id':'latest'}]},
          {'role':'tool','tool_call_id':'latest','content':'BEGIN '+('x'*4000)+' END'}]
    copied=deepcopy(rows);out,audit=fit_history(rows,None,ExactTokenizer(),650,'Fix widget')
    assert rows==copied and out[1:3]==rows[1:3]
    assert out[-1]['tool_call_id']=='latest' and CLIP in out[-1]['content']
    assert out[-1]['content'].startswith('BEGIN') and out[-1]['content'].endswith('END')
    assert audit['clipped_messages'][0]['message_index']==3 and token_count(out)<=650
    assert audit['original_prompt_tokens']>audit['prompt_tokens']


@pytest.mark.parametrize('issue,rows',[
    ('Fix widget',[{'role':'user','content':'unrelated '+'x'*1000}]),
    ('Fix widget',[{'role':'system','content':'x'*1000},{'role':'user','content':'Fix widget'}]),
    ('Fix widget',[{'role':'user','content':'Fix widget'},{'role':'assistant','content':'x'*1000}])])
def test_protected_issue_or_action_is_never_silently_clipped(issue,rows):
    original=deepcopy(rows)
    with pytest.raises(ValueError):fit_history(rows,None,ExactTokenizer(),500,issue)
    assert rows==original
