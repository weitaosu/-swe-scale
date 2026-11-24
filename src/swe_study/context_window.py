"""Explicit, audited history windowing with the model's exact chat tokenizer."""
from copy import deepcopy
import hashlib
import re

POLICY='history-window-v1'
NOTICE='[History window: older exchanges may be omitted and long tool output clipped. Re-read repository files when needed.]'
CLIP='\n[Tool output clipped by the recorded history-window policy; re-run a narrower command to inspect omitted content.]\n'


def fit_history(messages,tools,tokenizer,input_limit,issue):
    """Keep the issue/system prefix and recent complete action/observation groups.

    Never edit the initial issue or assistant tool calls. Every omission is marked
    in the resulting prompt and returned for recording alongside the full request.
    """
    def prompt(rows):return tokenizer.apply_chat_template(rows,tools=tools,tokenize=False,add_generation_prompt=True)
    def tokens(text):return tokenizer.encode(text,add_special_tokens=False)
    def count(rows):return len(tokens(prompt(rows)))
    original=deepcopy(messages);before=count(original)
    audit={'policy':POLICY,'input_limit':input_limit,'original_prompt_tokens':before,
           'prompt_tokens':before,'dropped_message_indices':[],'clipped_messages':[],'changed':False}
    if before<=input_limit:return original,audit
    normalized=re.sub(r'\s+',' ',issue).strip()
    if not normalized:raise ValueError('History window requires the frozen public issue')
    anchors=[i for i,m in enumerate(original) if m.get('role')=='user' and
             normalized in re.sub(r'\s+',' ',m.get('content','')).strip()]
    if not anchors:raise ValueError('Frozen public issue is absent; refusing history windowing')
    anchor=anchors[0];prefix=deepcopy(original[:anchor+1]);groups=[]
    for index in range(anchor+1,len(original)):
        message=deepcopy(original[index])
        if message.get('role')=='assistant' or not groups:groups.append([])
        groups[-1].append((index,message))
    # Put the notice in the system message without changing the task description.
    if prefix and prefix[0].get('role')=='system':prefix[0]['content']+='\n\n'+NOTICE
    else:prefix.insert(0,{'role':'system','content':NOTICE})
    def rows():return prefix+[message for group in groups for _,message in group]
    while len(groups)>1 and count(rows())>input_limit:
        audit['dropped_message_indices'].extend(index for index,_ in groups.pop(0))
    # If one recent exchange alone is too large, clip only observation content.
    # Preserve its head/tail, tool-call IDs and the entire associated action.
    if count(rows())>input_limit:
        observations=[(index,message) for group in groups for index,message in group
                      if message.get('role') in ('user','tool') and message.get('content')]
        observations.sort(key=lambda item:len(item[1]['content']),reverse=True)
        for index,message in observations:
            text=message['content'];encoded=tokens(text);lo=0;hi=len(encoded);best=None
            while lo<=hi:
                keep=(lo+hi)//2;head=(keep+1)//2;tail=keep//2
                message['content']=(tokenizer.decode(encoded[:head],skip_special_tokens=False)+CLIP+
                                    (tokenizer.decode(encoded[-tail:],skip_special_tokens=False) if tail else ''))
                if count(rows())<=input_limit:best=message['content'];lo=keep+1
                else:hi=keep-1
            message['content']=best if best is not None else CLIP
            audit['clipped_messages'].append({'message_index':index,'original_tokens':len(encoded),
                    'retained_content_tokens':len(tokens(message['content'])),
                    'original_content_sha256':hashlib.sha256(text.encode()).hexdigest()})
            if count(rows())<=input_limit:break
    after=count(rows())
    if after>input_limit:
        raise ValueError('Protected issue, instructions and latest action exceed the context budget; no task or action is truncated')
    audit.update(prompt_tokens=after,changed=True)
    return rows(),audit
