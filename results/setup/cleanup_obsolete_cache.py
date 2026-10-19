from pathlib import Path
import sys,re,json
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'src'))
from swe_study.common import setup_environment,read_json,write_json
setup_environment()
from swe_study.audit import Run
from swe_study.storage import StorageGuard
review=read_json('results/setup/cleanup_review_latest.json');graph=read_json(root/review['review']/'obsolete_cache_graph.json')
with Run('cleanup_cache',{'root_cache_id':graph['root'],'allowed_cache_ids':graph['ids'],'reference':'https://docs.docker.com/reference/cli/docker/buildx/prune/'}) as run:
    guard=StorageGuard()
    def command(args):
        record=run.command(args,guard=guard,timeout=180);return Path(record['stdout_path']).read_text()
    base=['docker','--config','.cache/docker_agent','buildx']
    def records(text):
        result={}
        for block in re.split(r'\n(?=ID:)',text):
            found=re.match(r'ID:\s+(\S+)',block)
            if not found:continue
            part=re.search(r'Parents?:([\s\S]*?)(?=Created at:)',block)
            result[found[1]]={'text':block,'parents':set(re.findall(r'[a-z0-9]{20,}',part[1])) if part else set()}
        return result
    original=records(command(base+['du','--verbose']));allowed=set(graph['ids']);derived={graph['root']}
    while True:
        expanded=derived|{key for key,value in original.items() if value['parents']&derived}
        if expanded==derived:break
        derived=expanded
    if derived!=allowed:raise ValueError('Obsolete cache dependency graph changed')
    for key in allowed:
        text=original[key]['text']
        if not re.search(r'^Shared:\s+false',text,re.M) or not re.search(r'^Reclaimable:\s+true',text,re.M):raise ValueError('Cache entry is shared or in use: '+key)
    images_before=set(command(['docker','image','ls','--no-trunc','--quiet']).splitlines())
    pending=set(allowed);prunes=[]
    while pending:
        leaves=sorted(key for key in pending if not any(key in original[other]['parents'] for other in pending))
        if not leaves:raise ValueError('Unexpected cache cycle')
        for key in leaves:
            result=command(base+['prune','--force','--filter','id='+key]);prunes.append({'id':key,'stdout':result});pending.remove(key)
    remaining=records(command(base+['du','--verbose']));removed=set(original)-set(remaining)
    if removed!=allowed:raise ValueError('Cache pruning removed unexpected records or did not finish: '+repr(removed))
    if set(command(['docker','image','ls','--no-trunc','--quiet']).splitlines())!=images_before:raise ValueError('Image set changed during cache cleanup')
    command(['docker','system','df']);storage=guard.snapshot()
    write_json(run.path/'verification.json',{'passed':True,'removed_cache_ids':sorted(removed),'removed_cache_count':len(removed),'other_cache_ids_preserved':len(remaining),'all_images_preserved':True,'prunes':prunes,'storage':storage,'note':'Only the exclusive descendants of the removed failed no-browser runtime were pruned, leaf-first. Shared parents and unrelated caches were kept.'})
    print({'passed':True,'cache_entries_removed':len(removed),'prunes':prunes,'storage':storage})
