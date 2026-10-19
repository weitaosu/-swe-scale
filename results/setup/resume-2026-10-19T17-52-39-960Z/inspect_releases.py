from pathlib import Path
import json,sys,urllib.parse,urllib.request
root=Path("/mnt/c/Users/weita/Desktop/SE in AI/proj")
sys.path.insert(0,str(root/"src"))
from swe_study.common import setup_environment,write_json,read_json
from swe_study.audit import Run
setup_environment()
with Run("source_inspection") as run:
    inventory=read_json("manifests/planned_downloads_4090.json")
    for asset in inventory["models"]:
        folder=root/asset["target"]/".cache/huggingface/download"
        commits={p.read_text().splitlines()[0] for p in folder.rglob("*.metadata")}
        if len(commits)!=1: raise ValueError("Local model metadata does not identify a single immutable snapshot")
        asset["revision"]=next(iter(commits))
        asset["revision_discovery"]="user-downloaded Hugging Face local snapshot metadata; checksum verification pending"
        run.event("local_snapshot_revision",repo=asset["repo"],revision=asset["revision"])
    write_json("manifests/planned_downloads_4090.json",inventory)
    for asset in inventory["datasets"]:
        url="https://huggingface.co/api/datasets/"+asset["repo"]+"/tree/main?recursive=true"
        with urllib.request.urlopen(url,timeout=30) as response: tree=json.load(response)
        write_json(run.path/(asset["repo"].split("/")[-1]+"_tree.json"),tree)
        paths=[x["path"] for x in tree if x["path"].endswith(".parquet")]
        run.event("dataset_tree",repo=asset["repo"],files=paths)
        print(json.dumps({"repo":asset["repo"],"split":asset["split"],"parquet_files":paths}))
        if "Trajectories" in asset["repo"]:
            query=urllib.parse.urlencode({"dataset":asset["repo"],"config":"default","split":asset["split"],"offset":0,"length":1})
            with urllib.request.urlopen("https://datasets-server.huggingface.co/rows?"+query,timeout=45) as response: sample=json.load(response)
            write_json(run.path/(asset["repo"].split("/")[-1]+"_first_row.json"),sample)
            row=sample["rows"][0]["row"]
            print(json.dumps({"schema":asset["repo"],"keys":list(row),"field_types":{k:type(v).__name__ for k,v in row.items()},"first_message":str(row.get("messages",[])[0])[:400] if row.get("messages") else None}))
