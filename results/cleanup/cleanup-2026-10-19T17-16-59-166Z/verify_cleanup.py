from pathlib import Path
import ast,hashlib,json,re
root=Path("C:/Users/weita/Desktop/SE in AI/proj")
audit=root/"results/cleanup/cleanup-2026-10-19T17-16-59-166Z"
before=json.loads((audit/"before_inventory.json").read_text(encoding="utf-8"))["files"]
plan=json.loads((audit/"cleanup_plan.json").read_text(encoding="utf-8"))
expected_removed=[row for row in before if any(row["path"]==t["path"] or row["path"].startswith(t["path"]+"/") for t in plan["targets"])]
remaining_removed=[row["path"] for row in expected_removed if (root/row["path"]).exists()]
assert not remaining_removed,remaining_removed
preserve=[row for row in before if row["path"].startswith(("results/","checkpoints/",".cache/tmp/tiny-qwen-fixture/",".cache/tmp/fixture_"))]
changed=[]
for row in preserve:
    p=root/row["path"]
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row["sha256"]: changed.append(row["path"])
assert not changed,changed
sources=[]
for folder in ["src","scripts","tests"]:
    for p in sorted((root/folder).rglob("*.py")):
        text=p.read_text(encoding="utf-8");tree=ast.parse(text,filename=str(p));compile(tree,str(p),"exec")
        imports=sorted({node.module or "" for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}|{alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names})
        sources.append({"path":p.relative_to(root).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"imports":imports})
missing=[];checked_links=0
for p in [root/"README.md",*sorted((root/"docs").glob("*.md"))]:
    for match in re.finditer(r"\[[^\]]*\]\((<[^>]+>|[^)]+)\)",p.read_text(encoding="utf-8")):
        target=match.group(1).strip().strip("<>")
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:",target) or target.startswith("#"): continue
        target=target.split("#",1)[0]
        if not target: continue
        checked_links+=1
        if not (p.parent/target).exists(): missing.append({"file":p.relative_to(root).as_posix(),"target":target})
assert checked_links>=20,checked_links
assert not missing,missing
report={"passed":True,"removed_preexisting_files":len(expected_removed),"removed_preexisting_bytes":sum(r["bytes"] for r in expected_removed),"preserved_evidence_files_checked":len(preserve),"preserved_evidence_changes":changed,"source_modules_syntax_checked":len(sources),"source_analysis":sources,"local_markdown_links_checked":checked_links,"broken_local_links":missing}
(audit/"verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps({k:v for k,v in report.items() if k!="source_analysis"},indent=2))
