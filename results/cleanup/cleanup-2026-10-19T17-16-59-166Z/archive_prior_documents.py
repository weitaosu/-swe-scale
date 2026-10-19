from pathlib import Path
import hashlib,json,zipfile
root=Path("C:/Users/weita/Desktop/SE in AI/proj")
output=root/"results/cleanup/cleanup-2026-10-19T17-16-59-166Z/prior_documents.zip"
files=["docs/first_two_milestones_plan.md","configs/models.yaml","configs/evaluation.yaml","README.md","docs/implementation_status.md","docs/rtx_4090_implementation_plan.md","docs/rtx_4090_constraints.md"]
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
    for name in files: archive.write(root/name,name)
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
print(json.dumps({'archive':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'files':files},indent=2))
