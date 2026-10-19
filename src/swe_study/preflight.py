from __future__ import annotations
import json
import os
import platform
import shutil
import subprocess
import sys
from .common import ROOT, write_json
from .audit import versions
from .storage import StorageGuard


def probe(args, timeout=20):
    try:
        p=subprocess.run(args,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout)
        return {"args":args,"returncode":p.returncode,"stdout":p.stdout.replace("\x00",""),"stderr":p.stderr.replace("\x00","")}
    except Exception as e:
        return {"args":args,"error":str(e),"returncode":None}


def memory_info():
    if os.name=="nt":
        import ctypes
        class Status(ctypes.Structure):
            _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ["total_physical","available_physical","total_pagefile","available_pagefile","total_virtual","available_virtual","available_extended"]]
        status=Status();status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)): raise OSError("Memory query failed")
        return {"total_physical_bytes":status.total_physical,"available_physical_bytes":status.available_physical}
    return {"total_physical_bytes":os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE"),"available_physical_bytes":os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")}


def run_preflight(run):
    report={"platform":platform.platform(),"cpu_count":os.cpu_count(),"versions":versions(),"storage":StorageGuard().snapshot()}
    report["processor"]=platform.processor()
    try: report["ram"]=memory_info()
    except Exception as e: report["ram"]={"error":str(e)}
    checks={"git":["git","--version"],"docker":["docker","version","--format","json"],"docker_info":["docker","info","--format","json"],"nvidia_smi":["nvidia-smi","--query-gpu=name,memory.total,memory.free,driver_version","--format=csv,noheader"]}
    if os.name=="nt": checks["wsl"]=["wsl.exe","--list","--verbose"]
    report["commands"]={}
    for name,args in checks.items():
        report["commands"][name]=probe(args)
        run.event("preflight_command",name=name,**report["commands"][name])
    docker=report["commands"].get("docker_info",{})
    try:
        info=json.loads(docker.get("stdout","{}"))
        report["docker_daemon_ready"]=bool(info.get("ServerVersion") and not info.get("ServerErrors"))
    except (ValueError,TypeError): report["docker_daemon_ready"]=False
    try:
        import torch
        gpu={"torch":torch.__version__,"compiled_cuda":torch.version.cuda,"available":torch.cuda.is_available(),"count":torch.cuda.device_count()}
        if gpu["available"]:
            p=torch.cuda.get_device_properties(0)
            gpu.update(name=p.name,total_memory_bytes=p.total_memory,capability=list(torch.cuda.get_device_capability()),bf16=torch.cuda.is_bf16_supported())
            gpu["tensor_check"]=(torch.ones(2,device="cuda")+1).cpu().tolist()
            gpu["free_memory_bytes"]=torch.cuda.mem_get_info()[0]
        report["gpu"]=gpu
    except Exception as e: report["gpu"]={"error":str(e)}
    write_json(run.path/"hardware.json",report)
    write_json("results/preflight/latest.json",report)
    write_json("manifests/versions.json",report["versions"])
    print(json.dumps(report,indent=2))
    return report
