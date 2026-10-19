from __future__ import annotations
import contextlib
import datetime as dt
import importlib.metadata
import json
import platform
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from .common import ROOT, canonical_hash, local_path, write_json

SECRET_KEYS = {"api_key", "hf_token", "authorization", "password", "access_token", "refresh_token"}


def redact(value):
    if isinstance(value, dict):
        return {k: "[REDACTED]" if str(k).lower() in SECRET_KEYS else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for stream in self.streams:
            if not getattr(stream,"closed",False):
                stream.write(text)
                stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:
            if not getattr(stream,"closed",False): stream.flush()
    def isatty(self): return False

def _linux_process_tree(parent_pid):
    records={}
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit(): continue
        try:
            fields=(folder/'stat').read_text().rsplit(')',1)[1].split()
            records[int(folder.name)]=(int(fields[1]),fields[19])
        except (OSError,ValueError,IndexError): continue
    selected={parent_pid}
    while True:
        expanded=selected|{pid for pid,(ppid,_) in records.items() if ppid in selected}
        if expanded==selected: break
        selected=expanded
    return {pid:records[pid][1] for pid in selected if pid in records}


def _signal_linux_processes(records,signum):
    for pid,started in reversed(list(records.items())):
        try:
            fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
            if fields[19]==started and fields[0]!='Z': os.kill(pid,signum)
        except (OSError,ValueError,IndexError): pass



class Run:
    def __init__(self, stage, config=None):
        self.id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.path = local_path(Path("results") / stage / self.id)
        self.path.mkdir(parents=True)
        self.config = redact(config or {})
        self.meta = {"run_id": self.id, "stage": stage, "started_at": utc(), "status": "running", "config": self.config,
                     "config_sha256": canonical_hash(self.config), "python": sys.version, "platform": platform.platform(),
                     "executable": sys.executable, "argv": sys.argv, "project_root": str(ROOT)}
        self.meta.update(pid=os.getpid(),parent_pid=os.getppid(),condition_id=os.environ.get("CS6158_CONDITION_ID"))
        self.lock = threading.Lock()
        self.stack = contextlib.ExitStack()
    def event(self, event, **fields):
        record = redact({"time": utc(), "event": event, **fields})
        with self.lock, (self.path / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    def __enter__(self):
        snapshot_files=[]
        for folder in ["src","scripts","configs","tests","manifests","docs"]:
            for item in sorted((ROOT/folder).rglob("*")):
                if item.is_file() and item.suffix in {".py",".sh",".ps1",".yaml",".json",".toml",".txt",".md"} and "__pycache__" not in item.parts:
                    snapshot_files.append(item)
        snapshot_files += [p for p in [ROOT/"pyproject.toml",ROOT/"README.md",ROOT/"AGENTS.md"] if p.exists()]
        with zipfile.ZipFile(self.path/"source_snapshot.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
            for item in snapshot_files: archive.write(item,item.relative_to(ROOT).as_posix())
        from .common import sha256_file
        self.meta["source_snapshot_sha256"]=sha256_file(self.path/"source_snapshot.zip")
        write_json(self.path / "run.json", self.meta)
        out = self.stack.enter_context((self.path / "stdout.log").open("w", encoding="utf-8"))
        err = self.stack.enter_context((self.path / "stderr.log").open("w", encoding="utf-8"))
        self.stack.enter_context(contextlib.redirect_stdout(Tee(sys.stdout, out)))
        self.stack.enter_context(contextlib.redirect_stderr(Tee(sys.stderr, err)))
        self.start = time.perf_counter()
        self.event("run_started", config=self.config)
        print(f"Run log: {self.path}")
        return self
    def __exit__(self, kind, exc, tb):
        self.meta.update(ended_at=utc(), elapsed_seconds=time.perf_counter()-self.start,
                         status="failed" if exc else "passed")
        if exc:
            self.meta["error"] = {"type": kind.__name__, "message": str(exc)}
            traceback.print_exception(kind, exc, tb)
        self.event("run_finished", status=self.meta["status"], error=self.meta.get("error"))
        self.stack.close()
        write_json(self.path / "run.json", self.meta)
        return False
    def command(self, args, cwd=None, timeout=120, guard=None):
        index = len(list(self.path.glob("command-*.json"))) + 1
        prefix = self.path / f"command-{index:03d}"
        started = time.perf_counter()
        result={"args":list(map(str,args)),"cwd":str(cwd or ROOT),"stdout_path":str(prefix.with_suffix(".stdout.log")),"stderr_path":str(prefix.with_suffix(".stderr.log"))}
        self.event("command_started", **result)
        process=None
        try:
            with prefix.with_suffix(".stdout.log").open("w",encoding="utf-8") as out, prefix.with_suffix(".stderr.log").open("w",encoding="utf-8") as err:
                options={"creationflags":subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=="nt" else {"start_new_session":True}
                process=subprocess.Popen(result["args"],cwd=cwd or ROOT,stdout=out,stderr=err,text=True,**options)
                self.event("process_started",pid=process.pid,args=result["args"])
                try:
                    while process.poll() is None:
                        if guard: guard.check()
                        if time.perf_counter()-started>timeout: raise TimeoutError(f"Command exceeded {timeout}s")
                        time.sleep(0.25)
                except BaseException:
                    if os.name=="nt":
                        subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],stdout=out,stderr=err,timeout=15,check=False)
                    else:
                        owned=_linux_process_tree(process.pid)
                        _signal_linux_processes(owned,signal.SIGTERM)
                    try: process.wait(10)
                    except subprocess.TimeoutExpired:
                        if os.name=="nt": process.kill()
                        else: _signal_linux_processes(owned,signal.SIGKILL)
                        process.wait()
                    if os.name!="nt": _signal_linux_processes(owned,signal.SIGKILL)
                    raise
            result["returncode"]=process.returncode
            if process.returncode:
                tail=prefix.with_suffix(".stderr.log").read_text(encoding="utf-8",errors="replace")[-4000:]
                raise RuntimeError(f"Command failed ({process.returncode}): {args[0]}\n{tail}")
            return result
        except BaseException as exc:
            result.update(error_type=type(exc).__name__,error=str(exc),returncode=process.returncode if process else None)
            raise
        finally:
            result["elapsed_seconds"]=time.perf_counter()-started
            write_json(prefix.with_suffix(".json"),result)
            self.event("command_finished",**result)


def versions():
    packages = {}
    for name in ["torch","transformers","peft","bitsandbytes","accelerate","huggingface-hub","datasets","pyarrow","tokenizers","fastapi","uvicorn","pytest"]:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {"python":sys.version,"executable":sys.executable,"packages":packages}
