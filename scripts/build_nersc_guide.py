"""Build a copyable Word runbook from the actual NERSC shell scripts."""
from pathlib import Path
import json
import re

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables"
OUT.mkdir(exist_ok=True)
PAGES = []


def page(title, *blocks):
    PAGES.append({"title": title, "blocks": list(blocks)})


def prose(text): return {"kind": "prose", "text": text}
def code(text): return {"kind": "code", "text": text.strip()}
def heading(text): return {"kind": "heading", "text": text}
def file_script(path):
    text = (ROOT / path).read_text().rstrip()
    return code(f"mkdir -p cloud/nersc\ncat > {path} <<'BASH'\n{text}\nBASH")


page("Download the models",
    prose("Run this guide in a Perlmutter terminal. Iris manages NERSC accounts and allocations; these A100 job examples target Perlmutter. Use one 40 GB A100 first, with two or four GPUs available later. [1, 2]"),
    prose("Prerequisite: the accompanying swe-study-a100-nersc-20261123.tar.gz source bundle is extracted as $SCRATCH/swe-study-a100. The final appendix gives the extraction commands if needed. Copy each code block in order, without adding shell prompt characters. The application source is in the bundle; every shell script used below is reproduced in this document."),
    prose("Start on the login node. This block downloads the official 7B snapshot at the project's frozen revision and checks its file sizes and LFS hashes. No GPU allocation is needed for downloading."),
    code(r'''export STUDY_ROOT="$SCRATCH/swe-study-a100"
cd "$STUDY_ROOT"
test -f scripts/cloud.py
mkdir -p logs
showquota'''),
    prose("Check showquota for at least about 100 GB of remaining project space, then run the download command. If less is available, stop before downloading. The script creates a small Python 3.12 bootstrap environment and keeps caches inside the project. The 7B snapshot is about 15.2 GB. [3]"),
    code("bash cloud/nersc/download_models.sh 7B"),
    prose("For the optional size comparison, download the other two models now. All three snapshots total about 24.5 GB. Repeated downloads verify and reuse matching files."),
    code(r'''bash cloud/nersc/download_models.sh 1.5B
bash cloud/nersc/download_models.sh 3B'''),
    heading("Install training dependencies and download datasets"),
    code(r'''bash cloud/nersc/setup_login.sh'''),
    prose("This installs the pinned training stack and downloads the five frozen datasets. It intentionally defers the CUDA check until a GPU job. Model/data preparation never silently falls back to CPU training. Keep this project directory in the same location after its storage baseline is created."),
)

page("Prepare the training data",
    prose("Run on the login node. Enter the CPU and GPU allocation account names shown in Iris; they may differ. CPU preparation is submitted to Slurm so tokenization does not run as a heavy login-node task. The environment variables remain available in this terminal. [1, 2]"),
    code(r'''read -rp "NERSC CPU allocation account: " CPU_ACCOUNT
read -rp "NERSC GPU allocation account: " GPU_ACCOUNT
export CPU_ACCOUNT GPU_ACCOUNT
cd "$STUDY_ROOT"
mkdir -p logs'''),
    prose("The following complete script is already in the source bundle. Paste the block to recreate it if needed. The quoted BASH delimiter keeps Slurm variables literal until job execution."),
    file_script("cloud/nersc/prepare.slurm"),
    code(r'''DATA_JOB=$(sbatch --parsable -A "$CPU_ACCOUNT" \
  cloud/nersc/prepare.slurm)
DATA_JOB=${DATA_JOB%%;*}
printf 'Data job %s\n' "$DATA_JOB"
squeue -j "$DATA_JOB"
# After the job leaves the queue:
sacct -j "$DATA_JOB" --format=JobID,State,ExitCode,Elapsed
tail -n 30 "logs/swe-data-${DATA_JOB}.out"'''),
    prose("Wait for COMPLETED with ExitCode 0:0 before GPU checks. At 32K, the existing audit retained 400 training and 43 validation histories, about 90.6% of the 489 traceable trajectories. At 16K it retained 229/30; at 8K, 37/9. Overlength histories are reported and excluded, never truncated. Confirm your generated tokenization manifest."),
)

page("Create the GPU job launcher",
    prose("Run this block on the login node. This reusable launcher requests one A100 40 GB in the shared queue and executes the cloud command supplied after its filename. One Slurm task launches torchrun internally. NERSC recommends the shared queue for one or two GPUs. [2, 4, 5]"),
    file_script("cloud/nersc/gpu_job.slurm"),
    heading("Check the installed CUDA and distributed training path"),
    code(r'''CHECK_JOB=$(sbatch --parsable -A "$GPU_ACCOUNT" \
  -J swe-check cloud/nersc/gpu_job.slurm \
  check-distributed --gpus 1)
CHECK_JOB=${CHECK_JOB%%;*}
squeue -j "$CHECK_JOB"
# After completion:
sacct -j "$CHECK_JOB" --format=JobID,State,ExitCode,Elapsed
tail -n 40 "logs/swe-check-${CHECK_JOB}.out"'''),
    prose("Continue when the job is COMPLETED, ExitCode is 0:0, and the diagnostic reports passed true. The tiny random-model diagnostic tests real NF4/LoRA training, checkpoint saving and fresh reload. It does not establish real 7B memory feasibility or model quality."),
)

page("Test memory and train the 7B adapter",
    prose("Submit from the login node after the CPU preparation and tiny GPU diagnostic pass. First test the real 7B model at 32K for only two optimizer steps. This setting covers substantially more trajectories than 8K, but its memory use must be measured on your A100."),
    code(r'''PROBE_JOB=$(sbatch --parsable -A "$GPU_ACCOUNT" \
  -J swe-32k-check cloud/nersc/gpu_job.slurm \
  train --size 7B --gpus 1 --max-length 32768 --max-steps 2 \
  --output checkpoints/cloud/qwen-7b-32k-check)
PROBE_JOB=${PROBE_JOB%%;*}
squeue -j "$PROBE_JOB"
# After completion:
sacct -j "$PROBE_JOB" --format=JobID,State,ExitCode,Elapsed
tail -n 40 "logs/swe-32k-check-${PROBE_JOB}.out"'''),
    prose("Continue only if the job completes and the fresh-process adapter reload passes. If 32K runs out of memory, keep the failed run's logs. Prepare at 16384 and use 16384 consistently for the new test, full run and inference, with fresh output names. More DDP GPUs do not pool VRAM for a single sequence."),
    heading("Run three training epochs on one GPU"),
    code(r'''TRAIN_JOB=$(sbatch --parsable -A "$GPU_ACCOUNT" \
  -J swe-7b-train --time=24:00:00 \
  cloud/nersc/gpu_job.slurm \
  train --size 7B --gpus 1 --max-length 32768 \
  --output checkpoints/cloud/qwen-7b-32k)
TRAIN_JOB=${TRAIN_JOB%%;*}
squeue -j "$TRAIN_JOB"
# These commands can be repeated while checking progress:
tail -n 30 "logs/swe-7b-train-${TRAIN_JOB}.out"
sacct -j "$TRAIN_JOB" --format=JobID,State,ExitCode,Elapsed'''),
    prose("The 24-hour wall limit is a scheduling allowance, not a runtime estimate. Defaults are NF4/BF16, LoRA rank 16/alpha 32, learning rate 1e-4, microbatch 1, gradient accumulation 8, and target-only assistant causal loss. The final adapter is selected by validation loss and independently reloaded. Do not submit the same output twice."),
)

page("Resume training and run inference",
    heading("Resume an interrupted training job"),
    prose("Run after the interrupted job has ended. Choose a complete checkpoint directory from the output. Replace the example checkpoint number below with one actually present. Reuse the same data, GPU count, context, output and epoch schedule; the saved plan checks these values."),
    code(r'''ls -d checkpoints/cloud/qwen-7b-32k/checkpoint-*
CHECKPOINT=checkpoints/cloud/qwen-7b-32k/checkpoint-50
sbatch -A "$GPU_ACCOUNT" -J swe-7b-resume --time=24:00:00 \
  cloud/nersc/gpu_job.slurm \
  train --size 7B --gpus 1 --max-length 32768 \
  --output checkpoints/cloud/qwen-7b-32k --resume "$CHECKPOINT"'''),
    prose("A complete checkpoint includes adapter, optimizer, scheduler, Trainer state and RNG files. If no complete checkpoint was saved, preserve the failed output and start with a fresh --output name. Do not delete the plan or reset storage accounting to bypass a failure."),
    heading("Generate with the trained adapter"),
    code(r'''INFER_OUT="results/nersc-adapter-$(date -u +%Y%m%dT%H%M%S).json"
INFER_JOB=$(sbatch --parsable -A "$GPU_ACCOUNT" \
  -J swe-infer --time=00:30:00 cloud/nersc/gpu_job.slurm \
  infer --size 7B --adapter checkpoints/cloud/qwen-7b-32k \
  --max-context 32768 --max-new-tokens 512 \
  --prompt "Write a Python function that merges two sorted lists." \
  --output "$INFER_OUT")
INFER_JOB=${INFER_JOB%%;*}
# After COMPLETED with ExitCode 0:0:
cat "$INFER_OUT"'''),
    heading("Generate with the baseline"),
    code(r'''sbatch -A "$GPU_ACCOUNT" -J swe-baseline --time=00:30:00 \
  cloud/nersc/gpu_job.slurm \
  infer --size 7B --max-context 32768 --max-new-tokens 512 \
  --prompt "Write a Python function that merges two sorted lists."'''),
    prose("Inference uses one GPU. Omitting --adapter selects the baseline. Both paths retain the same NF4/unquantized-layer precision and BF16 autocast. Named JSON outputs must be fresh; every invocation also writes an audited generation.json. Tool-like text from the model does not execute tools or constitute a SWE-bench score."),
)

page("Use two or four GPUs when needed",
    prose("Use additional GPUs for data parallel throughput after the one-GPU workflow passes. A full model replica lives on each GPU, so per-GPU context memory is unchanged. The default effective batch remains eight: accumulation is 8, 4 or 2 on one, two or four GPUs. [4, 5]"),
    heading("Two GPUs in the shared queue"),
    code(r'''sbatch -A "$GPU_ACCOUNT" -J swe-check-2 \
  --gpus-per-task=2 --cpus-per-task=64 \
  --export=ALL,STUDY_GPUS=2 cloud/nersc/gpu_job.slurm \
  check-distributed --gpus 2
# After the diagnostic passes, run a separate training condition:
sbatch -A "$GPU_ACCOUNT" -J swe-7b-2gpu --time=24:00:00 \
  --gpus-per-task=2 --cpus-per-task=64 \
  --export=ALL,STUDY_GPUS=2 cloud/nersc/gpu_job.slurm \
  train --size 7B --gpus 2 --max-length 32768 \
  --output checkpoints/cloud/qwen-7b-32k-2gpu'''),
    heading("Four GPUs in the regular queue"),
    code(r'''sbatch -A "$GPU_ACCOUNT" -J swe-check-4 --qos=regular \
  --gpus-per-task=4 --cpus-per-task=128 \
  --export=ALL,STUDY_GPUS=4 cloud/nersc/gpu_job.slurm \
  check-distributed --gpus 4
# After the diagnostic passes:
sbatch -A "$GPU_ACCOUNT" -J swe-7b-4gpu --time=24:00:00 \
  --qos=regular --gpus-per-task=4 --cpus-per-task=128 \
  --export=ALL,STUDY_GPUS=4 cloud/nersc/gpu_job.slurm \
  train --size 7B --gpus 4 --max-length 32768 \
  --output checkpoints/cloud/qwen-7b-32k-4gpu'''),
    prose("Choose one training condition; these optional examples are not an instruction to run all three. Keep --ntasks=1 and use the same GPU count on resume. NERSC shared GPU jobs support one or two GPUs; a regular-queue node is exclusive. Verify your allocation and queue access in Iris. [1, 4]"),
    heading("Optional model size comparison"),
    prose("Download 1.5B or 3B using the first page's model commands. Repeat CPU preparation with MODEL_SIZE=3B (or 1.5B), then use the same GPU commands with --size 3B and a fresh output such as checkpoints/cloud/qwen-3b-32k. Keep the task split and context policy common across comparisons."),
)

page("Keep the results and resolve common failures",
    prose("After GPU jobs finish, inspect metrics and create a backup. Perlmutter scratch is not backed up and is subject to purging; copy important adapters and evidence to an approved persistent destination. The archive below excludes base weights because their exact revisions can be downloaded again. [6]"),
    code(r'''cd "$STUDY_ROOT"
.envs/train/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("checkpoints/cloud/qwen-7b-32k/experiment.json")
m = json.loads(p.read_text())
for k in ["global_step", "best_eval_loss", "gpu_count", "gpu_hours",
          "train_examples", "validation_examples", "peak_allocated_bytes"]:
    print(k, m[k])
PY
mkdir -p backups
tar -czf "backups/adapter-$(date -u +%Y%m%dT%H%M%S).tar.gz" \
  checkpoints/cloud/qwen-7b-32k \
  checkpoints/cloud/qwen-7b-32k.plan.json manifests \
  data/metadata results/cloud_train results/cloud_train_rank0 \
  results/validate-checkpoint'''),
    prose("The 70 GB project-growth cap and 25 GB filesystem-free floor remain enabled. Project accounting ignores other users' filesystem growth; its directory-size sample is cached for up to 60 seconds. It does not replace NERSC user quotas: run showquota before large operations. Keep backups within the remaining budget. [3]"),
    prose("No CUDA device: run through the GPU launcher, not directly on the login node. Submission rejected: verify the CPU/GPU account, constraint, QOS and GPU count. Out of memory: use the 16K fallback below. Nonempty output: resume a complete checkpoint or choose a fresh output. For any other error, keep the job ID, Slurm .out/.err and results run directory."),
    heading("Prepare and test the 16K fallback"),
    code(r'''sbatch -A "$CPU_ACCOUNT" --export=ALL,CONTEXT=16384 \
  cloud/nersc/prepare.slurm
# After CPU preparation completes:
sbatch -A "$GPU_ACCOUNT" -J swe-16k-check \
  cloud/nersc/gpu_job.slurm \
  train --size 7B --gpus 1 --max-length 16384 --max-steps 2 \
  --output checkpoints/cloud/qwen-7b-16k-check'''),
    prose("If the fallback passes, train with --max-length 16384 and a fresh qwen-7b-16k output, then infer with --max-context 16384. Actual A100 memory use, distributed behavior and throughput must be verified by your submitted checks; local random-model tests do not establish benchmark quality."),
)

page("Script appendix for environment and model downloads",
    prose("These are the complete shell helpers invoked on the first page. They are included in the source bundle. Paste a complete block if you need to recreate the corresponding file. Do not execute the individual lines from inside a script separately."),
    file_script("cloud/nersc/env.sh"),
    file_script("cloud/nersc/download_models.sh"),
)

page("Script appendix for setup and the core launcher",
    prose("This complete setup helper installs the training environment on a login node, defers GPU validation and downloads the frozen datasets."),
    file_script("cloud/nersc/setup_login.sh"),
    prose("The Slurm launcher invokes this complete core shell entry point. It selects the training environment and preserves command arguments exactly as supplied."),
    file_script("cloud/run.sh"),
)

page("Source bundle and official references",
    heading("Extract the accompanying source bundle if needed"),
    prose("Use this only before the first page, in a fresh project directory. Place swe-study-a100-nersc-20261123.tar.gz and its .sha256 sidecar in your NERSC home directory using your usual file-transfer method. Do not copy a workstation environment or storage baseline."),
    code(r'''cd "$HOME"
sha256sum -c swe-study-a100-nersc-20261123.tar.gz.sha256
if [[ -e "$SCRATCH/swe-study-a100" ]]; then
  printf '%s\n' 'Target exists; preserve it and choose a fresh directory.'
else
  tar -xzf swe-study-a100-nersc-20261123.tar.gz -C "$SCRATCH"
  export STUDY_ROOT="$SCRATCH/swe-study-a100"
  cd "$STUDY_ROOT"
fi'''),
    heading("Official NERSC references"),
    prose("[1] Iris account and allocation guide\nhttps://docs.nersc.gov/iris/iris-for-users/"),
    prose("[2] Running jobs on Perlmutter\nhttps://docs.nersc.gov/systems/perlmutter/running-jobs/"),
    prose("[3] Storage quotas\nhttps://docs.nersc.gov/filesystems/quotas/"),
    prose("[4] Queue policy and charges\nhttps://docs.nersc.gov/jobs/policy/"),
    prose("[5] Training launchers\nhttps://docs.nersc.gov/machinelearning/launchers/"),
    prose("[6] Scratch backup and purge policy\nhttps://docs.nersc.gov/filesystems/perlmutter-scratch/"),
    prose("Project dependency pins: PyTorch 2.9.1 CUDA 12.6, Transformers 5.5.3, PEFT 0.18.1, bitsandbytes 0.49.2. The generated package freeze and downloaded asset manifests record the exact environment and revisions used by your run."),
)


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin = sec.bottom_margin = Inches(0.62)
    sec.left_margin = sec.right_margin = Inches(0.65)
    for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2"]:
        style = doc.styles[name]
        style.font.name = "Arial"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_after = Pt(6)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing = 1.04
    doc.styles["Title"].font.size = Pt(21)
    doc.styles["Heading 1"].font.size = Pt(16)
    doc.styles["Heading 2"].font.size = Pt(11)
    mono = doc.styles.add_style("Shell Code", 1)
    mono.font.name, mono.font.size = "Courier New", Pt(8.2)
    mono.font.color.rgb = RGBColor(0, 0, 0)
    mono.paragraph_format.line_spacing = Pt(9.6)
    mono.paragraph_format.space_after = Pt(7)
    mono.paragraph_format.space_before = Pt(2)
    mono.paragraph_format.keep_together = True
    mono.paragraph_format.widow_control = False
    mono.paragraph_format.left_indent = Inches(0.08)
    for index, pg in enumerate(PAGES):
        if index: doc.add_page_break()
        else:
            doc.add_paragraph("NERSC A100 Training and Inference", "Title")
        doc.add_paragraph(pg["title"], "Heading 1")
        for block in pg["blocks"]:
            if block["kind"] == "heading": doc.add_paragraph(block["text"], "Heading 2")
            else: doc.add_paragraph(block["text"], "Shell Code" if block["kind"] == "code" else "Normal")
    footer = sec.footer.paragraphs[0]
    footer.alignment = 2
    run = footer.add_run("NERSC A100 workflow  |  ")
    run.font.name, run.font.size = "Arial", Pt(8)
    field = OxmlElement("w:fldSimple");field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    doc.core_properties.title = "NERSC A100 Training and Inference"
    doc.core_properties.subject = "Copy and paste model download training and inference commands"
    doc.core_properties.author = ""
    path = OUT / "NERSC_A100_Training_and_Inference.docx"
    doc.save(path)
    (OUT / "nersc_guide_content.json").write_text(json.dumps(PAGES, indent=2) + "\n")
    # Verify that Word stores all code as literal selectable text, with no smart quotes.
    reread = Document(path)
    saved = [p.text for p in reread.paragraphs if p.style.name == "Shell Code"]
    expected = [b["text"] for pg in PAGES for b in pg["blocks"] if b["kind"] == "code"]
    assert saved == expected
    assert all(not re.search("[\u2018\u2019\u201c\u201d]", text) for text in saved)
    print(json.dumps({"docx": str(path), "planned_pages": len(PAGES), "code_blocks": len(saved)}))


if __name__ == "__main__": build()
