# Development environment constraints

Updated 2026-10-19. GPU training feasibility is validated for all three model sizes under the measured NF4/LoRA 8K settings. The outstanding five-task integration work is constrained by Docker environments and disk space.

## Validated and unavailable images

| Frozen development task | Current finding | Next action |
| --- | --- | --- |
| pvlib__pvlib-python-807 | Published image fails collection under NumPy 2. Dependency-only NumPy 1.26.4 derivative passes gold control; complete runtime cached. | Reuse pinned image/runtime. |
| pvlib__pvlib-python-1048 | Same NumPy incompatibility. Repaired runtime passes gold control; no benchmark source/test changes. | Reuse pinned repaired runtime. |
| pydicom__pydicom-1048 | Public linux/amd64 image manifest available; 1.623 GB compressed layers; not cached. | Pull, record digest/expanded size, validate gold control. |
| pylint-dev__astroid-1741 | Public linux/amd64 image manifest available; 0.979 GB compressed layers; not cached. | Pull, record digest/expanded size, validate gold control. |
| sqlfluff__sqlfluff-2907 | Public registry request returned HTTP 401 for the published image. Its accessibility is unresolved. | Check access, then build from the pinned task Dockerfile if necessary; validate gold control. |

Registry evidence: [read-only image audit](../results/development_image_audit/20261019T221340-e9f6a94c/images.json). A 401 does not distinguish a private image from an absent public repository. The frozen task list is unchanged; failures do not justify replacing tasks.

Repairs and original failures: [image overrides](../manifests/image_overrides.json), [1048 failed gold control](../results/grade/20261019T220416-957bc845/run.json), [1048 repaired gold control](../results/grade/20261019T221223-4c466ee4/grade_validation.json), [repair audit](../results/task_environment_repair/20261019T221216-425c93ad/repair_validation.json). Empty-prediction controls check denominator/empty handling; the official harness skips tests for them.

## Storage gate

The image audit measured 45.19 GB free on C:, corresponding to 57.69 GB conservative growth from the recorded baseline. The approved cap is 70 GB growth, leaving about 12.31 GB of growth headroom; the free-space floor is 25 GB. Docker's Ubuntu VHDX is on C: and is included. Values are decimal GB and can change with other machine activity.

Uncached image grading reserves 15 GB before pulling/building. The next pydicom gold control was stopped by this gate before any image download: [storage rejection](../results/grade/20261019T221411-5174df90/run.json). Cached grading reserves 2 GB because its images already exist. Continuous guards preserve the same hard cap and free-space floor in either case.

Measured complete runtimes are about 7.6 GB each, of which roughly 3.6 GB is runtime tooling above the task image; shared layers mean image sizes must not simply be summed as physical usage. The remaining three task images plus their runtimes may need roughly 20–30 GB persistent space, with an additional 5–10 GB of build/download headroom. This is a planning allowance, not a verified compressed-to-expanded conversion. SQLFluff builds have unpinned historical dependencies, so their final size and compatibility remain uncertain.

Before full five-task execution: make approximately 30–40 GB more working headroom available on the current volume, or select another volume with at least 50 GB available for an explicitly scoped project storage arrangement. Preserve existing user containers, model/data snapshots, checkpoints, and logs. Removing images alone may not return Windows space from a non-sparse VHDX; no broad prune, global migration, or VHDX compaction has been performed. Recheck storage before authorizing any new layout or cap.

## Work that continues within the existing allowance

Real-model API checks and an explicitly labeled smoke sequence use the first frozen development task and its already cached, gold-validated runtime. Each smoke episode has five agent steps and a 300-second task limit. Baseline/adapter conditions run sequentially with separate token ledgers and official patch export/grading. These diagnostics do not replace the five-task comparison or establish a benchmark score.

The interrupted full development sequence and early NumPy-incompatible attempts remain archived. Both milestone acceptance reports stay incomplete until the full documented checks pass.

## Executed negative grader control

A deliberately incorrect numerical derivative was inserted into the reference patch for the already validated pvlib__pvlib-python-1048 environment. The official harness applied this nonempty patch and executed all five target tests: four passed and exactly the derivative test failed, with no infrastructure failure. This confirms actual failing-test detection in addition to empty-prediction handling. The altered reference is grader-only and was never supplied to an agent. [Executed negative-control evidence](../results/executed_grader_negative/20261019T222237-60baffd4/negative_validation.json).

Final check after the four bounded diagnostics: 44.44 GB free, 58.45 GB conservative growth, 11.55 GB cap headroom. The full five-task gates remain. [Cleanup and storage verification](../results/progress_finalization/20261019T224903-f9f33887/finalization.json).
