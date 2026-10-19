# Cleanup report

Completed on 2026-10-19. Removed 31,954 disposable files (590,175,555 logical bytes): Python bytecode, temporary build/test files, package/test caches, and the superseded Windows validation environment. Temporary files and the old environment creation settings were archived before deletion.

Removed the failed no-browser Docker runtime and six exclusively associated build-cache records. Docker reported about 1.89 GB reclaimed inside WSL. This does not imply an equivalent reduction of the Windows VHDX file. Validated images, unrelated caches, and existing containers were preserved.

Verification compared 54,926 protected files against the pre-cleanup inventory, with no missing or changed files. Models, datasets, checkpoints, active Ubuntu environments, source, and historical experiment logs were retained. Package metadata checks passed for the active training, agent, and evaluation environments.

Latest recorded storage: 45.03 GB free on C: and approximately 12.15 GB remaining under the approved 70 GB incremental-growth cap. The next uncached image operation reserves 15 GB, so the full development run is still gated.

Evidence:

- [Main cleanup verification](../results/cleanup/20261019T225700-0e486164/verification.json)
- [Exact deletion plan](../results/cleanup/20261019T225700-0e486164/deletion_plan.json)
- [Temporary-file recovery archive](../results/cleanup/20261019T225700-0e486164/temporary_files_recovery.zip)
- [Protected-file verification](../results/cleanup/20261019T225700-0e486164/protected_files_verification.json)
- [Exclusive Docker-cache verification](../results/cleanup_cache/20261019T225901-1d945a7f/verification.json)
- [Refreshed inventory](../manifests/artifact_inventory.json)

Cache removal used exact ID filters documented in the [Docker buildx prune reference](https://docs.docker.com/reference/cli/docker/buildx/prune/#filter). No broad prune or VHDX compaction was performed.
