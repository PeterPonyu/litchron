# Desktop Cleanup Advisory — 2026-05-18

Generated as the deferred-component sidecar from the LitChron deep interview (`.omc/specs/deep-interview-litchron-llm-pseudotime.md`). User chose **"Defer cleanup, focus on system"** in Round 0, so this advisory is recorded for later action, not executed now.

**No deletions have been performed.** Each row is a candidate the user can review independently.

## Tier 1 — Largest reclaimable space (estimated >50 GB each)

| Path | Size | Action candidate | Risk | Notes |
|---|---|---|---|---|
| `.archive/GEO-DataHub/.git` | ~57 GB | `rm -rf` | low | MEMORY.md already labels `.git` as deletable; archive remains usable without git history |
| `.archive/GEO-DataHub/downloads/` | ~78 GB | `rm -rf` | low | MEMORY.md flags as deletable; raw GEO downloads can be re-fetched if needed |
| `.archive/GEO-DataHub/h5ad_output/` | ~73 GB | review then `rm -rf` | med | 73 GB of derived h5ad. Verify `CLOP-DiT/01b_integrate_geodh_h5ad.py` no longer reads this path before deletion |

**Tier 1 reclaim estimate: ~135 GB** (excluding h5ad_output until verified).

## Tier 2 — Floating worktrees that belong under `contrib/` per STRUCTURE.md

`STRUCTURE.md` mandates `contrib/<repo>/{main, worktrees/<N>-<slug>/}`. The following live at Desktop root and look like ad-hoc worktrees:

| Path | Size | Action | Notes |
|---|---|---|---|
| `opencode/` | 7.5 GB | move to `contrib/opencode/main/` | Working copy of fork's default branch |
| `opencode-24276` | 2.7 GB | check PR status; move to `contrib/opencode/worktrees/24276-<slug>/` or delete if merged | PR# 24276 |
| `opencode-24447` | 499 MB | same | PR# 24447 |
| `opencode-25315` | 560 MB | same | PR# 25315 |
| `opencode-26062` | 502 MB | same | PR# 26062 |
| `opencode-27831` | 507 MB | same | PR# 27831 |
| `opencode-28015` | 504 MB | same | PR# 28015 |
| `oh-my-openagent/` | 1.6 GB | move to `contrib/oh-my-openagent/main/` | Default branch checkout |
| `oh-my-openagent-3607` | – | check PR status; move to `worktrees/3607-<slug>/` or delete | PR# 3607 |
| `oh-my-openagent-3726` | – | same | PR# 3726 |
| `oh-my-openagent-3937` | – | same | PR# 3937 |
| `oh-my-openagent-4002` | – | same | PR# 4002 |

**Action recipe:**
```bash
# For each opencode-NNNNN worktree, after the PR closes/merges:
cd ~/Desktop/contrib/opencode/main
git worktree remove ~/Desktop/opencode-NNNNN     # if it's a real worktree of contrib/opencode/main
# OR if it's a separate clone:
rm -rf ~/Desktop/opencode-NNNNN
```

Tier 2 reclaim estimate: ~13 GB if all merged PRs are deleted.

## Tier 3 — Emberforge variants (likely dev parallels)

| Path | Size | Action |
|---|---|---|
| `emberforge/` | 19 GB | keep (active) |
| `emberforge-architecture-clean/` | 4.7 GB | review: is this still a working compare? |
| `emberforge-architecture-docs/` | 3.6 GB | review |
| `emberforge-buddy-clean/` | 5.1 GB | review |
| `emberforge-resume-main/` | 4.4 GB | review |
| `emberforge-translations/` | – | review |

**Decision needed from user**: are any of the five sibling forks superseded? If so, archive or delete. Potential reclaim: ~18–22 GB.

## Tier 4 — Lab project versioning

| Path | Size | Status candidate |
|---|---|---|
| `iaode/` | 434 MB | likely superseded by iAODE-LAB |
| `iAODE_dev/` | 2.5 GB | likely dev branch, possibly merged |
| `iAODE-LAB/` | 46 GB | active (keep) |
| `MCC-LAB/` | 3.3 GB | active (keep) |
| `MCC-revision/` | – | review |
| `MCC-previous/` | – | strong archive candidate (name implies superseded) |

User can move `MCC-previous/` to `labs/_previous/MCC/` per STRUCTURE.md, then archive after 90 idle days.

## Tier 5 — LaTeX and notebook intermediates

| Path | Size | Action |
|---|---|---|
| `CCVGAE_snLaTeX/` | 27 GB | run `latexmk -C` to clean intermediate `.aux/.log/.fls/.synctex.gz`; check for stray `_minted-*` cache directories |
| `.ipynb_checkpoints/` (root) | – | safe to delete; Jupyter regenerates |
| `unsloth_compiled_cache/` | – | safe to delete; unsloth regenerates on next run |
| `logs/` (root) | – | review for staleness; delete entries > 90 days |

LaTeX intermediates alone could reclaim multiple GB on `CCVGAE_snLaTeX` (typical 10–30% of project size for builds with figures).

## Tier 6 — Unknown or suspicious entries

| Path | Concern | Recommended action |
|---|---|---|
| `tocken.txt` | Typo of "token.txt" at Desktop root | **Inspect immediately** for secrets; if it contains tokens/keys, rotate and move to a secrets manager; if empty/stale, delete |
| `WC.` (3.6 GB) | Trailing-dot folder, unclear purpose | Inspect contents; rename or archive |
| `scivcd` | Unclear purpose | Inspect; archive if transient |
| `scMetaIntel-Hub-teamrun` | Likely a transient team-run output | Verify against `scMetaIntel-Hub/` then delete |
| `datasets/extra_preprocessed/` (2.8 GB) | Unclear vs primary `CancerDatasets/` etc. | Confirm provenance before action |
| `datasets/IRALL.h5ad`, `wtko0312.h5ad` | Orphan h5ads outside subdirs | Move into a labeled subfolder under `datasets/` |

## Tier 7 — v1 vs v2 dataset pairs

| Pair | Total size | Suspicion |
|---|---|---|
| `datasets/CancerDatasets/` (44 GB) + `CancerDatasets2/` (16 GB) | 60 GB | Possibly v1/v2 with overlap. Run `rsync -n --delete --checksum` between them or `du --inodes` to estimate dedup potential |
| `datasets/DevelopmentDatasets/` (13 GB) + `DevelopmentDatasets2/` (18 GB) | 31 GB | Same pattern |

Potential reclaim depends on overlap — could be 10–30 GB.

## Tier 8 — STRUCTURE.md migration backlog

Per `~/Desktop/STRUCTURE.md`, the following directories are scheduled to migrate to their canonical slots "at the natural moment". Migration is **not urgent**; record only:

| From | To |
|---|---|
| `oh-my-claudecode-main/`, `oh-my-codex-main/`, `oh-my-copilot/`, `oh-my-cursor/` | `tools/oh-my-*/` |
| `claude-code-src/`, `claw-code-parity-main/` | `tools/<name>/` |
| `scMetaIntel-Hub/`, `CLOP-DiT/`, `iAODE-LAB/`, `MCC-LAB/`, `Liora-LAB/`, `LiVAE-LAB/`, `MoCoO/`, `PanODE-Topic/`, `PanODE-DPMM/`, `emberforge/`, `scCCVGBen-assets/`, `CCVGAE_snLaTeX/` | `labs/active/<name>/` |

## Summary

If the user later approves cleanup execution, expected reclaim is **150–200 GB** dominated by `.archive/GEO-DataHub` deletions and dataset deduplication. Plus structural wins (moving worktrees into `contrib/`, surfacing the `tocken.txt` secret risk).

After LitChron scaffold is built, this file should move to `labs/active/litchron/docs/desktop-cleanup-advisory.md`.
