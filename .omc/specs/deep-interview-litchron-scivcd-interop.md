# Deep Interview Spec: litchron ↔ scivcd Interop via Meta-Plugin

## Metadata
- Interview ID: litchron-scivcd-interop-2026-05-19
- Rounds: 7 (Round 0 topology + 7 ETCLOVG dimension rounds)
- Final Ambiguity Score: 9.5%
- Type: brownfield
- Generated: 2026-05-19
- Threshold: 20%
- Initial Context Summarized: yes
- Status: PASSED
- Framework: ETCLOVG (Execution, Tools, Context, Lifecycle/Orchestration, Observation, Validation, Government) — user-supplied dimension lens applied to all 5 components

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 0.35 | 0.333 |
| Constraint Clarity | 0.90 | 0.25 | 0.225 |
| Success Criteria | 0.85 | 0.25 | 0.213 |
| Context Clarity | 0.90 | 0.15 | 0.135 |
| **Total Clarity** | | | **0.905** |
| **Ambiguity** | | | **0.095** |

## Topology

All 5 components confirmed active in Round 0. No deferrals.

| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| scivcd-plugin-mcp | active | Audit engine, 7-tool MCP, standalone plugin manifest | Covered by E/T/L/V — no internal changes required; keeps existing standalone plugin |
| litchron-mcp | active | 18-tool pipeline MCP; gains `emit_figure_script` tool + new `plugin/` directory | Covered by T (new tool), L (session-per-run), and packaging decision |
| cross-mcp-integration | active | Script-path handshake orchestrated by `/audit-litchron-run` skill in litchron's plugin | Covered by E (bundled meta-plugin), T (handshake shape), L (session lifetime) |
| routine-governance | active | Author-owned baselines, version-pinned scivcd, IGNORED decommissioned | Covered by V (baseline-relative gate) + G (author-owned policy) |
| shared-schema-contract | active | Convention-only: reuse scivcd's existing finding/baseline shapes + emit_figure_script return + run_dir audit log | Covered by C + O |

## Goal

Build a **meta-plugin in litchron's repo** that exposes the existing `mcp_litchron` MCP and the existing `scivcd` MCP under one Claude Code plugin manifest, plus a shared orchestration skill `/audit-litchron-run` that chains them via a **script-path handshake**: the skill calls a new `litchron.emit_figure_script` tool, then feeds the emitted driver to `scivcd.start_session`, then audits via the existing scivcd tools, all bounded by **one scivcd session per litchron run** and gated by a **baseline-relative validation policy** with **author-owned, self-healing baselines** persisted under `run_dir/audit/`.

Both MCPs stay independent (no import-time coupling, no shared schema package). The shared contract is convention: scivcd's existing finding/baseline JSON shapes + a new minimal `emit_figure_script` return shape. scivcd is pinned via a litchron `[audit]` pyproject extra.

## Constraints

- **Each MCP stays independent.** No Python imports across mcp_litchron ↔ scivcd at module-import time.
- **scivcd version is pinned** in litchron's `pyproject.toml` `[project.optional-dependencies] audit = ["scivcd==X.Y.Z"]`. Version bumps are deliberate PRs that re-bless baselines.
- **scivcd keeps its existing standalone `plugin/`** at `~/Desktop/scivcd/plugin/` for general matplotlib auditing. The new litchron meta-plugin lives at `~/Desktop/labs/active/litchron/plugin/`.
- **Session lifetime = run_id.** One `start_session` → audit all figures → one `end_session` per litchron run. Baseline keys are `(run_id, figure_name)`.
- **Convention-only schema.** No new Pydantic contract module, no JSON Schema in plugin manifest. Existing scivcd shapes pass through.
- **Audit log lives in run_dir.** `run_dir/audit/{figure_name}.findings.json` + `run_dir/audit/baseline.json`. No central audit DB.
- **Pre-commit auto-stages non-CRITICAL baseline deltas.** CRITICAL deltas block the commit.
- **IGNORED_CHECK_TYPES is decommissioned.** Today's `{"font_family_violation"}` migrates into baseline.json as pre-existing on first run. No new entries added going forward.
- **scripts/audit_figures.py is not deleted.** It remains the CI/pre-commit driver; the skill is the Claude-driven path. Both paths converge on the same baseline.json.

## Non-Goals

- **No shared in-process library coupling.** litchron does not import scivcd as a library (only as an MCP runtime dependency).
- **No third bridge repo.** No `scivcd-litchron-bridge` package.
- **No shared Figure registry.** No out-of-process pickle handoff of Figure objects.
- **No formal Pydantic/JSON-Schema cross-MCP contract.** Convention only.
- **No new scivcd tools.** The 7-tool surface (`start_session`, `audit_summary`, `audit_details`, `apply_autofix`, `save_baseline`, `diff_baseline`, `end_session`) is sufficient.
- **No reviewer-gated baseline workflow.** Author owns baseline diffs in the same commit as figure changes.
- **No soft-surface-only audit policy.** scivcd in CI is a blocker on regressions, not just a logger.
- **No re-scoring of the existing scivcd MCP design.** scivcd's `.omc/plans/scivcd-mcp-design-plan.md` stands; only US-011/US-012 verification is downstream.

## Acceptance Criteria

- [ ] `litchron/plugin/.claude-plugin/plugin.json` exists with `name: "litchron-audit"` and registers both MCPs.
- [ ] `litchron/plugin/.mcp.json` declares `mcpServers.litchron` (existing) and `mcpServers.scivcd` (referencing `python -m scivcd.mcp`).
- [ ] `litchron/plugin/skills/audit-litchron-run/SKILL.md` exists and documents the 4-step handshake.
- [ ] `mcp_litchron.tools.emit_figure_script(run_id, figure_name)` exists, returns `{script_path: str, figure_name: str, deps_pinned: bool}`, and is registered in `mcp_litchron/server.py`'s `build_registry()`.
- [ ] The emitted driver script is self-contained: loads run state, builds the named figure, leaves it in pyplot state (does not call `plt.close()`).
- [ ] Running `/audit-litchron-run` against an existing run_id produces `run_dir/audit/{figure_name}.findings.json` for every figure litchron knows how to build for that run, plus `run_dir/audit/baseline.json` if missing.
- [ ] `pyproject.toml` declares `[project.optional-dependencies] audit = ["scivcd==<pinned>"]`.
- [ ] Pre-commit hook (`.pre-commit-config.yaml`) runs `scripts/audit_figures.py` and auto-stages `run_dir/audit/baseline.json` when only non-CRITICAL deltas exist.
- [ ] Pre-commit hook fails the commit if any CRITICAL delta vs baseline is detected.
- [ ] `IGNORED_CHECK_TYPES` in `scripts/audit_figures.py` is emptied; the previously-suppressed `font_family_violation` findings appear in `run_dir/audit/baseline.json` after first run.
- [ ] scivcd's existing standalone `plugin/` is untouched.
- [ ] No `mcp_litchron` or `litchron` module imports `scivcd` at module import time (verified by grep).
- [ ] Skill end-to-end test: `claude --plugin-dir ~/Desktop/labs/active/litchron/plugin` and running `/audit-litchron-run --run-id wtko-hspc-2026-05-18` returns findings matching what `make audit` produces locally.
- [ ] scivcd-version-bump scenario: bumping the `[audit]` pin and re-running pre-commit produces a baseline diff PR that the author reviews before merging.

## Assumptions Exposed & Resolved

| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Audits should run in-process (status quo) | Round 1 forced an explicit execution-model choice across 4 alternatives | Bundled meta-plugin with each MCP independent |
| Figures should cross MCPs as live objects | Round 2 surfaced the script-path alternative which preserves audit fidelity | New `emit_figure_script` tool; live Figure objects never cross MCP boundary |
| One audit = one session (sessionless or per-figure assumed) | Round 3 forced session-lifetime decision | Session = run_id; baseline keys are (run_id, figure_name) |
| Hard gate on absolute MAJOR ceiling is correct (Contrarian) | Round 4 contrarian probe: "what if scivcd is a surfacer not a blocker?" | Hard gate retained but made baseline-relative — only NEW CRITICAL/MAJOR block; existing surfaces but doesn't block |
| Meta-plugin should live in scivcd or third repo | Round 5 packaging analysis showed litchron-as-host gives best UX for litchron users while preserving scivcd's standalone path | Meta-plugin in litchron; scivcd keeps standalone plugin |
| Cross-MCP contract needs formal Pydantic/JSON-Schema (Simplifier) | Round 6 simplifier probe: "what's the simplest version?" | Convention-only — reuse scivcd's existing JSON shapes; only new artifact is `emit_figure_script` return shape and run_dir audit log layout |
| IGNORED_CHECK_TYPES should stay for permanent rule disagreements | Round 7 G-policy analysis showed it competes with baseline.json and rots silently | Decommissioned — all currently-ignored findings move into baseline.json as pre-existing |
| Baseline changes need reviewer approval | Round 7 G-policy analysis | Author-owned — pre-commit auto-stages non-CRITICAL deltas in same commit as figure change |

## Technical Context

**scivcd (verified)** — `~/Desktop/scivcd/`:
- `scivcd/mcp/` has `__init__.py` (Agg backend verify), `__main__.py`, `server.py` (7 tools + USER_SCRIPT_LOCK), `runner.py` (in-process loader with asyncio.Lock + os.chdir + tempfile.mkdtemp), `sessions.py` (SessionStore with TTL 600s)
- `plugin/.claude-plugin/plugin.json` + `plugin/.mcp.json` + `plugin/skills/audit-figure/` already exist
- `pyproject.toml` has `[project.optional-dependencies] mcp = ["mcp>=1.27,<2.0", "tiktoken>=0.7,<1.0"]`
- Status: built; US-011 (pytest) and US-012 (manual acceptance via `claude --plugin-dir`) blocked on user shell actions — these stay blocked, not in scope here

**litchron (verified)** — `~/Desktop/labs/active/litchron/`:
- `mcp_litchron/` has `server.py` (build_registry, run_stdio_server, list_tools/call_tool dispatch), `tools.py` (18 pydantic-typed tools, module-level `_CACHE: AnnDataCache` + `_VERIFIER: CitationVerifier`, fcntl.flock state via `RunStateStore.update`), `cache.py`, `errors.py`
- `pyproject.toml` declares `mcp>=1.0` dependency; packages = `["litchron", "mcp_litchron"]`; Python `>=3.11,<3.14`
- `scripts/audit_figures.py` (existing) monkey-patches `plt.close`, imports `litchron.figures` + `mcp_litchron.tools` directly, runs `scivcd.detect_all_conflicts` in-process
- `Makefile` targets: `audit`, `audit-strict`, `audit-run RUN_ID=<id>`
- `.pre-commit-config.yaml` has local hook on `^litchron/figures\.py$`
- No `plugin/` or `.claude-plugin/` exists today
- Current audit ceiling: MAJOR=7 (3 colorblind in Okabe-Ito borderline + 2+ in pseudotime comparison strip)

**Integration today**: pure in-process Python imports. Both MCPs are bypassed in the audit path.

**Litchron's 18 MCP tools** (from MCP server registration): align_orderings, append_section, compare_orderings, compile_pdf, compute_litchron_pseudotime, compute_observations, finalize_run, load_h5ad, make_annotation_figure, propose_ordering, recompute_embeddings, report_status, run_baseline, search_crossref, search_europepmc, start_run, verify_doi, verify_pmid.

**Figures litchron produces per run** (from `litchron/figures.py`): `make_litchron_annotation_figure`, `make_pseudotime_comparison_strip` — these become the two `figure_name` values for `emit_figure_script`.

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| MetaPlugin | core domain | name, version, mcpServers, skills | lives in litchron/plugin/; registers ScivcdMCP + LitchronMCP |
| LitchronMCP | core domain | 18 existing tools + new `emit_figure_script` | produces FigureScript |
| ScivcdMCP | core domain | 7 tools (start_session/audit_summary/audit_details/apply_autofix/save_baseline/diff_baseline/end_session) | consumes FigureScript; produces AuditFinding + BaselineDelta |
| OrchestrationSkill | core domain | name=/audit-litchron-run, steps: 4-call chain per figure | orchestrates LitchronMCP → ScivcdMCP per RunId |
| Figure | supporting | run_id, figure_name, builder function | materialized in pyplot state by FigureScript |
| FigureScript | core domain | script_path, figure_name, deps_pinned | new return value of `emit_figure_script`; consumed by `scivcd.start_session(script_path=...)` |
| RunId | core domain | id string | session boundary; baseline namespace key |
| Session | supporting | scivcd session_id, run_id, figures: list[Figure] | one per RunId |
| AuditFinding | core domain | severity_level, type, detail, fix, figure_num | scivcd's existing shape; persisted to FindingsLog |
| BaselineKey | supporting | (run_id, figure_name) tuple | identifies a baseline entry |
| BaselineDelta | supporting | new[], removed[], unchanged[] | output of scivcd.diff_baseline; gates pre-commit |
| SeverityCeiling | external system | DECOMMISSIONED — replaced by BaselineDelta | no longer a first-class entity |
| MetaPluginManifest | supporting | plugin.json + .mcp.json + skills/audit-litchron-run/SKILL.md | lives at litchron/plugin/ |
| FindingsLog | core domain | path: run_dir/audit/{figure_name}.findings.json | per-figure JSON written by skill |
| RunDirAuditPath | supporting | run_dir/audit/ | directory containing FindingsLog + baseline.json |
| AutoStageHook | supporting | pre-commit hook behavior | auto-`git add` baseline.json on non-CRITICAL delta |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 8 | 8 | - | - | N/A |
| 2 | 10 | 2 (FigureScript, Session) | 0 | 8 | 80% |
| 3 | 11 | 1 (BaselineKey) | 0 | 10 | 91% |
| 4 | 12 | 1 (BaselineDelta) | 0 | 11 | 92% |
| 5 | 13 | 1 (MetaPluginManifest) | 0 | 12 | 92% |
| 6 | 15 | 2 (FindingsLog, RunDirAuditPath) | 0 | 13 | 87% |
| 7 | 16 | 1 (AutoStageHook) | 1 (SeverityCeiling → decommissioned but kept for trace) | 14 | 94% |

Convergence pattern: monotonic growth with stable core. SeverityCeiling is the only entity that *changed in meaning* (Round 4's contrarian probe redefined it from absolute MAJOR limit to baseline-relative delta). Stability ratio finishing at 94% indicates the domain model is solid.

## Interview Transcript
<details>
<summary>Full Q&A (Round 0 topology + 7 dimension rounds)</summary>

### Round 0 (Topology Gate)
**Q:** Is this 4-component topology right? Should any component be added, removed, merged, split, or explicitly deferred?
**A:** Add #5: Shared schema/contract.
**Topology:** 5 active components, no deferrals.

### Round 1 — E (Execution)
**Q:** Which execution model should anchor how scivcd and litchron 'work together'?
**A:** Bundled meta-plugin (each MCP independent, shared orchestration skill).
**Ambiguity:** 80%

### Round 2 — T (Tools, handshake)
**Q:** What's the orchestration handshake — what tools does the shared skill chain, and where does the Figure cross from litchron to scivcd?
**A:** Script-path handoff: new `emit_figure_script` tool in litchron; skill chains emit_figure_script → start_session → audit_summary → audit_details.
**Ambiguity:** 61%

### Round 3 — L (Lifecycle/Orchestration)
**Q:** What's the scivcd session lifetime in the meta-plugin world?
**A:** One session per litchron run; baselines naturally scoped to run_id.
**Ambiguity:** 61% (lifecycle locked but criteria still empty)

### Round 4 — V (Validation) [CONTRARIAN MODE]
**Q:** What does 'passing audit' mean? (Contrarian: what if scivcd in CI is a surfacer not a blocker?)
**A:** Hard gate, baseline-relative — pass = no NEW CRITICAL/MAJOR vs saved baseline for this run_id.
**Ambiguity:** 42%

### Round 5 — T (Tools, packaging)
**Q:** Where does the meta-plugin live, and do the individual MCPs keep standalone plugin manifests?
**A:** Meta-plugin in litchron; scivcd stays standalone.
**Ambiguity:** 38%

### Round 6 — C + O (Context + Observation) [SIMPLIFIER MODE]
**Q:** How formal is the cross-MCP schema, and what gets persisted to the run_dir audit log? (Simplifier: what's the simplest version?)
**A:** Convention-only + run_dir/audit/ log; only new artifact is emit_figure_script return shape.
**Ambiguity:** 23%

### Round 7 — G (Government)
**Q:** Pick the governance shape for baseline ownership, scivcd-version-bump handling, and the existing IGNORED_CHECK_TYPES set.
**A:** Author-owned baseline (self-healing, pre-commit auto-stages non-CRITICAL deltas); scivcd version-pinned in [audit] extra; IGNORED_CHECK_TYPES decommissioned into baseline.
**Ambiguity:** 9.5% ✓
</details>

## Status

**pending approval** — this spec has crystallized below the 20% ambiguity threshold. No execution begins until the user explicitly selects an execution path via the bridge below.
