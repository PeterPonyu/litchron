---
description: Audit all figures produced by a litchron run for publication-quality issues (text truncation, axes overlap, colorblind-unsafe palettes, effective-DPI problems, etc.). Use when a litchron run has completed and the user says "audit run", "check my figures", or "/audit-litchron-run". Chains the litchron MCP (to emit driver scripts) with the scivcd MCP (to load, audit, baseline, and diff each figure) in a single scivcd session scoped to the run.
---

# Audit LitChron Run

You have access to two MCP servers loaded by this plugin:

| Server    | Prefix              | Role                                                        |
| --------- | ------------------- | ----------------------------------------------------------- |
| `litchron`| `mcp__litchron__`   | Pipeline MCP; `emit_figure_script` emits a driver script per figure |
| `scivcd`  | `mcp__scivcd__`     | Audit MCP; stateful sessions holding matplotlib Figure objects |

## When to invoke

The user asks to audit figures for a litchron run. Typical phrasings: "audit run", "audit my figures", "check figures for run X", `/audit-litchron-run --run-id <id>`.

## Arguments

| Argument   | Required | Default              | Description                  |
| ---------- | -------- | -------------------- | ---------------------------- |
| `--run-id` | No       | *auto-discovered*    | The litchron run ID to audit |

If the user omits `--run-id`, resolve a default in this priority order:

1. Honor `$LITCHRON_AUDIT_RUN_ID` if it is set in the environment.
2. Otherwise, call `mcp__litchron__report_status` with no arguments to list runs and pick the most recently modified `run_id` under the project's `runs/` directory.
3. If neither yields a value, ask the user for `--run-id` explicitly — do not invent one.

Never hardcode a specific run_id (no fallbacks to dataset-specific defaults like dated identifiers); this skill must work across LitChron projects without modification.

## Figures audited

LitChron produces two named figures per run:

| `figure_name`              | Description                                   |
| -------------------------- | --------------------------------------------- |
| `annotation`               | Cell-type annotation UMAP/embedding figure    |
| `comparison_strip`         | Pseudotime comparison strip across methods    |

## Workflow

Follow these steps exactly. Do NOT skip steps or reorder them.

### Step 1 — Emit driver scripts (one call per figure)

For each `figure_name` in `["annotation", "comparison_strip"]`, call:

```
mcp__litchron__emit_figure_script(run_id=<run_id>, figure_name=<figure_name>)
```

Response shape: `{script_path: str, figure_name: str, deps_pinned: bool}`.

- `script_path` is an absolute path to a self-contained Python driver script that loads run state, builds the named figure, and leaves it in pyplot state (it does NOT call `plt.close()`).
- Collect both `script_path` values; you will need them in Step 2.
- If `deps_pinned` is `false`, warn the user that the driver script's dependencies are not pinned and results may differ from CI.

### Step 2 — Open one scivcd session per figure

The scivcd `start_session` tool is **per-script**, not per-run: each call loads one script and returns a `session_id` bound to the figures that script produces. Because each litchron figure has its own driver script, open **one session per figure**:

For each `(figure_name, script_path)` pair:

```
mcp__scivcd__start_session(script_path=<script_path>)
```

Optional fields you may pass when the script needs them:
- `cwd` — defaults to the script's parent directory; override if imports fail
- `pythonpath` — prepend project-local paths if the script imports litchron modules
- `timeout_seconds` — raise if the figure takes more than a few seconds to build

Response: `{session_id, figure_count, stdout_chars, stderr_chars}`.
- If the response has an `error` field, surface it and stop — do not retry blindly.
- If `stderr_chars > 0`, call `mcp__scivcd__get_script_output(session_id, stream="stderr")` to inspect before proceeding.
- Record the `session_id` for each figure; keep the mapping `figure_name → session_id`.

### Step 3 — Audit each figure

For each `(figure_name, session_id)`:

**3a. Summary first:**

```
mcp__scivcd__audit_summary(session_id=<session_id>, figure_index=0)
```

Response: `{critical, major, minor, info, total_issues, figure_index, cached}`.

If `critical + major == 0`, this figure is clean — note it and skip to Step 3c.

**3b. Details for figures with CRITICAL or MAJOR issues:**

```
mcp__scivcd__audit_details(session_id=<session_id>, figure_index=0, severity_floor="MAJOR")
```

Read each issue's `type`, `severity_level`, `detail`, and `elements`. Translate each to a one-line "what's wrong + suggested fix" for the user — do not dump raw dicts.

**3c. Write findings to run_dir:**

Determine `run_dir` from the `run_id` (litchron stores runs under its configured run directory). Write findings to:

```
<run_dir>/audit/<figure_name>.findings.json
```

The findings JSON shape follows scivcd's existing `audit_details` response — store the `issues` array plus metadata (`run_id`, `figure_name`, `session_id`, `critical`, `major`, `minor`, `info` counts).

### Step 4 — Baseline management

For each `(figure_name, session_id)`, check whether a baseline file exists at:

```
<run_dir>/audit/baseline.json
```

**If no baseline exists:**

```
mcp__scivcd__save_baseline(session_id=<session_id>, path="<run_dir>/audit/baseline.json", figure_index=0)
```

This snapshots the current audit as the initial baseline. Report `baseline_status: "created"`.

**If a baseline already exists:**

```
mcp__scivcd__diff_baseline(session_id=<session_id>, baseline_path="<run_dir>/audit/baseline.json", figure_index=0)
```

Response: `{ok, figure_index, diff: {has_new_critical, totals_added, totals_removed, figures[]}}`.

- If `diff.has_new_critical` is `true`, report this prominently — these are blocking regressions.
- Report any new MAJOR issues added vs baseline.
- If only non-CRITICAL deltas exist, report `baseline_status: "delta_non_critical"`.
- If CRITICAL deltas exist, report `baseline_status: "delta_critical_regression"`.

### Step 5 — Close all sessions

For each `session_id` opened in Step 2 (in any order):

```
mcp__scivcd__end_session(session_id=<session_id>)
```

Call `end_session` for every session opened, even if an earlier step failed. Response: `{ok, status, deferred}`.

## Acceptance

At completion, return a summary object to the user:

```json
{
  "run_id": "<run_id>",
  "figures_audited": ["annotation", "comparison_strip"],
  "critical_count": <total CRITICAL issues across all figures>,
  "major_count": <total MAJOR issues across all figures>,
  "baseline_status": "created" | "delta_non_critical" | "delta_critical_regression" | "unchanged",
  "findings_paths": [
    "<run_dir>/audit/annotation.findings.json",
    "<run_dir>/audit/comparison_strip.findings.json"
  ]
}
```

`baseline_status` values:
- `"created"` — no prior baseline existed; one was saved from this run
- `"delta_non_critical"` — baseline exists; new issues found but none are CRITICAL
- `"delta_critical_regression"` — baseline exists; at least one new CRITICAL issue detected (blocking)
- `"unchanged"` — baseline exists; no new issues of any severity

## Failure handling

- If `emit_figure_script` returns an error for a figure, skip that figure and note it in the summary.
- If `start_session` returns an `{error: ...}` response, surface it and do not open audit tools against that session.
- If a session expires mid-audit (`SessionNotFound`), note the figure as incomplete in the summary.
- Always call `end_session` for any session that was successfully opened, regardless of downstream failures.

## Token discipline

- Always call `audit_summary` before `audit_details` — skip `audit_details` for clean figures.
- Use `severity_floor="MAJOR"` unless the user explicitly requests minor/info issues.
- Use `include_elements=False` and `max_detail_chars=200` to trim large payloads if the figure has many issues.
