# litchron-audit plugin

Bundles the `litchron` MCP server, the `scivcd` MCP server, and the `/audit-litchron-run` orchestration skill under one Claude Code plugin.

## Required environment variable

The plugin's `.mcp.json` references `${LITCHRON_PYTHON}` for both MCP `command` fields — point it at the Python interpreter that has both `mcp_litchron` and `scivcd` installed.

```bash
# Conda env "dl" (LitChron development default)
export LITCHRON_PYTHON="$(conda run -n dl which python3)"

# Or directly (adjust the path for your machine):
export LITCHRON_PYTHON=/path/to/conda/envs/dl/bin/python
```

Set it before launching Claude Code:

```bash
LITCHRON_PYTHON="$(conda run -n dl which python3)" claude --plugin-dir /path/to/litchron/plugin
```

> Adjust the paths above for your machine: point `LITCHRON_PYTHON` at the Python interpreter in your environment, and `--plugin-dir` at this repo's `plugin/` directory.

Or persist it in your shell rc, or add it to Claude Code's settings:

```jsonc
// ~/.claude/settings.json
{
  "env": {
    "LITCHRON_PYTHON": "/path/to/conda/envs/dl/bin/python"
  }
}
```

If the variable is unset, both MCP servers will fail to start with a "command not found" error in Claude Code's startup log — that's intentional. Hardcoding `"python"` here would silently pick the wrong interpreter on most workstations.

## Required Python packages (in `$LITCHRON_PYTHON`'s env)

- `mcp_litchron` (this repo, install with `pip install -e .[audit]`)
- `scivcd` (pinned in `pyproject.toml` `[audit]` extra)
- `mcp>=1.0` (transitively required by both servers)

## Plugin contents

| File | Purpose |
|------|---------|
| `.claude-plugin/plugin.json` | Plugin manifest (name, version, license) |
| `.mcp.json` | MCP server registry (litchron + scivcd) |
| `skills/audit-litchron-run/SKILL.md` | Orchestration skill (5-step audit workflow) |
| `README.md` | This file |
