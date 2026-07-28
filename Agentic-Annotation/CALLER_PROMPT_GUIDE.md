# Caller Prompt Guide

Give your LLM (the **caller**) a command like the ones below. The caller spawns **one
`annotate-expert` agent per dataset, in parallel**, each with its **own `out` directory**, and each
runs `init → judge → descend → finalize` autonomously to write `obs['cellman_annotation']`.

- **`tissue`** is `species, organ`. It may be free text — the agent resolves it to the KG module.
- The ontology server is automatic (built into `helper.py`); you give no URL.

> **Caller rules (important — especially on Codex's manager/worker model).** The caller's ONLY job is
> to **spawn one agent per dataset and wait** for each agent's final result. The caller must **not**
> annotate anything itself: do not read an agent's `out` directory, run `helper.py`, inspect
> profiles, judge clusters, or decide descents. Each `out` is owned **exclusively** by its agent —
> a second brain touching it duplicates work and, on any write, corrupts `_annotate_state.json`
> (two loops on one directory). For a **single** dataset, just spawn one agent (or run it inline) and
> wait — do not also work the same directory in parallel.

---

## Claude Code

Invoke the skill with `/annotate-mode`:

```
/annotate-mode Annotate the h5ad files below using the annotate-expert agent.
Run each dataset as an INDEPENDENT agent IN PARALLEL — one agent per dataset, each with its
OWN --out directory.
Do NOT annotate, read any out directory, or run helper.py yourself — only spawn the agents and
wait for their results; each out directory belongs to its agent alone.

Datasets:
  [1] input  : mouse_brain.h5ad
      out    : dir/mouse_brain
      tissue : mouse, brain
  [2] input  : mouse_liver.h5ad
      out    : dir/mouse_liver
      tissue : mouse, liver

Common:
- virtual env    : /bce/groups/pnucolab/analysis/l202255179/UNNAMED/.venv
- cluster column : leiden
- note           : use GPU
```

## Codex

Invoke the skill with `$annotate-mode`; the expert agent is `annotate_expert`:

```
$annotate-mode Annotate the h5ad files below using the annotate_expert agent.
Run each dataset as an INDEPENDENT agent IN PARALLEL — one agent per dataset, each with its
OWN --out directory.
Do NOT annotate, read any out directory, or run helper.py yourself — only spawn the agents and
wait for their results; each out directory belongs to its agent alone.

Datasets:
  [1] input  : mouse_brain.h5ad
      out    : dir/mouse_brain
      tissue : mouse, brain
  [2] input  : mouse_liver.h5ad
      out    : dir/mouse_liver
      tissue : mouse, liver

Common:
- virtual env    : /bce/groups/pnucolab/analysis/l202255179/UNNAMED/.venv
- cluster column : leiden
- note           : use GPU
```

---

## Optional per-dataset fields

| field | meaning |
|---|---|
| `disease` | disease context (e.g. `glioblastoma`). Omit = no hint (the agent does **not** assume normal). |
| `output` | save path. Default `{input}_annotated.h5ad`. |

## Requirements (per input h5ad)

- `adata.X` = **log-normalised** (not raw counts, not scaled).
- `adata.var_names` = **gene symbols** (not Ensembl IDs).
- `adata.obs` has the **cluster column** named under `Common`.

`helper.py init` refuses input that violates the first two (override with `--force`).
