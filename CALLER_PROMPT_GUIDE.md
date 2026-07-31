# Caller Prompt Guide

Give your LLM (the **caller**) a command like the ones below. The caller spawns **one
`annotate-expert` agent per dataset, in parallel**, each with its **own `out` directory**, and each
runs `init → judge → finalize` autonomously to write `obs['cellman_annotation']`.

- **`tissue`** is `species, organ`. It may be free text — the agent resolves it to the KG module.

> The caller's ONLY job is
> to **spawn one agent per dataset and wait** for each agent's final result. The caller must **not**
> annotate anything itself.

---

## Claude Code

Invoke the skill with `/annotate-mode`:

> For a **single** dataset:

```
/annotate-mode Annotate the h5ad file below using the annotate-expert agent.

- input  : /path/mouse_brain.h5ad
- tissue : mouse, brain
- virtual env    : /path/to/your/.venv
- cluster column : leiden
- note           : use GPU
```

> For **multiple** datasets:
```
/annotate-mode Annotate the h5ad files below using the annotate-expert agent.
Run each dataset as an INDEPENDENT agent IN PARALLEL — one agent per dataset, each with its
OWN --out directory.
Do NOT annotate, read any out directory, or run helper.py yourself — only spawn the agents and
wait for their results; each out directory belongs to its agent alone.

Datasets:
  [1] input  : /path/mouse_brain.h5ad
      tissue : mouse, brain
  [2] input  : /path/mouse_liver.h5ad
      tissue : mouse, liver

Common:
- virtual env    : /path/to/your/.venv
- cluster column : leiden
- note           : use GPU
```

## Codex

Invoke the skill with `$annotate-mode`; the expert agent is `annotate_expert`:

> For a **single** dataset:
```
$annotate-mode Annotate the h5ad file below using the annotate_expert agent.

- input  : /path/mouse_brain.h5ad
- tissue : mouse, brain
- virtual env    : /path/to/your/.venv
- cluster column : leiden
- note           : use GPU
```

> For **multiple** datasets:
```
$annotate-mode Annotate the h5ad files below using the annotate_expert agent.
Run each dataset as an INDEPENDENT agent IN PARALLEL — one agent per dataset, each with its
OWN --out directory.
Do NOT annotate, read any out directory, or run helper.py yourself — only spawn the agents and
wait for their results; each out directory belongs to its agent alone.

Datasets:
  [1] input  : /path/mouse_brain.h5ad
      tissue : mouse, brain
  [2] input  : /path/mouse_liver.h5ad
      tissue : mouse, liver

Common:
- virtual env    : /path/to/your/.venv
- cluster column : leiden
- note           : use GPU
```

---

## Optional per-dataset fields

| field | meaning |
|---|---|
| `disease` | disease context (e.g. `glioblastoma`).|
| `output` | save path. Default {out}/{input}_annotated.h5ad |
| `use-rep` | embedding the neighbour graph is rebuilt on when a cluster is subdivided: an `obsm` key (e.g. `X_scVI`).|

## Requirements (per input h5ad)

- `adata.X` = **log-normalised** (not raw counts).
- `adata.var_names` = **gene symbols** (not Ensembl IDs).
- `adata.obs` has the **cluster column**.
