# Caller Prompt Guide

Give your LLM (the **caller**) a command like the ones below. The caller spawns **one
`annotate-expert` agent per dataset, in parallel**, each with its **own `out` directory**, and each
runs `init → judge → finalize` autonomously to write `obs['cellman_annotation']`.

- **`tissue`** is `species, organ`. It may be free text — it is resolved to a KG module before
  annotation starts (see [Which module gets used](#which-module-gets-used)).

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

## Which module gets used

The KG serves one module per `(species, organ)` — see the table in the [README](README.md). Your
`tissue` is matched against that list before `init`:

| your tissue | what happens |
|---|---|
| `mouse, brain` — a module of that exact name exists | used immediately, no question |
| `human, mammary gland` — a different name for a served module | you are **asked**: "the KG has no `mammary gland`; use `human/breast` (61 terms)?" |
| `mouse, cerebellum` — only the parent organ is served | you are **asked**: "the KG serves the parent `mouse/brain` (413 terms); use it?" |
| `human, gut` — several modules fit | you are **asked** to choose between `small_intestine` and `large_intestine` |
| `human, pineal gland` — nothing plausible | you are told the KG has no such module; annotation does not start |

You always have the last word:

- **pick one** of the offered modules → that one is used.
- **"use all of them"** → every chosen module is served together (their terms are merged into one
  index). Expect labels from across those organs; a wider candidate set makes cross-organ mislabels
  easier, so prefer one module when the tissue really is one organ.
- **name a different module** ("no, use `human/skin`") → your module is used verbatim, even if it was
  not offered. If that name does not exist on the server you are told, and asked again.

Annotation never starts on a module you have not confirmed, and no "close enough" organ is ever
substituted silently.

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
