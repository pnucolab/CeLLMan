# Caller Prompt Guide

Give your LLM (the **caller**) a command like the ones below. The caller spawns **one
`annotate-lead` agent per dataset, in parallel**, each with its **own `out` directory**, and each
runs `init → judge → finalize` autonomously to write `obs['cellman_annotation']`.

**By default each dataset is annotated three times and merged.** The lead spawns **three
`annotate-expert` judges**, hands all three the same round's profiles, and adjudicates their three
independent judgments per cell type — the deepest Cell-Ontology term that a majority of them support —
setting each label's confidence from how much they agreed. This is why annotation is reproducible
without making it rigid: the expert judgment stays free, and disagreement becomes a measurement instead
of a coin flip. You get **one** result either way — one annotated h5ad, one marker tree, one confidence
table — plus `consensus_report.md`, which lists exactly where the three disagreed. It costs about 3× the
tokens of a single run and roughly the same wall-clock (the judges run in parallel, and the heavy
compute runs once). Set `replicates : 1` to turn it off.

- **`tissue`** is `species, organ`. It may be free text — it is resolved to a KG module before
  annotation starts (see [Which module gets used](#which-module-gets-used)).

> The caller's ONLY job is to **spawn one lead per dataset and wait** for each lead's final result. The
> caller must **not** annotate anything itself, and must **not** spawn or manage the judges — the lead
> does that, because it is the one holding the profiles.

---

## Claude Code

Invoke the skill with `/annotate-mode`:

> For a **single** dataset:

```
/annotate-mode Annotate the h5ad file below using the annotate-lead agent.

- input  : /path/mouse_brain.h5ad
- tissue : mouse, brain
- virtual env    : /path/to/your/.venv
- cluster column : leiden
- note           : use GPU
```

> For **multiple** datasets:
```
/annotate-mode Annotate the h5ad files below using the annotate-lead agent.
Run each dataset as an INDEPENDENT agent IN PARALLEL — one agent per dataset, each with its
OWN --out directory.
Do NOT annotate, read any out directory, or run helper.py yourself — only spawn the agents and
wait for their results; each out directory belongs to its agent alone.
Settle the KG module for EVERY dataset first — ask me ONCE about all the tissues that need
confirmation — and only then start all datasets together.

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

Invoke the skill with `$annotate-mode`; the lead agent is `annotate_lead` and its judges are `annotate_expert`:

> For a **single** dataset:
```
$annotate-mode Annotate the h5ad file below using the annotate_lead agent.

- input  : /path/mouse_brain.h5ad
- tissue : mouse, brain
- virtual env    : /path/to/your/.venv
- cluster column : leiden
- note           : use GPU
```

> For **multiple** datasets:
```
$annotate-mode Annotate the h5ad files below using the annotate_lead agent.
Run each dataset as an INDEPENDENT agent IN PARALLEL — one agent per dataset, each with its
OWN --out directory.
Do NOT annotate, read any out directory, or run helper.py yourself — only spawn the agents and
wait for their results; each out directory belongs to its agent alone.
Settle the KG module for EVERY dataset first — ask me ONCE about all the tissues that need
confirmation — and only then start all datasets together.

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

## Reference runs (annotate to an answer-key)

When you have an expected set of cell types — a label list, or an already-annotated h5ad — invoke
`/annotate-reference-mode`. Identity is judged exactly as before (the same R1–R8 test against the full
Cell-Ontology graph); the answer-key only restricts which terms may become a **label**, and anything it
cannot explain becomes `Unknown`.

Reference runs are **not** ensembled — their labels are your answer-key's words, not the
Cell-Ontology's, so a consensus over the ontology hierarchy cannot read them. A reference run is one
`annotate-expert`, as before.

```
/annotate-reference-mode Annotate the h5ad below using the annotate-expert agent.

- input            : /path/mouse_kidney.h5ad
- tissue           : mouse, kidney
- virtual env      : /path/to/your/.venv
- cluster column   : leiden
- reference h5ad   : /path/ref.h5ad
- reference column : celltype_plot
- reference scope  : strict
```

**You declare the mode by which skill you invoke, not by a field in the prompt.** `/annotate-mode` and
`/annotate-reference-mode` are the two declarations. Your caller translates that choice into
`mode : reference` when it spawns the expert agent, because the agent is a separate process and does not
inherit the skill you loaded.

| field | meaning |
|---|---|
| `reference` **or** `reference h5ad` + `reference column` | the answer-key. Only the LABELS are read — the reference's expression never becomes evidence for the target data. |
| `reference scope` | `strict` (default) = label only with the answer-key's own terms; `refine` = its KG descendants become labelable too, so a lumped reference term can be resolved finer. It sets the label-space **depth**, not the mode. |
| `reference alias` | only when the answer-key's names differ from the KG's — and normally you do not write it: the agent builds it with `helper.py ref-resolve`, matching by marker agreement, and asks you only about the genuinely ambiguous ones. |

Belt and braces: the reference input switches the mode on its own too, so a caller that forgets to pass
the declaration still runs in reference-mode. The one combination that stops the run is a declared
reference-mode with **no** answer-key at all — it cannot be invented, so you are asked for it rather
than quietly annotated without one.

`finalize` additionally writes `reference_coverage.json`: how much of the answer-key was reproduced
(`recovery_rate`), what resolved deeper than it (`refine` only), and what it could not explain.

---

## Which module gets used

The KG serves one module per `(species, organ)` — see the table in the [README](README.md). Your
`tissue` is matched against that list before `init`:

| your tissue | what happens |
|---|---|
| `mouse, brain` — a module of that exact name exists | used immediately, no question |
| `human, mammary gland` — a different name for a served module | you are **asked**: "the KG has no `mammary gland`; use `human/breast` (61 terms)?" |
| `mouse, cerebellum` — finer than the KG; only the parent organ is served | you are **asked**: "the KG serves the parent `mouse/brain`; use it?" |
| `human, intestine` — coarser than the KG; it is split into several modules | you are **asked**: "the KG splits this into `small_intestine` and `large_intestine` — one of them, or both served together?" |
| `human, gut` — ambiguous wording | you are **asked** to choose between `small_intestine` and `large_intestine` |
| `human, pineal gland` — nothing plausible | you are told the KG has no such module; annotation does not start |

**With several datasets, the modules are settled for all of them before any annotation starts.** The
question comes once, as a single plan covering the whole batch — never one pop-up per dataset, and
never after some datasets are already running:

```
Modules for this run:
  [1] mouse_brain.h5ad   mouse, brain           -> mouse/brain     (exact match)
  [2] ts_breast.h5ad     human, mammary gland   -> human/breast?   (synonym — confirm)
Dataset [2] needs your confirmation before anything starts.
```

Once you answer, every dataset — the ones that matched exactly and the ones you confirmed — is spawned
together in parallel. So the batch always begins from one plan you have seen.

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
| `replicates` | how many independent judges annotate the dataset before they are merged. **Default 3.** `1` = a single run, no consensus, no `consensus_report.md` (the previous behaviour). An even number is rounded up, because a majority has to exist. Raising it above 3 costs proportionally more tokens for a small further gain. |
| `disease` | disease context (e.g. `glioblastoma`).|
| `output` | save path. Default {out}/{input}_annotated.h5ad |
| `use-rep` | embedding the neighbour graph is rebuilt on when a cluster is subdivided: an `obsm` key (e.g. `X_scVI`).|

## Requirements (per input h5ad)

- `adata.X` = **log-normalised** (not raw counts, and not scaled / z-scored / Pearson residuals).
- `adata.var_names` = **gene symbols** (not Ensembl IDs).
- `adata.obs` has the **cluster column**.

The first two are checked before anything else runs. If either fails, annotation **stops immediately**
and you are told exactly what was detected — for example *"63% of the non-zero values are whole numbers
(max=412)"* or *"row sums equal obs['total_counts'] — the rows are still that cell's transcript counts"* —
together with the fix (`sc.pp.normalize_total(adata, target_sum=1e4)` then `sc.pp.log1p(adata)`).

This is deliberately a hard stop rather than a warning: a run on raw counts or Ensembl IDs completes
normally and returns labels that are meaningless, which is far worse than failing in the first minute.
Nothing is written and no time is spent. If your matrix really is non-standard and you want to proceed
anyway, say so — that is the only way `--force` is used.
