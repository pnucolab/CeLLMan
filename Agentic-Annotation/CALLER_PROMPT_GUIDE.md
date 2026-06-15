# UNNAMED — Caller Prompt Guide

> The standard command a user gives their **caller** (their own LLM session) to run
> annotation. The caller uses this to decide which expert agent to invoke with which
> arguments. For the operating procedure, see `skills/annotate-mode/SKILL.md`.

---

## 1. Command template (user → caller)

Natural language + a parameter block. Just fill the parameter block accurately.

```
Annotate the {species}-{organ} data.

[UNNAMED annotate params]
species:        mouse                 # required. wing identity (e.g. mouse, human)
organ:          whole_brain           # required. wing identity (e.g. whole_brain, pancreas)
input_h5ad:     /path/to/my_clusters.h5ad   # required. absolute path of the h5ad to annotate
cluster_col:    leiden                # required. obs column holding the cluster labels (NOT a default — state it)
disease:        normal                # optional. omit = no disease hint given to the agent | normal | <disease> (e.g. glioblastoma, IBD). providing it improves accuracy via disease_context matching
output_h5ad:    (optional)            # save path. default {input}_annotated.h5ad
python:         /path/to/venv/bin/python   # required. the user venv that runs the helper
gpu_device:     0                     # optional. GPU index (rapids). leave empty → CPU (scanpy) auto
output_dir:     (optional)            # working dir. default = directory of input_h5ad
```

> **The ontology server is automatic — the user does NOT provide a URL.** The helper
> connects to the deployed Cell-Ontology server by default (it builds the wing URL
> from `species`/`organ`). Advanced override only: set `$CELLONTO_SERVER_URL` or pass
> the helper `--server-url`. Normal usage needs nothing.

---

## 2. Parameter detail

| Param | Required | Meaning | What the caller does |
|---|---|---|---|
| `species` | ✅ | species — half of the wing | picks `{species}-{organ}-expert` + server wing scope |
| `organ` | ✅ | organ — other half of the wing | same; if the server lacks the wing, refuse (see §4) |
| `input_h5ad` | ✅ | h5ad to annotate. cells × genes, needs `obs[cluster_col]`. If `obs['vf_norm']` is present, density-based Background is enabled | passed as helper `--h5ad` |
| `cluster_col` | ✅ | **cluster-label column name**. NOT hardcoded to `leiden` — the user names their column | passed as helper `--cluster-col` |
| `disease` | ⬜ | disease context. **Omitting it gives the agent no disease hint** (it does NOT assume normal); pass `normal` to state the sample is normal, or a disease name. Providing it improves accuracy. | hints the agent's disease_context handling (disease-specific term/state) |
| `output_h5ad` | ⬜ | output path. default `{input_basename}_annotated.h5ad` | passed as helper `--output` (auto-derived if omitted) |
| `python` | ✅ | venv that runs the helper | caller runs `{python} {plugin}/scripts/helper.py ...` (resolve `{plugin}` = the Agentic-Annotation dir; don't assume CWD) |
| `gpu_device` | ⬜ | rapids GPU index. empty → CPU (scanpy) auto | subcluster backend selection |
| `output_dir` | ⬜ | location of round working files. default = input dir | passed as helper `--out` |

### Auto-derived (no user input)
- **gene-naming convention**: derived from species (human=UPPERCASE CD3D / mouse=Title-case Cd3d).
- **wing name**: `{species}_{organ}`.
- **backend**: from gpu_device + whether rapids is installed (GPU→rapids, else scanpy).

---

## 3. What the caller does after receiving the command (summary)
1. Decide which expert agent to call from `species`/`organ`; check the server has the wing (tell the user if not).
2. Invoke the expert agent in `init` mode → the agent resolves the plugin's helper
   (`{plugin}/scripts/helper.py`, not CWD-relative) and runs init
   (`{python} {plugin}/scripts/helper.py init --h5ad {input_h5ad} --cluster-col {cluster_col} --out {output_dir} --species {species} --organ {organ}`).
   The server is auto-resolved (default Cell-Ontology server; no URL needed).
3. Receive the round summary, then `AskUserQuestion` (split the flagged clusters? next resolution?) → repeat.
   The agent's report says, per flagged cluster, **which split mechanism** it recommends:
   - **`signal_split`** (agent named K subtypes + their term_ids) → run
     `{python} helper.py signal_split --out {output_dir} --spec '{"<path>":["<tid>",...], ...}'`
     (marker-guided; for cortical layers / CA1-CA3 / co-located types Leiden can't separate).
   - **`subcluster`** (Leiden, structure unknown) → run `... subcluster --paths '[...]' --resolution <r>`.
     If a `signal_split` report says a cluster has `<2 separable targets`, fall back to `subcluster` (Leiden) for it.
4. **Validation loop (before finalize)**: run `{python} helper.py check --out {output_dir} --decisions ...`.
   If `_annotate_violations.json` has **error**-severity entries, hand them **back to the agent to re-decide only the
   flagged clusters**, then re-run check. Repeat until error-free. (helper never changes a decision — the
   agent does the re-deciding.) **warn**-severity entries are advisory and do not block finalize.
5. Finalize → `{python} helper.py finalize --out {output_dir} --decisions ... --output {output_h5ad or auto}`.
   - Writes the h5ad when error-free **and** the split-review gate is clear. `error` violations abort (override with `--force`).
   - **Split-review HOLD (exit 3):** if finalize reports large bare-class leaves that could still split
     (`_annotate_finalize_review.json`), do **not** force. `AskUserQuestion`: *"these N clusters can be split once
     more — split? (y/n + resolution)"*. **YES** → `subcluster --paths <review.paths> --resolution <rec>` → agent
     re-judges → finalize again. **NO** → re-run `finalize … --accept-large` to write as-is.

---

## 4. Example user command

```
Annotate the mouse whole_brain data.

[UNNAMED annotate params]
species:     mouse
organ:       whole_brain
input_h5ad:  /bce/.../analysis/Xenium_Prime_Mouse_Brain/my_clusters.h5ad
cluster_col: leiden
disease:     normal
python:      /bce/.../.venv/bin/python
gpu_device:  0
```
→ Output: `/bce/.../analysis/Xenium_Prime_Mouse_Brain/my_clusters_annotated.h5ad`
  (full copy of the input + `obs['celltype_annotation']`).
