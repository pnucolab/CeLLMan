---
name: domain-mode
description: Pure expert-judgment spatial-domain annotation for celltype-annotated spatial h5ad. Feature is neighbor cell-type composition (not gene expression); the leaf label is a spatial domain (anatomical region). Round-driven (build→subcluster*→finalize, +refine) with caller-managed user confirmation. The expert agent makes a 4-way per-niche judgment (domain / needs_subclustering / Background / Unknown) from anatomy knowledge. No metric scorer.
---

# Domain-mode skill (UNNAMED)

The **spatial twin** of `/annotate-mode`: `gene markers → celltype` becomes
**`neighbor cell-type composition → spatial domain`** (an anatomical/functional
region). Same round structure, same 4-way decision, same caller-managed user
confirmation, no metric scorer. Gene expression `X` is **not** used — only the
**neighbor cell-type composition**. You name regions from your **organ anatomy
knowledge** as the domain expert.

`obs['domain']` is written **only at finalize**, into the same h5ad that
annotate-mode produced, next to `obs['celltype_annotation']`.

## Prerequisite
An annotate-mode output h5ad (= `{input}_annotated.h5ad`) with:
- `obs['celltype_annotation']` (including Unknown/Background),
- `obsm['spatial']` — (N,2) µm coordinates.
Only these two are read (`X` unused). **No ontology-server fetch is needed** —
naming a region from composition is your anatomy knowledge.

## Required inputs
`H5AD_PATH` (celltype-annotated; gets `obs['domain']` at finalize), `OUTPUT_DIR`,
`SPECIES`, `ORGAN` (anatomy), `RADIUS` (neighbor radius µm, default 30),
`RESOLUTION` (set per round by the user), `PYTHON` (user venv), `GPU_DEVICE` (optional).

## Two-actor
| Actor | Role |
|---|---|
| **`scripts/domain_helper.py`** (mechanical) | radius graph + Gaussian-weighted neighbor composition, leiden niches (GPU rapids / CPU scanpy), per-niche profile, in-memory subcluster, finalize write-back + spatial majority smoothing |
| **expert agent (this skill)** | read `niche_profiles_round{k}.json`, make a **4-way** call per niche → `_domain_decisions.json` |
| **caller** | between rounds, `AskUserQuestion` (subcluster? resolution?) |

## Modes (one invocation = one round)
| mode | does |
|---|---|
| `build` | init: composition → leiden niches → `niche_profiles_round1.json` |
| `subcluster` | re-leiden inside flagged niches → sub-niche profiles (with `parent_ctx`) |
| `finalize` | broadcast niche→domain, spatial majority smoothing, write `obs['domain']`, `domain_annotation.csv` |
| `refine` / `refine-apply` | (optional) domain spatial-adjacency profile → apply the agent's relabel map |

## Step 1 — run (helper)
GPU → rapids, otherwise scanpy (auto, inside the helper). *(domain_helper is
currently rapids-based; a CPU fallback mirroring helper.py is planned.)*
```bash
{PYTHON} scripts/domain_helper.py build      --h5ad {H5AD_PATH} --out {OUTPUT_DIR} --radius {RADIUS} [--gpu {GPU_DEVICE}]
{PYTHON} scripts/domain_helper.py subcluster --out {OUTPUT_DIR} --paths '["3","7"]' --resolution {RESOLUTION}
{PYTHON} scripts/domain_helper.py finalize   --out {OUTPUT_DIR} --decisions {OUTPUT_DIR}/_domain_decisions.json
```

## Step 2 — per-niche judgment (the heart)
Each niche in `niche_profiles_round{k}.json`:

| channel | schema | meaning |
|---|---|---|
| `enriched` | `[[celltype,frac_in,frac_global,log2FC],…]` (top15) | celltypes over-represented vs global = region-identity signal (catches rare-but-defining types) |
| `dominant` | `[[celltype,frac_in],…]` (top10) | celltypes that quantitatively fill the niche (usually name the region) |
| `depleted` | `[[celltype,frac_in,frac_global,log2FC],…]` (log2FC≤−1) | absent celltypes → use to rule regions out |
| `bg` | `{unknown_frac, own_unknown_frac, entropy}` | high own_unknown_frac + no coherent dominant → Background |
| `spatial` | `{n_points, coherence}` | coherence ≈ ≥0.7 = contiguous territory (real domain); low = scattered/boundary |
| `parent_ctx` | `{path, enriched[:8], dominant[:8]}` (sub-niche) | parent composition anchor |

### Interpreting
- **Read `dominant` + `enriched` together**: dominant usually names the region
  (e.g. Matrix thalamic relay 0.53 → Thalamus); enriched catches sparse-but-defining
  types (e.g. striatal MSNs).
- A type missing from `enriched` is not necessarily absent (it may be globally
  common) → check `dominant`. True absence is read from `depleted`.
- `coherence` is the spatial sanity check. Low coherence + ≥2 region signatures
  mixed = lumping → needs_subclustering.

### 4-way decision
| Decision | When |
|---|---|
| **domain** | composition coherently names ONE region. Emit a clean anatomical `label` + `key_celltypes` + `confidence`. |
| **needs_subclustering** | the niche lumps ≥2 distinct regions (mutually exclusive territory signatures together, or two adjacent layers co-dominant) AND/OR low coherence. **Close-region rule**: if two adjacent layers/subfields are both strong, default to needs_subclustering. |
| **Background** | uninterpretable — high own_unknown_frac (niche mostly Unknown/Background celltype) or high entropy with no coherent dominant. Transition voxels. `label="Background"`. |
| **Unknown** | composition is coherent but the region cannot be named (too mixed yet not cleanly splittable, or outside your anatomy knowledge). `label="Unknown"`. |

Recognize region signatures from your organ anatomy knowledge (e.g. mouse_whole_brain:
oligo-dominant → white matter, L2/3 IT dominant → cortical L2/3, DG granule
dominant → dentate gyrus, D1+D2 MSN → striatum, VLMC+endothelial → meninges).

### `_domain_decisions.json` (cumulative, you write it)
```json
{"<niche_path>": {"decision":"domain|needs_subclustering|Background|Unknown",
  "label":"Cortical layer 2/3","key_celltypes":[...],"confidence":"high|medium|low",
  "rationale":"...","alternatives_considered":"..."}}
```
**Label hygiene**: clean literature-standard anatomical name only (no reasons,
symbols, or `(n=…)`). Put reasoning in rationale/alternatives_considered.

### Anti-patterns
- Do not force a boundary/transition niche into one region → needs_subclustering or Unknown.
- Do not collapse two co-dominant adjacent regions into one label.
- Do not call Background just because entropy is moderate (only when genuinely
  Unknown/Background-celltype dominated).
- Before Unknown, enumerate ≥2 candidate regions with rejection reasons.

## Step 3 — caller user confirmation
After you write decisions and stop, the caller runs `AskUserQuestion`: "subcluster
the N flagged niches? (Yes/No)" and, if Yes, "next resolution?" (no default; you
recommend, the user confirms). Yes → subcluster loop, No → finalize.

## Output (after finalize)
- `obs['domain']` added to the input annotated h5ad (next to celltype_annotation).
- `domain_annotation.csv`.
- (on refine) `obs['domain_type']`, `obs['domain_raw']`.

## Forbidden
- using gene expression (`X`) / writing the ontology / forced region labeling /
  collapsing co-dominant regions / Unknown without enumerating candidates /
  losing the input h5ad's existing obs.
