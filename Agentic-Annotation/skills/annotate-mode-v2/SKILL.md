---
name: annotate-mode-v2
description: Expert-judgment cell-type annotation of clustered scRNA / spatial h5ad, grounded in a Cell-Ontology wing (slim-index). Every decision is one test applied at four levels — a label is REAL only when the term's OWN markers are ENRICHED (elevated vs the whole dataset, not merely detected) AND DISCRIMINATIVE (its own low-degree markers, not shared/pan/ambient); assign the DEEPEST term that is REAL, FLOOR honestly to the class/region when none is, and never force-fit. Mixtures are not pre-detected: whenever a cluster is not a single clear identity, it is DESCENDED — re-Leidened on the marker axis of its KG children (a structural candidate set, no thresholds) and each sub-group re-judged; the distinct identities that emerge are the answer (1 → pure/label, ≥2 → split + recurse, none → floor), so a minority / balanced / cross-region mix all separate by the same call. Standard annotation runs autonomously to finalize; the user is consulted only in a free-form interactive phase after finalize. Writes obs['cellman_annotation'].
---

# annotate-mode-v2 — one identity test, four levels, marker-axis split

**You are the domain expert for one wing (species + organ).** The ontology wing you load (slim-index)
is your domain memory — each term's markers (required / defining / supporting / negative),
the class→region→subtype hierarchy, confusion pairs, context axes. `scripts/helper_v2.py` does all the
compute (global statistics, profiling, the marker-axis split, gates, the h5ad write) and hands you a
**profile** per cluster; **you supply the biological judgment.** The caller talks to the user — you never do.

Everything below is built on **one principle**, applied at four levels. Read §1 first; the rest is that
principle in full detail.

> **THE PRINCIPLE.** A label is **REAL** only when the term's **OWN** markers are **ENRICHED** (elevated
> vs the whole dataset — *not merely detected `>0`*; in spatial/KDE data almost everything is detected at
> ambient single-molecule level) **AND DISCRIMINATIVE** (its own low-degree markers — not ones it merely
> shares with siblings, not pan/lineage-wide, not ambient bleed). **Assign the DEEPEST term that is REAL;**
> when nothing deeper is REAL, **FLOOR honestly** to the confident class (for brain, the region) or
> **Unknown**. **Never force-fit** a specific term through shared/leaky markers.

---

## 1. The rules (the vocabulary — defined ONCE, referenced by name everywhere below)

Six named rules and a small shared vocabulary. Every case in §5–§9 is one of these rules applied at a
level (lineage / region / subtype / state). The engine (`helper_v2.py`) precomputes each rule's inputs and
exposes them as profile fields; the **verdict is always yours** — the engine gives evidence, not the answer.

### Shared terms (one name per concept — used throughout)
- **leaf** — a cluster (or sub-cluster) you are judging.
- **the reference** — the whole-dataset per-gene statistics (`_global_stats.npz`), computed once at `init`
  and reused by EVERY later round, so absolute identity never collapses on a small fragment.
- **share(gene)** — how many wing terms list the gene = its gene-degree. **OWN / DISCRIMINATIVE marker** =
  low share (`share ≤ 3`, `SPEC_SHARE_MAX`); **shared / pan / leaky marker** = high share. Only OWN markers
  identify a specific term; shared markers only place the lineage/class.
- **defining marker (DF)** — a curated marker for which *a single one ON is sufficient* evidence for the term.
- **lineage** — the class-tier root (`glutamatergic neuron`, `astrocyte`, `fibroblast`…). For neurons it is
  fixed by the canonical **transmitter**, not by cell count.
- **class / region-group / subtype** — hierarchy tiers from shallow to deep. **FLOOR** = the shallowest
  honest label (class, or a region-group node) you fall to when no deeper term is REAL.

### R1 — ENRICHED(gene)  ·  "is this marker actually elevated here, vs the whole dataset?"
The absolute-identity backbone; it survives fragmentation (a 150-cell fragment scores like the full band).
`gene` is **ENRICHED (ON)** when ANY of:
- **(a) robust** — exceeds the reference `mean + sd` in **≥30%** of the leaf's cells; OR
- **(b) fold** — global **fold ≥ 3×** (log2 ≥ 1.5) **AND** detected in **≥30%** of cells (catches
  *quiet-but-enriched* markers); OR
- **(c) rank** — rank-enrichment **`auc ≥ 0.85`** **AND** the gene is an OWN marker (`share ≤ 3`) of the term
  being tested (rescues markers so sparse they fail detection — 2 µm bins: `Rorb`→L4 auc 0.98, `Fezf2`→L5
  0.90 — yet detected <25%). **Guard:** branch (c) applies ONLY to a term's OWN low-share markers; a
  broad/shared gene (`Slc17a7`, `Pcp4`) still needs real detection, so spillover can never force-fit.
- **Region-gated relaxation (`auc_pos = 0.65`):** once a leaf's `dominant_region` is set (cross-region
  candidates are then structurally excluded, R4), a term's OWN **positive** markers count ENRICHED at the
  looser `auc ≥ 0.65` — so a quiet layer marker (`Rorb` auc ~0.68 in sparse Stereo-seq) still anchors its
  layer instead of collapsing to the generic region. **Negatives stay STRICT (0.85)** — a weak negative
  never vetoes. Targeted panels (Xenium, auc > 0.85) are unaffected.
- **`adaptive-floor` (sparse platforms):** on 0.5 µm SSAM the (a)/(b) thresholds are lowered to the data's
  own `ambient × F` (reference-free) to recover low-copy DF (`Rspo1`→L4). Set at `init`, persists to all rounds.

The engine's boolean for "gene ON" in `identity_scores` / anchors is exactly this rule. **"detected `>0`"
is NOT ENRICHED** — that is the single most common error; ambient single molecules are everywhere.

### R2 — OWN(term, gene)  ·  "is this the term's OWN discriminator, or a shared marker?"
`gene` is **OWN / DISCRIMINATIVE** for `term` iff `share(gene) ≤ 3` (low gene-degree). A shared/pan marker
(`Wfs1` listed on 6 CA1 terms; `Slc17a7` on all glut) can make a leaf *resemble* a specific type without
being it. **Only OWN markers move you below the class.** In the profile: anchor `excl_markers_on` = the OWN
markers actually ON; `identity_scores` `spec_n`/`spec_on` count OWN markers on-panel / ON.

### R3 — REAL(term)  ·  THE central gate ("is this label real, or a force-fit?")  [engine: `spec_pass`]
`term` names a real identity in this leaf **only when its OWN markers carry the identity**:
- **spec_cov ≥ 0.5** — at least **half of the term's OWN on-panel markers (R2) are ENRICHED (R1)** — the
  engine field `spec_pass`. A non-hippocampal leaf sharing only `Wfs1` scores spec_cov ≈ 0–0.33 for CA1
  (its OWN `Fibcd1`/`Spink8` OFF) → not REAL; a true CA1 ≈ 1.0 → REAL.
- **negatives clean** — none of the term's negative markers is ENRICHED (`neg_on == 0`); a negative leak
  rejects the term.
- **DF-override** — a curated **defining** marker is decisive among competitors. A term is **coherent** when
  **≥2 of its non-pan markers are ON, OR its defining marker is ON** (`n_on≥2 OR df_on` — *a defining marker
  satisfies coherence by itself*, which is exactly why a single-DF pathognomonic type still qualifies). When a
  coherent term's **defining marker is ENRICHED**, it **wins outright** over rival candidates ("a single DF is
  sufficient": `Fezf2`→L5, `Cux2`→L2/3, `Prox1`→DG, `Grik4`→CA3…). Two guards stop a *leaky* DF from
  hijacking: **(A)** the term's decisive-marker score ≥ 0.34× the best same-lineage competitor; **(B)** if the
  term has on-panel OWN markers, ≥1 must be ENRICHED (so a cortical leaf with `Wfs1` ON but `Fibcd1`/`Spink8`
  OFF is NOT CA1). **DF-override only picks the WINNER among candidates — that winner must STILL pass
  `spec_pass` above** (≥half of its *on-panel* OWN markers ON, with the DF now counted in the OWN set). So a
  lone DF carries a type only when the type's other OWN markers are off-panel/few; a type with several
  measurable OWN markers where only the DF fired is still nulled. **Dominant-broad-DF rescue:** a region-*broad*
  DF (an interneuron neurochemical class `Sst`/`Pvalb`/`Vip`, or glia/immune/vascular — types that recur across
  regions) that is the **top vs-rest marker with auc ≥ 0.85** makes its term coherent/REAL on its own, so
  `Sst⁺` interneurons surface instead of collapsing to the bare class.
- **panel-blind inertness** — if ALL of a term's OWN markers are **off-panel** (`spec_n == 0`), R3 is INERT
  (cannot reject); judge by the term's **supporting** markers instead (a type whose *required/defining* are
  all off-panel is judged on its supporting markers rather than force-rejected).

**When R3 fails for the deepest candidate, you do NOT descend to it** — you FLOOR (R6) or run the STATE
check (R5) first. This one rule is the antidote to every force-fit.

### R4 — DEEPEST(path)  ·  "how deep can I honestly go?"  [engine: `graph_evidence` descent + `verdict`]
Identity is a **graph descent**, not a flat pick from a ranked list. Descend the term's ancestor `path`
(class → region-group → subclass → subtype) and assign the **DEEPEST node that is REAL (R3)**. Stopping
early is honesty: a region-group (`hippocampal pyramidal neuron`) is a perfectly valid terminal label when
the exact subtype (CA1 vs CA2) is not REAL. The engine restricts anchors to subtypes of the leaf's
**`dominant_region`**, so a cross-region blip (a CA2 marker in a thalamic cluster) is excluded *before* you
judge — low-copy layer markers then survive with no magnitude floor. `verdict` summarizes the descent:
`clear_top` (one REAL best anchor → take it), `ambiguous_tie` (≥2 tie → R7), `class_or_unknown` (no REAL
anchor → R5 then R6).

### R5 — AMBIENT(candidate)  ·  "is this second signal a real identity, or bleed / a program / a state?"
The vector field is built by **KDE / Gaussian smoothing over a lattice** (and spatial transcripts diffuse
between neighbors), so **every leaf carries low-level signal from neighboring types** — sparse `Gad1`,
ubiquitous `Slc1a2`/`Apoe`, myelin `Mbp`/`Plp1` bleed, faint immune/vascular. A candidate identity is
**AMBIENT (not real)** — ignore it for splitting and never switch identity to it — when it is supported
**only by shared/pan markers (R2 fails) or only by another type's markers**, i.e. it is NOT REAL (R3) on
its own. This single rule covers what the engine surfaces as several fields:
- **`overlay_lineages`** — a second lineage present but with **zero** OWN required markers ENRICHED = a
  co-expressed activation/transition **program**, not an identity. Label the lineage that IS REAL; treat the
  overlay as at most a *state OF* it (an endothelial cell with a contractile program = EndMT-activated EC,
  not smooth muscle). **`lineage_repick`** = the engine already corrected a pan-marker hijack → honor
  `.to`, never revert to `.from`.
- **`lineage_unresolved` / `unresolved_to`** — ≥2 lineages whose strength rests on a marker they **share**
  (`Ptprc` = pan-leukocyte, in both myeloid and lymphocyte required) with no OWN codom-private marker →
  the evidence cannot tell them apart → label the common-ancestor class `unresolved_to` (`leukocyte`),
  never a coin-flip pick.
- **transition mimicry (`transition_candidates`)** — a leaf that keeps its **origin identity anchors** ON
  while expressing a **mimicked lineage's program** is the origin lineage *in a transition state*
  (`endothelial cell, mesenchymal-transitioning`), NOT the mimicked lineage and NOT a chimera.
- **`exclusive_mixture`, transmitter doublets, `subtype_groups`, `bimodal_subtypes`** are the OPPOSITE case
  — the second signal IS REAL (its OWN markers ON in a disjoint cell subset) → that is genuine multiplicity,
  handled by R6, not AMBIENT.

The test that separates AMBIENT from a real second identity is always R3: **does the second candidate's OWN
discriminative marker fire?** No → AMBIENT (single identity). Yes → R6 (split).

### R6 — DESCEND-AND-SEE  ·  "is this cluster one identity or a mixture — and of what?"  [engine: `descend`]
**Do NOT pre-detect the mixture. Splitting IS the detection.** When a cluster is not already a confirmed
single homogeneous type, **descend into its CURRENT KG node's children** (`helper_v2.py descend --node
<node>`): the engine builds a marker axis from the **union of ALL those children's discriminating markers**
(§6) and re-Leidens the cells; each sub-group is then re-judged (R1–R3). **What comes out IS the answer:**
- **1 distinct identity** (all sub-groups judge to the same term, or they don't separate) → the cluster was
  **pure** → label it (same-label sub-groups **merge** — this prevents tiling a continuous gradient).
- **≥2 distinct identities** → the cluster was **mixed** → keep the split and **recurse** each sub-group into
  its own resolved child node (region → subtype …).
- **no separation** despite the children's markers being on the axis → a genuine **detection floor** (signal
  ≈ ambient, e.g. microglia) → stop at this node honestly (R8).

**Why this replaces the whole detection zoo.** "Is a 10% minority / a balanced 50-50 / a cross-region mix / a
codominant-lineage doublet present?" are NOT separate questions with separate thresholds — they are all just
**"descend and see which sub-groups appear."** The candidate set is **structural** (the KG children of the
current node — §6 breaks the chicken-and-egg: you never need to know what is mixed to build the axis; absent
children contribute ~zero markers), and the reclustering is **per-cell**, so it separates a minority, a 50-50,
and a cross-region pair with the SAME logic — **no `mixed`/`ambiguous_tie`/`bimodal`/`exclusive_mixture`/fold/
fraction/sep thresholds.** (The engine still emits those detection fields; **ignore them for the split
decision** — descend-and-see is the method. They remain at most as hints.)

**Descend ONE level at a time** (lineage → region-groups first, then a resolved region → its subtypes):
region-first concentrates each region's cells so a subtype signal, diluted at the mixed level, only emerges
**after** the region split. **Always descend to check** even a clear-looking cluster — a pure one descends to
1 identity and stops (cheap), so a hidden minority is never missed by a detection threshold, and a diluted
region-floor is never left buried (this is why `dominant_region` gating is NOT used to drop a minority region).

**Do NOT over-split:** a descent that yields 1 identity (or same-label sub-groups) **merges and stops** —
never tile a homogeneous population. **EXCEPTION:** a declared **context axis** (§5 Step 2b) splits a
single-lineage population by spatial pole (zonation/layer) even though the poles do not differ by intrinsic
markers — that is the one split that descend-and-see cannot see, so it is driven by the `context_axis` field.

### R7 — STATE  ·  "before any floor/Unknown, is a state term the real identity?"  [engine: `is_state`, `disease_candidates`]
State terms (activation / inflammatory / proliferating / transitional / reactive / disease-injury — anything
the wing declares `tier=state`) are **deliberately excluded from the anchor descent and from `top`** (their
broad reactive markers would force-fit normal cells). The consequence you MUST account for: a `null top` /
`class_or_unknown` verdict occurs **even when a state IS the answer**. So **before emitting any FLOOR or
Unknown**, read `identity_scores.ranked` for `is_state=true` and the per-leaf `disease_candidates` block
(surfaced for EVERY leaf whether or not `--disease` was passed), and apply R3 to the best state term, **plus
a specificity gate: ≥1 of its OWN markers must be low-share AND rank-enriched (`n_specific_on ≥ 1`)** — so a
state carried only by pan-reactive markers (`Gfap`/`Vim`/`Apoe`, shared across many states) does NOT qualify
and falls back to the normal counterpart. If it passes, **the state term IS the label** (a first-class
terminal, not "less real"). **disease is a HINT, never a filter** — assign a state when its OWN markers win
even with `--disease` omitted; never dismiss one "because it's a state" / "in-vitro program" / "no disease
declared."

### R8 — FLOOR  ·  "nothing deeper is REAL — what is the honest label?"
A FLOOR is a **last resort you EARN by descending, never a default you reach for.** Emitting a class /
region-group floor is FORBIDDEN until you have actually **`descend`ed that node and the descent yielded no new
separation** — every sub-group re-judged to the same parent, i.e. no child's OWN markers pass R3 even after the
marker-axis reclustering CONCENTRATED them (§6). Floor-without-descend is the single most common failure and is
a HARD violation (`check` flags it): a `class_floor` / region-floor on a leaf that was never descended, or that
still carries a `mixture_subtypes` / `candidate_subtypes.children_resolvable` / `df_present`-subtype signal, is
INVALID — descend it first. **The bar for floor is EVIDENCE, not effort saved:** your `rationale` must name
WHICH children you descended into and the concrete reason each failed to separate (markers off-panel; markers
detected but ≈ ambient after concentration = detection floor; a declared ContextAxis continuum; a genuine
`split_no_gain`). "It looked generic" / "no subtype was obviously on at the mixed level" is NOT a reason —
dilution hides every minority at the mixed level; that is exactly what descend un-dilutes.

Only THEN: the confident **lineage class** (or a region-group node) is the honest terminal `celltype`
(`class_floor`) if the lineage is clear; **Unknown** only when even the lineage is unclear (after listing ≥3
considered terms each with a concrete rejection). A confident wrong subtype (CA1 on a hypothalamic cluster) is
worse than an honest class — but an UNDESCENDED floor is worse than both, because it silently buries recoverable
subtypes (whole cortex → generic `neocortical glutamatergic neuron` with L5/L6 unrecovered inside it). **Never
downgrade a confident class/region to Unknown** — a class or region-group node is a valid final label, *after
the descent that earned it.*

---

### The many engine fields are ONE test — the map
Each field the profile exposes is R1–R8 applied at a level. Read this once and the field names stop
multiplying:

| engine field / case-name | rule | what it blocks / decides |
|---|---|---|
| `spec_pass`, `spec_cov`, `SPEC_UNSUPPORTED`, `REQUIRED_ABSENT_ONPANEL`, worked-ex I | **R3** | force-fit via shared/leaky markers |
| `share`, `excl_markers_on`, `spec_n` | **R2** | shared marker mistaken for an OWN discriminator |
| ON / fold-ON / auc-ON / `auc_pos` / `adaptive-floor` | **R1** | "detected `>0`" mistaken for enriched |
| `verdict`, `dominant_region`, anchor `path`, "stop at region" | **R4** | going deeper than the evidence defends |
| neuronal-priority, `overlay_lineages`, `lineage_repick`, `lineage_unresolved`, transition mimicry, ambient bleed | **R5** | ambient bleed / a program / a shared-marker lineage mistaken for a real 2nd identity |
| `descend` (marker-axis recluster of KG children) → count distinct re-judged identities | **R6** | a mixture confirmed as one type — OR a single population over-split. *(The old detection fields `codominant_lineages`/`ambiguous_tie`/`bimodal`/`exclusive_mixture`/`DF_MULTI_SUBTYPE`… are superseded — descend-and-see replaces them; ignore them for splitting.)* |
| `is_state`, `disease_candidates`, `n_specific_on`, STATE CHECK | **R7** | a real state buried under the bare class |
| `class_or_unknown`, `class_floor`, `DIFFERENTIAL_UNSUPPORTED` | **R8** | a confident class downgraded to Unknown, or a magnitude-only over-split |

---

## 2. Roles (do not cross)

| actor | responsibility |
|---|---|
| **expert agent = you** | Run `helper_v2.py`, read profiles, judge each leaf by R1–R8 (§5), drive the split yourself, write the decisions JSON + per-round CSV. **Never ask the user.** Standard annotation is autonomous. |
| **caller** | Relays your helper calls back-to-back during standard annotation (no per-round user prompt). **After finalize**, runs the free-form interactive phase with the user (§8). |
| **`helper_v2.py`** | The **single, self-contained backend** — all compute (profiling, the identity engine, gates), the marker-axis split, and the final h5ad write, in one script. **Validates** your decisions (`check`/`finalize`) and returns violations — it **never silently changes a decision**. Every command (`init`/`subcluster`/`descend`/`check`/`finalize`) runs in this one file; it does not call any other helper. |

---

## 3. Run the helper (exact commands)

`{HELPER}` = absolute path to `scripts/helper_v2.py` (resolve once:
`find . -path '*Agentic-Annotation/scripts/helper_v2.py'`). `{python}` = the user venv. **Run the WHOLE
workflow through `helper_v2.py`** — it is the single, self-contained entry point (all commands run in this one file). The
ontology server is automatic — pass `--species`/`--organ`; the helper writes `{out}/_ontology_index.json`.

**Output directory (hard rule):** `{out}` = an `annotation_out/` directory inside the caller's output
location (`--out {base_out}/annotation_out`; `mkdir -p` it before `init`). All artifacts and the final h5ad
land there. Pass the same `--out` to every round.

```bash
mkdir -p {out}
# process-profile / process-apply — Step P (BEFORE any clustering; segmentation-free spatial only).
#   Run FIRST, on the raw downsample_vector() h5ad, before pca/neighbors/leiden. See §5 Step P.
#   profile writes {out}/_process_profile.json and NEVER labels; you judge, then apply.
#   applicable:false ⇒ this wing has no process terms ⇒ skip Step P entirely and cluster everything.
{python} {HELPER} process-profile --h5ad {input}.h5ad --out {out} --species {sp} --organ {og}
#   --cand-pct 10  --neg-pct 10   (defaults; candidate = pos_score top N%, remove = neg_score top M%)
{python} {HELPER} process-apply   --h5ad {input}.h5ad --out {out} --species {sp} --organ {og} \
    --decisions '{"neuropil": true, "fiber tract": false}'      # true = you judged this class REAL
#   → writes obs['cellman_class'] (background|cell|process), obs['cellman_process'] (which process: neuropil /
#     fiber tract / '' for non-process), and uns['cellman_partition'] (params/decisions/evidence). No score
#     columns are persisted.

# process-cluster — pca/knn/leiden on the obs['cellman_class']=='cell' vectors ONLY, IN PLACE on
#   the SAME h5ad (no subset file, nothing to merge later). process/background get leiden=''.
#   rapids-singlecell when a GPU is present, automatic scanpy fallback otherwise (scanpy is very slow
#   at 10^5 vectors — GPU is the normal path). --gpu N picks the device (default 0), --gpu -1 forces CPU.
#   Defaults: --n-pcs 50 --n-neighbors 15 --metric euclidean --resolution 1.0 (the values you tell the user).
#   The CALLER asks the USER (§5 Step P.1, free-form) and passes whatever they type. NEVER run this
#   without asking — clustering (and now HVG) is ALWAYS a user decision, even if the input already has them:
{python} {HELPER} process-cluster --h5ad {input}.h5ad --out {out} --species {sp} --organ {og} \
    --hvg --n-top-genes 2000 \
    --n-pcs 50 --n-neighbors 15 --metric euclidean --resolution 1.0
#   HVG (default PCA mode only) — feature selection for the embedding, computed on the CELL vectors only
#   (Step P's cellman_class=='cell'; process/background excluded). NEVER subsets the h5ad; only sets
#   var['highly_variable'] so PCA uses it, X keeps ALL genes (the engine must see every KG marker).
#   The panel mirrors select_gene_panel: **HVG − technical genes ∪ (KG markers present in data)** —
#   tech genes (mt-/Rps/Rpl/Hba/Hbb, human MT-/RPS/…/MALAT1/lncRNA…) are dropped, and any KG wing marker
#   that IS in the data but HVG missed is re-added (KG bypasses the denylist).
#   --species = species denylist (caller already knows it). The KG wing is REUSED, not re-fetched: it reads
#   {out}/_ontology_index.json that process-profile/init already saved (so NO server URL here; --organ/
#   --ontology are only fallbacks if that file is missing). KG unavailable ⇒ graceful HVG−tech only. Flags:
#     --hvg --n-top-genes N      compute HVG (N genes) here → panel = HVG−tech ∪ KG-in-data → PCA on it
#     --use-existing-hvg         start from var['highly_variable'] already in the h5ad, then same panel adjust
#     (neither)                  no HVG — PCA on all genes (use_highly_variable=False)
#   Four MUTUALLY-EXCLUSIVE ways to feed the graph (pick per the user's answer):
#     (default)                  PCA → kNN → leiden on the cell vectors (HVG options apply here)
#     --use-rep X_scVI           use an existing obsm embedding, SKIP PCA+HVG (kNN/resolution still apply)
#     --use-existing-neighbors   reuse the kNN graph already in the h5ad, SKIP PCA+kNN (only resolution applies)
#                                (add --neighbors-key KEY if the stored graph is under a non-default key)
#     --use-existing-leiden COL  the h5ad already has clusters in obs[COL] — use as-is, SKIP everything
#   Run this even when Step P does not apply (no cellman_class ⇒ every vector is a cell).
#   It records metric/n_neighbors/use_rep in uns['cellman_cluster_params'], and `init` INHERITS them —
#   do NOT pass --subcluster-metric / --subcluster-n-neighbors (those exist only for an h5ad that was
#   clustered outside this skill).

# … then the celltype engine (init → subcluster/descend/check → finalize) annotates the cell leiden
# clusters and writes obs['cellman_annotation'] for the CELL vectors only. FINALLY:

# process-finalize — Step P.2 + merge. Run on the finalized {input}_annotated.h5ad so that EVERY vector
#   has a cellman_annotation value. It subtypes each REAL process class into its KG typed children
#   (glutamatergic / GABAergic / … neuropil, else the bare class) and merges them + 'Background' into
#   obs['cellman_annotation'], leaving the cell celltype labels untouched. Without this, process/background
#   vectors keep cellman_annotation='' — the annotation is INCOMPLETE.
{python} {HELPER} process-finalize --h5ad {out}/{input}_annotated.h5ad --out {out} --species {sp} --organ {og}
#   Skip only when Step P did not apply (no process/background vectors exist).

# init — compute the reference (global stats) + fetch ontology + profile the input's Leiden clusters.
#   --disease "<name>"     pass when the input is a disease/cancer dataset (HINT only, R7).
#   --subcluster-metric / --subcluster-n-neighbors  DO NOT PASS after process-cluster — init inherits the
#     initial clustering's metric/n_neighbors from uns['cellman_cluster_params'] automatically. Supply them
#     ONLY for an h5ad clustered outside this skill, where they must MATCH that clustering.
#   --adaptive-floor                               sparse platforms (0.5µm SSAM): lower R1 thresholds to ambient×F (persists to all rounds).
{python} {HELPER} init --h5ad {input_h5ad} --cluster-col {cluster_col} \
   --out {out} --species {species} --organ {organ} [--disease "{disease}"] \
   [--adaptive-floor]      # (--subcluster-metric/-n-neighbors: only for externally clustered h5ad)

# subcluster — MARKER-AXIS split of a NAMED multiplicity (R6). --candidates = the REAL distinct terms to separate.
{python} {HELPER} subcluster --out {out} --paths '["4.8"]' \
   --candidates '["CA2 pyramidal neuron, hippocampus","glutamatergic neuron, fasciola cinerea"]' \
   [--resolution 1.0] [--subcluster-n-neighbors 15] [--gpu 0]
#   per-path candidates: --candidates '{"4.8":["CA2 …","FC …"], "11.2":["D1-MSN","D2-MSN"]}'
#   0/1 candidate on a path → automatic whole-transcriptome PCA fallback (discovery subclustering).

# descend — UNCONDITIONAL KG descent of a class/region FLOOR leaf (R4/R8), region→subtype, kg-descent style.
{python} {HELPER} descend --out {out} --paths '["cl3"]'   --node "glutamatergic neuron"                 # → its KG children (regions)
{python} {HELPER} descend --out {out} --paths '["cl3.2"]' --node "neocortical glutamatergic neuron"     # → its layers

# check — validate your decisions (helper never edits them; you fix the flagged clusters).
{python} {HELPER} check --out {out} --decisions {out}/_annotate_decisions.json
# finalize — write {input}_annotated.h5ad (only when error-free AND the split-review gate is clear, §7.1).
{python} {HELPER} finalize --out {out} --decisions {out}/_annotate_decisions.json [--accept-large]
```

If the server 404s for this wing → **stop and tell the caller** (no guessing). Read the ontology from
`{out}/_ontology_index.json` — schema `{terms:{id:{L,T,P,R,S,N,D,DF}}, cp, oq, ca, sa}`: **L**=label,
**T**=tier, **P**=parent ids, **R**=required, **S**=supporting, **N**=negative, **D**=disease
context, **DF**=defining (one is sufficient), **ca**=context axes (§5 Step 2b), **sa**=state axes
(base-agnostic cell states, §5 Step 2c), **cp**=confusion pairs, **oq**=open questions, **GM**=grounding
PMIDs (cited in the finalize rationale only).

---

## 4. Profile fields — what you read (`{out}/cluster_gene_profiles_round{k}.json`)

First skim `{out}/cluster_overview_round{k}.json` (`{path:{n, dominant_lineages, top_up[:6],
specificity_peak}}`) for the side-by-side view. Then per leaf, read these — each feeds a rule from §1:

| field | feeds | how to use |
|---|---|---|
| **`graph_evidence`** ★ PRIMARY | R2·R4·R5·R6 | `{lineage, lineage_ambiguous, codominant_lineages[], lineage_repick, overlay_lineages[], lineage_unresolved, unresolved_to, dominant_region[], verdict, anchors[≤8], cp[≤4], confusion_unresolved, mixture_subtypes[]}`. `lineage` = the class by a **transmitter hard gate** (the lineage whose canonical transmitter is ENRICHED, else the strongest PRESENT transmitter; **neuronal lineages take priority when any neuronal transmitter is present** — a leaky glut marker never pulls a `Gad⁺` cluster into a glut region; pan-neuronal `Camk2a`/`Snap25` excluded). `dominant_region[]` = the dominant region-group(s); **anchors are restricted to its subtypes** (R4). Each anchor = `{term_id, label, tier, excl_on, excl_n, excl_markers_on[], neg_on, neg_markers_on[], df_on, df_absent, df_onpanel[], n_on, path[]}` — `excl_markers_on` = the OWN markers ON (R2); `path` = class→…→term (R4 stop points); `neg_on>0` → reject (R3). `df_absent` = the term has on-panel DFs but none ON — **advisory**, a caution not a demotion (a `df_absent` anchor with `excl_on≥2` is genuine; with `excl_on==1` it is a likely force-fit). `verdict` ∈ `clear_top`/`ambiguous_tie`/`class_or_unknown`/`confusion_unresolved` (R4/R6). |
| `identity_scores` *(supplementary — NOT the decision)* | R1·R3·R6 | absolute-fold summary `{top, ranked[≤5], class_fallback, mixed, dominant_lineage, codominant_lineages}`; each candidate `{term_id,label,tier,n_on,cov,spec,neg,spec_n,spec_on,spec_cov,spec_pass}`. `n_on` = the term's **non-pan discriminative markers (R∪S∪DF, PAN excluded) that are ENRICHED** (R1) — broader than `spec_on`, which is only the OWN (low-share) subset; `df_on` = its defining marker is ENRICHED; `spec_pass` = R3. **`top` is nulled by the engine when the dec-selected term FAILS R3** → that means "no normal subtype is REAL" → run R7 then R8, do NOT reach into `ranked` for a force-fit. `mixed` = ≥2 codominant lineages, counting only R3-passing terms (R6). Cross-check `top` against `hp`/`ontology_gene_table`/`lineage_scan` — never treat it as the verdict. |
| `hp` | R1 | `[[gene, mean_in, rank, pct_in], …]` top 50 — absolute expression (HK removed). A gene absent from `up` is NOT absent; broad/uniform markers show here. The decisive cross-check when `up` collapses. |
| `up` | R2·R6 | `[[gene, auc, mean_in, mean_out], …]` AUC≥0.6 — vs-rest contrast (vs siblings in subcluster rounds). Specificity + sub-structure. **Demoted for identity** — a flat `up` never vetoes a clear absolute identity (fragmentation collapses `up`→0.5). |
| `dn` | R3 | depleted genes — supports ruling OUT (soft; KDE smear). |
| `ontology_gene_table` | R1·R3 | `{gene:{rank,mean,auc,panel}}` for every R/S/N/DF marker with absolute mean. `panel=false` ⇒ off-panel — you may NOT reject a type for it (R3 panel-blind). Use `mean` to confirm a marker is genuinely on even at auc~0.5. |
| `lineage_scan` | R5·R8 | `{lineage:[[gene,auc,pct_in],…]}` canonical lineage markers (incl. pan) — authoritative for "is this lineage present" and the class-level FLOOR. |
| `candidate_subtypes` | R4·R6 | ranked terms whose defining markers appear, parent-pooled for sub-leaves; `children_resolvable` = 1-level children whose distinguishing markers are present → the split-deeper signal (R6). |
| `identity_evidence` | Step 0 | `{density, specificity_peak, lineage_markers_present, housekeeping_share, cross_lineage}` — the Background test. |
| `compartment` | Step 0.5 | `{low_frac, synaptic_top[], n_synaptic, neuropil_neg_on[], n_neuropil_neg, myelin_top[], n_myelin, fibertract_neg_on[], n_fibertract_neg, no_specific_anchor, candidate, compartment_type, compartment_term_id, compartment_label}` — tissue process (KG `tier='process'`); `candidate==true` ⇒ emit `compartment_label` (`neuropil` / a typed `… neuropil` / `fiber tract`) with `compartment_term_id`. Empty (no fields fire) for non-CNS wings. (`neuropil` key kept as a back-compat alias.) |
| `exclusive_mixture` | R6 | reciprocally-exclusive same-lineage DFs each ON in a disjoint subset → `needs_subclustering`. |
| `transition_candidates` / `transition_note` | R5 | origin-anchor-retained + mimicked-program → label the transition term, do NOT split or call the mimicked lineage. |
| `context_axis` | R-context | `[{axis, poles[], mixed, dominant_pole, suggested_subtype, pole_confidence, pmids}]` — zonation/layer subtyping (§5 Step 2b); follow the engine's pole. |
| `state_axis` | R-state | `[{name, markers_on[], n_on, score, pmids}]` — base-agnostic **cell states** (proliferating / interferon-stimulated / hypoxic …) whose OWN program markers are ENRICHED here, ranked by enrichment. Each is INDEPENDENT (multiple can be active); appended to the label as `, <state>` (§5 Step 2c). Not a base identity, not a split trigger. |
| `subtype_groups` | R6 | parents whose ≥2 finer sibling subtypes co-occur → split-deeper / sibling-mixture signal. |
| `parent_ctx` | R3·R4 | `{path, up, hp}` parent signature — **for sub-leaves you MUST pool this** (within-parent re-ranking erases the shared class signal). |
| `sibling_contrast` | R6 | within-parent discriminators — for deciding splits / choosing between close siblings, never for denying identity. |
| `flags` | corroborating | `{multi_lineage, layer_mixing, pan_class_only, background_score}` — **`multi_lineage` is ambient-prone** (raw `up`), never a chimera call on its own (use `mixed`, R6). `pan_class_only` = only class markers passed at gate level — re-check `identity_scores`/`candidate_subtypes` before treating as unresolved. |
| `n`, `n_out`, `n_sub` | sizing | `n` = TRUE leaf cell count (use for rare-type/size sanity). `n_sub` = cells used for AUC (init subsamples ≤150k; global stats stay on full data). `warn=n_out_too_small` ⇒ treat AUC cautiously. |

Gene matching is **case-insensitive** throughout (`Gad1` ≡ `GAD1`).

---

## 5. The decision algorithm — apply the rules per leaf (first terminal call wins)

Produce the 7-line scaffold below for EVERY leaf; it forces the synthesis and becomes your
`rationale`/`alternatives_considered`.

```
S1 CELL?      : density.pct / low_frac / lineage_markers.n / graph_evidence.lineage → cell? y/n  (Step 0)
S2 LINEAGE    : graph_evidence.lineage=__  ; apply R5 (drop overlay/repick/unresolved/ambient 2nd lineages) ; ≥2 lineages EACH REAL(R3)? → R6 split
S3 REAL?      : per candidate anchor, R3 : OWN markers ENRICHED(R1) ≥half? negatives clean? DF-override?  → which terms are REAL
S4 DEEPEST    : R4 : deepest REAL node on the best anchor's path ; stop at region if the subtype is not REAL
S5 STATE      : R7 : before any FLOOR/Unknown — any is_state / disease_candidates term REAL + n_specific_on≥1, beating normal?
S6 RESOLVE    : clear single REAL → label (R4) ; ELSE DESCEND-AND-SEE (R6): descend the node's KG children → 1 identity=label/merge, ≥2=split+recurse, no separation=FLOOR (R8, after STATE R7)
S7 FLAGS      : subclass_resolved ; confidence ; reason_code ; disease ; split mechanism = descend (marker-axis)
```

### Step P — WHAT KIND OF THING IS THIS VECTOR? (pre-Leiden; segmentation-free spatial only)
Segmentation-free sampling hits things that are **not cells**. A **process** is tissue built of cell
*processes* rather than cell bodies — defined as *"has mRNA but is not a nucleus / cell body"*; the mRNA is
there because it was **transported out of the soma**. KG `tier='process'` (neuropil, fiber tract; retinal
plexiform layers / podocyte foot processes / alveolar septa later). If these are clustered as if they were
cells, the compartment is lost for good — Step 0.5 cannot recover it once a cell-type anchor forms.

**Applicability.** `process-profile` reports `applicable:false` when the wing has no `tier='process'` term
⇒ **skip Step P**, cluster everything, continue at Step 0. Never invent process classes for such a wing.

**Classes come from the KG structure** — the direct children of the process root. Never hardcode names.

**The rule (already computed for you; you only judge):** candidate = `pos_score` top `cand_pct`% (the
class's own R∪DF); remove candidates whose `neg_score` (its N ∪ **all ancestors' N** — the generic nucleus
axis MALAT1 lives on the root and every class inherits it) is in the top `neg_pct`%. Survivors = `process`.

**Step P.1 — the caller asks the user for the HVG + clustering parameters (FREE-FORM typing, not a menu).** You
(the expert) judge the partition and run `process-apply`, then **report the result to the caller** — you never
talk to the user. **This is a HARD STOP: the caller MUST ask and WAIT for the user's typed reply before ANY
`process-cluster` run. Running PCA/kNN/Leiden (or HVG) with defaults without asking is a BUG — it is never
autonomous, even though the rest of annotation is.** The caller asks **as plain conversational text** — it MUST
NOT use `AskUserQuestion` or any selection/multiple-choice UI. There are no preset options to pick from; the
user types values (or a phrase like "use my X_scVI", "reuse the existing clusters").

**First, inspect the input and TELL the user what is already there — never silently reuse it.** Report which of
these the h5ad already carries: `var['highly_variable']` (how many genes), an embedding in `obsm` (e.g.
`X_pca`, `X_scVI`), a stored kNN graph (`uns['neighbors']`), and any cluster column(s) in `obs` (e.g.
`leiden`). Then the caller shows the partition (counts per class, removal ratio, panel gaps), states the
DEFAULTS, lists what pre-computed pieces exist, and invites free text — asking, for each stage, **reuse what's
there OR recompute (and from which stage)**:

> This h5ad already has: HVG (2000 genes) / obsm X_scVI / obs leiden. I will NOT reuse any of it unless you say so.
> Choose per stage —
> **HVG**: compute fresh (say how many genes, default 2000), reuse the existing `highly_variable`, or none (PCA on all genes).
> **Embedding**: PCA (default n_pcs 50), or use an existing `obsm` key (e.g. `X_scVI` — PCA + HVG then skipped).
> **kNN**: default n_neighbors 15, metric euclidean — or reuse the stored kNN graph.
> **Leiden**: default resolution 1.0.
> Type whatever you want (e.g. "HVG 3000, PCA, n_neighbors 15, resolution 2"), or "use my X_scVI, resolution 1", or "keep the leiden that's already in there".

Map the user's typed answer to ONE `process-cluster` invocation:
- HVG: "hvg 3000" → `--hvg --n-top-genes 3000`; "use the HVG that's there" → `--use-existing-hvg`; "no hvg / all genes" → (omit both). HVG applies ONLY in the default PCA mode and is computed on the **cell vectors only** (Step P's `cell` class). It never subsets — X keeps all genes so the engine sees every KG marker. The panel = **HVG − technical genes ∪ (KG wing markers present in data)** (like `select_gene_panel`): pass `--out {out} --species {sp} --organ {og}` — the same species/organ the caller was given, and `--out` so it REUSES the wing already fetched to `{out}/_ontology_index.json` (by process-profile/init) instead of re-fetching. No server URL needed. KG unavailable ⇒ graceful (HVG − tech only).
- plain params → `--n-pcs --n-neighbors --metric --resolution` (unstated ones fall back to the defaults above)
- "use my X_scVI / don't run PCA" → `--use-rep X_scVI` (PCA+HVG skipped; `--n-neighbors/--metric/--resolution` still apply)
- "use the kNN graph that's already there" → `--use-existing-neighbors` (PCA+kNN skipped; only `--resolution` applies; `--neighbors-key` if non-default)
- "use my existing clusters in column C" → `--use-existing-leiden C` (everything skipped; the column is taken as-is for the cell vectors)

The four graph modes are mutually exclusive (HVG options combine only with the default PCA mode). **Do not
silently pick defaults, and do not silently reuse whatever the input already contains** — n_neighbors=30 once
merged away rare clusters (recovered at 15, or resolution 2–4). The subcluster rounds must use the SAME
metric/n_neighbors as this initial clustering — automatic (`init` inherits them from
`uns['cellman_cluster_params']`), so do not re-supply them.
**When Step P does not apply** (`applicable:false`), the caller still tells the user that annotation is
proceeding straight to celltypes and asks the same fixed question.

**Annotation targets are TWO columns, on the one h5ad:** `leiden` (the cell vectors) and `cellman_process`
(the process vectors — which process). The agent does BOTH a celltype annotation (on `leiden` clusters) and a
process annotation (Step P.2, subtyping the process vectors). Nothing is subset out and nothing is merged
back — both stream into `cellman_annotation`, which ends up holding a value for **every** vector:
`Background` (background), the process subtype (process), or the celltype (cell).

**obs schema written by Step P + engine (final):**
- `cellman_class` — `background` | `cell` | `process`
- `cellman_process` — which process (`neuropil` / `fiber tract` / …), `''` for non-process
- `leiden` — clustering of the `cell` vectors only (`''` for process/background)
- `cellman_annotation` — **all vectors**: `Background` / process subtype / celltype
(No `cellman_score_*` columns are persisted.)

**Judge each class REAL or not, per `{out}/_process_profile.json`:**
- `removal.ratio` — `neg_score` top M% removes M% of ALL vectors by construction, so **chance = 1.0**.
  **>1** = the candidates really do contain somata and the filter is removing them (the gate works).
  **≈1** = the filter is *uninformative* — it is deleting a random M%. **<1** = the candidates are already
  soma-depleted, i.e. the positive axis alone did the job (fine — the filter is then confirmatory, not
  load-bearing; say so in the rationale).
- `pos_missing` / `neg_missing` / `delta_missing` — markers absent from the panel. **A class whose axis is
  mostly missing is NOT evidence of absence** — it is a panel gap; report it, do not conclude.
- **Anatomy is the real check.** A REAL class is spatially coherent and lands where the tissue actually is
  (neuropil → cortical mantle / hippocampal molecular layers / cerebellar molecular layer; fiber tract →
  corpus callosum / fimbria). A class that is speckled everywhere is not real, whatever its ratio.
- **No force-fit (R3 applies here too).** If a class is not REAL, pass `false` — those vectors stay `cell`
  and the agent still gets to judge them after Leiden. Erring toward `cell` is *recoverable*; erring toward
  `process` is **not**. Be strict here and permissive downstream.

`background` = **out-of-tissue**, decided by SPACE, not markers: vectors that sit OUTSIDE the tissue mass —
isolated floaters beyond the Otsu boundary that leaked through downsampling. `process-apply` builds a tissue
footprint from the vector positions themselves (rasterize → morphological close → largest connected component
→ fill holes) and flags vectors outside it. NO marker/density test — inside already-downsampled tissue there
is essentially no empty voxel, the lineage-max axis floors near 0 so it cannot detect emptiness, and the
nucleus axis varies by region so thresholding it invents regional artifacts. `--close-iter` (default 3) tunes
how aggressively in-tissue gaps are bridged. `cell` = **everything in-tissue that is not process**, by
exclusion — no positive test. Then `process-cluster` with the user's parameters and continue at Step 0.

**Step P.2 — subtype the process vectors from the KG** (they never enter Leiden): for each class marked
REAL, judge its `subtypes[]` deltas by R1/R3 exactly as for a celltype, and emit the DEEPEST REAL one
(`glutamatergic neuropil` …) or the bare class. A subtype whose delta is mostly `delta_missing` is a panel
gap, NOT an absence — floor to the bare class and say so.

**Hard-won constraints — violating any of these silently produces nonsense (2026-07-17):**
- **NEVER `k-of-n` / AND / OR over markers.** Yield is driven by the *gene count*, not biology: on one
  dataset "all 7 neuropil markers in their top 25%" gave 0.05% and at top 10% gave **0**, while "both 2
  fiber-tract markers" gave 16% — that is `0.25^7` vs `0.25^2`, not marker quality. Use the standardized
  composite (per-gene z → mean → **re-standardize**), which is invariant to set size.
- **NEVER use detection (`>0`).** KDE blur puts every marker in ~100% of vectors (Snap25 99.9%, Mbp 99.99%,
  Malat1 99.9% at 8µm). "Expressed" separates nothing; only **level** does.
- **Density is for `background` only.** It is legitimate there ("nothing here" IS a density claim) but must
  never gate a process class: cortical L2–L6 neuropil is soma-mixed and therefore *medium* density, so a
  low-density gate deletes it by construction — and near a lesion (e.g. amyloid plaque) glia pile in and
  raise density, making "process is depleted near the lesion" true by circularity.
- **Do not expect a generic soma marker to exist.** Cajigas 2012 found 2,550 transcripts in the neuropil, so
  "mRNA found only in the soma" is a near-empty category. MALAT1 works because of a *mechanism* (nuclear
  retention — never exported), not a gradient. A class is separable from its cell only when that specific
  cell has soma markers of its own (fiber tract has PLP1/SOX10/OLIG2; neuropil has nothing, so its filter
  is expected to read `ratio<1`).

### Step 0 — IS IT A CELL? → may emit **Background**
Background ONLY when ALL hold: `density.pct` bottom-quartile (or `low_frac` high) AND `specificity_peak`
null AND `lineage_markers_present.n == 0` AND `graph_evidence.lineage == null`. **`n ≥ 500` or a top-level
leaf (path has no ".") ⇒ never Background** → continue. `density==null` ⇒ use the identity conditions only.

### Step 0.5 — IS IT A PROCESS? (cluster-level safety net; spatial only) → may emit `neuropil` / `… neuropil` / `fiber tract`
**Step P already ran at the vector level**; this is the *recoverable-error* net for process vectors that were
left in `cell` and clustered. Same KG `tier='process'` terms (driven from the wing slim-index,
no hardcoded gene set; **absent for non-CNS wings** ⇒ this step never fires). Emit the `compartment.candidate`
term (decision `celltype`, `label=compartment_label`, `term_id=compartment_term_id`) when
`compartment.candidate == true`. The engine fires it under one gate — low density (`low_frac ≥ 0.5`) AND **no
specific cell-type anchor** — then picks by which process markers are **ENRICHED (R1, `up`)**:
**Caveat:** this cluster-level gate is weaker than Step P by design — a cell-type anchor blocks it, which is
exactly how whole fiber tracts were lost before Step P existed. Step P, not this, is the primary path.
Both compartments use the same **positive AND NOT-negative** form — the positive markers alone cannot separate an
acellular compartment from the cell that *makes* those transcripts, so each carries a **soma/nucleus negative**:
- **neuropil** — synaptic markers dominate (`n_synaptic ≥ 3`: Syp/Syt1/Snap25/Nrgn/Camk2a/Map2/Dlg4) AND the
  soma/nucleus negative (`n_neuropil_neg == 0`; Malat1 — a nuclear-**retained** lncRNA, so it is present wherever
  a nucleus is and absent from the acellular mesh) is **NOT** enriched (enriched ⇒ a neuron cell body, whose
  soma also carries the synaptic transcripts, not the neuropil). If one transmitter subtype's delta is genuinely
  enriched, `compartment_label` deepens to that **typed neuropil** (glutamatergic / GABAergic / dopaminergic /
  cholinergic / serotonergic); else it stays bare `neuropil`.
- **fiber tract** — myelin markers dominate (`n_myelin ≥ 3`: Mbp/Plp1/Mobp/Mag/Cnp) AND the oligo-**soma**
  negatives Sox10/Olig2 are **NOT** enriched (those ⇒ real oligodendrocyte cell bodies, not a tract).

Both negative sets come from the wing slim-index — `N` of the term **∪ `N` of all its ancestors**, so the
generic nucleus axis (MALAT1) declared once on the process root reaches every class. A wing whose process
terms declare no `N` anywhere simply skips that half of the gate.

If a specific anchor IS present (real somata, even at low density) ⇒ NOT a compartment → continue. scRNA-seq
(real single cells) never fires this. `reason_code="neuropil"` for any neuropil-type, `"fiber_tract"` for a tract.

### Step 1 — LINEAGE (the floor you are sure of) — apply **R5**
Confirm `graph_evidence.lineage` (corroborated by `lineage_scan`). If none resolves → Step 4 (Unknown). Then
**apply R5** to remove every *non-identity* second lineage: honor `lineage_repick.to`; label the REAL lineage
and treat `overlay_lineages` as a program/state OF it; a `lineage_unresolved` pair → its `unresolved_to`
common-ancestor class; a transition-mimicry leaf → its origin-transition term. A weak ambient bleed of a
competing transmitter is contamination, not a second identity. **But if ≥2 lineages are EACH REAL (R3)** —
their OWN transmitters/markers strongly co-enriched (glut `Slc17a7` + GABA `Gad1`; astrocyte `Gfap/Aqp4` +
VLMC `Col1a1/Vtn`) — → **R6 split is MANDATORY, confirming one celltype is FORBIDDEN**, EVEN IF the engine's
`mixed`/`lineage_ambiguous` flags are false (they can miss a strongly-enriched non-neuronal lineage — trust
the OWN markers you can see). Never rationalize one lineage's strong OWN markers as "part of" another's
signature.

### Step 2 — DESCEND to the deepest REAL identity → **celltype** — apply **R3 then R4**
Read `verdict` + `anchors`. For each anchor apply **R3** (its OWN markers ENRICHED, negatives clean,
DF-override); among the REAL anchors apply **R4** (deepest defensible node on the path). Then by verdict:
- **`clear_top`** → `celltype` = the DEEPEST REAL term on its path (usually the top anchor). If the exact
  subtype is not REAL but its region is → **stop at the region node** (`hippocampal pyramidal neuron`) —
  `subclass_resolved = (chosen.tier != "class")`.
- **CP check (mandatory when `graph_evidence.cp` names your term):** apply its `discriminators` — confirm the
  type's OWN distinguishing markers (not shared) and clean negatives; else fall to the safer side / region.
- **NOT a single clear REAL identity** (`ambiguous_tie`, or a leaf that resolves only to a region/class) →
  do NOT pick, pre-classify a tie, or pre-detect a mixture → **DESCEND-AND-SEE (R6)**: descend the node's KG
  children (marker-axis) and let the reclustering show whether it is ≥2 identities (split + recurse) or one
  (label). A region/class floor is usually region-DILUTION, not a true floor — descend and SEE.
- **`class_or_unknown`** → no REAL anchor → **R7 STATE CHECK first**; then **DESCEND-AND-SEE (R6)** into the
  lineage/region children. Only if the descent yields **no separation** (children's markers on the axis but
  cells don't split) is it a genuine floor → **R8** (class FLOOR if lineage clear, else Unknown).

`confidence`: **high** (anchor `excl_on≥3` or `excl_on/excl_n≥0.7`, clean negatives) · **medium**
(`excl_on` 1–2, or you stopped at a region) · **low** (single OWN marker, panel-limited).

### Step 2b — CONTEXT-AXIS subtyping (when the wing declares `ca`; subtypes differ by POSITION, not intrinsic markers)
Some subtypes are NOT distinguished by their own cell-intrinsic markers — only by **position along an
anatomical axis** (liver lobule peri↔pericentral; kidney cortico-medullary; gut crypt↔villus). Their
discriminating signal is the dominant **neighboring** cell type's markers (a periportal LSEC carries
periportal *hepatocyte* markers `Cyp2f2/Hal`) — which §Step 1/R5 would call ambient for *lineage*, but here
**that context IS the subtype evidence**. Applies ONLY when (i) the wing's slim-index declares a `ca` axis,
(ii) the leaf's lineage is confirmed by its OWN intrinsic markers (R5 unchanged — lineage never comes from
ambient), and (iii) that lineage is in the axis's `applies_to`. **Follow the engine's `context_axis` field
mechanically** (poles scored graded — weak zonal signal counts; a lean ≥1.15× sets `dominant_pole`):
- **`mixed==true`** (both poles ≥0.4× each) AND a large top-level parent (n≳1500, no ".") → **split**
  (`needs_subclustering`, reason `context_axis`) — labeling the whole parent by its lean buries the minority
  pole. `check` enforces this (`CONTEXT_AXIS_SPLIT`).
- **else `dominant_pole` set** → label `suggested_subtype` directly (`reason_code="context_axis"`,
  `confidence=pole_confidence` — **low for a weak lean is still preferred over the bare class** on this
  bounded same-lineage axis). Use the engine's pole; do NOT re-judge from the subtype's (off-panel)
  intrinsic markers, do NOT downgrade a weak lean to class.
- **no pole present** → stop at the lineage class. Context can NEVER change the lineage (a `Glul⁺`
  hepatocyte is a pericentral *hepatocyte*, never "pericentral LSEC").

### Step 2c — STATE-AXIS (base-agnostic cell states appended to the label)  [engine: `state_axis`, slim-index `sa`]
Some biology is a **cross-cutting cell STATE**, not a cell type — proliferating, interferon-stimulated,
hypoxic, senescent, mesenchymal/EMT, stemness, estrogen-response, angiogenic, M2-polarization. A `CD8 T cell`
and a `luminal epithelial cell` can BOTH be *proliferating*; encoding that as a separate `proliferating CD8
T cell` Term would duplicate the same `MKI67` program across dozens of types. Instead the wing declares each
state ONCE as a base-agnostic **State** with its own program signature (`sa` in the slim-index), and the
engine evaluates them **independently of the base identity**:
- **After the base celltype is decided** (Steps 1–2), read `state_axis` — each active state has ≥2 of its
  OWN program markers **ENRICHED** (R1: elevated vs the dataset, NOT mere detection; state markers like
  `MKI67`=proliferation are program markers, not base markers).
- **Append EVERY active state** to the label, comma-separated, in a **canonical (alphabetical) order** so the
  same state-set is always one label: `CD8 T cell, interferon-stimulated, proliferating`. Multiple states
  co-occur — do not pick one. (Enrichment score is a tie-break only; it does not set the label order.)
- **LINEAGE GATE (`sa[i].applies_to`).** A state may be **lineage-restricted** — M2-polarization is a
  *myeloid* program, EMT/mesenchymal an *epithelial* program, estrogen-response a *luminal* program. Such a
  state carries `applies_to` (a lineage whitelist); it is kept **only when the base celltype OR one of its
  IS_A ancestors is in `applies_to`** — so ambient `CD163` bleed can never make a T cell "M2-polarization".
  States with **no** `applies_to` are **global** (proliferating / interferon-stimulated / hypoxic /
  senescence / stemness / angiogenic) — any lineage. `finalize` applies this gate on the FINAL base label.
- A state is **NOT a base identity and NOT a split trigger** — it never changes the celltype and never causes
  a descend. A cluster is not split into "proliferating vs not"; the whole cluster's base gets the state tag.
- **Drop a state redundant with the base** (a `mesenchymal cell` base is not re-tagged `mesenchymal`); EMT and
  mesenchymal share one signature (deduplicated). **Enriched-only** — never tag a state on mere detection of
  one marker (that is ambient); ≥2 program markers must be genuinely elevated.
- **Mechanics:** the engine surfaces `state_axis` (evidence); `finalize` auto-appends the active states to the
  base label (or the agent's explicit `dec["states"]` override). Disease/injury states that remain per-base
  (malignant, and the FUZZY states) are handled by `disease_candidates`/R7, NOT here — this axis is only the
  base-agnostic program states in `sa`.

### Step 3 — DESCEND-AND-SEE (the only split path) — apply **R6**
When the leaf is not a single clear REAL identity, **do NOT try to detect what is mixed** — `descend` into the
current KG node's children (§6) and re-judge the sub-groups (`reason_code="multi_identity"` when ≥2 distinct
emerge). The reclustering IS the mixture test: **1 distinct identity → label/merge (pure), ≥2 → split +
recurse each, no separation → floor.** The candidates are the KG children (structural — you never name them
from a detection). Descend one level at a time (region → subtype); always descend even a clear-looking leaf so
a hidden minority / diluted region is not missed. Never tile a homogeneous population — same-label sub-groups
merge and stop.

### Step 4 — NOTHING RESOLVES → **Unknown** — apply **R8**
No lineage AND no REAL anchor AND no state passes R7 AND a descent yields no separation, not Background — after
listing ≥3 considered terms each with a concrete rejection. **Class-floor vs Unknown = whether the LINEAGE is
clear** (clear lineage + descent that doesn't separate → class FLOOR; lineage itself unclear → Unknown). Never
downgrade a confident class/region to Unknown.

### Step 5 — anti-loop termination
If this leaf is already a child of a prior `descend` (path has ".") and the descent STILL yields no new
separation (all sub-groups re-judge to the same parent class) → stop: label the common-ancestor class
(`reason_code="split_no_gain"`), else Unknown. Never Background here (Step 0 already ran). This is the honest
detection floor — descend and SEE, then fall to the floor ONLY after a descent actually failed to separate.

### disease handling
`--disease` is a HINT (R7), never a filter. Omitted → state candidates still surfaced (`hint=false`), assign
a state when its OWN markers win, else the best normal type; never assume normal. `== <name>` → matching
states flagged `hint=true` and sorted first; assign one only when it passes R3 + the specificity gate. Never
invent a non-ontology label; `open_question_flag=true` if the chosen term is in `oq`.

---

## 6. The split — MARKER-AXIS recluster (`helper_v2.py subcluster` / `descend`)

Splitting is **not** whole-transcriptome Leiden. A mixture's distinguishing signal is often a handful of
low-copy markers (`Grik4`, `Fezf2`, `Drd1` vs `Drd2`) — across thousands of genes (PCA) it is diluted and
the mixture won't separate. So the flagged leaf is re-Leidened on the **marker axis**: cells represented by
ONLY the candidates' discriminating markers, making the cell-type difference dominant. (This un-dilutes
present-but-swamped markers; it does NOT rescue markers that are simply under-detected — that is a low-copy
floor needing a larger bin/bandwidth.)

### `descend` — the standard split (KG children, structural, no detection)
**Standard annotation splits ONLY by `descend`** (`helper_v2.py descend --node <label>`): the engine takes
that KG node's **direct children** as the candidate set (from the slim-index — you do NOT name them, do NOT
detect what is mixed), builds their marker axis (below), re-Leidens, and re-profiles each sub-group. You
descend whenever a leaf is **not a single clear REAL identity** — a floor, a tie, or just to verify a
clear-looking leaf — recursively: lineage → region-groups; a resolved region → its subtypes/layers.

This is **dilution-proof and threshold-free**: a big `glutamatergic neuron` floor pools neocortical +
thalamic + hippocampal glut, so at the mixed mean every region/layer marker is flat (`Cux2`~0.003, auc~0.5)
and NO aggregate detector could fire — but clustering ON the region children's marker axis FIRST
**concentrates** each region's cells, so re-judgment then resolves them (proven: this exact floor, descended,
split into real cortical layers — `Cux2`→L2/3 0.89, `Fezf2`→L5 0.98). The mixture only becomes visible AFTER
the split, so `descend` breaks the chicken-and-egg by using the **STRUCTURAL** candidate set (KG children),
never a detected one — a 10% minority, a 50-50, and a cross-region pair all separate by the same call.

> **STOP a `descend` branch** when it yields **no new resolution** — every sub-group re-judges to the SAME
> parent class (no child's OWN markers pass R3 even after concentration). THAT is a genuine detection floor
> (OPC `Pdgfra`~0.001; L6b `Nxph4`=0; microglia ≈ ambient) — keep the parent label honestly. **Descend and
> SEE — fall to the floor ONLY after a descent actually failed to separate.**

### `subcluster --candidates` — the INTERACTIVE-phase tool (user names the split target)
`helper_v2.py subcluster --candidates '["…","…"]'` splits a leaf on an **explicitly named** candidate set —
used only in the free-form interactive phase (§8) when the user asks "split D1 vs D2 here" or "is there a
Treg?". Same marker-axis engine as `descend`, just with a hand-named list instead of the KG children. A path
with **0/1** candidate → whole-transcriptome PCA Leiden (discovery subclustering).

> Rule of thumb: **standard annotation = `descend` (KG children, structural) recursively; the interactive
> phase = `subcluster --candidates` (user-named) or PCA discovery.**

### How the marker axis is built (KG-structure-aware)
Axis genes = **union of the candidates' full discriminating signatures**, where each candidate's full
signature comes directly from the curated+normalized KG:
1. **Ancestor-walk (delta reconstruction).** The KG is normalized — a term stores only its OWN
   discriminators; shared markers live on ancestors (CA1 stores `Wfs1/Spink8`; parent `hippocampal pyramidal
   neuron` stores the shared `Zbtb20`). So a candidate's FULL signature = its own ∪ **all ancestors' (IS_A
   P-chain)** `defining ∪ required` markers.
2. **No specificity heuristic on the axis.** No `share`/`region_share` gene-degree filter, no
   measured-broadness cut — the curated KG already demoted broad markers to supporting, so DF∪R IS the
   specific signal; a broadness cut would only drop genuine local discriminators.
3. **Drop ONLY genes shared by ALL candidates** (for siblings the ancestors cancel; for cross-branch
   candidates the differing ancestors BECOME the discriminators). Keep genes shared by SOME. Drop off-panel
   genes; a candidate left with no unique gene is absorbed into the parent.

Then: subset X to the axis genes → **centering-free per-gene scaling** (divide each gene by the 90th
percentile of its non-zero values — 0-preserving, so rare markers don't sweep no-signal cells) → kNN →
**Leiden on that axis** → sub-clusters.

> **Precondition:** the input X must be **normalized** (per-cell library-size normalized — SSAM vectors
> already are; per-cell total ≈ constant). The axis does per-gene scaling only, so raw-count input would let
> sequencing depth drive the clustering.

### The purity guard is at JUDGMENT, not at the split
The marker-axis Leiden applies **no identity gate** — it only produces cleaner sub-clusters. The guard is
**R3 at re-judgment**: each sub-leaf is re-profiled against the SAME reference and judged by §5 — a specific
type is assigned only if it is REAL. So a spurious sub-cluster is caught at judgment (falls to class /
Unknown), never force-labeled. A sub-group that still mixes → `descend` again (recurse).

### Command notes
- `descend --node <label>` takes the node's **direct KG children** (from the slim-index) as candidates — no
  need to name them. `--candidates` = a list (all paths) or `{path:[labels]}`.
- `--resolution` default 1.0 (the axis is low-dimensional); raise if a mixture stays merged, lower if it
  fragments. `--subcluster-n-neighbors` default 15. **Low-copy panels need a larger bin/bandwidth, NOT lower
  resolution.**
- the re-profiling/judgment after a split uses the SAME engine (`helper_v2.py` is self-contained — the
  identity engine, profiling, and gates all live in it), so a marker-axis split adds no new judgment path.

---

## 7. Merge + resolution + the finalize split-review gate

**Parent-constrained descent.** A subcluster's children are searched ONLY within the parent's KG subtree —
a child of a *neocortical* cluster can only be a neocortical subtype, never pontine/cerebellar. This is why
the KG is a graph: the parent's position bounds the child's search, killing the over-split failure where a
tiny pan-neuronal fragment force-fits a distant region.

**Fragment-escape (label-follows-own-evidence).** When a parent is a weakly/mis-resolved FLOOR (its own
`excl_on==0`), a child must NOT blindly inherit it. The engine relaxes the constraint when a child scores a
DIFFERENT region whose DF is ENRICHED and beats the parent-region. **Rule: a child's label follows ITS OWN
`graph_evidence`, never the parent's inherited floor when they differ** (a child of a "midbrain" parent with
`verdict=clear_top` on `L4 IT` → label L4 IT). Inherit the parent label ONLY when the child has no REAL
identity of its own AND its markers are consistent with the parent.

**Same-label merge.** When a split gives all (or a contiguous run of) sub-leaves the SAME `top` identity →
give them all that label; same label = same annotation → they **merge** in the output. A split that yields
one identity gained nothing (`split_no_gain`) → terminalize; a split that yields ≥2 REAL distinct identities
is real → keep each.

**Differential-discriminator requirement (the over-split antidote, R6-negative).** A subtype S (child of
class C) is valid only with a **differential discriminator** — ≥1 marker that is ENRICHED *and* distinguishes
S from C (a marker of S that is not equally ON in C or the sibling). Present → keep S (L4=`Rorb`, CA3=`Grik4`,
D1=`Drd1`). Absent (only a higher *level* of markers S SHARES with C) → collapse to the parent class; same-
identity siblings merge. Enforced at `check` by `DIFFERENTIAL_UNSUPPORTED` (intrinsic) / `CONTEXT_AXIS_LONELY_
POLE` (context poles, judged cross-leaf so genuine two-pole zonation survives).

**Resolution is CONVERGED, not predicted.** Leiden γ has no closed-form map to #clusters — the right
granularity is converged using the identity check as the oracle. Each split candidate carries its own
`start_resolution = 0.25·k` (k = expected #types, floor 0.1, ceil 2.0). Start LOW (avoid over-fragmentation),
then across rounds: a sub-leaf still flagged `SHOULD_SPLIT_MIXTURE`/`DF_MULTI_SUBTYPE` (≥2 subtype markers
co-present) → raise γ **×1.5** and re-split; sub-leaves with no distinct identity (`split_no_gain`) → same-
label merge, stop. Converge γ toward marker-purity, don't guess it.

### 7.1 Finalize split-review gate

> **⛔ MUST-DESCEND HARD BLOCK (un-bypassable).** Before anything else, `finalize` counts every leaf you floored
> to a class / region-group (`subclass_resolved=false`) that is large & dense and **was NEVER descended**
> (top-level, no "." in the path) and whose node has ≥2 KG children. If any exist it **BLOCKS the write —
> `--accept-large` does NOT bypass it** (only an explicit `--force` + a recorded reason). This is the mechanical
> form of R8/§8: you may not `init → judge → floor → finalize` without a single descend. It is **threshold-free
> and reads no markers** — because a real subtype is DILUTED at the whole-cluster mean (a Fibrous astrocyte is
> `Gfap` auc≈0.5 across the mixed astrocyte pool, 0.99 only AFTER the split concentrates it), so NO signal-based
> gate can see it at round 1; the only correct rule is *descend every floor to CHECK*. `descend --node "<the
> leaf's class/region label>"` each one, re-judge; a descent that genuinely fails to separate leaves depth≥1
> sub-leaves that are no longer top-level → the block clears and the honest floor finalizes. Never reach for
> `--accept-large` / `--force` to escape it — descend.

`finalize` also flags large leaves that could still split into REAL normal subtypes (decided `celltype`,
`n ≥ 1500`, high density `low_frac<0.6`, depth ≤ 1) in three cases, each with an **ambient guard** (a
child/sibling counts ONLY if its DEFINING marker is ENRICHED — `df_present` — so KDE-bleed false siblings
are filtered):
- **`children`** (`subclass_resolved=false`): ≥2 REAL children resolvable here.
- **`siblings`** (`subclass_resolved=true`): the chosen subtype co-occurs with ≥1 other REAL sibling subtype.
- **`bimodal`**: a balanced 50/50 anti-correlated sibling-region mix the mean/fold gates miss (R6).

These HOLD the write (`_annotate_finalize_review.json`). With descend-and-see (§6) you have already descended
every non-single-identity leaf, so this engine gate is now largely **redundant** — if it still HOLDs a path,
**`descend` it** (KG children), re-judge, finalize again; fall to the floor label only after a descent yields
no separation. (The gate's detection heuristics — `df_present`/`children`/`siblings`/`bimodal` — are
superseded by descend-and-see; they only act as a backstop reminder to descend.)

**`floored_over_anchor` HOLD — a RELABEL, not a split.** Separately, `finalize` HOLDs any leaf you floored to a
CLASS / region-group **ancestor** when the engine's OWN `verdict` was `clear_top` with a deeper subtype
**anchor** carrying **≥2 of its own exclusive markers ON** (`excl_on≥2`) and no strong negative leak
(`neg_strong=0`) — i.e. the DEEPEST-REAL test (R1–R3) already passed for that subtype and you stopped short.
This is the frequent failure on **dense whole-transcriptome Stereo-seq**: a marker-axis `descend` there
fragments into cross-region ambient, so it is tempting to floor big glut/DG/thalamic clusters to
`glutamatergic neuron` — burying L5/L6/L4/L2-3 IT, subiculum, claustrum, DG-granule, thalamic-relay calls the
engine had already anchored. The fix is **not** a re-cluster (which would re-fragment): just **reassign the
decision's `label`+`term_id` to the anchor** and finalize again (it won't re-flag — chosen == anchor clears it).
Only `--accept-large` past it when you have a real reason to floor (a negative-marker leak, an unresolved
confusion partner) and record it in the rationale. Grounded entirely in the engine's already-region-coherence-
gated signals (`verdict`/`excl_on`/`neg_strong`/anchor `path`), so it never flags a cross-region force-fit.

---

## 8. Control flow — AUTONOMOUS standard annotation, then interactive

**Standard annotation runs autonomously; NO user prompts mid-run.** You drive each next action from the
profiles and chain rounds until every leaf is celltype / Background / Unknown, then finalize. The caller only
relays your helper calls. The user is engaged **only after finalize**.

> **THE ITERATION IS THE JOB — round 1 is NEVER the end.** `init` gives you round-1 profiles of the *input*
> Leiden clusters; those are almost always COARSE (a whole cortex in one cluster, all glia in another). Judging
> round 1 and finalizing is the failure mode this skill exists to prevent — it buries every subtype (L5/L6
> inside `neocortical glutamatergic neuron`, oligo/astro subtypes inside the bare class). **You MUST `descend`
> every leaf that is not already a confirmed single deepest subtype, and keep chaining rounds until each leaf is
> either (a) a deepest REAL subtype, or (b) a floor you EARNED by a descent that failed to separate (R8).** A run
> that finalizes with class/region-floor leaves and NO descend rounds is WRONG on its face — go back and descend.
> Do not stop because the profiles "look reasonable"; a coarse floor always looks reasonable. Push to the deepest
> honest label for EVERY leaf.

1. `helper_v2.py init` → profile the input's Leiden clusters (round 1 — coarse; the START, not the answer).
2. Judge each leaf by §5 (R1–R8) → a **single clear REAL identity at the DEEPEST tier** (celltype) / **Background**
   / **Unknown**, or **needs descend** (anything else — a floor, a region-group, a tie, OR a clear-looking leaf
   you have not yet verified by descending). **A region-group or class label is `needs descend`, not a terminal
   answer, until you have descended it.**
3. **Descend-and-see (no user prompt, the ONLY split path):** `descend --node <the leaf's current node>`
   (lineage → region-groups; a resolved region → its subtypes/layers). The engine reclusters the cells on the
   KG children's marker axis and re-profiles each sub-group. Re-judge (step 2). **The distinct identities that
   emerge ARE the answer:** 1 → label/merge (pure), ≥2 → keep + **recurse** each, no separation → floor. You do
   **NOT** name candidates or pre-detect the mixture — the KG children are the candidates (structural), and the
   reclustering separates a minority / 50-50 / cross-region mix uniformly. **STOP a descend branch only when it
   yields no new separation** (all sub-groups re-judge to the same parent class — a genuine detection floor) →
   keep the parent label honestly. Never force-fit. Descend even clear-looking leaves so a hidden minority /
   diluted region is not missed.
4. `helper_v2.py check` → fix any flagged leaves → `finalize` → writes `obs['cellman_annotation']`. **This
   ends standard annotation.** Report the outcome to the caller.

### Interactive phase (after finalize — the only user touchpoint)
The caller presents the finalized outcome (English; table + explanation) and offers **① done** or **②
user-directed exploration**. This is free-form — the user asks in their own words; you translate it into
targeted marker-axis work and answer **honestly** (BIAS GUARD: the data decides, never manufacture a positive
to satisfy the user; never editorialize the *sample* as normal/diseased from the absence of a state label —
that is an evidence/coverage outcome, not a determination). Examples:
- **"Is there really no <type X>?"** → find the leaf carrying its signal → `subcluster --candidates
  '["<X>","<sibling>"]'` on that leaf → if a sub-leaf is REAL (R3) → label & report; if X's markers are not
  above background anywhere → say so plainly.
- **"Re-annotate only the astrocytes whose positive markers are ON."** → `subcluster` those paths with the
  astrocyte-subtype candidates → the REAL sub-leaves get the finer label, the rest stay `astrocyte`.
- **"Split D1 vs D2 MSN in the striatal cluster."** → `subcluster --candidates '["D1-MSN","D2-MSN"]'`.

Each refinement = targeted `subcluster`/`descend` + re-judgment (§5) + `finalize`. A finalize AFTER
interaction writes to `cellman_free_annotation` (`finalize … --column cellman_free_annotation`), keeping
`cellman_annotation` intact. If a requested type is genuinely absent, report that honestly.

---

## 9. Validation loop — you own every fix (`check` / `finalize`)

After you write decisions the caller runs `check` → `_annotate_violations.json` (`{cluster_path, check,
severity, message, suggestion}`); helper **changes nothing** — you revise the flagged clusters until clean.
Each violation is a rule from §1 restated as a check. At `finalize`, only **error** severity blocks
(override `--force`); **warn** is advisory — the call is always yours.

**R3 (force-fit):** `SPEC_UNSUPPORTED` (`spec_pass==false` subtype — drop to class/Unknown unless
`hp`/supporting show its OWN markers dominate) · `REQUIRED_ABSENT_ONPANEL` (**HARD block**: the term's OWN
on-panel required/defining is OFF — you may NOT label it by reading raw `up`/`hp`; follow the engine's
`clear_top` or fall to the common class; off-panel-required types like CA1 are exempt) · `DEFINING_ABSENT`
(warn: `df_absent` with `excl_on<2` — a single non-defining marker, likely force-fit) · `OFF_EVIDENCE`
(warn: chosen term not in `anchors` and in a different region than `clear_top` — justify with OWN evidence or
take the engine's anchor).

**R5 (ambient / mislabel):** `GLIAL_MISLABEL` (**error**: `up` top-8 glial/vascular-dominant with NO
neuronal marker, yet labelled a neuron → relabel to the real non-neuronal type read from `up`; real neurons
with ambient myelin are exempt — `up`=own markers, not absolute `Plp1`) · `INTERNEURON_MISLABEL` (warn: `up`
has enriched `Gad1/2` + ≥2 interneuron-subclass markers but labelled glutamatergic → relabel to `cortical
GABAergic interneuron`, unless glut is clear and Gad is ambient) · `UP_CROSS_LINEAGE` (≥2 lineages in `up`
but single celltype — chimera? split or re-check).

**R6 (multiplicity):** `SHOULD_SPLIT_DEEPER` / `SIBLING_RESOLVABLE` / `DF_MULTI_SUBTYPE` (≥2 subtypes' DFs
fold-ON — split in-round, don't wait for finalize) / `SHOULD_SPLIT_MIXTURE` (`mixture_subtypes` ≥2 weakly
co-present — split at high-enough resolution) · `CONFUSION_UNRESOLVED` (resolve per the 3-way action) ·
`CONTEXT_AXIS_SPLIT` (mixed context parent left at class → split) · `DIFFERENTIAL_UNSUPPORTED` (warn:
magnitude-only over-split → relabel to the parent class; a real subtype survives because its differential
marker is ON) · `CONTEXT_AXIS_LONELY_POLE` (warn: single-pole noise-lean → parent class).

**R4/R8:** `NEEDS_SUB_REMAINING` (give a terminal call per Step 5) · `CONTEXT_AXIS_SUBTYPE` (a `dominant_pole`
exists but left at bare class → relabel to `suggested_subtype`) · `MISSING_DECISION` / `BAD_TERM_ID` /
`RS_LT2_BUT_RESOLVED` / `NEGATIVE_LEAK` (reconsider as the suggestion hints).

**Advisory reconsider-prompts** (`check` warn, not forced): `RESCUE_IGNORED` (engine rescued an embedded
vascular/immune anchor but you labeled parenchymal — re-weigh, glia may be KDE smear) · `NO_ANCHOR_SPECIFIC`
(engine found no anchor = `class_or_unknown` but you assigned a specific type — re-check R3, else FLOOR).

---

## 10. Output you write

### 10.1 `{out}/_annotate_decisions.json` (cumulative)
```json
{ "<cluster_path>": {
  "decision": "celltype | needs_subclustering | Background | Unknown",
  "label": "<literature-standard name | 'Unknown' | 'Background'>",
  "term_id": "<ontology id if celltype, else ''>",
  "subclass_resolved": true,
  "confidence": "high | medium | low",
  "reason_code": "multi_identity | sibling_resolvable | pan_class_only | class_floor | split_no_gain | panel_limited | context_axis | neuropil | fiber_tract | transition_background | ''",
  "provisional_subclass": "<best-guess subclass if class-final, else ''>",
  "key_markers": ["geneA","geneB"],
  "rationale": "<condensed S1–S7: which rules fired, markers ON, why>",
  "alternatives_considered": "<other terms: why each rejected (R3/R5/R6)>",
  "open_question_flag": false
}}
```
**Label hygiene (hard):** clean literature name only — no `(n=…)`, `"?"`, `"possible"`, reason text, or `+`.
Unknown→"Unknown", Background→"Background", compartment→its `compartment_label` (`neuropil` / `… neuropil` /
`fiber tract`, with `compartment_term_id`). Reasoning goes in `rationale`.

### 10.2 `{out}/expert_judgment_round{k}.csv`
`cluster_path,n_cells,decision,label,term_id,subclass_resolved,confidence,reason_code,key_markers,rationale`.

### 10.3 Final layout after `finalize`
Helper writes `{input}_annotated.h5ad` (input copy + `obs['cellman_annotation']`), packs every other artifact
into `annotation_artifacts.tar.gz`, and leaves `annotation_out/` with exactly: the annotated h5ad,
`expert_judgment_annotation.csv` (loose), and the tarball. `finalize` also writes `annotation_rationale.md`
(English, one paragraph per leaf: the rule path that fired, the markers ON, competing terms rejected, and the
chosen term's grounding PMIDs from `GM`).

---

## 11. Self-check — run BEFORE writing each decision (helper validates the same)
1. Every gene in `key_markers` appears in this leaf's `up ∪ hp ∪ parent_ctx ∪ ontology_gene_table(present)`.
2. `celltype` ⇒ `term_id` exists in `_ontology_index.json`; `label` == that term's `L` (never invent).
3. You did NOT deny a clear absolute identity because `up`/`sibling_contrast` was flat (R1 backbone survives
   fragmentation; `up` is demoted).
4. You corroborated the call across ≥2 axes (absolute + one of hp / ontology-mean / lineage_scan / supporting).
5. **R3:** a subtype (`subclass_resolved=true`) has `spec_pass==true` — ≥half its OWN markers ENRICHED — OR
   you defended a `spec_pass==false` term from `hp`/supporting. Otherwise → class FLOOR / Unknown.
6. No unexplained negative leak (`neg_on>0`) except pan-neuronal modality on neuron terms.
7. **R6:** `needs_subclustering`+`multi_identity` ⇒ ≥2 REAL co-dominant identities (`mixed==true` or ≥2 REAL
   distinct anchors), NOT `flags.multi_lineage` or ambient bleed alone; and you did NOT split a single-
   dominant-lineage cluster merely because no subtype passed (→ FLOOR, R8).
8. Sub-leaf ⇒ `parent_ctx` pooled; the same `term_id` never carries two labels; split mechanism = marker-axis
   `subcluster`/`descend`.
9. **R7:** before any `class_floor`/`Unknown`, you inspected `ranked` for `is_state` + `disease_candidates`
   and confirmed no state passes R3 + `n_specific_on≥1`. A `null top` was NOT read as "→ class".
10. Before `Unknown`: `alternatives_considered` lists ≥3 candidates each with a concrete rejection.

## 12. Forbidden (each has caused real misannotations)
- Treating **"detected `>0`" as ENRICHED** (R1) — ambient single molecules are everywhere; require the
  robust/fold/rank bar.
- **Force-fitting the nearest specific term** when its OWN markers are mostly OFF (R3) — CA1 on a
  hypothalamic leaf sharing `Wfs1`; a subtype on a pan marker with its DF off (`REQUIRED_ABSENT_ONPANEL` is a
  HARD block). A confident wrong region is worse than an honest class.
- **Denying a clearly-present absolute identity because vs-sibling `up` is flat** (R1) — that collapse is
  exactly what the backbone fixes.
- **Calling a chimera on ambient/KDE bleed** (R5) when one lineage dominates; or **switching identity to an
  overlay lineage** because its program is high-magnitude (honor `lineage_repick.to`).
- **Splitting a single-dominant-lineage cluster because no subtype passed** (R6/R8) — that tiles a continuous
  gradient into meaningless fragments; FLOOR at the dominant subtype or its class. (And the converse: when ≥2
  identities genuinely co-resolve, DO split — merge cleans redundancy.)
- **⛔ Stopping at a class/region FLOOR without `descend`ing** (R4/R8 — THE failure this skill exists to prevent).
  A floor is EARNED by a descent that failed to separate, never chosen to save a round. Emitting a `class_floor`
  / region-group label on a leaf you never descended — or one still carrying `mixture_subtypes` /
  `children_resolvable` / a subtype in `df_present` — is INVALID; descend it first. A floor is usually
  region/subtype DILUTION (the minority is invisible at the mixed mean and ONLY appears after the marker-axis
  reclustering concentrates it), so "nothing was obviously on" is never a reason. **Every** class/region floor's
  `rationale` must name the children you descended into and why each failed (off-panel / detection-floor /
  ContextAxis continuum / `split_no_gain`). Finalizing with floor leaves and zero descend rounds = wrong run.
- **Finalizing after round 1** (or after too few descends) — round 1 is the coarse *input* clustering; treating
  it as the answer buries every subtype. Chain descend rounds until each leaf is a deepest REAL subtype or an
  EARNED floor.
- **Burying a `tier=state` identity under the bare class** because you stopped at `null top` without R7; or
  dismissing a state "because it's a state / in-vitro program / no `--disease`" (R7).
- Forcing a chimera into one label; picking the higher-AUC of two close passing siblings (→ split);
  concluding Unknown without ≥3 enumerated candidates; downgrading a confident class to Unknown.
- Whole-transcriptome forced marker assignment (`signal_split`) — it does not exist here (contaminates).
- Inventing `term_id`/`label`, putting markers/reasons in `label`, writing the h5ad (only finalize), or
  asking the user (only the caller does).

---

## 13. Worked examples (copy this reasoning shape)

**A — strong absolute identity survives fragmentation → celltype (R1·R3·R4).** A 150-cell fragment;
`hp`=Grik4/Npy2r/Cpne4 high; CA3's OWN markers ENRICHED (R1 fold/rank), negatives clean → R3 REAL; `up`≈0.5
(flat vs siblings — demoted). → **celltype, CA3 pyramidal, high.** The flat `up` does NOT veto it.

**B — the term's OWN markers are off-panel → panel-blind, judge by supporting (R3 inert).** Cortical
fragment; L2/3's OWN required/defining (`Cux2/Rorb`) are off-panel or unmeasured here, so `spec_n==0` → R3
cannot reject. Its **supporting** markers (`Lamp5/Dact2/Otof`) are strongly ENRICHED and lineage=
glutamatergic → judge on supporting → **L2/3 IT, medium** (noted: called on supporting, required off-panel).
Do NOT FLOOR to the bare class just because the required didn't fire — R3 is inert here, not failed.

**C — genuine chimera → split (R6).** `up`=[Mbp 0.81, Plp1 0.78, Aqp4 0.74, Gfap 0.70, Cx3cr1 0.66]; n=17k.
Oligo + astro + microglia each REAL (OWN markers strong, comparable) → `codominant_lineages` → **split**,
`--candidates '["oligodendrocyte","astrocyte","microglial cell"]'`.

**D — ambient bleed is NOT a chimera → celltype (R5).** A cortical band (n=18k) glut-dominant; `up` also has
Pvalb/Lamp5 (sparse interneurons), Slc1a2/Apoe (astro), Mbp/Plp1 (myelin). `flags.multi_lineage`=[glut,GABA,
astro,oligo] but only glut is REAL (others' OWN markers not ENRICHED) → `mixed=false` → **NOT a chimera** →
FLOOR to a glut layer if one is REAL, else `glutamatergic neuron`. Do NOT split.

**E — class FLOOR → descend, not stop (R4/R8).** Top-level `glutamatergic neuron` blob (n=36k); every layer
marker flat at the mixed mean → `verdict=class_or_unknown`. Do NOT label the bare class → **`descend --node
"glutamatergic neuron"`** → sub-leaves concentrate each region → re-judge resolves L2/3 (`Cux2` 0.89), L5
(`Fezf2` 0.98), CA fields. Fall to the class only if a descend branch yields no separation.

**F — force-fit via shared markers → FLOOR, not the subtype (R3).** A non-hippocampal glut leaf; `up`=Wfs1,
Crym, Pdyn, Cartpt (neuroendocrine). `Wfs1/Crym` are also on CA1 terms, so `identity_scores` offers only
hippocampal candidates — but CA1's OWN `Fibcd1/Spink8` are flat → `spec_cov≈0.33`, R3 fails, engine nulls
`top`. → **celltype `glutamatergic neuron`, class_floor, low** (noted: CA1 a force-fit). Contrast A: a real
CA3 has `Grik4/Cpne7` ON → R3 passes → keep CA3.

**G — over-split → merge (R6-negative).** Parent split into 7 sub-leaves; six score `top`=CA3, one DG
granule → label the six **CA3** (merge in output), the one **dentate gyrus granule cell**; do not split the
six further (`split_no_gain`).
