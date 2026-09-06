---
name: annotate-mode
description: Expert-judgment cell-type annotation of clustered scRNA / spatial h5ad, grounded in a Cell-Ontology module (slim-index). Every decision is one test applied at four levels — a label is REAL only when the term's OWN markers are ENRICHED (elevated vs the whole dataset, not merely detected) AND DISCRIMINATIVE (its own low-degree markers, not shared/pan/ambient); assign the DEEPEST term that is REAL, FLOOR honestly to the class/region when none is, and never force-fit. Mixtures are not pre-detected: whenever a cluster is not a single clear identity, it is DESCENDED — re-Leidened on the marker axis of its KG children (a structural candidate set, no thresholds) and each sub-group re-judged; the distinct identities that emerge are the answer (1 → pure/label, ≥2 → split + recurse, none → floor), so a minority / balanced / cross-region mix all separate by the same call. Standard annotation runs autonomously to finalize; the user is consulted only in a free-form interactive phase after finalize. Writes obs['cellman_annotation'].
---

# annotate-mode — one identity test, four levels, marker-axis split

**You are the domain expert for one module (species + organ).** The ontology module you load (slim-index)
is your domain memory — each term's markers (required / defining / supporting / negative),
the class→region→subtype hierarchy, confusion pairs, context axes. `scripts/helper.py` does all the
compute (global statistics, profiling, the marker-axis split, gates, the h5ad write) and hands you a
**profile** per cluster; **you supply the biological judgment.** The caller talks to the user — you never do.

> **ENGLISH ONLY — every word the user reads.** Reports, tables, cluster rationales, questions,
> warnings, error relays, the post-finalize conversation (§8): all English, **whatever language the user
> wrote in.** A prompt in Korean, Japanese, Chinese or any other language is still answered in English —
> match its content, never its language, and never apologise for or explain the choice. This is not a
> style preference: labels come from the Cell-Ontology, whose terms exist only in English, so a
> translated report silently detaches the prose from the term it names and makes the judgment
> unverifiable. Cell-type names, marker genes and rule names (R1–R8) are **never** translated or
> transliterated. The one thing that may be quoted in the user's language is a literal string from their
> own data (an `obs` value, a file path).

Everything below is built on **one principle**, applied at four levels. Read §1 first; the rest is that
principle in full detail.

> **THE PRINCIPLE.** A label is **REAL** only when the term's **OWN** markers are **ENRICHED** (elevated
> vs the whole dataset — *not merely detected `>0`*; in spatial/KDE data almost everything is detected at
> ambient single-molecule level) **AND DISCRIMINATIVE** (its own low-degree markers — not ones it merely
> shares with siblings, not pan/lineage-wide, not ambient bleed). **Assign the DEEPEST term that is REAL;**
> when nothing deeper is REAL, **FLOOR honestly** to the confident class (for brain, the region) or
> **Unknown**. **Never force-fit** a specific term through shared/leaky markers.

---

> **A REFERENCE (answer-key) was supplied? → this is the wrong SKILL.** If the caller handed a set of
> expected cell types — a label list, or an annotated reference h5ad + column — the run is
> **annotate-reference-mode**: read `skills/annotate-reference-mode/SKILL.md` instead. It is this same
> method with one delta (the label space is restricted to the answer-key, and anything the answer-key
> cannot explain becomes Unknown rather than an out-of-reference class floor). **The caller declares
> `mode : reference` when the user invoked `/annotate-reference-mode`**, and the presence of a reference
> input switches the mode on its own as well — either signal is enough, so a forgotten declaration never
> silently turns a reference run into an ordinary one.

## 1b. One mode — clustering-based annotation. Tissue **processes** are out of scope.

This skill has **exactly one method**: marker-axis Leiden → judge each cluster by R1–R8 (REAL/DEEPEST/FLOOR).
Everything in §1 and §5–§13 is that method. The engine is `init` / `subcluster` / `descend` / `check` /
`finalize` and the unit of decision is **a cluster**, which gets **one** celltype.

**Tissue processes are NOT annotated by this skill.** The module may serve `tier='process'` terms (neuropil,
fiber tract, …), but they are **not celltypes** and there is no path to label them here:
- The engine **excludes** `tier='process'` terms from lineage/anchor derivation, so they never appear in
  `graph_evidence` / `identity_scores` / `verdict`. That is deliberate — without it a process term is picked up
  as a lineage root and a real oligodendrocyte gets called `fiber tract`.
- `check` raises `PROCESS_IN_CLUSTERING` (warn) if a process `term_id` ever ends up in your decisions.
- **A process shares its positive markers with the soma that made the transcripts** (neuropil's synaptic markers
  *are* the neuron's), so a positive marker axis cannot separate it — gathering the neurons and re-clustering
  splits them into neuron subtypes, never into neuropil-vs-not. Do not attempt it.

If the user asks for a process, say plainly that this version does not support it. **Never force-fit a process
term onto a cluster**, and never invent a process label. (A per-vector scoring engine for processes is under
development and **not part of this release** — `helper.py score-*` is rejected. Its known defects are
fixed-quota candidates, no agent-judgeable material, and unjudged auto-subtyping.)

---

## 1. The rules (the vocabulary — defined ONCE, referenced by name everywhere below)

Six named rules and a small shared vocabulary. Every case in §5–§9 is one of these rules applied at a
level (lineage / region / subtype / state). The engine (`helper.py`) precomputes each rule's inputs and
exposes them as profile fields; the **verdict is always yours** — the engine gives evidence, not the answer.

### Shared terms (one name per concept — used throughout)
- **leaf** — a cluster (or sub-cluster) you are judging.
- **the reference** — the whole-dataset per-gene statistics (`_global_stats.npz`), computed once at `init`
  and reused by EVERY later round, so absolute identity never collapses on a small fragment.
- **share(gene)** — how many module terms list the gene = its gene-degree. **OWN / DISCRIMINATIVE marker** =
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
- **`adaptive-floor` (sparse platforms):** on a 0.5 µm KDE vector field the (a)/(b) thresholds are lowered to the data's
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
single homogeneous type, **descend into its CURRENT KG node's children** (`helper.py descend --node
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

### R7 — DISEASE/INJURY/MALIGNANT ARE ORDINARY CELLTYPES (same level, no fallback)  [engine: R3/R4]
There is **no separate "state-term identity" step and no before-floor fallback.** Disease, injury,
metaplastic, dedifferentiated and malignant identities are ordinary **celltype nodes** (`tier=class`/`subclass`)
carrying their OWN discriminative markers, so they are scored **at the SAME level as normal types** in
`identity_scores` (`top`/`ranked`) and chosen by the SAME test — R3 REAL (own markers ENRICHED + DISCRIMINATIVE)
then R4 DEEPEST. You **never pick a disease celltype because a normal one is absent** — you pick it when ITS
OWN markers WIN, exactly like any celltype. A node carried only by pan-reactive markers (`Gfap`/`Vim`/`Apoe`)
simply **fails R3** (not discriminative) and is not chosen; the honest label is then the normal counterpart or
the FLOOR (R8), as for any under-supported candidate. (Base-agnostic PROGRAMS — proliferating / IFN / hypoxic
— are NOT identities: they live in `state_axis` and are used ONLY in the interactive phase, §8.)

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
| (disease/injury/malignant celltypes score WITH normal in `top`/`ranked`) | **R7** | disease is an ordinary celltype at the same level — no state-fallback field |
| `class_or_unknown`, `class_floor`, `DIFFERENTIAL_UNSUPPORTED` | **R8** | a confident class downgraded to Unknown, or a magnitude-only over-split |

---

## 2. Roles (do not cross)

| actor | responsibility |
|---|---|
| **expert agent = you** | Run `helper.py`, read profiles, judge each leaf by R1–R8 (§5), drive the split yourself, write the decisions JSON + per-round CSV. **Never ask the user.** Standard annotation is autonomous. **You own your `--out` directory exclusively.** |
| **caller (= the manager/main session)** | Spawns **one expert agent per dataset** and then **WAITS** for that agent's final result — nothing else. The caller runs **exactly two** `helper.py` commands, both read-only and both BEFORE any agent exists: `modules` and `resolve-module` (§3 MODULE RESOLUTION — they take no `--out`, touch no h5ad and write nothing). Beyond those it **does NOT** run `helper.py`, read the agent's `--out` directory, inspect profiles, judge clusters, decide descents, or make any annotation call of its own. Doing so duplicates the agent's work and, if it writes, **corrupts `_annotate_state.json`** (two loops on one directory). Each `--out` is owned by exactly one expert agent. The caller's only jobs: **settle the KG module of every dataset first** (§3 MODULE RESOLUTION — one confirmation round covering the whole batch, before a single agent is spawned), **translate the invoked skill into a mode and pass it on** — the user declares the mode by choosing `/annotate-mode` or `/annotate-reference-mode`, not by a prompt field, and the caller carries that choice to the agent as `mode : reference` in the **spawn prompt** (the agent is a separate process and does not inherit the skill the caller loaded), spawn + wait, then relay the finished result to the user and run the post-finalize interactive phase (§8). For a single dataset it may run the expert inline instead of spawning — but still **one owner only**, never a second parallel brain on the same directory. **Everything it says to the user is in English, regardless of the language the user wrote in** (see the rule at the top). |
| **`helper.py`** | The **single, self-contained backend** — all compute (profiling, the identity engine, gates), the marker-axis split, and the final h5ad write, in one script. **Validates** your decisions (`check`/`finalize`) and returns violations — it **never silently changes a decision**. Every command (`modules`/`resolve-module`/`init`/`subcluster`/`descend`/`check`/`finalize`) runs in this one file; it does not call any other helper. `modules`/`resolve-module` are the read-only pre-`init` pair the CALLER runs (§3); everything else belongs to the one agent that owns `--out`. |

---

## 3. Run the helper (exact commands)

`{HELPER}` = absolute path to `scripts/helper.py` (resolve once:
`find . -path '*CeLLMan/scripts/helper.py'`). `{python}` = the user venv. **Run the WHOLE
workflow through `helper.py`** — it is the single, self-contained entry point (all commands run in this one file). The
ontology server is automatic — pass `--species`/`--organ`; the helper writes `{out}/_ontology_index.json`.

**INPUTS the caller gathers from the user's prompt BEFORE `init` (clustering is the user's job — done in their
notebook — so the h5ad arrives already clustered):**
- **`{input_h5ad}`** — path to the h5ad to annotate. **Preconditions `init` enforces before anything
  else:** `adata.X` is **log-normalised** (not raw counts, not scaled/z-scored/Pearson residuals) and
  `adata.var_names` are **gene symbols** (not Ensembl IDs). Violate either and `init` exits with
  `[init][ERROR:INPUT_MATRIX]`, writing nothing — the agent stops and returns `INPUT_MATRIX_REJECTED`
  with the engine's diagnosis, and **you relay that diagnosis to the user verbatim**. It is not advisory:
  the run would otherwise complete and produce garbage labels (markers are matched by symbol; the
  marker-axis statistics assume lognorm). `--force` downgrades it to a warning — pass it **only** when
  the user has explicitly asked to proceed on non-standard input, never to get past the gate yourself.
- **`{cluster_col}`** — the **obs column holding the clusters/annotation to annotate** (e.g. `leiden`). The user
  names it in the prompt (*"cluster column: leiden"* / *"annotate the `seurat_clusters` column"*). It is a
  REQUIRED input — `init --cluster-col {cluster_col}` reads exactly that column, and every distinct value in it
  becomes a leaf to annotate. If the user did not state which column, the caller **asks** (the one question it
  must resolve up front); do not guess `leiden` when the h5ad has several candidate columns.
- **`{species}` / `{organ}`** — the KG module, resolved by the **MODULE RESOLUTION** protocol below before
  `init` is run. The user's tissue is free text; `--organ` takes the **canonical** module name.
- **`mode`** — **not a field in the user's prompt; it comes from WHICH SKILL they invoked.** The user
  declares the mode by choosing `/annotate-mode` or `/annotate-reference-mode`; you translate that choice
  into `mode : reference` in the **spawn prompt**, because the agent is a separate process and does not
  inherit the skill you loaded. Set it whenever the reference variant was invoked, or whenever you pass
  any reference input. (The reference input switches the mode on its own too — either signal alone is
  enough, so a forgotten declaration cannot silently downgrade a reference run to an ordinary one.)
- **reference inputs** *(reference runs only)* — `reference` (a label-list file) **OR**
  `reference h5ad` + `reference column`; optionally `reference scope` (`strict` default / `refine` —
  label-space depth, **not** the mode switch) and `reference alias`. Relay the user's paths verbatim.
  Passing `mode : reference` with **no** reference input makes the agent stop with
  `REFERENCE_INPUT_MISSING` — an answer-key cannot be invented, so ask the user for it before spawning.

**MODULE RESOLUTION (the caller does this BEFORE spawning the agent — it is the caller that talks to the user):**

1. **Ask the KG — do not do this by hand.** One call turns the user's wording into the vocabulary the KG
   actually serves, and reports the server it resolved against. **The MCP tool is the default path:**

   ```
   mcp__cellman-kg__resolve_module(tissue="{the user's wording}", species="{species}")
   mcp__cellman-kg__list_modules(species="mouse")     # the full served list, to show coverage
   ```

   **Fallback — use it whenever the `cellman-kg` tools are not present in this host** (the server was not
   registered; nothing else about the run changes):

   ```bash
   {python} {HELPER} resolve-module --species {species} --tissue "{the user's wording}"
   #   (or, in the SKILL's own "mouse, brain" form:  resolve-module --tissue "mouse, brain")
   #   {python} {HELPER} modules [--species mouse]
   ```

   **The two are the same code** — the MCP server shells out to that very command — so they return the
   same answer against the same server. Pick whichever is available; never run both to compare.
   It returns `{status, organ|candidates[{organ,n_terms,match_type}], can_serve_together, server_url, served_organs}`.
   The normalisation (lowercase, separators, articles, `brain tissue` → `brain`) and the synonym /
   parent / umbrella / ambiguous tables are **in the command**, so you, the agent and the engine cannot
   drift apart. An MCP host may expose the same thing as `resolve_module` — identical answer, same code.
2. **`status="exact"` → use it silently.** `brain` → `brain`, `Large Intestine` → `large_intestine`. No
   question is asked; a name that already IS a module is never second-guessed.
3. **RECORD `server_url` and pass it to the agent**, which passes it to `init --server-url`. The built-in
   default and a deployment's `$CELLONTO_SERVER_URL` are often different addresses; without this the
   module you confirmed and the wing the engine fetches can come from **different servers**, silently.
4. **`status="candidates"` → ASK the user and WAIT.** Each candidate's `match_type` says which case it is —
   these are the four, and `resolve-module` has already picked the right one for you:
   - **different name** (`match_type=synonym`) — `mammary gland` → `breast`, `colon` → `large_intestine`,
   - **finer than the KG (`match_type=parent_of_subtype`)** — the user names a part, the KG serves only the whole
     (`cerebellum`/`hippocampus` → `brain`, `white adipose tissue` → `adipose_tissue`). One candidate:
     confirm that the parent module is acceptable for this sample.
   - **coarser than the KG (`match_type=umbrella_children`, `can_serve_together=true`)** — a region the KG **split into
     several modules** (`intestine` → `small_intestine` + `large_intestine`; `CNS` → `brain` +
     `spinal_cord`; `immune` → `blood` + `bone_marrow` + `lymph_node` + `spleen` + `thymus`). List
     **every** child module and offer both readings explicitly: **one of them**, or **all of them served
     together**. Say which way you would lean and why — all of them when the sample genuinely spans the
     parts (a whole gut roll, whole CNS section), one of them when the user probably has a single part.
   - **plainly ambiguous** (`match_type=ambiguous`) — the word maps to different organs by intent (`gut` may be either
     intestine; `throat` → `trachea`/`tonsil`/`mouth`).
   Present the candidate module(s) with one line of WHY each was proposed, and state plainly that the KG has
   no module under the name the user wrote. Never start `init` on an unconfirmed module.
5. **The user's answer decides — always:**
   - picks one → that module.
   - **asks for several / "use them all"** → serve them ALL: `--organ brain,spinal_cord` (comma-separated). The
     helper fetches each wing and merges them into one slim-index (common-lineage terms de-duplicated). The
     label set then spans those organs — say so, since a wider candidate set makes cross-organ force-fit easier.
   - **names a module you did not propose** → use the user's module verbatim. Do not argue it back to your own
     candidate. Only if that name 404s do you report it and ask again (the helper's error lists the served modules).
6. **`status="none"`** (the KG has no such tissue) → tell the user and stop. Never annotate from
   guesswork, and never silently substitute a "close" organ. A `match_type=token` candidate is the
   weakest kind (a served name appeared as a word of the input, e.g. `left kidney` → `kidney`) — treat
   it as a question, never as a match.

The agent re-validates whatever `(species, organ)` it is handed. If it is not an exact served module and the
caller did not record a user confirmation, the agent **stops and returns `MODULE_CONFIRM_REQUIRED` + candidates**
instead of guessing — the caller then asks the user and re-spawns it with the confirmed module(s).

**BATCH RULE — several datasets: resolve ALL modules first, then start ALL agents.** Module resolution is a
**barrier**, not a per-dataset step. With N datasets the caller:

1. Runs `resolve-module` for **every** dataset first, and spawns **nothing** yet. Datasets that come back
   `status="exact"` are settled silently; the rest collect candidates.
2. If **any** dataset needs confirmation, asks **one** question covering them all — never one question per
   dataset, and never a question after some agents are already running. Show the whole plan so the user sees
   what is settled and what is being asked:

   ```
   Modules for this run:
     [1] mouse_brain.h5ad   mouse, brain            -> mouse/brain            (exact match)
     [3] mouse_liver.h5ad   mouse, liver            -> mouse/liver            (exact match)
     [2] ts_breast.h5ad     human, mammary gland    -> human/breast?          (synonym — confirm)
     [4] gut.h5ad           mouse, gut              -> small_intestine OR large_intestine? (choose, or both)
   Datasets [2] and [4] need your confirmation before anything starts.
   ```
3. Applies the answers (a pick, several modules, or a module the user names instead), then spawns **all N
   agents at once**, in parallel, each with its own `--out`.

No dataset may start annotating while another dataset's module is still unconfirmed — an exact-match dataset
waits at the same barrier as an ambiguous one, so every run in the batch begins from the same confirmed plan.
The only exception is a dataset the user drops from the run; the rest then start together.
- **`{disease}`** (optional) — pass `--disease "<name>"` only if the user says the data is disease/cancer (R7 hint).
- **`{output}`** (optional) — a save path the user names for the annotated h5ad (*"output: /data/kidney_done.h5ad"*,
  *"save it to …"*). Pass it as `--output {output}` on `init` and it persists to `finalize`; passing it on
  `finalize` instead also works and wins. Omitted, the h5ad is written to
  `{out}/{input_basename}_annotated.h5ad` — **inside the `--out` directory, not beside the input.** Relay the
  user's path verbatim; do not invent one, and do not move or copy the file afterwards to "fix" the location.
- **`{use_rep}`** (optional) — an `obsm` key the user names for subdividing (*"use rep: X_scVI"*, *"subcluster on
  the scVI embedding"*). Pass it through as `--use-rep {use_rep}`; **never substitute one of your own.** Omitted,
  the helper uses `X_pca`, or recomputes normalize + log1p + PCA per group when `obsm` has no `X_pca`. It only
  takes effect where a marker axis cannot be built (0/1 KG candidates → the PCA fallback below), so it changes
  nothing on a normal marker-axis descend. An integrated embedding is never selected implicitly, because it
  changes which splits are reachable — so if the user mentions one, relaying it is the only way it is used.

**Output directory (hard rule):** `{out}` = an `annotation_out/` directory inside the caller's output
location (`--out {base_out}/annotation_out`; `mkdir -p` it before `init`). All artifacts and the final h5ad
land there. Pass the same `--out` to every round.

```bash
# modules / resolve-module — the ONLY commands that run BEFORE init (no --out, no h5ad, read-only).
#   Normally you reach these through the MCP tools (mcp__cellman-kg__resolve_module / list_modules);
#   this is the fallback form for a host where the cellman-kg server is not registered.
#   resolve-module turns free text into the served vocabulary and reports the server_url it used;
#   pass that server_url to init --server-url so the LLM and the engine read the SAME KG.
{python} {HELPER} modules [--species {species}]
{python} {HELPER} resolve-module --species {species} --tissue "{free text}"

mkdir -p {out}
# The input h5ad ALREADY carries a cluster/annotation column (leiden or any obs column the user names —
#   clustering is the user's job, done in their notebook). You annotate that column directly:
#   init → judge → descend/subcluster → check → finalize. There is no process/pre-Leiden partition step.

# init — compute the reference (global stats) + fetch ontology + profile the input's clusters.
#   --cluster-col {cluster_col}   the obs column the USER named in the prompt (leiden / any annotation column).
#   --disease "<name>"     pass when the input is a disease/cancer dataset (HINT only, R7).
#   --subcluster-metric / --subcluster-n-neighbors  init inherits the input clustering's metric/n_neighbors
#     from uns['cellman_cluster_params'] if present. Supply them ONLY when they must MATCH an external
#     clustering that did not record those params.
#   --adaptive-floor                               sparse platforms (0.5µm KDE vector field): lower R1 thresholds to ambient×F (persists to all rounds).
#   --use-rep <obsm key>   ONLY when the user named an embedding to subdivide on (e.g. X_scVI). Decided here
#     and persisted for every later round. Omitted = X_pca, else a per-group normalize+log1p+PCA recompute.
#     Applies solely to the PCA fallback (0/1 candidate); a marker-axis split ignores it. An unknown key aborts.
{python} {HELPER} init --h5ad {input_h5ad} --cluster-col {cluster_col} \
   --out {out} --species {species} --organ {organ} [--disease "{disease}"] \
   [--adaptive-floor] [--use-rep {use_rep}] [--output {output}]   # (--subcluster-metric/-n-neighbors: only for externally clustered h5ad)

# subcluster — MARKER-AXIS split of a NAMED multiplicity (R6). --candidates = the REAL distinct terms to separate.
{python} {HELPER} subcluster --out {out} --paths '["4.8"]' \
   --candidates '["CA2 pyramidal neuron, hippocampus","glutamatergic neuron, fasciola cinerea"]' \
   [--resolution 1.0] [--subcluster-n-neighbors 15] [--gpu 0]
#   per-path candidates: --candidates '{"4.8":["CA2 …","FC …"], "11.2":["D1-MSN","D2-MSN"]}'
#   0/1 candidate on a path → automatic whole-transcriptome PCA fallback (discovery subclustering).
#     This is the ONLY route that reads --use-rep (set at init): the named obsm key, else X_pca, else a recompute.

# descend — UNCONDITIONAL KG descent of a class/region FLOOR leaf (R4/R8), region→subtype, kg-descent style.
{python} {HELPER} descend --out {out} --paths '["cl3"]'   --node "glutamatergic neuron"                 # → its KG children (regions)
{python} {HELPER} descend --out {out} --paths '["cl3.2"]' --node "neocortical glutamatergic neuron"     # → its layers

# check — validate your decisions (helper never edits them; you fix the flagged clusters).
{python} {HELPER} check --out {out} --decisions {out}/_annotate_decisions.json
# finalize — write the annotated h5ad (only when error-free AND the split-review gate is clear, §7.1).
#   Destination: --output here > --output given at init > {out}/{input_basename}_annotated.h5ad.
#   Pass --output ONLY when the user named a path; never redirect the write on your own.
{python} {HELPER} finalize --out {out} --decisions {out}/_annotate_decisions.json [--accept-large] [--output {output}]

```

If the server 404s for this module → **stop and tell the caller** (no guessing). Read the ontology from
`{out}/_ontology_index.json` — schema `{terms:{id:{L,T,P,R,S,N,D,DF,DS,DP}}, cp, oq, ca, sa}`: **L**=label,
**T**=tier, **P**=parent ids (IS_A ∪ REFINES — the module hierarchy; DEVELOPS_INTO is NOT a parent, so
developmental cells never inherit markers), **R**=required, **S**=supporting, **N**=negative, **D**=disease
context, **DF**=defining (one is sufficient), **DS**=developmental_stage (precise: Carnegie/pcw/Theiler/E-day),
**DP**=developmental_period (`preimplantation`/`embryonic`/`fetal`/`neonatal`/`postnatal`/`adult`) — **present
only on developmental cells; use it to decide adult-vs-fetal**: in fetal/embryo data a `DP=fetal` term (radial
glia, hepatoblast, fetal cardiomyocyte …) is the correct answer, not its adult counterpart. **ca**=context axes
(§5 Step 2b), **sa**=state axes (base-agnostic cell states + transition-guard, §5 Step 2c), **cp**=confusion
pairs, **oq**=open questions, **GM**=grounding PMIDs (cited in the finalize rationale only). **State is a `sa`
axis, NOT a `tier=state` term** (that tier is retired): base-agnostic programs (proliferating / IFN-stim /
EMT / reactive) live in `sa` as EVIDENCE and are applied ONLY in the interactive phase (§8), never in standard.
Disease-identity terms (malignant / injury / metaplastic …) are ordinary `class/subclass` anchors gated by R3
at the same level as normal — never a separate tier, never a fallback.

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
| `exclusive_mixture` | R6 | reciprocally-exclusive same-lineage DFs each ON in a disjoint subset → `needs_subclustering`. |
| `transition_candidates` / `transition_note` | R5 | origin-anchor-retained + mimicked-program → label the transition term, do NOT split or call the mimicked lineage. |
| `context_axis` | R-context | `[{axis, poles[], mixed, dominant_pole, suggested_subtype, pole_confidence, pmids}]` — zonation/layer subtyping (§5 Step 2b); follow the engine's pole. |
| `state_axis` | R-state | `[{name, markers_on[], n_on, n_onpanel, score, pmids}]` — base-agnostic **PROGRAMS** (proliferating / interferon-stimulated / hypoxic …) with ≥80% of their on-panel markers ENRICHED here. **EVIDENCE ONLY — standard annotation NEVER appends these.** Used solely by the interactive phase (§8) to subdivide a base by state on user request. Not a base identity, not a split trigger. |
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
S5 (no state step) : disease/injury/malignant are ordinary class/subclass celltypes scored WITH normal in S3/S4 (R7) — no separate state check, no fallback. state_axis PROGRAMS are interactive-only (§8), never in standard.
S6 RESOLVE    : clear single REAL → label (R4) ; ELSE DESCEND-AND-SEE (R6): descend the node's KG children → 1 identity=label/merge, ≥2=split+recurse, no separation=FLOOR (R8)
S7 FLAGS      : subclass_resolved ; confidence ; reason_code ; disease ; split mechanism = descend (marker-axis)
```


### Step 0 — IS IT A CELL? → may emit **Background**
Background ONLY when ALL hold: `density.pct` bottom-quartile (or `low_frac` high) AND `specificity_peak`
null AND `lineage_markers_present.n == 0` AND `graph_evidence.lineage == null`. **`n ≥ 500` or a top-level
leaf (path has no ".") ⇒ never Background** → continue. `density==null` ⇒ use the identity conditions only.


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
- **`class_or_unknown`** → no REAL anchor → **DESCEND-AND-SEE (R6)** into the lineage/region children (disease/
  injury/malignant subtypes are among those children, scored WITH normal — R7, no separate state step). Only if
  the descent yields **no separation** (children's markers on the axis but cells don't split) is it a genuine
  floor → **R8** (class FLOOR if lineage clear, else Unknown).

`confidence`: **high** (anchor `excl_on≥3` or `excl_on/excl_n≥0.7`, clean negatives) · **medium**
(`excl_on` 1–2, or you stopped at a region) · **low** (single OWN marker, panel-limited).

### Step 2b — CONTEXT-AXIS subtyping (when the module declares `ca`; subtypes differ by POSITION, not intrinsic markers)
Some subtypes are NOT distinguished by their own cell-intrinsic markers — only by **position along an
anatomical axis** (liver lobule peri↔pericentral; kidney cortico-medullary; gut crypt↔villus). Their
discriminating signal is the dominant **neighboring** cell type's markers (a periportal LSEC carries
periportal *hepatocyte* markers `Cyp2f2/Hal`) — which §Step 1/R5 would call ambient for *lineage*, but here
**that context IS the subtype evidence**. Applies ONLY when (i) the module's slim-index declares a `ca` axis,
(ii) the leaf's lineage is confirmed by its OWN intrinsic markers (R5 unchanged — lineage never comes from
ambient), and (iii) that lineage is in the axis's `applies_to`. **Follow the engine's `context_axis` field
mechanically** (poles scored graded — weak zonal signal counts; a lean ≥1.15× sets `dominant_pole`):
- **`mixed==true`** (both poles ≥0.4× each) AND a large top-level parent (n≳1500, no ".") → **split**
  (`needs_subclustering`, reason `context_axis`) — labeling the whole parent by its lean buries the minority
  pole. `check` enforces this (`CONTEXT_AXIS_SPLIT`).
- **`dominant_pole` set** (the engine passed its POSITIVE-EVIDENCE gate: the winning pole has ≥1 of its OWN
  on-panel context markers ON **and** every pole of the axis is measurable on this panel) → label
  `suggested_subtype` directly (`reason_code="context_axis"`, `confidence=pole_confidence`). A weak-but-
  *positive* lean is still preferred over the bare class on this bounded same-lineage axis; do NOT re-judge
  from the subtype's off-panel intrinsic markers. **Use ONLY the engine's `context_axis` poles — do NOT
  substitute other neighbour markers or spatial-neighbour composition the axis does not declare** (that is
  how a leaf gets force-called to a pole whose own markers are off-panel). **NOTE — own vs neighbor:**
  context_markers are the dominant *neighbour's* genes read as ambient; when they are the cell's OWN zonal
  genes (hepatocyte `Glul`/`Cyp2f2`) the pole is strong, but when only a neighbour's genes are borrowed via
  KDE blur (a periportal LSEC via hepatocyte markers) keep `confidence=low` and SPLIT a large parent rather
  than blanket-labelling it.
- **`dominant_pole` NOT set** — either no pole is present, OR the engine withheld the pole because a pole's
  context markers are entirely off-panel (the axis cannot discriminate the zones) or the winner had no
  positive on-panel marker (*assignment by exclusion*) → **floor to the lineage class.** (kidney cortico-
  medullary: mTAL's own `Cldn10`/`Ptgs2` off-panel → 'medulla' cannot be positively confirmed → label
  *thick ascending limb cell*, not mTAL, even when the cortex markers are also low.) Context can NEVER
  change the lineage (a `Glul⁺` hepatocyte is a pericentral *hepatocyte*, never "pericentral LSEC").

### Step 2c — STATE-AXIS is NOT decided in standard annotation (interactive only)  [engine: `state_axis`, slim-index `sa`]
Some biology is a **cross-cutting cell PROGRAM**, not a cell type — proliferating, interferon-stimulated,
hypoxic, senescent, mesenchymal/EMT, stemness, estrogen-response, angiogenic, M2-polarization. A `CD8 T cell`
and a `luminal epithelial cell` can BOTH be *proliferating*; encoding that as a separate `proliferating CD8
T cell` Term would duplicate the same `MKI67` program across dozens of types. The module declares each once as
a base-agnostic **State** with its own program signature (`sa` in the slim-index).
- **STANDARD annotation does NOT append states.** The finalized label is the **base celltype ONLY** (`CD8 T
  cell`, never `CD8 T cell, proliferating`). The engine surfaces `state_axis` — a State is active when **≥80%
  of its on-panel program markers are ENRICHED** (R1: elevated vs the dataset, not mere detection) — purely as
  **EVIDENCE**. Do not add it to the label; do not let it change the celltype or trigger a descend.
- **State determination is an INTERACTIVE step (§8), only on the user's request** ("subdivide the tumor by
  state", "which of these are proliferating?"). Only THEN do you set `dec["states"]` from the `state_axis`
  evidence and finalize to `cellman_free_annotation`. The **lineage gate** (`sa[i].applies_to` — M2=myeloid,
  EMT=epithelial, estrogen-response=luminal; empty = global) and the **within-base gate** (≥2 of its markers
  enriched vs same-base siblings) are applied there, on the final base label. State order is canonical
  (alphabetical); EMT≡mesenchymal signatures are deduplicated; drop a state redundant with the base.
- (Disease/injury/metaplastic/malignant **IDENTITIES** are a different thing — ordinary class/subclass
  celltypes judged at the same level as normal, R7 — NOT program states, and NOT deferred.)

**STATE IS STILL A DURING-DETERMINATION SPLIT-GUARD (this part STAYS in standard).** A state whose program *resembles
  another lineage* — EMT (epithelial → mesenchymal program `VIM/CDH2/ZEB1`), EndMT, dediff, DAM
  (microglia → macrophage-like `APOE/TREM2`) — makes a single cell carry **two lineages' signals at once**, so
  the naive `mixed`/codominant test (R5/R6) could wrongly SPLIT one EMT⁺ epithelial cell into an
  epithelial-vs-mesenchymal "mixture." To stop that, a `sa` state carrying **`mimics`** (the lineage it
  resembles) **+ `anchor_markers`** (the origin identity it retains) is folded into `transition_candidates`
  (R5): when the leaf keeps the origin anchors ON **and** expresses the mimicked program (≥2 of `Qspec` ON), it
  is the **origin lineage in that state** (`epithelial cell, epithelial-mesenchymal transition`), **NOT a
  mixture — do NOT split.** So state, like context, participates in the celltype decision (as a split-guard),
  and is *additionally* appended after. Same-cell co-expression (anchors + program together) = state; disjoint
  subsets (origin-only vs mimicked-only cells) = real mixture → R6 split. `_mimicry_terms` unifies the old
  Term-level `A`/`MP` transition terms with these `sa`-derived ones.

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
Disease/injury/metaplastic/malignant identities are **ordinary class/subclass celltypes** with their own
discriminative markers — judged by the SAME R3/R4 test at the SAME level as normal types (R7), never as a
before-floor fallback, and chosen only when THEIR markers WIN. Never assume normal, but also never pick a
disease celltype because a normal one is merely absent. `--disease` is a retired hint (no longer filters or
surfaces a separate candidate set). Never invent a non-ontology label; `open_question_flag=true` if the chosen
term is in `oq`.

---

## 6. The split — MARKER-AXIS recluster (`helper.py subcluster` / `descend`)

Splitting is **not** whole-transcriptome Leiden. A mixture's distinguishing signal is often a handful of
low-copy markers (`Grik4`, `Fezf2`, `Drd1` vs `Drd2`) — across thousands of genes (PCA) it is diluted and
the mixture won't separate. So the flagged leaf is re-Leidened on the **marker axis**: cells represented by
ONLY the candidates' discriminating markers, making the cell-type difference dominant. (This un-dilutes
present-but-swamped markers; it does NOT rescue markers that are simply under-detected — that is a low-copy
floor needing a larger bin/bandwidth.)

### `descend` — the standard split (KG children, structural, no detection)
**Standard annotation splits ONLY by `descend`** (`helper.py descend --node <label>`): the engine takes
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
`helper.py subcluster --candidates '["…","…"]'` splits a leaf on an **explicitly named** candidate set —
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

> **Precondition:** the input X must be **normalized** (per-cell library-size normalized — KDE vectors
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
- the re-profiling/judgment after a split uses the SAME engine (`helper.py` is self-contained — the
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

> **MUST-DESCEND HARD BLOCK (un-bypassable).** Before anything else, `finalize` counts every leaf you floored
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

1. `helper.py init --cluster-col {cluster_col}` → profile the input's clusters in the **user-named column**
   (§3 INPUTS; `leiden` or whatever the user gave) — round 1 is coarse, the START not the answer.
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
4. **MODULE EXPANSION — the one pause before finalize (skip when nothing is Unknown).** A tissue does not
   stop at the module boundary: mouse skin carries panniculus carnosus, and the mouse/skin wing has no muscle
   term at all, so a real skin section leaves a 5,771-cell `Ttn`-top cluster at `lineage=None` / `top=None`
   → Unknown, purely because `Ttn` is absent from the skin index. When Unknown leaves remain, ask whether
   another wing explains them — but **never add one on your own**:

   ```bash
   {python} {HELPER} suggest-modules --out {out} --decisions {out}/_annotate_decisions.json --species {species}
   ```
   It reads what the Unknown leaves EXPRESS (from the profiles — no data is re-read), sweeps the other
   modules the KG serves, and ranks the ones whose terms could name them.

   > **SAME SPECIES, ALWAYS — expansion crosses the ORGAN boundary, never the SPECIES one.** A *mouse* skin
   > run that needs muscle takes **mouse** muscle, never human muscle: the markers are that organism's gene
   > symbols, so a cross-species wing would be nonsense on this panel. The species is taken from the module
   > the run was confirmed on — **omit `--species`**; passing one that contradicts the index is a hard error
   > (`SPECIES_MISMATCH`), and the merge refuses a species mix as well. Only the organ varies. A candidate term
   counts only when (a) it is **not already in your index**, (b) ≥1 of the markers carrying it is a gene your
   module could not even see, and (c) its OWN required/defining markers are ON (one is enough when the panel
   measures only one, and it is a DEFINING marker). Ranking is by **marker specificity KG-wide**, not by
   cluster size — a broad marker on a big cluster must not outrank a narrow one.

   **Then JUDGE the output — it is evidence, not an answer.** A marker can be shared between organs for real
   biological reasons: on that same skin run the top-ranked candidate was `thymus` via `FOXN1`, on a
   `Krt1`-high keratinocyte cluster — `Foxn1` is a hair-follicle gene too, and the proposal is simply wrong.
   The `muscle` candidate underneath it (`skeletal muscle cell` via `Ttn`, auc 0.88, claimed by ONE term in
   the whole KG) is the real one. Discard what your module knowledge rejects.

   **Ask the caller, then WAIT — do not finalize.** Emit, and stay alive for the answer:
   ```
   MODULE_EXPANSION_PROPOSED
   unknown: <n leaves, n cells>
   propose: [{organ, why, unknown_leaves, evidence_markers}]     # one or several modules
   ```
   The caller puts it to the user ("*…so I recommend also serving the muscle module; shall I proceed?*").
   - **approved** (possibly a subset) → `helper.py expand-modules --out {out} --add muscle[,…]` — the
     organ name only; the species is the run's own → it merges
     the wing (your primary module keeps its hierarchy and its marker degrees) and **re-profiles ONLY the
     Unknown leaves** — every decided leaf is FROZEN, so an expansion can add answers but never rewrite a
     call your module already earned. Re-judge just those leaves by R1–R8 (descend as usual), then finalize.
     A leaf the added wing still cannot name **stays Unknown** — that is a result, not a failure.
   - **declined** → finalize as is, with those leaves Unknown. Record it and never ask again.

   Propose only when it is worth a question: no Unknown, or only a scatter of tiny fragments with no
   candidate clearing the gate → skip this step entirely and finalize.

5. `helper.py check` → fix any flagged leaves → `finalize` → writes `obs['cellman_annotation']`. **This
   ends standard annotation.** Report the outcome to the caller, naming every module that was served.

### Interactive phase (after finalize — the only user touchpoint)
The caller presents the finalized outcome (table + explanation) and offers **① done** or **②
user-directed exploration**. This is free-form — the user asks in their own words, in **any** language, and
**every reply is still English** (top rule): read their language, answer in English. You translate the request
into targeted work and answer **honestly** (BIAS GUARD: the data decides, never manufacture a positive
to satisfy the user; never editorialize the *sample* as normal/diseased from the absence of a state label —
that is an evidence/coverage outcome, not a determination). Every refinement here is the **same** method —
`subcluster` / `descend` + re-judgment by R1–R8. Examples:
- **"Is there really no <type X>?"** → find the leaf carrying its signal → `subcluster --candidates
  '["<X>","<sibling>"]'` on that leaf → if a sub-leaf is REAL (R3) → label & report; if X's markers are not
  above background anywhere → say so plainly. *(clustering-based)*
- **"Re-annotate only the astrocytes whose positive markers are ON."** → `subcluster` those paths with the
  astrocyte-subtype candidates → the REAL sub-leaves get the finer label, the rest stay `astrocyte`. *(clustering-based)*
- **"Split D1 vs D2 MSN in the striatal cluster."** → `subcluster --candidates '["D1-MSN","D2-MSN"]'`. *(clustering-based)*
- **"Subdivide the tumor by state" / "which of these are proliferating / interferon-stimulated?"** → **the ONLY
  place states are applied.** Read `state_axis` (the engine's ≥80%-coverage program evidence) on the target
  leaves; for those where a program passes the lineage gate (`applies_to`) and the within-base gate (≥2 markers
  enriched vs same-base siblings), set `dec["states"]` = the active program names and `finalize --column
  cellman_free_annotation` → the base label gains `, <state>` (`CD8 T cell, proliferating`). Leaves the user did
  not ask about, and `cellman_annotation`, stay base-only. *(state = program overlay, never a new identity)*
- **"Gather the neurons and find the neuropil / fiber tract."** → **NOT SUPPORTED (§1b).** These are
  `tier='process'` terms, not celltypes. Report plainly that this version does not annotate tissue processes.
  Do **NOT** `subcluster` toward them — a process shares its positive markers with the soma that made the
  transcripts, so a marker axis cannot separate it, and any label you produce would be a force-fit.

Each refinement = a targeted `subcluster`/`descend` + re-judgment (§5) + `finalize`. A finalize AFTER
interaction writes to `cellman_free_annotation` (`--column cellman_free_annotation`), keeping
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
  "reason_code": "multi_identity | sibling_resolvable | pan_class_only | class_floor | split_no_gain | panel_limited | context_axis | ''",
  "provisional_subclass": "<best-guess subclass if class-final, else ''>",
  "key_markers": ["geneA","geneB"],
  "rationale": "<condensed S1–S7: which rules fired, markers ON, why>",
  "alternatives_considered": "<other terms: why each rejected (R3/R5/R6)>",
  "open_question_flag": false
}}
```
**Label hygiene (hard):** clean literature name only — no `(n=…)`, `"?"`, `"possible"`, reason text, or `+`.
Unknown→"Unknown", Background→"Background". Reasoning goes in `rationale`. (**Never emit a `tier='process'`
term — neuropil / fiber tract / … — as a celltype decision; this version does not annotate processes, §1b.**)

### 10.2 `{out}/expert_judgment_round{k}.csv`
`cluster_path,n_cells,decision,label,term_id,subclass_resolved,confidence,reason_code,key_markers,rationale`.

### 10.3 `{out}/celltype_markers.yaml` — write this yourself at finalize

The machine-readable marker panel downstream mapping consumes (`key_markers` in the CSV is only a 3–5 gene
summary of a decision — too thin to build a centroid from). It is a **tree**, not a flat list: every label you
emitted plus **its ancestors**, nested by `children`.

```yaml
endothelial cell:
  required: [Kdr]
  supporting: [Egfl7, Emcn, Vwf]
  children:
    arterial endothelial cell:
      defining: [Dkk2, Sema3g, Efnb2, Gja5]
      supporting: [Cxcl12]
    glomerular endothelial cell:
      defining: [Ehd3]
      supporting: [Kdr, Icam2, Emcn, Podxl]
epithelial cell:
  defining: [Epcam]
  children:
    collecting duct epithelial cell:        # ancestor only — you never called a cluster this
      children:
        collecting duct principal cell:
          defining: [Aqp2]
          required: [Avpr2]
          supporting: [Scnn1g, Aqp4]
```

**Why a tree.** A celltype's full marker set is *the union along the path from the root down to its node*. The
KG puts a marker on the parent when ALL of that parent's children share it, and on the children when only some
do — so the path carries what a flat file would have to duplicate. Written flat, `Epcam` alone would land in 18
of the 27 mouse-kidney entries and stop discriminating anything. The tree says it once, in the right place.

**Hard rules.**
1. **Dataset genes only.** Every gene must be `panel=true` in `ontology_gene_table` — actually measured in this
   dataset. Off-panel KG markers are dropped here (they are still fine to *reason* about under R3
   panel-blindness, but they cannot go in this file).
2. **Transcribe each node's KG tiers verbatim** (`DF`/`R`/`S`) — do not compare against the ancestors and do
   not subtract. A gene may legitimately sit on BOTH a parent and a child (`Prox1` is on
   `dentate gyrus glutamatergic neuron` and on `dentate gyrus granule cell`): the parent carries it as the
   lineage-level marker, the child as its own identity marker. Subtracting it empties the child of the very
   gene that defines it, and consumers union along the path anyway, so a repeat costs nothing. Within ONE node
   a gene appears in one tier only — strongest wins (DF > R > S).
3. Never list a gene that is a NEGATIVE marker of that node or of any of its ancestors.
4. **Omit empty tiers and empty nodes' bodies** — a node that contributes no on-panel marker is written with
   `children:` alone. Do not pad with `[]`, do not substitute markers you did not have.
5. Labels `Unknown` / `Background` get no node.
6. One node per label, not per cluster — many clusters can share a label (25 did for
   `distal convoluted tubule cell` in the mouse kidney run); they resolve to one term, so one node.
7. **Ancestors only, not siblings.** Include a parent because one of your labels descends from it — never add
   the parent's other children that this dataset did not show.

### 10.4 Final layout after `finalize`
Helper writes `{input}_annotated.h5ad` (input copy + `obs['cellman_annotation']`), packs every other artifact
into `annotation_artifacts.tar.gz`, and leaves `annotation_out/` with exactly: the annotated h5ad,
`expert_judgment_annotation.csv` (loose), `celltype_markers.yaml` (loose, §10.3), and the tarball. `finalize` also writes `annotation_rationale.md`
(English, one paragraph per leaf: the rule path that fired, the markers ON, competing terms rejected, and the
chosen term's grounding PMIDs from `GM`).

---

## 11. Self-check — run BEFORE writing each decision (helper validates the same)
0. **At `finalize` only:** `celltype_markers.yaml` (§10.3) exists and is a TREE. Verify by ENUMERATION, not by
   impression: take the set of labels you actually emitted, add every ancestor of each, and check that set is
   EXACTLY the set of node names in the file — no label missing (a called label with no node makes it invisible
   to the mapper), no sibling you never called. Then per node: tiers copied from the KG verbatim, every gene
   `panel=true`, no negative marker of that node or any ancestor, no `[]` padding, tiers not invented.
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
9. **R7:** disease/injury/malignant celltypes were scored WITH normal in `top`/`ranked` (same level, R3/R4) —
   not picked as a before-floor fallback, and not deferred. A `null top` was NOT read as "→ class" (descend).
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
- **Stopping at a class/region FLOOR without `descend`ing** (R4/R8 — THE failure this skill exists to prevent).
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
