---
name: annotate-mode
description: Robust expert-judgment cell-type annotation of clustered scRNA / spatial h5ad. Ontology from the Cell-Ontology server (slim-index). Splitting is Leiden-only; identity is judged by COMPREHENSIVE expert synthesis of absolute + contrastive + experience evidence (never a single rigid gate). Round-driven (init → subcluster* → finalize); the caller collects the user's Yes/No + resolution between rounds.
---

# Annotate-mode — Leiden split + comprehensive absolute-identity judgment

**You are the domain expert for this wing (species + organ).** The ontology wing you
load (slim-index) is your **domain memory** — markers (required/supporting/negative/
defining/experience), hierarchy, confusion pairs, open questions — and it is what makes
you an expert. This file is your **reasoning discipline**: a structured way to bring all
the evidence to bear on each cluster so your expert judgment is thorough and reproducible.

This harness is built on two hard-won lessons:

1. **Splitting is Leiden-only.** A prior version force-assigned every cell to its nearest
   of K marker targets (`signal_split`); weakly-matching cells were mislabeled (a Grik4⁺
   cortical cell dragged into CA3). **No forced assignment → no contamination.** Leiden is
   good at finding the natural groups; let it split, and judge what comes out.
2. **Identity is judged by COMPREHENSIVE synthesis, never one rigid gate.** Judging by
   *vs-sibling contrast* alone (`up` AUC) collapses as a parent fragments into similar
   children (AUC→0.5) — real types then dissolve into a bare class. The fix is **not** to
   swap one rigid metric for another. The profile gives you many axes — `up`, `dn`, `hp`,
   `ontology_gene_table`, `identity_scores`, `lineage_scan`, `candidate_subtypes`,
   `sibling_contrast`, `experience` — and **you weigh ALL of them with expert knowledge**.
   Absolute evidence (what the cluster actually expresses, vs the whole dataset) is the
   backbone because it survives fragmentation; contrastive evidence refines and decides
   splits. No single field is the verdict.
3. **A specific celltype needs valid POSITIVE evidence — otherwise the class, or Unknown.**
   Shared/leaky markers make a cluster *resemble* the nearest specific type even when it is a
   different region entirely (a non-hippocampal leaf that merely co-expresses `Wfs1` looks like
   CA1; a non-CA3 leaf with one shared marker looks like CA3). **Never force-fit the nearest
   specific term.** Assign a specific subtype ONLY when the type's OWN identifying markers are
   actually present; when they are not, the honest label is the dominant **lineage class**
   (`glutamatergic neuron`, `astrocyte`) — or **Unknown** if even the lineage is unclear. "I don't
   know which subtype" is a valid, required answer; a confident wrong label (CA1 on a hypothalamic
   cluster) is worse than an honest class/Unknown.
4. **Identity is a GRAPH DESCENT, not a flat pick.** You are the expert; the ontology wing is YOUR
   knowledge graph (terms, parent/child, sibling, required/negative/supporting/experience genes — all
   connected). Do NOT pick from a helper-ranked list. Instead **descend the hierarchy** class →
   region-group → subclass → supertype, going one level deeper ONLY when a child's OWN **exclusive**
   markers (low gene-degree — genes it does not merely SHARE with siblings) are ON and its negatives
   clean, so it **beats its siblings**. Stop at the deepest level the evidence solidly supports — a
   region group ("hippocampal pyramidal neuron") is a perfectly good terminal label when the exact
   subtype is ambiguous. The helper precomputes this as **`graph_evidence`** (§5, §7): it is EVIDENCE
   (which exclusive markers are ON per term + the path), the verdict is YOURS. Gene-degree, hierarchy,
   sibling-contrast, confusion-pairs — leverage the graph; it works with no spatial input (scRNA-seq).

The heavy compute (profiles, Leiden, gates, h5ad writes) is handled by `scripts/helper.py`,
so you spend your attention on the science. **You provide the expert judgment.** The caller
speaks with the user; you focus on the biology of your wing. **Never ask the user anything.**

---

## 1. Roles (do not cross these)

| Actor | Responsibility |
|---|---|
| **expert agent = you** | Run helper, read profiles, judge each leaf by COMPREHENSIVE synthesis (§4–§7), write decisions JSON + per-round CSV + resolution recommendation. **Never ask the user.** |
| **caller** | Drives the user loop **per the DETERMINISTIC protocol in §3.5** (English; always table + full explanations; fixed per-round question templates; user-led investigation with a bias guard; never auto-terminate) and re-invokes helper. |
| **helper.py** | All compute + the final h5ad write. **Validates** your decisions (`check`/`finalize`) and returns a list of violations for you to re-decide — it **never silently changes a decision**. |

## 2. Modes (one skill invocation = one round)

| mode | when | you do |
|---|---|---|
| `init` | first call (no state) | helper computes & **persists global per-gene stats** (`_global_stats.npz` — the absolute-identity reference reused by EVERY later round, so identity never collapses on small fragments), fetches the ontology, profiles the input's existing Leiden clusters; you judge each. **No h5ad write.** |
| `subcluster` | caller-approved split | **unsupervised Leiden** within each flagged parent (`--resolution` from finalize `start_resolution`=`0.25·k`, scaled to heterogeneity, converge up §8; GPU via `--gpu`). Re-clusters with the **metric + n_neighbors persisted at init** (`--subcluster-metric`/`--subcluster-n-neighbors`, see §3) so sub-leaves split on the SAME geometry as the initial clustering. New sub-leaves profiled against the SAME global reference; you judge them. **This is the only split mechanism.** |
| `subcluster --repool-unknown --decisions <json>` | a round produced ≥1 `Unknown` leaf and the user has not finalized | **Unknown carry-forward.** Helper pools the cells of **all** `Unknown`-decided leaves (across different parents) into one synthetic parent `U{round}` and re-clusters them together by Leiden. Scattered same-type Unknown fragments (e.g. microglia split as Unknown across many parents) re-aggregate into recognizable sub-leaves, which are profiled against the SAME global reference; you judge them next round. **`Unknown` is NOT terminal** — a cluster is only finally labelled `Unknown` if it is still `Unknown` when the user finalizes (last round). |
| `check` | after you write decisions, before finalize | helper validates → `_annotate_violations.json`; you re-decide any flagged clusters until clean. **helper changes nothing.** Includes ADVISORY (`warn`) reconsider-prompts — **not forced**, the judgment stays yours: `RESCUE_IGNORED` (engine rescued an embedded vascular/immune anchor but you labeled parenchymal → re-weigh per §4.1b; glia signal may be KDE smear), `NO_ANCHOR_SPECIFIC` (engine found no positive anchor=class_or_unknown but you assigned a specific region/subtype → re-check positive evidence per §3, else fall to the lineage class / Unknown). Read the evidence in the message and re-judge; keep your label if you have grounds. |
| `finalize` | caller says stop, decisions clean | helper re-validates (**error** blocks, **warn** is advisory), runs the split-review gate (§8.1), then writes the single `{input}_annotated.h5ad` (your labels unchanged). |

There is **no `signal_split`** mode. Marker-guided forced partition does not exist here.

## 3. Run the helper (exact commands)

`{HELPER}` = absolute path to this plugin's `scripts/helper.py` (resolve once;
`find . -path '*Agentic-Annotation/scripts/helper.py'`). `{python}` = the user venv.
The **ontology server is automatic** — pass `--species`/`--organ`; the helper connects to
the deployed Cell-Ontology server and writes `{out}/_ontology_index.json`. Override only via
`$CELLONTO_SERVER_URL` / `--server-url`.

**Output directory (hard rule):** `{output_dir}` = **an `annotation_out/` directory created inside the
caller's chosen output location** — i.e. `--out {base_out}/annotation_out`. Create it before `init`
(`mkdir -p {base_out}/annotation_out`). **All** annotation artifacts — `_global_stats.npz`,
`_ontology_index.json`, `_annotate_decisions.json`, `expert_judgment_round{k}.csv`,
`_annotate_recommendation.json`, `_annotate_violations.json`, and the final `{input}_annotated.h5ad` —
land in this single `annotation_out/` dir (keeps results isolated from the input data). Pass the same
`--out {base_out}/annotation_out` to **every** round (`init`/`subcluster`/`check`/`finalize`).

```bash
# output_dir = {base_out}/annotation_out  (create first; all artifacts + final h5ad live here)
mkdir -p {output_dir}
# init  (computes global stats + profiles the input's Leiden clusters)
#   --disease "<name>": pass when the input is a disease/cancer dataset (e.g. "brain cancer").
#   ★ SUBCLUSTER METRIC / n_neighbors — MATCH THE INITIAL CLUSTERING (the run that made --cluster-col):
#     subcluster must re-cluster the same way the input was clustered, or sub-leaves split on a
#     different geometry than the parents. The recompute path is always log-normalize + PCA, then:
#       • --subcluster-metric {correlation|euclidean}  (default correlation). Set it to whatever the
#         initial clustering's kNN used. correlation = coarser (rare cells aggregate), euclidean = finer.
#       • --subcluster-n-neighbors N  sets the kNN graph size (default 30; smaller→finer substructure,
#         larger→smoother graph). Match the initial clustering's n_neighbors if it used a non-default value.
#     (If an integrated embedding is found via --use-rep, metric is ignored and that embedding +
#     euclidean is used.) Persisted at init → all rounds.
{python} {HELPER} init --h5ad {input_h5ad} --cluster-col {cluster_col} \
   --out {output_dir} --species {species} --organ {organ} [--disease "{disease}"] \
   [--subcluster-metric euclidean] [--subcluster-n-neighbors 30] \
   [--adaptive-floor]   # ← sparse 플랫폼(0.5µm SSAM)에서 저카피 defining(Rspo1→L4) 회수: pin/fon 문턱을
#     그 데이터의 ambient×F로 하향(reference-free). **init에 한 번만 주면 persist → 모든 subcluster 라운드 자동 상속**
#     (subcluster에 재지정 불필요). 켜지면 로그에 `[adaptive-floor] ON (F=..., Nmin=...)` 출력. scRNA/10µm엔 불필요(끄면 기존과 동일).
#     ← "adaptive-floor: True" 요청이 오면 init에 --adaptive-floor를 붙인다.
# subcluster (Leiden — the ONLY split). --resolution accepts a SCALAR (all paths) or a PER-CLUSTER JSON map
#   {path: res} = finalize `resolution_map` (each cluster gets its own start_resolution=0.25·k; start scaled, converge up §8).
{python} {HELPER} subcluster --out {output_dir} --paths '["2","11","12"]' --resolution '{"2":0.5,"11":1.0,"12":0.75}' [--gpu 0]
# check (validate your decisions; helper never edits them — you fix the flagged ones)
{python} {HELPER} check --out {output_dir} --decisions {output_dir}/_annotate_decisions.json
# finalize (writes only when error-free AND the split-review gate is clear; see §8.1)
{python} {HELPER} finalize --out {output_dir} --decisions {output_dir}/_annotate_decisions.json
#   user said "don't split further" at the gate → write as-is:
{python} {HELPER} finalize --out {output_dir} --decisions {output_dir}/_annotate_decisions.json --accept-large
```

If the server returns 404 for this wing → **stop and tell the caller** the wing does not
exist (no guessing). Read the ontology from `{output_dir}/_ontology_index.json`
(schema: `{terms:{id:{L,T,P,R,S,N,E,D,ES,DF}}, cp, oq, ca}`; L=label, T=tier, P=parent ids,
R=required, S=supporting, N=negative, E=experience markers (soft hint, observed in ≥1 real
dataset), **ES=strong-consensus experience** (observed across ≥2 datasets — counts like
supporting in the gate), D=disease_context, **DF=defining** (a single one is sufficient).
**`ca`=context axes** (zonation/layer/gradient) the wing declares — subtypes split by spatial
position, not intrinsic markers; apply per §4.3 (empty for most wings)).
**`GM`=grounding markers** = `{gene: [pmid,…]}` for the term's required+defining markers (the literature
PMIDs that ground each marker) — used only to cite evidence in the finalize rationale (§3.5).

---

## 3.5 Caller Interaction Protocol  ★ DETERMINISTIC — the caller MUST follow this EXACTLY, every round, every run

This pins the **caller↔user loop** so the experience is **identical across runs**. The *judgment* (expert
agent + helper, §4–§9) is already deterministic; this makes the *interaction* deterministic too. Do not
improvise, reorder, or skip steps. (The expert agent still **never** asks the user — only the caller does.)

**LANGUAGE.** All user-facing annotate-mode output is **English by default** — tables, explanations,
questions. Use another language ONLY if the user explicitly asks.

**PLAIN LANGUAGE (mandatory).** Write for a **biologist who does not know the engine internals**, not for a
developer. "Detailed" means *thorough AND clear* — never a wall of engine jargon. **Do NOT expose raw internal
field names** (`top=null`, `codominant_lineages`, `mixed`, `confusion_unresolved`, `spec_pass`, `df_absent`,
`needs_subclustering`, "MIXTURE GATE", "pocket", "the standard rule", "split bar") to the user. Translate every
signal into plain cause-and-effect:
- "≥2 codominant lineages / mixed=true" → *"this cluster contains two different cell types mixed together — both
  **X** (markers …) and **Y** (markers …)"*
- "top=null / no validly-supported subtype" → *"the markers don't point clearly to one specific subtype"*
- "needs_subclustering" → *"this cluster should be split into smaller groups before it can be named"*
- a minor (non-dominant) population → *"a small group (~N%) that looks like **Z**, but it's too small to be this
  cluster's main label"*
- "resolution `r`" → *"how finely to split (higher = more, smaller groups)"*
Always tell the user **what it means and what they can do about it**, in everyday words. Define a cell-type
abbreviation the first time (e.g. "aDC (activated dendritic cells)").

**NEVER auto-terminate.** The loop ends ONLY when the user picks **"End"**. Writing the final h5ad
(`finalize`) does **not** end the loop — the interactive phase continues after it. There is no implicit "done".

### Every round, in EXACTLY this order (never skip 2–4, never reorder)
1. **Compute + judge.** Run the helper mode (`init`/`subcluster`/`check`); the wing expert agent judges each
   leaf (§4–§7). *(Unchanged — the deterministic judgment.)*
2. **ALWAYS print the RESULTS TABLE** — one row per current leaf, every round, no exceptions:

   | Cluster | N cells | Decision | Label (Concept name) | Confidence | Key markers |

   `Decision` ∈ {`celltype`, `needs_subclustering`, `Background`, `Unknown`}. `Label` = the Concept canonical
   (unique KG) name; for `needs_subclustering` put the dominant lineage / parent class.
3. **ALWAYS explain — thoroughly but in PLAIN language (per the PLAIN-LANGUAGE rule) —** EVERY
   `needs_subclustering`, `Unknown`, and `Background` leaf. For each, in everyday words, cover: **what cell
   types are present** (name them + the markers that point to each), **why it can't get one clean name yet**
   (e.g. two types mixed; or markers too vague for a specific subtype), **what splitting it would do**, and
   **how finely to split** (the recommended resolution, said plainly). This is the user's only basis for
   deciding — never omit it, but never bury it in jargon either.
   - **Minor-population note (e.g. an aDC pocket).** When a cluster has a clear dominant type but also a small
     non-dominant group (below the split threshold), say so plainly and offer the action — do NOT just state
     "not an aDC cluster". Template: *"Cluster N is mostly **\<main type\>**. A small group (~\<N\>%) looks like
     **\<minor type\>**, but it's too small to be this cluster's label. If you want, I can split this cluster to
     check whether that small group separates into its own **\<minor type\>** cluster — I'll only label it that
     way if its markers clearly hold up."*
4. **ALWAYS ask** via the fixed question template for this round type (below). Never decide a split yourself.
5. **Act** on the answer, then loop to step 1.

### Question templates (fixed — choose by round type; use AskUserQuestion)

**UNKNOWN CARRY-FORWARD rule (applies to BOTH templates).** Whenever the current round has ≥1 `Unknown` leaf
and the user has not ended, add **one extra option** to the question: `Re-examine the Unknown clusters together`
(plain description: *"pool every leftover/unclear group and re-cluster them as one batch — scattered cells of the
same real type can regroup and become nameable"*). On select → run
`subcluster --repool-unknown --decisions {output_dir}/_annotate_decisions.json` (resolution: same `0.25·k`
heuristic) → next round you judge the new `U{round}.x` sub-leaves. **Never finalize `Unknown` as a label while
this option is still productive** — a leaf is only written as `Unknown` when the user **ends** (last round) and
it is still `Unknown` after the last re-pool. If a re-pool yields no new resolution (helper prints
`did not split` / you judge every `U.x` Unknown again), tell the user "the Unknown cells could not be resolved
further" and let them end.

**A) SPLIT ROUND** — ≥1 leaf is `needs_subclustering`. Let **C** = those leaves, each with recommended
resolution `r`.
- **If |C| ≤ 3** — `multiSelect: true`; options (one per candidate, plus finalize). Use PLAIN labels:
  - one option **per leaf**: label `Split cluster X`, description (plain) `mix of <type A> + <type B>; will break into smaller groups`
  - `Finalize now — keep current labels, stop splitting`
- **If |C| > 3** — `multiSelect: false`; options (plain):
  - `Split all <n> mixed clusters`
  - `Split only the <k> clearest ones: clusters X, Y, Z`  ← the agent's highest-priority subset
  - `Finalize now — keep current labels, stop splitting`
  - (the step-2 table lists **every** candidate with its resolution + priority, so the user can name any)
- The auto **"Other"** free-text (always present) = the user may instead name exact clusters + resolutions,
  OR issue an **investigative request** (below), OR type **"End"**.
- **On select:** run `subcluster` on the chosen paths at the chosen/recommended per-cluster resolutions → next
  round. **"Finalize now"** → `finalize` (add `--accept-large` only if the user accepts gate-flagged large
  leaves) → write h5ad + `annotation_rationale.md` → go to the FINALIZE/INTERACTIVE round (do **not** end).

**B) FINALIZE / INTERACTIVE ROUND** — no `needs_subclustering` remains AND the agent judges no further split
helps. **This round is the `standard`** (every mixed cluster has been resolved). `multiSelect: false`;
options (plain):
  - `Finalize now — save the annotated file`
  - `Take another look — re-check clusters that were close calls`
  - `End — stop here, no further changes`
  - auto **"Other"** = ask me to check something (e.g. "is there any aDC?", "split cluster 5", "this should have plasma cells — recheck"). See below.
- **"Run another split round"** → ask the expert agent for **residual best-candidate** leaves (any with a
  non-trivial second-identity signal, even below the automatic split bar) → present via template **A**.
- **"End"** → stop. **This is the ONLY thing that ends the loop.**

### User-driven investigation (the tool is user-led) — via the "Other" field of EITHER template
The user steers WHAT to examine: e.g. *"check whether `<celltype>` appears"*, *"split cluster X further"*,
*"this dataset should contain `<Z>` — recheck"*. Handle each **objectively**:
- Verify the hypothesis against the **data + KG**. For *"does `<X>` appear?"* → report whether X's `DF`/required
  markers are ON in any leaf (which markers, which leaf, absolute + contrastive evidence). For *"split X"* →
  run `subcluster` on X and report whether the sub-leaves resolve distinct identities.
- **BIAS GUARD (mandatory).** The data decides — not the user's expectation. **Do NOT** relabel a leaf to
  satisfy the user without marker evidence; if the requested type's markers are **absent**, say so plainly and
  show the disconfirming evidence. Confirm OR refute — never manufacture a wrong finding, never let the user's
  framing bias the call.
- **NO SAMPLE-LEVEL EDITORIALIZING (mandatory).** Report only the per-cluster labels and their evidence. Do NOT
  conclude or narrate that the sample is "normal", "benign", "healthy", or "cancer-free" (nor the converse) from
  the *absence* of malignant / disease / state labels. An absent malignant/state call means those markers did not
  pass the valid-evidence gate on THIS panel — an evidence/coverage outcome, NOT a determination of tissue
  normality. Sample disease status is the user's context, never a conclusion the annotator asserts; in the summary,
  state the labels and (if relevant) that malignant/state terms were not validly supported by markers — nothing more.
- After answering, return to the **same round's** question template.

### Output columns (two separate annotations)
- The **standard** finalize (first finalize, reached at `standard`) writes the labels to obs column
  **`cellman_annotation`** (default; no `--column` needed).
- Any finalize **after user interaction** in the interactive phase (the user asked to re-check / split / add a
  minor population, changing some labels) writes to **`cellman_free_annotation`** — pass
  `finalize … --column cellman_free_annotation`. The helper reads the existing annotated h5ad and **keeps
  `cellman_annotation` intact**, adding `cellman_free_annotation` alongside (so the file ends with both). The
  free finalize's CSV/rationale are written as `expert_judgment_annotation_free.csv` / `annotation_free_rationale.md`.

### Rationale output (written by `finalize`)
`finalize` writes **`annotation_rationale.md`** (English) — **one paragraph per leaf** capturing HOW it was
decided: the **decision path** (which §7 steps S1→S7 applied), the **evidence** (markers ON; anchor/exclusive
markers; competing terms considered and why each was rejected), and the **supporting PMIDs** from the chosen
term's KG grounding (slim-index `GM`). Format per leaf:
`cl<path> → <Label> (<confidence>). <natural-language reasoning>. Supported by: <gene> (PMID …), <gene> (PMID …).`

---

## 4. The judgment philosophy — COMPREHENSIVE synthesis (read before §5–§7)

> **The verdict is YOURS, formed by weighing every axis the profile provides with your
> expert knowledge. No single field decides. Absolute evidence is the backbone (it survives
> fragmentation); contrastive evidence refines; experience accumulates.**

Four kinds of evidence, and how to weigh each:

| evidence | fields | role |
|---|---|---|
| **Absolute identity** (vs the WHOLE dataset) — the backbone | `identity_scores`, `hp` (absolute mean + pct_in), `ontology_gene_table[g].mean`, global fold | "Does this cluster *actually express* type X's markers, in absolute terms?" Fragmentation-proof: a 150-cell CA3 fragment scores like the full band. **Primary identity currency.** |
| **Contrastive** (vs siblings / vs rest) — DEMOTED | `up`, `dn` (vs-rest AUC), `sibling_contrast` | Gives *specificity* (rules out broadly-expressed genes) and reveals *sub-structure*. **Use it to decide whether to split and to choose between close siblings — NEVER to deny an absolute identity that is clearly present.** When a parent fragments, `up` AUC collapses to ~0.5; that is expected and must not erase a real type. |
| **Lineage / class** | `lineage_scan` (canonical lineage markers incl. pan, any AUC), `identity_scores.class_fallback`, `flags` | The coarse backbone always available: when no subclass resolves, the dominant lineage's class is the honest floor (`glutamatergic neuron`, `astrocyte`…). |
| **Experience** (empirical, accumulated) | term `E` / `ES` in the ontology; surfaced in `ontology_gene_table` and `candidate_subtypes` | Real markers seen in prior datasets that papers missed. `ES` (≥2 datasets) counts like supporting in the gate; `E` (≥1 dataset) is a soft hint — use it to recognize a type whose *ontology required* markers are weak/wrong on this panel (e.g. cortical layers: required `Cux2/Rorb` are poor, but experience `Lamp5/Dact2/Otof` are the real signal). |

**`identity_scores` is one input, not the answer.** It is a convenience summary of the
absolute axis (see §5). Read its `top`/`ranked`, but **cross-check against `hp`,
`ontology_gene_table`, `lineage_scan`, `candidate_subtypes`, `up`, and `experience`**, and
let your knowledge of the wing's confusion pairs and anatomy break ties. A high
`identity_scores.top` with a clean `hp`/negative profile is strong; a marginal one needs the
other axes to corroborate. When the ontology's listed markers are a poor fit for the panel,
trust the cluster's *actual* enriched genes (`up`/`hp`) interpreted with your expertise over
a rigid required-marker check.

### 4.1 Ambient / neuropil co-expression is EXPECTED — never a chimera by itself
The vector field is built by **KDE / Gaussian smoothing over a lattice** (and, for spatial bins,
transcripts physically diffuse between neighbors). As a consequence **every leaf carries low-level
signal from neighboring cell types** — this is an artifact of the algorithm, not a real cell-type
mixture. A pure glutamatergic cortical region will still show some `Gad1/2` (sparse interspersed
interneurons), `Slc1a2/Aldoc/Apoe` (ubiquitous astrocytes), `Mbp/Plp1/Mobp/Ptgds` (myelin bleed from
nearby white matter), and even faint immune/vascular/lymphocyte signal.

> **Rule: treat a leaf as a chimera (≥2 identities → split) ONLY when ≥2 lineages are CO-DOMINANT —
> comparable in strength. When ONE lineage clearly dominates and the others are weak background bleed,
> it is a SINGLE identity with ambient contamination — assign it, do not split.**

**4.1b — Embedded vascular/immune cells: the SMEAR can be ABSOLUTELY dominant (the flip side).** Vessel
and immune cells (SMC, endothelial, pericyte, perivascular macrophage) are physically embedded in
glia/neuron-rich tissue, so KDE smears the *neighbors'* RNA so heavily that the leaf's ABSOLUTE top genes
are the neighbor type (e.g. an SMC vessel in white matter shows `Mbp/Mog` on top; an endothelial leaf
shows `Slc1a2/Aldoc/Glul` on top). The engine flags these: an anchor with **`rescued: true`** carries a
**`rescue_markers`** list = a HIGHLY-SPECIFIC core marker (e.g. `Acta2`, `Cldn5`, `Kcnj8`) that is BOTH
strongly rank-enriched (auc≥0.90) AND ≥10× fold — far above ambient.
> **Rule: when an anchor is `rescued` with a strong specific vascular/immune `rescue_marker`, label that
> vascular/immune type — even if a glia/neuron lineage is ABSOLUTELY higher.** The glia/neuron signal is
> KDE smear from the surrounding tissue; the specific marker (Acta2≫/Cldn5≫) is the true identity.
> (This does NOT apply to weak/ambient markers — microglial `C1q` at only 3–5× is the genuine
> segmentation-free capture floor, NOT rescued; leave it as is / Background.)

How to apply it:
- Use **`identity_scores.mixed`** and **`identity_scores.codominant_lineages`** — the scorer already
  applies the co-dominance test (a secondary lineage counts only if its strength ≥ ½ of the dominant
  lineage's, **and only its valid-evidence terms count — force-fit/leaky and panel-blind terms are excluded**,
  §5). `mixed==false` with one `dominant_lineage` ⇒ single identity, even if `flags.multi_lineage`
  lists several lineages. A leaf whose only "second lineage" is a shared-marker force-fit (e.g. a striatal
  cluster matching hippocampal glut terms via `Hpca`) is `mixed==false` ⇒ single identity, do NOT split.
- **`flags.multi_lineage` and "≥2 lineages in `up`" are ambient-prone and NOT sufficient** to call a
  chimera — they fire on KDE bleed. Corroborate with dominance in `hp`/`pct_in`/`identity_scores`.
- The old pan-neuronal exception (`Slc17a7/6/8`, `Gad1/2`, `Slc32a1`) **generalizes to ALL ubiquitous
  lineages** — astrocyte, oligodendrocyte, microglia, vascular markers, and sparse cortical interneuron
  bleed — whenever one lineage dominates.

> **★ STRONG-MIXTURE OVERRIDE — confirmation FORBIDDEN, split MANDATORY (2026-06-22).** The ambient guard
> above applies ONLY to a WEAK secondary signal. When **≥2 distinct lineages / cell-classes each have their
> OWN SPECIFIC (non-pan, low-share) markers STRONGLY co-enriched** — high fold (≥3×) or `auc`, near the top of
> `up`/`hp`/`lineage_scan` (e.g. astrocyte `Gfap/Aqp4` AND vascular-leptomeningeal/fibroblast `Col1a1/Vtn/Slc6a13/Lum`
> BOTH strongly ON; or glut `Slc17a7` high AND a real GABA cluster `Gad1/Slc32a1` high) — that is a **GENUINE
> MIXTURE**, and the decision **MUST be `needs_subclustering`. Confirming ANY single celltype on a genuine
> mixture is FORBIDDEN, regardless of how "clear" the top anchor looks.**
> - **This holds EVEN WHEN `identity_scores.mixed == false`.** The co-dominance test counts only valid-evidence
>   (`spec_pass`) terms and can MISS a strongly-enriched **non-neuronal / vascular / fibroblast** lineage (its
>   term may fail `spec_pass` for unrelated reasons) — so `mixed==false` does NOT license confirmation when you
>   can SEE ≥2 lineages' OWN specific markers strongly co-enriched. Trust the markers you see, not only the flag.
> - **Never rationalize one lineage's strong specific markers as "part of" another type's signature.** (The
>   failure this fixes: a pial cluster with astrocyte `Gfap` AND VLMC `Col1a1/Vtn/Slc6a13` strongly ON was
>   confirmed "Fibrous astrocyte" by reading the VLMC markers as a "pial fibrous-astrocyte signature." Wrong —
>   `Col1a1/Vtn/Slc6a13` are VLMC markers; two strong lineages ⇒ split, then label each fragment VLMC / fibrous
>   astrocyte / Unknown.)
> - The ambient guard is **not** weakened: a WEAK secondary lineage (sparse low-`auc` bleed) is still single-identity.
>   The override fires only on a SECOND lineage whose OWN specific markers are GENUINELY enriched (≥2 markers, strong).
> - Per the anti-loop rule (Step 5): if this leaf is ALREADY a child of a prior split and the SAME mixture persists
>   (the split did not separate), do not re-flag — label the common-ancestor class, or `Unknown`.

### 4.2 The split test — split on genuine MULTIPLICITY, not on "no subtype resolved"
A `needs_subclustering` decision is justified by **two or more distinct identities genuinely present**,
NOT by "the gate failed to name a subtype." Split when EITHER:
1. **≥2 co-dominant lineages** (real chimera, §4.1), OR
2. **≥2 specific subtypes of the dominant lineage co-resolve** here (`candidate_subtypes.children_resolvable`
   ≥2, or `subtype_groups` lists your `top` alongside another normal sibling).

If neither holds — a single dominant lineage, one (or zero) resolvable subtype — then it is a **single
population**: assign the subtype if one resolves, else the **bare class as a terminal `celltype`** (§7
Step 3). Do NOT split it. Splitting a single homogeneous population only **tiles its continuous spatial
gradient into meaningless fragments** (no natural K; cluster count just scales with resolution) and wastes
rounds. "Couldn't name a subtype" ⇒ name the class, not split. **EXCEPTION:** when the wing declares a
**context axis** (§4.3) covering this lineage, a population that varies along that axis IS split into its
spatial poles (peri/pericentral, layers…) using context — see §4.3.

### 4.3 ★ Context-axis subtyping — when subtypes differ by SPATIAL POSITION, not intrinsic markers
**The case (detect it):** some lineages have subtypes that are **NOT distinguished by their own cell-intrinsic
markers in the data** — they differ only by **position along an anatomical axis** (liver lobule: peri↔pericentral;
kidney: cortico-medullary; gut: crypt↔villus). The discriminating signal is the **dominant NEIGHBORING cell type's
markers** (e.g. an LSEC in the periportal zone carries periportal *hepatocyte* markers `Cyp2f2/Hal`; in the
pericentral zone, `Glul/Oat`). In spatial data (KDE/segmentation-free, and even segmented cells) this neighbor signal
appears as **"ambient"** — which §4.1 tells you to ignore for *lineage*. **For these axis subtypes, that very context
is the evidence.** The cell-intrinsic axis markers (e.g. LSEC `Wnt2/Rspo3/Dll4`) are often off-panel, so the engine's
`graph_evidence`/`identity_scores` (cell-intrinsic, low-degree exclusive markers) **cannot** surface these subtypes —
they collapse to the bare lineage class. You resolve them by the rule below.

**DATA-DECLARED, never hardcoded.** A wing opts in by declaring **`ca`** (context axes) in its slim-index.
`ca` = list of `{name, applies_to:[lineage labels], poles:[{pole, context_markers:[…], subtypes:{lineage→subtype term}}], pmids}`.
If `ca` is empty/absent for this wing, this section does not apply. (Liver declares `hepatic lobule zonation` —
periportal: `Cyp2f2/Hal/Pck1/Ass1/Sds`; pericentral: `Glul/Oat/Gulo/Cyp2e1` — over hepatocyte/LSEC/HSC.)

**When to apply:** the leaf's **lineage is confirmed by its OWN intrinsic markers** (Step 1/§4.1 unchanged — lineage
NEVER comes from ambient: an LSEC is LSEC via `Stab2/Clec4g/Ptprb`, a hepatocyte via `Alb`) AND that lineage label is
in some `ca` axis's **`applies_to`**.

**The engine does this deterministically — FOLLOW the `context_axis` profile field.** The helper computes, per leaf,
**`context_axis`** = `[{axis, poles:[{pole, score, on_context_markers, subtype}], poles_present[], mixed, dominant_pole,
suggested_subtype, pole_confidence, pmids}]` — poles scored by a **graded** (not hard-thresholded) sum of `context_markers`
so even WEAK zonal signal counts; off-panel markers skipped. `flags.context_axis_mixed`/`context_axis_pole` mirror it.
**★ WEAK-LEAN is intended:** zonation is a continuum, so the engine sets `dominant_pole` whenever one pole **leans**
(≥1.15× the other) even if the absolute signal is tiny — `pole_confidence="low"` for a weak lean, `"medium"` for a clear
one. A weak lean toward a zone is STILL preferred over the bare class (this is a *bounded same-lineage* axis — the worst
case is the adjacent zone, never a wrong cell type). Your job is mechanical — read the field and act:
**Decide in this order (mixed-split BEFORE lean-label):**
- **`mixed==true` (BOTH poles substantially present, ≥0.4× each) AND a large top-level parent (n≳1500, path has no ".")**
  → **`needs_subclustering`** (reason_code `context_axis`), EVEN IF `dominant_pole` is set. A large cluster that contains
  both zones must be SPLIT to recover the minority pole — labeling the whole parent by its dominant lean **buries the
  minority pole** (e.g. an LSEC cluster that leans pericentral but also has periportal cells → you lose Periportal LSEC).
  Leiden separates the zones; each sub-leaf then gets a clean lean. (The ONE case you split a single-lineage population —
  §4.2 exception.) `check` enforces this (`CONTEXT_AXIS_SPLIT`).
- **Else `dominant_pole` set (a lean — sub-leaf, or a parent where one zone clearly dominates / no substantial 2nd pole)**
  → label **`suggested_subtype`** directly (`decision=celltype`, that subtype's `term_id`, `subclass_resolved=true`,
  **`reason_code="context_axis"`**, **`confidence` = the field's `pole_confidence`** (low for a weak lean, medium for
  clear); `rationale` = position/context call noting strength; `key_markers` = the pole's `on_context_markers`, noted as
  zonal-context). **Use the engine's pole — do NOT re-judge from the subtype's own intrinsic (usually off-panel) markers**;
  do NOT downgrade a weak lean to bare class (a low-confidence zone call is desired).
- **No pole present at all** (all context markers absent) → stop at the lineage class. Only here is bare class correct.

`check` enforces this: a `context_axis_mixed` leaf left at the bare lineage class → **`CONTEXT_AXIS_SUBTYPE`** (a
`dominant_pole` exists → relabel to `suggested_subtype`) or **`CONTEXT_AXIS_SPLIT`** (mixed parent → subcluster).

**Guards (never corrupts identity):**
- Context-axis subtyping assigns ONLY the **subtype within an already-confirmed lineage**; it can **never change the
  lineage**. A `Glul`+ hepatocyte stays a (pericentral) *hepatocyte* — never "pericentral LSEC" (LSEC needs
  `Stab2/Clec4g`, absent in hepatocyte). Apply the pole ONLY to the `subtypes` entry for the leaf's OWN lineage.
- Use context ONLY for declared axes (`ca`) and lineages in `applies_to`. Everywhere else §4.1 stands: ambient is
  contamination, ignored.
- Honest grounding: context markers are the NEIGHBOR cell type's markers (grounded by the axis `pmids`), used as spatial
  position evidence — not claimed as the leaf's intrinsic markers.

---

## 5. Profile fields — read once, carefully (`{out}/cluster_gene_profiles_round{k}.json`)

Before judging individual clusters, skim `{out}/cluster_overview_round{k}.json`
(`{path:{n, dominant_lineages, top_up[:6], specificity_peak}}`) — the side-by-side view a
human expert uses to know, e.g., which cluster is *the* microglia vs one of several sharing a
marker. Then per leaf:

| field | shape | how to use |
|---|---|---|
| **`graph_evidence`** ★ PRIMARY | `{lineage, lineage_ambiguous, codominant_lineages[], lineage_repick, overlay_lineages[], lineage_unresolved, unresolved_to, dominant_region[], verdict, anchors[≤8], cp[≤4], confusion_unresolved, mixture_subtypes[]}` where each anchor = `{term_id, label, tier, excl_on, excl_n, excl_markers_on[], neg_on, neg_markers_on[], df_on, df_absent, df_onpanel[], n_on, n_disc, path[]}` | **★ The redesigned identity substrate (§7) — REGION-COHERENT graph-descent over the KG you built (2026-06-13).** `lineage` = the class, by a **transmitter HARD gate**: the lineage whose canonical transmitter is ENRICHED (rejects ambient myelin Mbp → not oligodendrocyte) or, if none is enriched (cortical Slc17a7 is everywhere), the strongest PRESENT transmitter by expression — and **neuronal lineages take priority when any neuronal transmitter is present** (a leaky glut marker like Wfs1 can never pull a Gad+ cluster into a hippocampal region). **`dominant_region[]`** = the dominant region-group(s) (neocortical / thalamic / hippocampal pyramidal …) under that lineage, chosen by expression-weighted enriched EXCLUSIVE markers; **anchors are restricted to subtypes of `dominant_region`** so a cross-region blip (CA2 in a thalamic cluster) is structurally excluded — low-abundance layer markers (Fezf2→L5) survive without any magnitude floor. When `dominant_region` is empty (heterogeneous/weak cluster) anchors fall back to all subtypes of the lineage. `lineage_scan` corroborates. **`lineage_ambiguous`** = `true` when **`codominant_lineages`** (≥2 lineages of comparable transmitter strength — glut+GABA doublet / co-release) — then do NOT descend confidently; suspect doublet→`needs_subclustering`, or stop at the common ancestor / bare class (§7 Step 1). **`anchors`** = specific terms (in that lineage) whose OWN **exclusive markers** are ON and negatives clean, **ranked by exclusive support**. **Exclusive markers = low gene-degree** (`share` = #terms listing the gene; degree 1 = exclusive/decisive, high = leaky/pan — NOT evidence for a specific child). `excl_on`/`excl_n` = how many of the term's exclusive markers are ON / exist on-panel; **`excl_markers_on`** = the actual ON exclusive genes (CA1 = `Fibcd1/Spink8`; CA3 = `Grik4/Cpne7`; CA2 = `Rgs14/Amigo2`). `path` = the term's ancestor chain (class → region-group → … → term) — use it to **STOP at a region node when the exact subtype is ambiguous**. `neg_on` = negative-marker leak (>0 ⇒ reject the term). **`df_absent` (DF-absent flag, 2026-06-15)** = the term has on-panel **defining** markers (`df_onpanel[]`, "a single one is sufficient") but **NONE is ON** — the anchor rests on non-defining markers (e.g. cl37 OB-mitral: `Cdhr1` ON but defining `Tbx21` OFF). This is an **advisory flag, NOT a demotion**: anchors are still ranked by `excl_on`/ratio/`n_on` first (df_absent only breaks an exact tie). *(A hard demotion was tried and reverted — it let the generic region node, which has no DF so is never df_absent, beat real cortical layers whose defining marker is merely quiet in sparse data → mass collapse to `neocortical glutamatergic neuron`.)* Use it as a **caution**: a `df_absent` anchor with `excl_on==1` (single non-defining marker) is a likely force-fit (`_validate` → `DEFINING_ABSENT`) — prefer a competitor whose defining/exclusive markers are ON; but a `df_absent` anchor with `excl_on≥2` (e.g. L4 = `Rorb`+`Whrn` ON, `Rspo1` quiet) is genuine — its required markers carry the identity, so KEEP it. (auc≥0.85 counts a quiet-but-enriched defining marker as ON, so a real `Grik4`→CA3 at auc 0.92 is not flagged.) **★ Region-gated positive bar (`auc_pos`=0.65, 2026-06-17):** once `dominant_region` is set (cross-region blips already excluded), a term's OWN POSITIVE markers (exclusive/defining/n_on) count as ON at the looser `auc_pos` instead of 0.85 — so a quiet-but-rank-enriched layer marker (sparse Stereo-seq: a real L4 leaf has `Rorb` auc~0.68, `Rspo1`~0.72, mean~0.01) anchors its layer. Without this the whole cortex collapses to generic `neocortical glutamatergic neuron` because no child subtype passes the strict bar and the region node surfaces via fallback. NEGATIVES stay STRICT (0.85) — a weak negative never vetoes. Targeted panels (Xenium: layer markers auc>0.85) are unaffected; only sparse low-copy data benefits. A resulting `excl_on==1` single-marker layer call is region-bounded (worst case = wrong cortical layer, never cross-region) — judge it with `hp`/anatomy. **`verdict`** = `clear_top` (unique best anchor — descend to it) / `ambiguous_tie` (≥2 tie — split if distinct & strong, else stop at region/class) / `class_or_unknown` (no exclusive anchor — class or Unknown) / **`confusion_unresolved`** (the top anchor and a documented ConfusionPair PARTNER both anchor). **`confusion_unresolved`** field = `{pair, terms[], discriminators, action, mixture, resolves_to, top_distinct_on[], partner_distinct_on[]}` — resolved 3 ways by whether each member's OWN (non-shared) exclusive markers are ON: **`action="split"`** (BOTH members' distinct markers ON ⇒ a real MIXTURE of both cell types, e.g. cl64 = CA2 `Cacng5/Map3k15/Acan` + FC `Necab2/Scgn` both ON → **`needs_subclustering`**, then label each fragment CA2 / FC / Unknown / hippocampal class) / **`action="resolve"`** (only one side's distinct markers ON ⇒ the discriminator favors `resolves_to` → label that type) / **`action="unknown"`** (only shared markers ON, neither's distinct markers ⇒ unseparable by markers → **label `resolves_to`, the deepest COMMON ANCESTOR class** of the pair, e.g. `hippocampal pyramidal neuron` for CA2-vs-FC — NOT a forced pick of either, and NOT Unknown when a common class exists. 2026-06-18: this is the honest call when a sibling's defining marker is **off-panel** — e.g. FC's `Necab2` is defining but unmeasured in a panel, so FC can't be confirmed over CA2 → both fall to `hippocampal pyramidal neuron`. Only when there is NO common-ancestor class does it fall to Unknown.) / `no_lineage`. **`cp`** = confusion pairs touching the path: APPLY their `discriminators` before committing. **This is gene-degree-aware, hierarchy-aware, and scRNA-compatible (no spatial): a leaf descends to CA1 ONLY if CA1's exclusive `Fibcd1/Spink8` are ON, not because it shares the leaky `Wfs1`.** See §7. |
| `identity_scores` *(DEMOTED — supplementary fold summary, NOT the decision; use `graph_evidence` for identity)* | `{top, ranked[≤5], class_fallback, mixed, n_strong_lineages, dominant_lineage, codominant_lineages}` where each candidate = `{term_id,label,tier,n_on,cov,spec,neg,spec_ok,spec_cov,spec_pass}` | **Absolute identity summary** (the backbone axis). `n_on`=how many of the term's non-pan discriminative markers are **ON**. **ON = exceeds global mean+sd in ≥30% of cells, OR (global fold ≥~3× [log2≥1.5] AND detected in ≥30% of cells), OR (rank-enrichment `auc≥0.85` AND the gene is a low-share specific marker, share≤3)** — the fold branch catches *quiet-but-enriched* markers; the **auc branch (RC2, 2026-06-14)** rescues markers so sparse they fail even detection-in-25% (2µm subcellular bins: `Rorb`→L4 auc 0.98, `Fezf2`→L5 auc 0.90, `Dcx`→immature-granule auc 0.72 — yet detected <25%) so L4/L5/immature-granule resolve. **GUARD: the auc branch applies ONLY to a term's OWN low-share (≤3) specific markers — broad/shared genes (`Slc17a7`/`Pcp4`) still need real detection, so it cannot force-fit via spillover.** `spec`=Σ log2-fold of ON markers; **`wspec`**=specificity-weighted spec (each marker weighted 1/#terms-listing-it); **`dec`**=the single most *decisive* marker = max(specificity×fold) — a unique high-fold marker (Fezf2→L5, Rasgrf2/Stard8→L2/3, Grik4→CA3) drives this. Developmental/reactive STATE terms (Neuroblast, Radial glia, reactive/aging/DAM/IFN) are excluded from `top`/co-dominance (shared broad markers, not discrete populations) — **but (2026-06-22) a state term whose OWN markers dominate (`n_on≥2`, `neg==0`, `spec_pass`) is still surfaced in `ranked` flagged `is_state`, and every leaf carries a `disease_candidates` block (all wing state terms with ON markers, `hint=true` for `--disease` matches). disease is a HINT, not a filter: assign a state term when its markers win even with `--disease` omitted (§7).** **`top` is chosen by the DECISIVE marker WITHIN the robust `dominant_lineage`** — so one unique specific marker beats many broadly-shared ones (Tbr1/Satb2/Slc17a7). **DF-override**: if a **coherent** term (n_on≥2) in the dominant lineage has a **DEFINING (`DF`) marker** that is fold-ON, that term wins outright ("a single defining marker is sufficient" — Fezf2→L5 ET, Cux2→L2/3, Rspo1→L4, Foxp2→L6 CT, Cplx3→L6b, Prox1→DG, Wfs1→CA1, Grik4→CA3…). This is what lets the exact cortical layer / hippocampal field resolve despite shared broad markers. (The coherence requirement guards against a *lone* defining marker firing on ambient signal in an over-split fragment — e.g. it stops Rgs14→CA2 over-calls.) **★ DOMINANT broad-defining rescue (2026-06-23) — CASE+RULE.** *Case:* a cell type whose discriminative identity rests on a **region-BROAD** defining marker (the *type itself* recurs across regions — interneuron neurochemical classes `Sst`/`Pvalb`/`Vip`, glia, immune, vascular). `region_share` (a proxy for "specific to a region") wrongly tags such a marker as pan and strips it from the `disc` set, so the type has too few non-pan disc markers to be built as a candidate → it is never scored → the cluster collapses to its bare lineage **class** even though the defining marker is its dominant signal (an Sst⁺ interneuron cluster, up=`Sst`/`Npy`/`Cort`, fell to `cortical GABAergic interneuron`, leaving the agent nothing to subcluster on). *Rule:* a defining marker encodes curated sufficiency that the `region_share` heuristic must not veto — so a **region-broad DF that DOMINATES the cluster (top vs-rest up-marker, `auc≥0.85`) makes its term a candidate and satisfies coherence on its own**, and counts toward its `spec` set. Region-**specific** DFs (low-share `Fezf2`/`Rorb`/`Grik4`) are unchanged (already in `disc`); a non-dominant or leaky broad DF cannot fire (must be a top up-marker), and the VALID-EVIDENCE gate + guard A still bound it. Net effect: interneuron neurochemical subtypes now surface in `top`/`ranked` so you can label `Sst+`/`Pvalb+`/`Vip+` (or subcluster a genuine mix) instead of the bare class. **DF-override corroboration (helper guards A+B, 2026-06-11):** a DF win additionally requires (A) the term's `dec` be competitive with the best same-lineage candidate (≥0.34× it), AND (B) if the term has on-panel high-specificity (low-share) discriminative markers, ≥1 must be ON. This stops a *leaky* defining marker — curated as one term's DF in the ontology but biologically also expressed elsewhere (e.g. `Wfs1` is listed only on CA1 yet is also present in cortex/amygdala) — from hijacking a higher-evidence competitor: a cortical/amygdalar leaf with `Wfs1` fold-ON but its specific `Fibcd1/Spink8` OFF is no longer mislabelled CA1 (the override falls through to the genuinely decisive candidate). If a term's specific markers are ALL off-panel, guard B is inert ⇒ panel-blind fallback preserved. **★ VALID-EVIDENCE gate on EVERY candidate (`spec_n`/`spec_on`/`spec_cov`/`spec_pass`, 2026-06-12) — the over-call antidote:** a specific celltype is only honest when the leaf actually expresses THAT type's OWN identifying markers, not when it merely shares a few broad/leaky markers. Per term: **`spec_n`** = its on-panel high-specificity (share≤SPEC_SHARE_MAX=3) discriminative markers; **`spec_on`** = how many are ON; **`spec_cov`** = `spec_on/spec_n`; **`spec_ok`** = `(spec_on≥1 OR spec_n==0)` (the DF-guard-B threshold); **`spec_pass`** = `(spec_cov≥0.5 OR spec_n==0)` — i.e. **≥ half of the type's own specific markers are present.** A leaky marker carried as plain `Required`/`DF` (e.g. `Wfs1` is `R`+`DF` on 6 CA1 terms, so its share=6 keeps it OUT of the spec set) does NOT count toward `spec_on`. So a NON-hippocampal leaf that merely shares `Wfs1`/`Crym` scores `spec_cov≈0–0.33` for CA1 (its `Fibcd1`/`Spink8` OFF) while a real CA1 ≈1.0; a non-CA3 leaf ≈0.29 for CA3 (`Grik4`/`Cpne4`/`Nectin3` OFF) while a real CA3 ≈0.71. **The helper ENFORCES this: if the dec-selected `top` is a SPECIFIC (non-class) term that FAILS `spec_pass`, `top` is set to `null`** ⇒ this leaf has no validly-supported subtype → fall to its dominant-lineage **CLASS** (`class_fallback`, §7 Step 3) or **Unknown** (Step 4). The rejected candidate stays in `ranked` (marked `spec_pass=false`) so you can RESCUE it ONLY if `hp`/experience clearly show the type's OWN markers dominate; `ranked` lists `spec_pass=true` terms first. (`spec_n==0` ⇒ all specific markers off-panel ⇒ gate INERT, panel-blind: judge by `E`/`ES`/supporting per §4 — this preserves cortical layers whose required are off-panel.) **Principle (§4 lesson 3): a present `top` has valid evidence; a null `top` means "no NORMAL subtype is validly supported" → name the class or Unknown, never force-fit the nearest specific term. Still corroborate a present `top` with `hp`/negatives/anatomy; if it names the wrong subtype of the RIGHT lineage, fall to `class_floor`, never to a split.** ★ **EXCEPTION — null `top` does NOT mean "go straight to class": you MUST first run the §7 Step-2.5 STATE CHECK.** `tier=state` terms are excluded from `top` BY DESIGN (their broad reactive markers would force-fit normal cells), so a `null top`/`class_or_unknown` verdict ALSO occurs when the true identity is a state — surfaced instead in `ranked` flagged `is_state` and in `disease_candidates`. If a state term passes the Step-2 test (`spec_pass`, `neg==0`, beats the normal/class alternative), IT is the validly-supported label, not the bare class. `cov`=n_on/(its markers); `spec`=Σ global log-fold of the ON markers (real specificity, pan excluded); `neg`=its negative markers that are ON (a leak ⇒ wrong/contaminated). **STRONG** = `n_on≥2 and neg==0`. *strength-before-depth* applies (a deep subclass outranks a class only when itself STRONG). **`dominant_lineage`** = the strongest lineage-root present; **`codominant_lineages`** = lineage-roots whose strength ≥½ the dominant's; **`mixed`= (≥2 codominant lineages)** — i.e. a *real* chimera, NOT mere ambient bleed (§4.1). **The co-dominance/`mixed` test counts ONLY valid-evidence terms (`spec_pass` AND `spec_n>0`)** — the same gate that nulls a force-fit `top` — so a force-fit term (shared/leaky markers) or a panel-blind term CANNOT inflate a false second lineage (a single striatal cluster sharing `Hpca`/`Wfs1` with hippocampal glut terms is NOT flagged `mixed`). It falls back to all-coherent only when NO term qualifies. `class_fallback`=dominant lineage's class term. **Treat as one strong input — corroborate with the fields below.** |
| `hp` | `[[gene, mean_in, rank, pct_in], …]` top 50 | **Absolute expression** (HK removed): what this cluster most strongly/broadly expresses. A gene absent from `up` is NOT absent — broad/uniform markers fall below the 0.6 AUC cutoff but show here. The decisive cross-check when `up` collapses. |
| `up` | `[[gene, auc, mean_in, mean_out], …]` AUC≥0.6, desc | **vs-rest contrast** (local in subcluster rounds = vs siblings). Specificity + sub-structure signal. **Demoted for identity** — do not let a flat `up` veto a clear absolute identity. |
| `dn` | same, AUC≤0.4 | depleted; supports ruling OUT (not a hard gate — KDE smear). |
| `ontology_gene_table` | `{gene:{rank,mean,auc,panel}}` | every R/S/N/E/ES marker of this wing with its **absolute mean** + AUC + panel flag. `panel=false` ⇒ off-panel (not measured): you may NOT use its absence to reject a type (panel-blind fallback). Use the **mean** (absolute) to confirm a marker is genuinely on even when its AUC is ~0.5. |
| `lineage_scan` | `{lineage: [[gene, auc, pct_in], …]}` | canonical lineage markers (incl. pan) regardless of cutoff → the **class-level identity** when no subclass resolves. Authoritative for "is this lineage present". |
| `candidate_subtypes` | ranked `[{term_id,label,tier, n_req_present,n_nonpan_present,n_req,n_req_offpanel, required:[[gene,auc,present,on_panel,is_pan]], children_resolvable:[…]}]` | the ontology terms whose defining markers actually appear here, with per-term evidence (parent-pooled for sub-leaves so an over-split homogeneous subtype is not lost) and **`children_resolvable`** = 1-level children whose distinguishing markers are present → the **split-deeper** signal. |
| `identity_evidence` | `{density{median,pct,low_frac}, specificity_peak, lineage_markers_present{n,genes}, housekeeping_share, cross_lineage{…}}` | raw signals for the Background test (§7 Step 0). |
| `neuropil` | `{low_frac, synaptic_top[], n_synaptic, no_specific_anchor, candidate}` | **spatial neuropil signal (§7 Step 0.5).** `candidate==true` (low density + ≥3 synaptic/dendritic markers dominant + no specific cell-type anchor) ⇒ label **`neuropil`** (tissue compartment, not a cell). |
| **`exclusive_mixture`** | `[{pair:[labelX,labelY], markers:[[X-genes],[Y-genes]], frac_exclusive:[fX,fY], verdict:"split"}]` | **★ MUTUAL-EXCLUSIVITY mixture (§7 S2.55).** Two same-lineage siblings whose **defining markers are reciprocally exclusive** (X.DF ⊂ Y.negative & vice-versa — CD4/CD8, arterial/venous EC) each ON in a DISJOINT cell subset (`fX`,`fY` ≥0.15) with low double-positive. Invisible to MEAN-based `top` (both look diluted → class_floor) but a real per-cell mixture ⇒ **`needs_subclustering`**. Derived from KG (no hardcoding); complements `bimodal_subtypes` (low-share positive markers) by using the reciprocal pos↔neg relation, so it catches high-share lineage markers. Empty unless a genuine balanced exclusive mix is present. |
| **`transition_candidates`** / `transition_note` | `[{term_id, label, D, anchor_retained[], acquired_on[], mimics:[{lineage,program}], n_anchor, n_acquired}]` + a note string | **★ STATE-TRANSITION / lineage-mimicry (anchor-tiebreaker, §7 S2.6).** Surfaced when a KG transition Term (has `MIMICS_PHENOTYPE`+identity-anchor) has **its origin-identity anchors ON** (`anchor_retained`) AND **the mimicked lineage's program ON** (`acquired_on`). Means: this leaf is the **origin lineage in a transition state** (`label`), NOT the `mimics` lineage and NOT a chimera — anchors are the retained-origin tiebreaker. Label the transition Term; do not split (S2.5) or call the mimicked lineage. Empty/absent ⇒ rule inactive (no anchor retained, or wing has no mimicry Terms). Ranked by anchor then acquired support. |
| **`context_axis`** | `[{axis, poles:[{pole,score,on_context_markers,subtype}], poles_present[], mixed, dominant_pole, suggested_subtype, pole_confidence, pmids}]` | **★ DETERMINISTIC zonation/layer subtyping (§4.3).** Present only when the leaf's lineage is in a declared `ca` axis's `applies_to`. Poles scored **graded** (weak signal counts). `dominant_pole`+`suggested_subtype` ⇒ label that subtype (`reason_code=context_axis`, `confidence=pole_confidence` — **low = a weak lean, still preferred over bare class** on this bounded axis); `mixed` (near-tie, no lean) ⇒ large parent → `needs_subclustering`. Follow the engine's pole — don't re-derive from intrinsic (off-panel) markers. |
| `subtype_groups` | `[{parent_id,parent_label,child_term_ids,children}]` | parents whose ≥2 finer sibling subtypes co-occur here → maximal-depth recursion target. **Also the sibling split-deeper signal**: if your chosen `top` is one of the `child_term_ids` and ≥1 OTHER normal sibling is co-listed, the leaf mixes two subtypes (e.g. CA3+CA1) → split (Step 2 sibling check). |
| `parent_ctx` | `{path, up[:10], hp[:10]}` | parent signature. **For sub-leaves you MUST pool parent_ctx** — within-parent re-ranking erases the shared class signal. |
| `sibling_contrast` | `[[gene, pct_in, max_sibling_pct, delta], …]` | within-parent discriminators — for deciding splits / choosing between close siblings, not for denying identity. |
| `flags` | `{multi_lineage, layer_mixing, pan_class_only, background_score}` | pre-computed split signals (corroborating, not deciding). **`multi_lineage` is computed from raw `up` and is AMBIENT-PRONE** — KDE/neuropil bleed (glia, interspersed interneurons) makes it list lineages that are not real co-populations. For the chimera decision use `identity_scores.mixed`/`codominant_lineages` (co-dominance-tested), NOT this flag alone (§4.1). `pan_class_only` means "only class markers passed" — at the gate level; with the fold-aware ON a real subtype often resolves anyway, so re-check `identity_scores`/`candidate_subtypes` before treating it as unresolved. |
| `signal_targets_suggested` | ranked list | informational only (no marker-guided split here); ignore for the decision. |
| `n`, `n_out`, `n_sub` | ints | `n` = **TRUE** leaf cell count (use this for absolute-size / rare-type sanity). `n_sub` = cells actually used for AUC/rank — for speed `init` profiles a **stratified subsample** (≤150k; small clusters kept whole, large ones proportionally; global gmu/gsd stay on full data). `warn=n_out_too_small` (subsample out-group <20) ⇒ treat AUC cautiously. |

Gene matching is **case-insensitive** throughout (mouse `Gad1` ≡ human `GAD1`).

---

## 6. Reasoning Scaffold — produce this for EVERY cluster

Condense the result into `rationale` / `alternatives_considered`. This is how an expert keeps
the judgment thorough and reproducible — and it forces the comprehensive synthesis of §4.

```
S1 IS-IT-A-CELL : density.pct=__  lineage(specificity_peak, lineage_markers.n, graph_evidence.lineage)=__ → cell? y/n
S2 LINEAGE      : graph_evidence.lineage=__ (the floor you are sure of) ; lineage_ambiguous? __ (if T → doublet/common-ancestor, not a confident single call) ; ambient bleed lineages ignored (§4.1) ; overlay_lineages=__ (program/state, NOT identity — never switch to these) ; lineage_repick? honor .to not .from ; lineage_unresolved? → label unresolved_to (common-ancestor class) else Unknown, never a confident codom pick
S2.5 MIXTURE GATE: ≥2 lineages/cell-classes with their OWN specific markers STRONGLY co-enriched? (check up/hp/lineage_scan, not just identity_scores.mixed — §4.1 strong-mixture override). If YES → decision = needs_subclustering, STOP here (confirming a single celltype is FORBIDDEN on a genuine mixture). **BUT first run S2.6 — a lineage-mimicry transition (one cell) masquerades as an origin+mimicked chimera.**
S2.55 EXCLUSIVE-MIXTURE CHECK: leaf has `exclusive_mixture` (verdict `split`)? = two SAME-lineage sibling subtypes whose **mutually-exclusive defining markers** (X's DF is Y's negative & vice-versa — e.g. CD4 vs CD8, arterial GJA5 vs venous ACKR1) are EACH ON in a substantial DISJOINT cell subset (per-cell, low double-positive). At cluster-MEAN both look "low/diluted" so `top` nulls to class_floor — **but both-low-and-similar for mutually-exclusive markers = a MIXTURE, not a uniform ambiguous population.** → decision = **needs_subclustering** (then label each fragment, e.g. CD4SP / CD8SP). This complements S2.5 (which only fires on ≥2 different lineage ROOTS); here the two subtypes share a root so S2.5's `mixed` is false. Derived from KG structure (no hardcoding).
S2.6 TRANSITION CHECK (anchor-tiebreaker): leaf has a `transition_note` / `transition_candidates`? Each = an origin lineage's state Term that retains **identity-anchor markers** (`anchor_retained`) WHILE expressing a **mimicked lineage's program** (`acquired_on`). RULE: when origin anchors persist, this is NOT the mimicked lineage and NOT a chimera — it is **"<origin lineage>, <transition state>"** (e.g. EndMT endothelial keeps PECAM1/CDH5 while expressing VIM/ACTA2 → `endothelial cell, mesenchymal-transitioning`, NOT mesenchymal/fibroblast; tuft-like mTEC keeps EPCAM/KRT5 while expressing POU2F3 → mimetic TEC, NOT a tuft cell). So: if a `transition_candidate`'s anchors are clearly ON and its acquired program ON, **label that transition Term** (do NOT split as mixture in S2.5, do NOT call the `mimics` lineage). Only when the origin anchors are ABSENT (pure mimicked program) is it the real mimicked lineage. Evidence-strong: the engine surfaces & ranks these; you confirm anchors are genuinely the leaf's (not ambient) before committing.
S3 ANCHORS      : verdict=__ ; top anchors w/ excl_markers_on = __ (the type's OWN exclusive markers actually ON?)
S4 DESCEND      : path of the best anchor = class→region→…→term ; how deep can I DEFEND? (stop at region if subtype ambiguous)
S5 CP / NEG     : any graph_evidence.cp on the path? apply discriminators ; any neg_on leak? reject
S5.5 STATE CHECK: BEFORE any class-floor/Unknown — identity_scores.ranked is_state=true + disease_candidates: any state term whose OWN markers dominate (spec_pass, neg==0, beats normal)? (states are NOT graph_evidence anchors & excluded from top → null top does NOT mean class; §2.5). If YES → label that state term (regardless of whether --disease was passed).
S6 RESOLVE      : clear_top → deepest defensible term ; ambiguous_tie(distinct+strong) → needs_subclustering ; ambiguous_tie(weak) → region/class ; class_or_unknown → §2.5 STATE CHECK FIRST → state term if its markers win, else class (lineage clear) / Unknown
S7 FLAGS        : subclass_resolved(T/F) ; confidence(high/med/low) ; reason_code ; disease ; split mechanism = subcluster(Leiden)
```

---

## 7. Decision Algorithm — graph-descent over the KG you built (run in order; first terminal call wins)

> **You are the expert; the ontology wing is YOUR knowledge graph.** Identity = descend the hierarchy
> (class → region-group → subclass → supertype) **as deep as the leaf's evidence SOLIDLY supports**, judged by
> each term's OWN **exclusive markers** (low gene-degree — markers it does NOT merely share), never by
> shared/leaky markers. **`graph_evidence` is your primary substrate** (§5): `{lineage, verdict, anchors[…],
> cp}`. Each anchor = a specific term whose exclusive markers are ON, carrying `excl_markers_on` and its
> `path` (class→…→term). `identity_scores` is now only a supplementary absolute-fold summary — do NOT treat
> its `top` as the decision. **NO spatial input** — context is gene expression only (works for scRNA-seq).

### Step 0 — IS IT A CELL? (Background) → may emit **Background**
Background ONLY when ALL hold: `identity_evidence.density.pct` bottom-quartile (or `low_frac` high) AND
`specificity_peak` null AND `lineage_markers_present.n == 0` AND `graph_evidence.lineage == null`. **init /
large**: `n ≥ 500` or a top-level leaf (path has no ".") ⇒ never Background → continue. `density==null` ⇒
use the identity conditions only. Else continue.

### Step 0.5 — IS IT NEUROPIL? (tissue compartment, not a cell) → may emit label **`neuropil`**
**Spatial only** (segmentation-free KDE/SSAM): the vector field samples not just cell bodies but the
**neuropil** — the dense dendrite/axon/synapse mesh *between* somata. Neuropil lattice points form their own
cluster: synaptic markers high, but NO cell-type identity (the cell-body markers live in the dense soma
clusters). Labeling such a cluster as its lineage (`glutamatergic neuron`) is WRONG — it double-counts the
neuropil of neurons whose somata are already labeled elsewhere, and pollutes the cell-type map.
Emit **`neuropil`** (decision `celltype`, **`label="neuropil"`, `term_id=""`**, `reason_code="neuropil"`) when
the helper's **`neuropil.candidate == true`** — i.e. ALL of: (a) **low density** (`neuropil.low_frac ≥ 0.5`),
(b) **synaptic/dendritic markers dominate** `up` (`neuropil.n_synaptic ≥ 3` — Camk2a/Map2/Dlg4/Syp/Nrgn…),
(c) **no specific cell-type anchor** (`neuropil.no_specific_anchor` — no subclass/supertype/cluster anchor; the
cell-body markers are absent). This is DISTINCT from Background (which is null signal / no lineage — neuropil
HAS a clear neuronal lineage) and from a lineage class (which is cell bodies). If a **specific anchor IS present**
(real somata, even at low density) ⇒ NOT neuropil → continue to Step 1–2 and label the cell type. scRNA-seq has
no neuropil lattice points (real single cells) ⇒ this practically never fires there. When in doubt (synaptic but
some cell-type marker present), prefer the cell type / class over neuropil.

### Step 1 — LINEAGE (the floor you are sure of)
Confirm **`graph_evidence.lineage`** (the dominant class — `glutamatergic neuron` / `GABAergic neuron` /
`astrocyte` / …), corroborated by `lineage_scan`. If no lineage resolves → **Step 4 (Unknown)**. The lineage
class is the SHALLOWEST honest label; you descend below it ONLY with valid exclusive evidence (Step 2). A
secondary lineage that is only weak **ambient/neuropil bleed** (§4.1: `Gad1/2`, `Slc1a2/Aldoc/Apoe`,
`Mbp/Plp1/Mobp/Ptgds`, scattered immune/vascular under ONE dominant lineage) is contamination, NOT a second
identity — do not split on it. A real second lineage shows its OWN exclusive subtype markers at comparable
strength (handled in Step 3). **★ But a real second lineage does NOT require `lineage_ambiguous`/`mixed`==true to
be honored: if you can SEE ≥2 lineages' OWN specific markers STRONGLY co-enriched (§4.1 strong-mixture override),
the decision MUST be `needs_subclustering` — confirming a single celltype is FORBIDDEN — even when the helper's
`mixed`/`lineage_ambiguous` flags are false (they can miss a strongly-enriched non-neuronal/vascular lineage).**

**★ NON-IDENTITY LINEAGE OVERLAY — `graph_evidence.overlay_lineages` / `lineage_repick` (2026-06-23, CASE+RULE).** *Case:* a cluster's discriminative identity is lineage **B** (B's OWN non-pan required markers ON), but a second lineage **A**'s program is also present — yet A is supported by **broadly-shared (pan) required markers ONLY, with ZERO discriminative (non-pan) required marker ON.** Because the engine picks lineage by a single required marker's max strength, A's high-magnitude pan marker can even out-score or hijack B. *Rule:* a lineage in **`overlay_lineages`** (present, but no discriminative required ON) is a **co-expressed program — an activation/transition STATE — or ambient, NOT a competing identity.** Label **B** (the lineage with its own discriminative markers ON); treat A as at most an activation state OF B — e.g. an endothelial cell carrying a contractile/matrix program = **EndMT-activated EC** (not smooth-muscle/myofibroblast); an epithelial cell with a mesenchymal program = **EMT-activated epithelium**; a leukocyte resolvable only by a shared pan marker = ambiguous, prefer the common class / Unknown over a coin-flip. **Do NOT switch identity to an overlay lineage just because its program is high-magnitude.** **`lineage_repick`** = the engine ALREADY corrected such a hijack (the pan-only lineage had won the raw max; it was re-assigned to the discriminative-identity codominant lineage): **honor `lineage_repick.to`, never revert to `.from`.** This is the SAME discriminative-vs-pan logic as Step 3's valid-evidence gate, applied at the lineage level — so a REAL second cell type is *not* in `overlay_lineages` (it carries its own discriminative markers → surfaces as a mixture/split, handled above), and a real B-with-A-state is labeled B, not mis-switched to A.

**★ INDISCRIMINABLE LINEAGES — `graph_evidence.lineage_unresolved` / `unresolved_to` (2026-06-23, CASE+RULE).** *Case:* ≥2 codominant lineages whose strength is carried by a marker they **SHARE** (e.g. `PTPRC` is in BOTH `myeloid cell`'s and `lymphocyte`'s required set — a pan-leukocyte marker that says "leukocyte" but cannot tell myeloid from lymphocyte), and the chosen lineage has **NO codom-private** ON required marker. The evidence literally cannot distinguish the codominant lineages. *Rule:* the engine sets **`lineage_unresolved=true`** and **`unresolved_to`** = the deepest common-ancestor class of the codominant lineages (or `null` if the wing has none). **Label `unresolved_to` if present (e.g. `leukocyte`/`immune cell`) — else `Unknown`. Do NOT confidently pick one codominant lineage** (the prior failure: a `PTPRC`-only 33-cell cluster with no `CD3`/`CD8`/`NKG7` and no `CD68`/`LYZ` was called `lymphocyte` off `ZEB2` alone — `ZEB2` is not lymphoid-specific). This is **not** triggered by a real doublet: glut+GABA carry DIFFERENT private transmitters (`Slc17a7`∈glut only, `Gad1`∈GABA only), so each keeps a codom-private marker → not unresolved → handled as a doublet/split above. `unresolved_to` being a class is the honest floor — informative ("it IS a leukocyte") without over-claiming a subtype the markers don't support.

> **★ Lineage strength & ambiguity (helper Fix 2/3).** `graph_evidence.lineage` is chosen by **transmitter
> STRENGTH** (summed fraction-on of the lineage's canonical markers — Slc17a7/6/8 for glut, Gad1/2·Slc32a1 for
> GABA, Th/Ddc for DA…), NOT by raw marker count, and NOT from pan-neuronal markers (Camk2a/Snap25 are excluded
> — they confirm "neuron", not "which lineage"). So a faint ambient bleed of a competing transmitter cannot
> establish a false lineage. **When `graph_evidence.lineage_ambiguous == true`** (`codominant_lineages` lists
> ≥2 lineages of comparable transmitter strength — a real glut+GABA doublet or co-release), do **NOT** descend
> confidently into the named lineage's anchors. Instead: (a) if the two are mutually-exclusive fast
> transmitters (glut vs GABA) → suspect a **doublet/mixed bin → `needs_subclustering`** (§4.1), or (b) **stop at
> the common ancestor** (e.g. `neuron`) / name the bare class, or (c) reconcile with `up`/`experience`/anatomy
> before committing. Never silently take the strongest of two co-dominant lineages as a confident single call.

### Step 2 — DESCEND to the deepest VALIDLY-SUPPORTED identity → **celltype**
Read `graph_evidence.verdict` + `anchors` (each = a specific term whose OWN exclusive markers are ON, with
`excl_markers_on` + ancestor `path`):

- **verdict = `clear_top`** — the top anchor's exclusive markers are clearly ON and it beats every
  alternative. → **`celltype` = the deepest term on its `path` whose exclusive evidence you can DEFEND** —
  usually the top anchor itself (CA1 via `Fibcd1/Spink8`; CA3 via `Grik4/Cpne7`; CA2 via `Rgs14/
  Amigo2`). **Go as deep as the evidence is solid; if the EXACT subtype is uncertain but its region/group is,
  STOP at the region node on the path** (e.g. label `hippocampal pyramidal neuron` when CA1-vs-CA2 is
  genuinely unclear) — stopping early is HONESTY, not failure. `subclass_resolved = (chosen.tier!="class")`.
- **CP check (mandatory when present):** if `graph_evidence.cp` lists a confusion pair involving your chosen
  term, APPLY its `discriminators` BEFORE committing — confirm the type's OWN distinguishing markers (not the
  shared ones) and that its negatives are clean; otherwise fall to the safer side / region / class the cp
  names (e.g. a `Wfs1`-sharing non-hippocampal leaf → NOT CA1; a deep-cortical leaf → `neocortical`, not
  piriform).
- **verdict = `ambiguous_tie`** — ≥2 anchors tie at the top exclusive support. TWO cases:
  - **Genuine multiplicity** — the tied anchors are DISTINCT identities EACH with REAL exclusive markers
    (e.g. a strong glut anchor AND a strong oligo anchor) → **needs_subclustering** (Step 3).
  - **Weak ties (the force-fit trap)** — each anchor rests on a single shared/leaky marker with no clear
    winner → do **NOT** pick one. Fall to the **deepest COMMON ancestor** on their paths (a region group), or
    the **lineage CLASS** if they share no region. This is the "never force-fit the nearest specific term"
    rule — a non-hippocampal leaf with weak scattered anchors → its lineage class, not a fake CA1/piriform.
- **verdict = `class_or_unknown`** — no NORMAL region anchor has exclusive support. **Before you fall to class,
  run the mandatory STATE CHECK (Step 2.5): `tier=state` terms are NOT graph_evidence anchors, so the descent
  returns `class_or_unknown` (and `identity_scores.top` is often `null`) EVEN WHEN an activation/disease/
  developmental STATE is the true identity.** If Step 2.5 finds a validly-supported state term → label THAT (a
  state term is a valid terminal `celltype`). Only if no state term qualifies → the **lineage CLASS** as a terminal
  `celltype` (`subclass_resolved=false`, `reason_code="class_floor"`) if the lineage is clear; else **Unknown**
  (Step 4). NEVER reach into a weak candidate to name a specific type with no exclusive evidence.

`confidence`: **high** (anchor `excl_on≥3` or `excl_on/excl_n≥0.7`, clean negatives) · **medium** (anchor
`excl_on` 1–2, or you stopped at a region group) · **low** (single exclusive marker, panel-limited).

### Step 2.5 — ★ STATE / activation / disease terms are FIRST-CLASS labels (MANDATORY before any class-floor/Unknown)
**Why this step exists (the structural failure it fixes):** `tier=state` terms (any activation / inflammatory /
proliferating / transitional / reactive / regenerating / senescent / disease-or-injury state your wing declares) are
**deliberately excluded from the graph_evidence anchor descent AND from `identity_scores.top`/co-dominance** (their
broad reactive markers would otherwise force-fit normal cells, §5). The CONSEQUENCE you MUST account for: the §7
descent verdict comes back `class_or_unknown` and `identity_scores.top` comes back **`null`** *even when a state term
is the single best-supported identity in the leaf*. If you read "null top → class" or "no anchor → class" literally,
you WRONGLY bury a real state under the bare lineage class. **So before emitting ANY `class_floor` (Step 2) or
`Unknown` (Step 4), run this check:**
1. Read **`identity_scores.ranked`** for entries flagged **`is_state=true`**, AND the per-leaf **`disease_candidates`**
   block. BOTH are surfaced for EVERY leaf regardless of whether `--disease` was passed (see §5 / disease handling).
   `identity_scores.top` being `null` does NOT mean "nothing resolved" — it means the dec-`top` was suppressed; the
   real winner may be sitting in `ranked` flagged `is_state`. **Always read `ranked`, never stop at `top`.**
2. Apply the **Step-2 test to the best state term**: its OWN required/specific markers dominate (high `n_on`,
   `spec_pass==true`, `neg==0`) AND it BEATS the best normal alternative (higher `spec`/decisive evidence). A state
   term that fails `spec_pass` (only shared/pan markers ON while its OWN defining markers are OFF) does NOT qualify —
   do not force-fit it either (this is the same valid-evidence gate as for normal subtypes).
   **★ SPECIFICITY GATE (2026-07-02) — mandatory: `disease_candidates[i].n_specific_on ≥ 1`.** A disease/reactive
   state is only real when its OWN **distinguishing** marker is ON. `n_specific_on` counts its REQUIRED/DEFINING
   markers that are BOTH low-share (not pan-reactive — Gfap/Serpina3n/Vim/Cd44/Apoe are shared across many states →
   excluded) AND rank-enriched (`auc≥0.85`, not ambient). If `n_specific_on==0` (only shared reactive/pan or ambient
   markers ON — i.e. a long marker list winning by raw `n_on` count) the state is **NOT** the identity → **fall back
   to the NORMAL counterpart** (reactive/fibrous astrocyte, or the bare lineage class), never the disease state. This
   blocks healthy-tissue false positives (a 23-cell Gfap⁺ white-matter astro fragment called
   `Disease-associated astrocyte` / DAM on a normal brain). `n_on` / list length alone NEVER qualifies a state.
3. If it passes → **label that state term** (`decision=celltype`, its `term_id`, `subclass_resolved=true`,
   `confidence` per evidence; note the sample/disease context in `disease_context`). A state term is a fully valid
   terminal label — it is NOT "less real" than a normal subtype.
4. If NO state term passes → proceed to class-floor / Unknown as normal.

**Forbidden rationalizations (each has caused a real misannotation):** dismissing a candidate "because it is a state",
"because it's an on-chip / in-vitro / culture program, not a real cell type", or "because no `--disease` was declared".
None of these is a rule. The MARKER EVIDENCE governs, full stop (§5, disease handling). When several state terms
compete, separate them by each term's OWN discriminating markers (not the shared ones); never collapse distinct states
into one, and do not prefer a disease/cancer-specific state on a non-matching sample unless its own markers clearly
dominate.

### Step 3 — SPLIT on genuine multiplicity (symmetric with "single → don't split")
Emit **`needs_subclustering`** (`reason_code="multi_identity"`) ONLY when **≥2 GENUINELY DISTINCT identities
co-occur**, EACH with its OWN exclusive markers ON at comparable strength — i.e. `graph_evidence.verdict ==
ambiguous_tie` with the tied anchors being real, distinct types (different lineages or different regions,
each `excl_on≥2`-ish). Examples: a real oligodendrocyte band fused with a glut band (`Mbp/Mobp/Plp1` AND
glut exclusive markers both strong); a coarse mega-blob where CA1 (`Fibcd1/Spink8`) AND CA3 (`Grik4/Cpne4`)
exclusive markers are BOTH strongly present. Splitting separates them and same-identity fragments merge (§8).

**Do NOT split on:** (a) ambient/neuropil bleed (§4.1 — weak glia/interneuron/vascular co-signal under ONE
dominant lineage); (b) "couldn't name the exact subtype" — that is Step 2 **stop-at-region/class**, NOT a
split; (c) a single homogeneous population (splitting just tiles it into meaningless fragments). **Splitting
and NOT-splitting are enforced EQUALLY**: one clear identity → terminalize (Step 2); ≥2 distinct exclusive-
supported identities → split; weak/ambiguous → stop at the region or class (Step 2), never split-to-guess.

### Step 4 — NOTHING RESOLVES → **Unknown**
No lineage (`graph_evidence.lineage==null`) AND no exclusive anchor **AND no state term passes Step 2.5**, not
Background — after honestly listing ≥3 considered terms (from `graph_evidence.anchors` + `candidate_subtypes` +
`identity_scores.ranked` incl. `is_state` + `disease_candidates`) each with a concrete rejection (no exclusive/own
marker ON / negative leak / shared-marker-only force-fit). **You MUST have run the Step-2.5 STATE CHECK first** — a
state term can be the answer when no normal anchor resolves. **The Step-2-class-floor vs Step-4-
Unknown split is whether the LINEAGE is clear:** clear lineage + no exclusive anchor → **class-floor**
(`glutamatergic neuron`, Step 2 `class_or_unknown`); lineage itself unclear → **Unknown**. "Don't know" is
the required answer — NEVER force-fit the nearest specific term. **Do NOT downgrade a confident class/region
to Unknown** (a class or a region-group node CAN be the final label).

### Step 5 — unsplittable / doublet termination (anti-loop)
A `needs_subclustering` is a request to the caller to split. If this leaf is **already a child of a
prior split** (path has ".") and STILL shows the same mixture as its `parent_ctx` (the split did not
separate) → do **not** flag again: if the mixed types share a common ancestor class, label that class
(`subclass_resolved=false`, `reason_code="split_no_gain"`); else `Unknown`. **Never `Background` here**
(Step 0 already ran; these are real cells).

### disease handling (Step 7 of the scaffold)
**★ `disease` is a HINT, never a filter (2026-06-22).** The normal scoring core (`graph_evidence`/`identity_scores`)
excludes `tier=state` terms from `top`/chimera (their broad reactive markers would force-fit normal cells). So for
**every leaf — regardless of whether `--disease` was passed — the helper ALWAYS surfaces a `disease_candidates`
block**: the wing's state terms (every `tier=state` / dev-reactive term) whose markers are ON in this leaf, each
with **on-markers / n_on / n_markers** and a `hint` flag. A strong state term (its own markers dominant) also
appears in `identity_scores.ranked` flagged `is_state`. **You are the judge**: if a state term's OWN markers
genuinely dominate (Step-2 test: enough of its specific markers ON, negatives clean, it beats the normal
alternative), **assign it — even if `--disease` was omitted**. Do not skip a state call just because no disease was
declared; the marker evidence governs.
- `--disease` **omitted** → no hint, but state candidates are STILL surfaced (`hint=false`). Consider all terms
  regardless of `D`; assign a state term when its markers win, else the best normal type. Never assume normal.
- `--disease == normal` → treat as a hint toward normal: prefer `D=="normal"`; assign a state term only if its
  markers are unambiguously dominant (a true disease state in a "normal" sample is still real if the data says so).
- `--disease == <name>` → matching state terms are flagged `hint=true` and sorted first. For each leaf read
  `disease_candidates`: if one fits (Step 2 — its OWN markers ON, negatives clean, beats the normal alternative),
  assign it; else label the nearest **normal** type (`disease_context` still records the sample context).
- Never invent a non-ontology label. If the wing has **no** state term that fits, fall back to the nearest normal
  type / Unknown (build the state type first; build-mode §3/§7).
- `open_question_flag=true` if the chosen term is in the wing's `oq`.

---

## 8. ★ MERGE + resolution — handling imperfect clustering (apply every subcluster round)

**★ PARENT-CONSTRAINED subclustering (helper, 2026-06-14).** A subcluster's children are searched ONLY within
the **parent's KG subtree** — `graph_evidence` for a child of cluster C is restricted to descendants of C's
resolved **dominant region-group** (or its lineage class if no region resolved). A child of a *neocortical*
cluster can therefore ONLY be a neocortical subtype (L4/L5/L6…), NEVER pontine/cerebellar/midbrain — and genuine
SIBLINGS stay findable (a minority CA1 inside a CA3 cluster: both under hippocampal pyramidal). **This is WHY the
KG is a graph: the parent's position bounds the child's search.** It structurally kills the over-split failure
where a tiny cortical fragment with only pan-neuronal signal (Snap25/Nrgn) force-fit a distant region.

**★ FRAGMENT-ESCAPE + label-follows-own-evidence (helper 2026-06-18; rule 2026-06-19).** The constraint above
assumes the parent was resolved correctly. When a parent is a **weakly/mis-resolved floor** (its own `excl_on==0`,
a markerless region/class force-fit — e.g. a cortical cluster mislabelled `midbrain glutamatergic neuron`), a child
must NOT blindly inherit that floor. The helper **fragment-escape** already relaxes `allow` when a child scores a
DIFFERENT region whose DEFINING marker is ON and whose score beats the parent-region — so the child's
`graph_evidence` will then surface its true region. **Your labeling rule: a child's label follows ITS OWN
`graph_evidence` (verdict/dominant_region/top anchor), never the parent's inherited floor when they differ.** If a
child of a "midbrain" parent has `verdict=clear_top` on `L4 IT … neocortex` (Cux2/Rorb ON), label it L4 IT — the
parent floor was the error. Inherit the parent label ONLY when the child has no resolvable identity of its own
(class_or_unknown) AND its markers are consistent with the parent.

- When a parent is split and **all** (or a contiguous run of) its new sub-leaves get the **same `top`
  identity** in Step 2 → give them all that label. They are one population Leiden happened to fragment;
  in the output (same label = same annotation) they **merge**. Do not split them further (`split_no_gain`).
- This is also the stop rule: a split that yields one identity across its children gained nothing →
  terminalize at that identity. A split that yields ≥2 distinct identities is real → keep each.

**★ A subtype must be DIFFERENTIALLY supported — the over-split rule (2026-06-23).** The merge above keys on the
*label*; this rule keys on the *evidence*, and catches the harder case of two **different** labels that are really one
population. **CASE — "magnitude-only subtype":** a leaf is given a subtype S (a child of class C), or sibling leaves
get different subtypes of one C, but no marker that is *specific to S vs its parent/sibling* is actually ON — the only
difference is a higher *level* of markers S SHARES with C. **RULE:** a subtype is valid only with a **differential
discriminator** — ≥1 marker that is ON *and* distinguishes S from C (a marker of S that is not a marker of C, not
equally ON in the sibling). Present → keep the subtype (L4=Rorb, CA3=Grik4, D1=Drd1, Opalin⁺ oligodendrocyte, liver
periportal=Cyp2f2 — all carry a real differential). Absent → the leaf collapses to the validly-supported **parent
class**, and same-identity siblings merge. This is enforced at `check` by **`DIFFERENTIAL_UNSUPPORTED`** (intrinsic
subtypes) and **`CONTEXT_AXIS_LONELY_POLE`** (context-axis poles, judged cross-leaf so genuine two-pole zonation
survives) — §9. The rule is wing-agnostic and data-driven (no per-celltype hardcoding): the *data* decides whether a
distinction is real by whether its differential marker fires.

**★ Resolution is CONVERGED, not predicted (2026-06-15).** Leiden γ has NO closed-form map to #clusters or
*which* subtypes split — it is data-dependent (Traag 2019; γ in CPM = a density threshold, in modularity it
carries a resolution-limit √(2m) that UNDER-resolves sub-types inside a LARGE cluster). So there is no formula
γ(k); the right granularity is **converged**, using the pipeline's unique oracle — the IDENTITY check.
- **PER-CLUSTER resolution, not one global value.** Each split candidate carries its OWN `start_resolution` =
  `_res_for(k)` = **0.25·k** (k = that cluster's expected #types), **floor 0.1, ceil 2.0** (k=2→0.5, 4→1.0, 6→1.5).
  Pass `finalize.resolution_map` as `--resolution '{"2":0.5,"15":1.0}'` so each flagged cluster splits at ITS own
  resolution in ONE call (a heterogeneous L2/3+L4 blob gets higher γ than a clean 2-way doublet). A scalar
  `--resolution 0.6` still works (applies to all). The old single `0.12·n` (capped 0.5) UNDER-split large blobs.
- **Start LOW to avoid over-fragmentation, raise where mixtures persist.** High γ on sparse 2µm bins makes many
  ambient fragments (→ Unknown noise). So start at the scaled value and let the round loop ADD resolution only
  where the data demands it — the real mechanism is the convergence, not the seed.
- **Converge across rounds** (the seed is non-critical because of this): after a split, (a) a sub-leaf still
  flagged `SHOULD_SPLIT_MIXTURE` / `DF_MULTI_SUBTYPE` (≥2 subtype markers co-present) ⇒ still under-resolved →
  raise γ **×1.5** and re-split it; (b) sub-leaves with no distinct identity (`split_no_gain`) ⇒ over-fragmented →
  same-label sub-leaves MERGE in the output (§8 above), stop. You converge γ toward marker-purity, not guess it.

### 8.1 Split-review gate at finalize (caller asks the user once more)
`finalize` flags **large leaves that could still split into real normal subtypes** (decided `celltype`,
`n≥--large-min-n` default 1500, high density `low_frac<0.6`, `reason_code≠panel_limited`, depth
`≤--max-split-depth` default 1) in two cases — each entry carries `split_kind`:
- **`children`** (`subclass_resolved=false`): **≥2 normal children resolvable here**.
- **`siblings`** (`subclass_resolved=true`): the chosen subtype co-occurs with **≥1 other normal sibling
  subtype** in `subtype_groups` (a minority sibling mixed in, e.g. CA1 inside a CA3 leaf — invisible to
  absolute `identity_scores` by dilution, so this gate is the backstop).
**Ambient guard (both kinds):** a child/sibling counts ONLY if its **DEFINING marker is fold-ON in the leaf** (`df_present`). Shared/ambient markers (KDE neuropil bleed) make `subtype_groups`/`children_resolvable` list false siblings (e.g. BLA/PAG/thalamic in a cortical leaf, CA3 in a DG leaf); these are filtered out, so the gate no longer forces spurious extra rounds on a correctly-resolved single region.
**★ BIMODAL backstop (`bimodal_subtypes`, 2026-06-23) — a THIRD trigger for balanced sibling-region mixtures the two above miss.** The mean/fold-based `mixture_subtypes` (and the `children`/`siblings` gates) miss a *balanced 50/50* mix of two **sibling-region** subtypes: each pole's markers are diluted below the fold/auc bar AND the 2nd pole sits OUTSIDE `dominant_region`, so the leaf terminalizes at class_floor and never enters the γ-escalation loop (arterial+venous EC lumped as `endothelial cell`; alveolar+adventitial/peribronchial fibroblast lumped as `fibroblast`). `_bimodal_split` (helper, only when `verdict≠clear_top` AND `mixture_subtypes` empty) tests **PER-CELL** whether ≥2 region-groups under the lineage have **ANTI-CORRELATED** programs (corr≤−0.3) each high in a **disjoint** cell subset (≥15% each), with the same ≥2-distinct-low-share-exclusive-markers and disease-context=normal guards. This is region-INDEPENDENT (does not touch region-gating) so it is purely ADDITIVE; continuum co-expression (corr≥0: SMC↔pericyte) and dropout clusters (no high subpop: low-UMI lymphocyte) are NOT flagged. A `bimodal_trigger` leaf becomes a split candidate; the (already-existing) round loop then γ-escalates and `converged_stop` terminates it honestly if it turns out non-separable.
These go into `_annotate_finalize_review.json` and **HOLD** (no write). The caller
asks the user split-or-accept: **YES** → `subcluster --paths <review.paths> --resolution
<review.recommended_resolution>` → you re-judge → finalize again; **NO** → `finalize … --accept-large`.

---

## 9. Validation loop — you own every fix (`check` / `finalize`)

After you write decisions, the caller runs `check`; it writes `_annotate_violations.json`
(`{cluster_path, check, severity, message, suggestion}`). helper **changes nothing** — **you** revise
only the flagged clusters and the caller re-runs `check` until clean. Violation types:
`MISSING_DECISION` · `NEEDS_SUB_REMAINING` (give a terminal call per Step 5 — never Background) ·
`BAD_TERM_ID` (fix the id or change the decision) · `UP_CROSS_LINEAGE` (≥2 lineages in `up` but single
celltype — chimera? split or re-check) · `REQUIRED_ABSENT_ONPANEL` (the term's own non-pan **on-panel**
required is absent — relabel to a type whose required/experience is present, or a broader class; types
whose required are all off-panel, e.g. CA1, are exempt and identified by supporting/experience).
**★ This is a HARD block, not advisory (2026-06-19): if a subtype's DEFINING/REQUIRED on-panel marker is OFF
(e.g. `dopaminergic neuron` needs `Th`/`Slc6a3` and both are OFF), you may NOT label it that subtype by
reading raw `up`/`hp` markers — that is exactly the force-fit this gate exists to stop. NEVER override the
engine's `graph_evidence.verdict=clear_top`/`dominant_region` with a hand-picked term whose own required is
absent; the engine already rejected it. A non-canonical claim (e.g. dopaminergic neurons in the medial/lateral
habenula, where the standard biology is cholinergic-glutamatergic) needs the type's OWN required markers ON,
not just a shared one like `Nr4a2`/`Ddc`. Follow the engine call or fall to the common class.** ·
`SPEC_UNSUPPORTED` (a `subclass_resolved=true` subtype whose chosen term FAILS the valid-evidence gate —
`spec_pass==false`, < half its own specific markers ON; a force-fit through shared/leaky markers, e.g. a
non-hippocampal leaf labelled CA1 by sharing `Wfs1`, or a non-CA3 leaf on CA3 via one shared marker → drop to
class-floor / Unknown, or keep ONLY if `hp`/experience show the type's own markers dominate) ·
`SHOULD_SPLIT_DEEPER` (≥2 child subtypes resolvable → split) · `SIBLING_RESOLVABLE` (the chosen subtype
co-occurs with ≥1 normal sibling subtype, e.g. CA1 mixed into a CA3 leaf → `needs_subclustering` unless a
split would not separate them) · `DF_MULTI_SUBTYPE` (≥2 subtypes' DEFINING markers are **fold-ON** in this
leaf — a fold-based signal that fires even on **sparse/2µm** data where strict-ON quiet markers miss it; e.g.
CA1+CA2+CA3 co-present in one hippocampal cluster → split it **in-round**, don't wait for the finalize gate) ·
`SHOULD_SPLIT_MIXTURE` (graph_evidence.`mixture_subtypes` ≥2 — the dominant region's child subtypes each have
their OWN exclusive marker **WEAKLY co-present** (auc≥0.6, below strict-ON), i.e. an **under-clustered mixture**
whose diluted markers no single anchor caught: cl2=D1-MSN`Drd1`+D2-MSN`Drd2`, cl15=L2/3`Cux2`+L4`Rorb`, deep-layer
L5+L6. → `needs_subclustering` at a **sufficiently high resolution** to separate them (over-splitting is safe —
same-label sub-leaves merge; under-splitting is lossy). A cluster with NO subtype's own marker present is NOT
flagged ⇒ no noise) ·
`GLIAL_MISLABEL` (**★ ERROR — blocks finalize**: the leaf's vs-rest `up` top-8 is glial/vascular-DOMINANT
(≥3 of oligo/astro/microglia/vascular/choroid/ependymal markers) with **NO** neuronal/pan-neuronal/own-identity
marker, yet labelled a NEURON — a fiber-tract/vascular/glia fragment that the parent-constraint force-fit into a
neuronal subtype. **Relabel to the real non-neuronal type read from `up`** (oligodendrocyte/astrocyte/microglial
cell/endothelial cell/pericyte/vascular leptomeningeal cell/VSMC/choroid plexus/ependymal — label is FREE;
graph_evidence may show only neuronal anchors because of the parent-constraint, but `up` decides). Real neurons
with ambient myelin are exempt (`up`=Cplx3/Nxph3 own markers, not absolute Plp1/Mbp). Must be 0 at finalize.) ·
`INTERNEURON_MISLABEL` (warn: `up` top-6 has a GABA transmitter (Gad1/Gad2) **ENRICHED** + ≥2 interneuron-subclass
markers (Pvalb/Sst/Vip/Erbb4/Pthlh…) but the label is **glutamatergic** — a cortical GABA interneuron mis-routed to
a glut layer because ambient Slc17a7 drove the lineage gate. If `up` is interneuron-dominant → relabel to
**cortical GABAergic interneuron** (Pvalb+/Sst+/Vip+); if the glut identity is clear and Gad is ambient (L4/L5 with
dense PV neighbours) keep glut and say why.) ·
`OFF_EVIDENCE` (warn: the chosen term is **NOT in `graph_evidence.anchors`** and sits in a DIFFERENT region than
the engine's `clear_top` anchor — a raw-marker override of the engine. Justify with the term's own-identity
evidence, or take the engine's anchor.) ·
`DEFINING_ABSENT` (warn: the chosen term is `df_absent` with `excl_on<2` — it rests on a single non-defining
marker, a likely force-fit. Prefer a competitor whose defining/exclusive markers are ON, or drop to the class.) ·
`CONFUSION_UNRESOLVED` (the chosen term and its documented ConfusionPair partner both anchor — resolve per the
3-way `confusion_unresolved.action` in §5: split / resolve / Unknown; never force-fit one side.) ·
`RS_LT2_BUT_RESOLVED` / `NEGATIVE_LEAK` (reconsider as the suggestion hints) ·
**`CONTEXT_AXIS_SUBTYPE`** (a `context_axis` leaf left at the bare lineage class though a `dominant_pole` exists →
relabel to `suggested_subtype`, reason_code=context_axis; §4.3) · **`CONTEXT_AXIS_SPLIT`** (a `context_axis_mixed`
parent left at the bare class → `needs_subclustering` to separate the poles; §4.3) ·
**`DIFFERENTIAL_UNSUPPORTED`** (warn — the **over-split antidote**, §8: the chosen **subtype** term has discriminative
markers that distinguish it from its **parent class**, but NONE are ENRICHED here — the apparent subtype rests only on
a higher *magnitude* of markers SHARED with the parent/siblings, i.e. a level-only over-split. **Relabel to the parent
class** named in the suggestion; same-identity siblings then merge. A *real* subtype keeps its label because its own
differential marker is ON — L4=Rorb, CA3=Grik4, Opalin⁺ oligodendrocyte all survive; only `Protoplasmic astrocyte`
with `Sparcl1`/`Plcd4` OFF, etc., collapse to `astrocyte`.) · **`CONTEXT_AXIS_LONELY_POLE`** (warn — the context-axis
counterpart of the same rule: the leaf is labelled one **pole** of a context axis, but the OPPOSITE pole is assigned to
**no** cluster anywhere — the axis did not actually separate anything, so this is a single-pole noise-lean, not real
zonation. **Relabel to the parent class**; the pole returns only when both poles genuinely appear. This is a cross-leaf
check, so true zonation where both poles are present — liver periportal↔pericentral — is preserved.). The `suggestion` is
advisory; the final call is always yours. At `finalize`, only **error**-severity violations block (override
`--force`); **warn** are recorded, non-blocking.

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
  "reason_code": "multi_identity | sibling_resolvable | pan_class_only | class_floor | split_no_gain | panel_limited | sibling_ambiguity | transition_background | ''",
  "provisional_subclass": "<best-guess subclass if class-final, else ''>",
  "key_markers": ["geneA","geneB"],
  "rationale": "<condensed S1–S7: identity_scores.top n_on/cov/spec + hp/experience corroboration + why>",
  "alternatives_considered": "<other ranked/candidate terms: why rejected>",
  "open_question_flag": false
}}
```
**Label hygiene (hard):** clean literature name only — no `(n=…)`, `"?"`, `"possible"`, reason text,
or `+` symbols. Unknown→"Unknown", Background→"Background". Reasoning goes in `rationale`/`alternatives_considered`.

### 10.2 `{out}/expert_judgment_round{k}.csv`
columns: `cluster_path,n_cells,decision,label,term_id,subclass_resolved,confidence,reason_code,key_markers,rationale`.

### 10.3 `{out}/_annotate_recommendation.json`
`{"resolution": <float 0.1–1.5>, "rationale": "<≤200 chars>"}` — for the next Leiden round. **Bias
toward higher resolution** (split freely; merge fixes over-splitting). The split mechanism is always Leiden.

### 10.4 Final layout after `finalize` (helper does this automatically)
On `finalize` the helper writes `{input}_annotated.h5ad` into `{out}` (= `annotation_out/`), then
**packs every other artifact** (`_global_stats.npz`, all `_annotate_*.json`, `_ontology_index.json`,
profiles, per-round CSVs) into a single `annotation_artifacts.tar.gz` and deletes the loose files.
**`annotation_out/` ends with exactly three things:**
- `{input}_annotated.h5ad` — the result (uncompressed)
- `expert_judgment_annotation.csv` — kept loose (**merge step reads it as input**)
- `annotation_artifacts.tar.gz` — everything else, compressed

---

## 11. Self-check — run BEFORE writing each decision (helper validates the same)
1. Every gene in `key_markers` appears in this leaf's `up ∪ hp ∪ parent_ctx ∪ ontology_gene_table(present)`.
2. `celltype` ⇒ `term_id` exists in `_ontology_index.json`; `label` == that term's `L` (never invent).
3. You did NOT deny `identity_scores.top` / a clear absolute identity because `up`/`sibling_contrast`
   was flat — that is the demoted local contrast; absolute + comprehensive evidence wins.
4. You DID corroborate the call across ≥2 axes (absolute + one of hp / ontology-mean / lineage_scan /
   experience) — no single-field over-call.
4b. A *subtype* `celltype` (subclass_resolved=true) has `spec_pass==true` for its term — ≥ half its OWN
   specific markers actually ON, not just a broad/leaky shared marker (Wfs1/Crym/Pcp4). The helper nulls a
   `spec_pass==false` `top`; if you label a `spec_pass==false` term anyway, you have defended it from
   `hp`/experience (the type's own markers dominate). Otherwise ⇒ class-floor or Unknown.
5. No unexplained negative leak (`neg>0` on the chosen top, or N marker AUC≥0.6 / hp pct_in≥0.9) —
   except pan-neuronal modality for neuron terms. `neg>0` ⇒ usually `needs_subclustering`.
6. `needs_subclustering` mechanism is `subcluster` (Leiden) — there is no signal_split.
7. Before `Unknown`: `alternatives_considered` lists ≥3 candidates each with a concrete rejection.
8. Sub-leaf ⇒ `parent_ctx` pooled; the same `term_id` never carries two different labels.
9. `needs_subclustering`+`multi_identity` ⇒ `identity_scores.mixed==true` (≥2 **co-dominant** lineages),
   NOT `flags.multi_lineage` or ambient glia/interneuron bleed alone (§4.1).
10. You did NOT send a single-dominant-lineage cluster to `needs_subclustering` merely because no subtype
   passed the gate — used the dominant subtype (fold-aware) or `class_floor`, and split ONLY when ≥2
   specific subtypes genuinely co-resolve (§4.2).
11. ★ Before writing `class_floor` or `Unknown`, you RAN the §2.5 STATE CHECK — inspected `identity_scores.ranked`
   for `is_state=true` and the `disease_candidates` block, and confirmed no state term passes the Step-2 test
   (`spec_pass`, `neg==0`, beats the normal alternative). A `null` `identity_scores.top` was NOT read as "→ class":
   you checked `ranked` for a state winner. (This applies whether or not `--disease` was passed.)

## 12. Report to the caller (after init/subcluster, then STOP)
Print: round number, #new leaves, decision tally, cumulative tally, the FLAGGED
(`needs_subclustering`) paths, and the recommended resolution. The split mechanism is always Leiden.
Do NOT ask the user. Background clusters are excluded from FLAGGED. At **finalize**, if helper HOLDs
(split-review gate, §8.1), relay `split_candidates` + `recommended_resolution` to the caller.

## 13. Forbidden (each has caused real misannotations)
- Using `signal_split` / any marker-guided forced assignment (does not exist here — causes contamination).
- **Denying a clearly-present absolute identity because vs-sibling contrast (`up`) is flat** — that
  collapse is exactly what this paradigm fixes.
- Reducing the judgment to a single rigid gate (`identity_scores.top` alone, or `required`-markers alone).
  Synthesize ALL axes (§4) with expert knowledge; trust the cluster's actual enriched genes over a poor
  ontology marker list (the experience axis is there for exactly this).
- Collapsing a STRONG subtype to a bare class (only Step 3 `class_floor`/`split_no_gain`/`panel_limited`,
  i.e. when no STRONG subtype exists, justifies a bare class — never override a STRONG subtype).
- **Calling a chimera (`multi_identity`) on KDE/neuropil AMBIENT bleed** (glia/interneuron/vascular co-signal)
  when one lineage clearly dominates — §4.1. Decide chimeras from `identity_scores.mixed`/`codominant_lineages`,
  NOT from `flags.multi_lineage` or a few ambient markers in `up`.
- **Splitting a single-dominant-lineage cluster just because no subtype passed the gate** (`pan_class_only`).
  That tiles a continuous spatial gradient into meaningless fragments (§4.2). Terminalize at the dominant
  subtype (fold-aware) or its class (Step 3); split ONLY when ≥2 specific subtypes genuinely co-resolve.
- Over-splitting anxiety in the OTHER direction: when ≥2 identities genuinely co-resolve, do split — merge (§8) cleans redundancy.
- Assigning a term whose **on-panel required** markers are absent on a pan marker alone; or rejecting a
  type for an **off-panel** required marker (panel-blind fallback — use supporting/experience).
- **Force-fitting the nearest specific term when its OWN specific markers are mostly OFF** (`spec_pass==false`,
  §4 lesson 3) — e.g. labelling a hypothalamic/amygdalar leaf "CA1" because it shares `Wfs1`/`Crym`, or a
  non-CA3 leaf "CA3" via one shared marker, while `Fibcd1`/`Spink8` (CA1) or `Grik4`/`Cpne4`/`Nectin3` (CA3)
  are flat. A real type has ≥ half its own specifics ON. `spec_pass==false` ⇒ the dominant-lineage **class**
  (`glutamatergic neuron`) if the lineage is clear, else **Unknown** — never the leaky specific subtype.
- **Labelling a specific celltype with no valid positive evidence at all** rather than saying Unknown — a
  confident wrong region (CA1 on a non-hippocampal cluster) is worse than an honest class/Unknown.
- ★ **Burying a `tier=state` identity under the bare lineage class (or under Unknown) because you stopped at
  `identity_scores.top==null` / `verdict=class_or_unknown` without running the §2.5 STATE CHECK** — states are
  excluded from `top`/anchors BY DESIGN, so a `null top` is EXPECTED when a state is the answer; the winner is in
  `ranked` flagged `is_state` and in `disease_candidates`.
- ★ **Dismissing a candidate "because it is a state", "because it's an on-chip/in-vitro/culture program", or
  "because no `--disease` was declared".** None of these is a rule. State terms are first-class labels; `--disease`
  is only a HINT (often withheld on purpose to test unaided recall). The marker evidence governs (§2.5, §5,
  disease handling). Assign the state term when its OWN markers dominate; keep the normal type when they don't.
- Forcing a chimera into one label; collapsing co-dominant lineages; picking the higher-AUC of two close
  passing siblings (→ needs_subclustering).
- Concluding `Unknown` without enumerating ≥3 candidates; downgrading a confident class to Unknown.
- Inventing `term_id`/`label`, putting markers/parentheses/reasons in `label`, writing the h5ad (only
  finalize/helper), or asking the user (only the caller).

---

## 14. Worked examples (copy this reasoning shape)

**A — strong absolute identity, survives fragmentation → celltype.** A 150-cell Leiden fragment;
`identity_scores.top`=CA3 (n_on=4, cov=1.0, neg=0); `hp` shows Grik4/Npy2r/Cpne4 high; `up` AUC≈0.5 (flat
vs siblings — DEMOTED, ignored for identity); `children_resolvable` empty.
S3 absolute strong + S5 hp corroborates + clean negatives → **celltype, CA3 pyramidal, subclass_resolved=true,
high.** (The flat `up` does NOT veto it — that is the collapse this paradigm prevents.)

**B — ontology required is wrong for the panel; experience rescues.** A cortical fragment; ontology L2/3
`required=[Slc17a7,Cux2,Rorb]` but Rorb is an L4 marker and Cux2 weak here; `hp`/`up` show **Lamp5, Dact2,
Otof, Nectin3** high — the term's **experience (`E`)** markers; `lineage_scan`=glutamatergic.
S5 synthesis: required is a poor panel fit (panel-blind), but the **experience axis** matches strongly →
**celltype, L2/3 IT, medium** (note in rationale: called on experience markers, not required). Do NOT fall
to bare class just because `Cux2/Rorb` required didn't fire.

**C — pan-class only, large: split ONLY if ≥2 subtypes co-resolve.** Top-level glutamatergic blob (n=36k);
`hp` Slc17a7 strong; `flags.pan_class_only`. Branch on the §4.2 split test:
- If `candidate_subtypes.children_resolvable` lists **≥2** subtypes (e.g. CA1 + CA3 + layer markers all
  present) → **needs_subclustering, pan_class_only** (a real mega-blob; fragments then score single identities
  and merge). 
- If only ONE dominant-lineage subtype resolves (fold-aware) → **celltype = that subtype**. If NONE resolves
  → **celltype = "glutamatergic neuron" (bare class), `class_floor`** — do NOT split a single homogeneous
  region (that would tile its gradient, §4.2).

**D — GENUINE chimera (co-dominant) → needs_subclustering.** `up`=[Mbp 0.81, Plp1 0.78, Aqp4 0.74, Gfap 0.70,
Cx3cr1 0.66]; n=17k. Oligo(Mbp/Plp1) + astro(Aqp4/Gfap) + microglia(Cx3cr1) are all STRONG with **comparable
spec** → `identity_scores.codominant_lineages`=[oligo,astro,microglia] (≥2 co-dominant) → **needs_subclustering,
multi_identity.** (Contrast with G: here the glia are CO-DOMINANT, not a weak bleed under one dominant lineage.)

**E — over-split → merge.** Parent split into 7 sub-leaves; six score `top`=CA3 (Step 2), one scores DG
granule. → label the six **CA3 pyramidal** (they merge in output), the one **dentate gyrus granule cell**;
do not split the CA3 six further (`split_no_gain`).

**F — Background.** `up` empty; density.pct=6, low_frac=0.7; specificity_peak null; lineage_markers.n=0;
`identity_scores.top` null; n=120 (final round). S0 → **Background.** (If n≥500 at init → needs_subclustering.)

**G — ambient bleed is NOT a chimera (single dominant lineage) → celltype, not split.** A coherent cortical
band (n=18k); `hp`/`pct_in` dominated by glutamatergic signal; `up` also shows Pvalb/Lamp5/Cck (sparse
interspersed interneurons), Slc1a2/Aldoc/Apoe (astrocytes), Mbp/Plp1 (myelin bleed). `flags.multi_lineage`=
[glut,GABA,astro,oligo] — but `identity_scores.dominant_lineage`=glutamatergic and `codominant_lineages`=
[glutamatergic] only (the others' `spec` ≪ ½ glut) → `mixed=false`. §4.1 → **NOT a chimera**; the glia/
interneuron signal is KDE/neuropil bleed. Proceed to Step 2/3 on glut: if a layer subtype resolves
(fold-aware) → that layer `celltype`; else **"glutamatergic neuron", `class_floor`.** Do NOT split.

**H — single layer region, subtype markers sub-threshold → class-floor (NOT split).** Top-level cortical
band (n=23k), spatially one contiguous strip; layer marker Rorb (fold 4×, detected 70%) exceeds mu+sd in
only ~24% of bins (fails the absolute gate) but **passes the fold-aware ON** (log2≥1.5 ≈3× AND detected≥30%).
`identity_scores.dominant_lineage`=glutamatergic, `mixed=false`. Now synthesize: if Rorb + a co-marker
corroborate an upper-layer IT subtype (and its negatives are clean) → **celltype = L4/L2-3 IT (medium)**.
If `top` names a different glut subtype that `hp`/negatives do NOT corroborate (a shared-broad-marker
artifact, e.g. a hippocampal or brainstem glut term in a cortical band) → reject it and fall to
**"glutamatergic neuron", `class_floor`** — single label. Either way **never `needs_subclustering`**:
splitting one homogeneous band just tiles its gradient (§4.2).

**I — force-fit via shared/leaky markers → valid-evidence gate → class-floor (NOT the specific subtype).**
A non-hippocampal glutamatergic leaf (e.g. a hypothalamic/amygdalar region, n=13k). `up` top = `Wfs1, Crym,
Gpr101, Pdyn, Cartpt, Dlk1, Penk` — its real signature is neuroendocrine; but `Wfs1`/`Crym` are also listed
on CA1 terms, so `identity_scores` offers ONLY hippocampal candidates (CA1 dorsal, subiculum, …). The dec-top
is `CA1 dorsal pyramidal` but **`spec_cov≈0.33` (`spec_pass=false`)** — CA1's OWN specifics `Fibcd1`/`Spink8`
are flat (`hp` has none of them). The helper has **nulled `top`**. S2: no validly-supported subtype. S3: the
dominant lineage IS clear (glutamatergic) → **celltype = "glutamatergic neuron", `class_floor`, low**, noting
in `rationale` that CA1/subiculum were force-fits (shared `Wfs1`, specific `Fibcd1/Spink8` absent). Do NOT
label CA1. If even the lineage were unclear → **Unknown** (Step 4). Contrast with **A**: a real CA3 has
`Grik4/Cpne7/Npy2r` (its own specifics) ON (`spec_cov≈0.71`) → `spec_pass=true` → keep CA3.
