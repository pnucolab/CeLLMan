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
| **caller** | Between rounds asks the user (Yes/No split + next resolution) and re-invokes helper. |
| **helper.py** | All compute + the final h5ad write. **Validates** your decisions (`check`/`finalize`) and returns a list of violations for you to re-decide — it **never silently changes a decision**. |

## 2. Modes (one skill invocation = one round)

| mode | when | you do |
|---|---|---|
| `init` | first call (no state) | helper computes & **persists global per-gene stats** (`_global_stats.npz` — the absolute-identity reference reused by EVERY later round, so identity never collapses on small fragments), fetches the ontology, profiles the input's existing Leiden clusters; you judge each. **No h5ad write.** |
| `subcluster` | caller-approved split | **unsupervised Leiden** within each flagged parent (`--resolution` from finalize `start_resolution`=`0.25·k`, scaled to heterogeneity, converge up §8; GPU via `--gpu`). New sub-leaves are profiled against the SAME global reference; you judge them. **This is the only split mechanism.** |
| `check` | after you write decisions, before finalize | helper validates → `_annotate_violations.json`; you re-decide any flagged clusters until clean. **helper changes nothing.** |
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
{python} {HELPER} init --h5ad {input_h5ad} --cluster-col {cluster_col} \
   --out {output_dir} --species {species} --organ {organ} [--disease "{disease}"]
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
(schema: `{terms:{id:{L,T,P,R,S,N,E,D,ES,DF}}, cp, oq}`; L=label, T=tier, P=parent ids,
R=required, S=supporting, N=negative, E=experience markers (soft hint, observed in ≥1 real
dataset), **ES=strong-consensus experience** (observed across ≥2 datasets — counts like
supporting in the gate), D=disease_context, **DF=defining** (a single one is sufficient)).

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
rounds. "Couldn't name a subtype" ⇒ name the class, not split.

---

## 5. Profile fields — read once, carefully (`{out}/cluster_gene_profiles_round{k}.json`)

Before judging individual clusters, skim `{out}/cluster_overview_round{k}.json`
(`{path:{n, dominant_lineages, top_up[:6], specificity_peak}}`) — the side-by-side view a
human expert uses to know, e.g., which cluster is *the* microglia vs one of several sharing a
marker. Then per leaf:

| field | shape | how to use |
|---|---|---|
| **`graph_evidence`** ★ PRIMARY | `{lineage, lineage_ambiguous, codominant_lineages[], dominant_region[], verdict, anchors[≤8], cp[≤4], confusion_unresolved, mixture_subtypes[]}` where each anchor = `{term_id, label, tier, excl_on, excl_n, excl_markers_on[], neg_on, neg_markers_on[], df_on, df_absent, df_onpanel[], n_on, n_disc, path[]}` | **★ The redesigned identity substrate (§7) — REGION-COHERENT graph-descent over the KG you built (2026-06-13).** `lineage` = the class, by a **transmitter HARD gate**: the lineage whose canonical transmitter is ENRICHED (rejects ambient myelin Mbp → not oligodendrocyte) or, if none is enriched (cortical Slc17a7 is everywhere), the strongest PRESENT transmitter by expression — and **neuronal lineages take priority when any neuronal transmitter is present** (a leaky glut marker like Wfs1 can never pull a Gad+ cluster into a hippocampal region). **`dominant_region[]`** = the dominant region-group(s) (neocortical / thalamic / hippocampal pyramidal …) under that lineage, chosen by expression-weighted enriched EXCLUSIVE markers; **anchors are restricted to subtypes of `dominant_region`** so a cross-region blip (CA2 in a thalamic cluster) is structurally excluded — low-abundance layer markers (Fezf2→L5) survive without any magnitude floor. When `dominant_region` is empty (heterogeneous/weak cluster) anchors fall back to all subtypes of the lineage. `lineage_scan` corroborates. **`lineage_ambiguous`** = `true` when **`codominant_lineages`** (≥2 lineages of comparable transmitter strength — glut+GABA doublet / co-release) — then do NOT descend confidently; suspect doublet→`needs_subclustering`, or stop at the common ancestor / bare class (§7 Step 1). **`anchors`** = specific terms (in that lineage) whose OWN **exclusive markers** are ON and negatives clean, **ranked by exclusive support**. **Exclusive markers = low gene-degree** (`share` = #terms listing the gene; degree 1 = exclusive/decisive, high = leaky/pan — NOT evidence for a specific child). `excl_on`/`excl_n` = how many of the term's exclusive markers are ON / exist on-panel; **`excl_markers_on`** = the actual ON exclusive genes (CA1 = `Fibcd1/Spink8`; CA3 = `Grik4/Cpne7`; CA2 = `Rgs14/Amigo2`). `path` = the term's ancestor chain (class → region-group → … → term) — use it to **STOP at a region node when the exact subtype is ambiguous**. `neg_on` = negative-marker leak (>0 ⇒ reject the term). **`df_absent` (DF-absent flag, 2026-06-15)** = the term has on-panel **defining** markers (`df_onpanel[]`, "a single one is sufficient") but **NONE is ON** — the anchor rests on non-defining markers (e.g. cl37 OB-mitral: `Cdhr1` ON but defining `Tbx21` OFF). This is an **advisory flag, NOT a demotion**: anchors are still ranked by `excl_on`/ratio/`n_on` first (df_absent only breaks an exact tie). *(A hard demotion was tried and reverted — it let the generic region node, which has no DF so is never df_absent, beat real cortical layers whose defining marker is merely quiet in sparse data → mass collapse to `neocortical glutamatergic neuron`.)* Use it as a **caution**: a `df_absent` anchor with `excl_on==1` (single non-defining marker) is a likely force-fit (`_validate` → `DEFINING_ABSENT`) — prefer a competitor whose defining/exclusive markers are ON; but a `df_absent` anchor with `excl_on≥2` (e.g. L4 = `Rorb`+`Whrn` ON, `Rspo1` quiet) is genuine — its required markers carry the identity, so KEEP it. (auc≥0.85 counts a quiet-but-enriched defining marker as ON, so a real `Grik4`→CA3 at auc 0.92 is not flagged.) **`verdict`** = `clear_top` (unique best anchor — descend to it) / `ambiguous_tie` (≥2 tie — split if distinct & strong, else stop at region/class) / `class_or_unknown` (no exclusive anchor — class or Unknown) / **`confusion_unresolved`** (the top anchor and a documented ConfusionPair PARTNER both anchor). **`confusion_unresolved`** field = `{pair, terms[], discriminators, action, mixture, resolves_to, top_distinct_on[], partner_distinct_on[]}` — resolved 3 ways by whether each member's OWN (non-shared) exclusive markers are ON: **`action="split"`** (BOTH members' distinct markers ON ⇒ a real MIXTURE of both cell types, e.g. cl64 = CA2 `Cacng5/Map3k15/Acan` + FC `Necab2/Scgn` both ON → **`needs_subclustering`**, then label each fragment CA2 / FC / Unknown / hippocampal class) / **`action="resolve"`** (only one side's distinct markers ON ⇒ the discriminator favors `resolves_to` → label that type) / **`action="unknown"`** (only shared markers ON, neither's distinct markers ⇒ truly unseparable → **Unknown**, never force-fit one) / `no_lineage`. **`cp`** = confusion pairs touching the path: APPLY their `discriminators` before committing. **This is gene-degree-aware, hierarchy-aware, and scRNA-compatible (no spatial): a leaf descends to CA1 ONLY if CA1's exclusive `Fibcd1/Spink8` are ON, not because it shares the leaky `Wfs1`.** See §7. |
| `identity_scores` *(DEMOTED — supplementary fold summary, NOT the decision; use `graph_evidence` for identity)* | `{top, ranked[≤5], class_fallback, mixed, n_strong_lineages, dominant_lineage, codominant_lineages}` where each candidate = `{term_id,label,tier,n_on,cov,spec,neg,spec_ok,spec_cov,spec_pass}` | **Absolute identity summary** (the backbone axis). `n_on`=how many of the term's non-pan discriminative markers are **ON**. **ON = exceeds global mean+sd in ≥30% of cells, OR (global fold ≥~3× [log2≥1.5] AND detected in ≥30% of cells), OR (rank-enrichment `auc≥0.85` AND the gene is a low-share specific marker, share≤3)** — the fold branch catches *quiet-but-enriched* markers; the **auc branch (RC2, 2026-06-14)** rescues markers so sparse they fail even detection-in-25% (2µm subcellular bins: `Rorb`→L4 auc 0.98, `Fezf2`→L5 auc 0.90, `Dcx`→immature-granule auc 0.72 — yet detected <25%) so L4/L5/immature-granule resolve. **GUARD: the auc branch applies ONLY to a term's OWN low-share (≤3) specific markers — broad/shared genes (`Slc17a7`/`Pcp4`) still need real detection, so it cannot force-fit via spillover.** `spec`=Σ log2-fold of ON markers; **`wspec`**=specificity-weighted spec (each marker weighted 1/#terms-listing-it); **`dec`**=the single most *decisive* marker = max(specificity×fold) — a unique high-fold marker (Fezf2→L5, Rasgrf2/Stard8→L2/3, Grik4→CA3) drives this. Developmental/reactive STATE terms (Neuroblast, Radial glia, reactive/aging/DAM/IFN) are excluded from `top`/co-dominance (shared broad markers, not discrete populations). **`top` is chosen by the DECISIVE marker WITHIN the robust `dominant_lineage`** — so one unique specific marker beats many broadly-shared ones (Tbr1/Satb2/Slc17a7). **DF-override**: if a **coherent** term (n_on≥2) in the dominant lineage has a **DEFINING (`DF`) marker** that is fold-ON, that term wins outright ("a single defining marker is sufficient" — Fezf2→L5 ET, Cux2→L2/3, Rspo1→L4, Foxp2→L6 CT, Cplx3→L6b, Prox1→DG, Wfs1→CA1, Grik4→CA3…). This is what lets the exact cortical layer / hippocampal field resolve despite shared broad markers. (The coherence requirement guards against a *lone* defining marker firing on ambient signal in an over-split fragment — e.g. it stops Rgs14→CA2 over-calls.) **DF-override corroboration (helper guards A+B, 2026-06-11):** a DF win additionally requires (A) the term's `dec` be competitive with the best same-lineage candidate (≥0.34× it), AND (B) if the term has on-panel high-specificity (low-share) discriminative markers, ≥1 must be ON. This stops a *leaky* defining marker — curated as one term's DF in the ontology but biologically also expressed elsewhere (e.g. `Wfs1` is listed only on CA1 yet is also present in cortex/amygdala) — from hijacking a higher-evidence competitor: a cortical/amygdalar leaf with `Wfs1` fold-ON but its specific `Fibcd1/Spink8` OFF is no longer mislabelled CA1 (the override falls through to the genuinely decisive candidate). If a term's specific markers are ALL off-panel, guard B is inert ⇒ panel-blind fallback preserved. **★ VALID-EVIDENCE gate on EVERY candidate (`spec_n`/`spec_on`/`spec_cov`/`spec_pass`, 2026-06-12) — the over-call antidote:** a specific celltype is only honest when the leaf actually expresses THAT type's OWN identifying markers, not when it merely shares a few broad/leaky markers. Per term: **`spec_n`** = its on-panel high-specificity (share≤SPEC_SHARE_MAX=3) discriminative markers; **`spec_on`** = how many are ON; **`spec_cov`** = `spec_on/spec_n`; **`spec_ok`** = `(spec_on≥1 OR spec_n==0)` (the DF-guard-B threshold); **`spec_pass`** = `(spec_cov≥0.5 OR spec_n==0)` — i.e. **≥ half of the type's own specific markers are present.** A leaky marker carried as plain `Required`/`DF` (e.g. `Wfs1` is `R`+`DF` on 6 CA1 terms, so its share=6 keeps it OUT of the spec set) does NOT count toward `spec_on`. So a NON-hippocampal leaf that merely shares `Wfs1`/`Crym` scores `spec_cov≈0–0.33` for CA1 (its `Fibcd1`/`Spink8` OFF) while a real CA1 ≈1.0; a non-CA3 leaf ≈0.29 for CA3 (`Grik4`/`Cpne4`/`Nectin3` OFF) while a real CA3 ≈0.71. **The helper ENFORCES this: if the dec-selected `top` is a SPECIFIC (non-class) term that FAILS `spec_pass`, `top` is set to `null`** ⇒ this leaf has no validly-supported subtype → fall to its dominant-lineage **CLASS** (`class_fallback`, §7 Step 3) or **Unknown** (Step 4). The rejected candidate stays in `ranked` (marked `spec_pass=false`) so you can RESCUE it ONLY if `hp`/experience clearly show the type's OWN markers dominate; `ranked` lists `spec_pass=true` terms first. (`spec_n==0` ⇒ all specific markers off-panel ⇒ gate INERT, panel-blind: judge by `E`/`ES`/supporting per §4 — this preserves cortical layers whose required are off-panel.) **Principle (§4 lesson 3): a present `top` has valid evidence; a null `top` means "no subtype is validly supported" → name the class or Unknown, never force-fit the nearest specific term. Still corroborate a present `top` with `hp`/negatives/anatomy; if it names the wrong subtype of the RIGHT lineage, fall to `class_floor`, never to a split.** `cov`=n_on/(its markers); `spec`=Σ global log-fold of the ON markers (real specificity, pan excluded); `neg`=its negative markers that are ON (a leak ⇒ wrong/contaminated). **STRONG** = `n_on≥2 and neg==0`. *strength-before-depth* applies (a deep subclass outranks a class only when itself STRONG). **`dominant_lineage`** = the strongest lineage-root present; **`codominant_lineages`** = lineage-roots whose strength ≥½ the dominant's; **`mixed`= (≥2 codominant lineages)** — i.e. a *real* chimera, NOT mere ambient bleed (§4.1). **The co-dominance/`mixed` test counts ONLY valid-evidence terms (`spec_pass` AND `spec_n>0`)** — the same gate that nulls a force-fit `top` — so a force-fit term (shared/leaky markers) or a panel-blind term CANNOT inflate a false second lineage (a single striatal cluster sharing `Hpca`/`Wfs1` with hippocampal glut terms is NOT flagged `mixed`). It falls back to all-coherent only when NO term qualifies. `class_fallback`=dominant lineage's class term. **Treat as one strong input — corroborate with the fields below.** |
| `hp` | `[[gene, mean_in, rank, pct_in], …]` top 50 | **Absolute expression** (HK removed): what this cluster most strongly/broadly expresses. A gene absent from `up` is NOT absent — broad/uniform markers fall below the 0.6 AUC cutoff but show here. The decisive cross-check when `up` collapses. |
| `up` | `[[gene, auc, mean_in, mean_out], …]` AUC≥0.6, desc | **vs-rest contrast** (local in subcluster rounds = vs siblings). Specificity + sub-structure signal. **Demoted for identity** — do not let a flat `up` veto a clear absolute identity. |
| `dn` | same, AUC≤0.4 | depleted; supports ruling OUT (not a hard gate — KDE smear). |
| `ontology_gene_table` | `{gene:{rank,mean,auc,panel}}` | every R/S/N/E/ES marker of this wing with its **absolute mean** + AUC + panel flag. `panel=false` ⇒ off-panel (not measured): you may NOT use its absence to reject a type (panel-blind fallback). Use the **mean** (absolute) to confirm a marker is genuinely on even when its AUC is ~0.5. |
| `lineage_scan` | `{lineage: [[gene, auc, pct_in], …]}` | canonical lineage markers (incl. pan) regardless of cutoff → the **class-level identity** when no subclass resolves. Authoritative for "is this lineage present". |
| `candidate_subtypes` | ranked `[{term_id,label,tier, n_req_present,n_nonpan_present,n_req,n_req_offpanel, required:[[gene,auc,present,on_panel,is_pan]], children_resolvable:[…]}]` | the ontology terms whose defining markers actually appear here, with per-term evidence (parent-pooled for sub-leaves so an over-split homogeneous subtype is not lost) and **`children_resolvable`** = 1-level children whose distinguishing markers are present → the **split-deeper** signal. |
| `identity_evidence` | `{density{median,pct,low_frac}, specificity_peak, lineage_markers_present{n,genes}, housekeeping_share, cross_lineage{…}}` | raw signals for the Background test (§7 Step 0). |
| `subtype_groups` | `[{parent_id,parent_label,child_term_ids,children}]` | parents whose ≥2 finer sibling subtypes co-occur here → maximal-depth recursion target. **Also the sibling split-deeper signal**: if your chosen `top` is one of the `child_term_ids` and ≥1 OTHER normal sibling is co-listed, the leaf mixes two subtypes (e.g. CA3+CA1) → split (Step 2 sibling check). |
| `parent_ctx` | `{path, up[:10], hp[:10]}` | parent signature. **For sub-leaves you MUST pool parent_ctx** — within-parent re-ranking erases the shared class signal. |
| `sibling_contrast` | `[[gene, pct_in, max_sibling_pct, delta], …]` | within-parent discriminators — for deciding splits / choosing between close siblings, not for denying identity. |
| `flags` | `{multi_lineage, layer_mixing, pan_class_only, background_score}` | pre-computed split signals (corroborating, not deciding). **`multi_lineage` is computed from raw `up` and is AMBIENT-PRONE** — KDE/neuropil bleed (glia, interspersed interneurons) makes it list lineages that are not real co-populations. For the chimera decision use `identity_scores.mixed`/`codominant_lineages` (co-dominance-tested), NOT this flag alone (§4.1). `pan_class_only` means "only class markers passed" — at the gate level; with the fold-aware ON a real subtype often resolves anyway, so re-check `identity_scores`/`candidate_subtypes` before treating it as unresolved. |
| `signal_targets_suggested` | ranked list | informational only (no marker-guided split here); ignore for the decision. |
| `n`, `n_out` | ints | sizes; `n_out<20` ⇒ `warn` (treat AUC cautiously). |

Gene matching is **case-insensitive** throughout (mouse `Gad1` ≡ human `GAD1`).

---

## 6. Reasoning Scaffold — produce this for EVERY cluster

Condense the result into `rationale` / `alternatives_considered`. This is how an expert keeps
the judgment thorough and reproducible — and it forces the comprehensive synthesis of §4.

```
S1 IS-IT-A-CELL : density.pct=__  lineage(specificity_peak, lineage_markers.n, graph_evidence.lineage)=__ → cell? y/n
S2 LINEAGE      : graph_evidence.lineage=__ (the floor you are sure of) ; lineage_ambiguous? __ (if T → doublet/common-ancestor, not a confident single call) ; ambient bleed lineages ignored (§4.1)
S3 ANCHORS      : verdict=__ ; top anchors w/ excl_markers_on = __ (the type's OWN exclusive markers actually ON?)
S4 DESCEND      : path of the best anchor = class→region→…→term ; how deep can I DEFEND? (stop at region if subtype ambiguous)
S5 CP / NEG     : any graph_evidence.cp on the path? apply discriminators ; any neg_on leak? reject
S6 RESOLVE      : clear_top → deepest defensible term ; ambiguous_tie(distinct+strong) → needs_subclustering ; ambiguous_tie(weak) → region/class ; class_or_unknown → class (lineage clear) / Unknown
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

### Step 1 — LINEAGE (the floor you are sure of)
Confirm **`graph_evidence.lineage`** (the dominant class — `glutamatergic neuron` / `GABAergic neuron` /
`astrocyte` / …), corroborated by `lineage_scan`. If no lineage resolves → **Step 4 (Unknown)**. The lineage
class is the SHALLOWEST honest label; you descend below it ONLY with valid exclusive evidence (Step 2). A
secondary lineage that is only weak **ambient/neuropil bleed** (§4.1: `Gad1/2`, `Slc1a2/Aldoc/Apoe`,
`Mbp/Plp1/Mobp/Ptgds`, scattered immune/vascular under ONE dominant lineage) is contamination, NOT a second
identity — do not split on it. A real second lineage shows its OWN exclusive subtype markers at comparable
strength (handled in Step 3).

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
- **verdict = `class_or_unknown`** — no anchor has exclusive support. → the **lineage CLASS** as a terminal
  `celltype` (`subclass_resolved=false`, `reason_code="class_floor"`) if the lineage is clear; else **Unknown**
  (Step 4). NEVER reach into a weak candidate to name a specific type with no exclusive evidence.

`confidence`: **high** (anchor `excl_on≥3` or `excl_on/excl_n≥0.7`, clean negatives) · **medium** (anchor
`excl_on` 1–2, or you stopped at a region group) · **low** (single exclusive marker, panel-limited).

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
No lineage (`graph_evidence.lineage==null`) AND no exclusive anchor, not Background — after honestly listing
≥3 considered terms (from `graph_evidence.anchors` + `candidate_subtypes`) each with a concrete rejection
(no exclusive marker ON / negative leak / shared-marker-only force-fit). **The Step-2-class-floor vs Step-4-
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
**Wiring:** the caller passes `--disease "<name>"` to `init` (e.g. a brain-cancer dataset → `--disease "brain cancer"`).
The helper persists it in state (all rounds) and, for every leaf, surfaces a **`disease_candidates`** block in the
profile — the wing's state terms whose `D` (disease_context) matches that disease, each with its **on-markers /
n_on / n_markers** computed against this leaf (same ON definition as identity_scores). This exists because the
normal scoring core (`graph_evidence`/`identity_scores`) deliberately excludes `tier=state` terms from `top`
(they share broad reactive markers); `disease_candidates` re-surfaces the *matching* ones so you can judge them.
- `disease` **omitted** → no disease hint; do NOT assume normal; consider all terms regardless of `D`. (no `disease_candidates` block)
- `disease == normal` → prefer `D=="normal"` terms; don't assign disease-state terms.
- `disease == <name>` → for each leaf, **read `disease_candidates`**: if one's markers genuinely fit (run Step 2
  on it — enough of its OWN markers ON, negatives clean, beats the normal alternative), assign that disease-state
  term; if none fits, label the **normal** type it most resembles (`disease_context` still records the sample
  context). Never invent a non-ontology label. If the wing has **no** state term for this disease, the block is
  empty → fall back to the nearest normal type / Unknown (the state type must be built first; build-mode §3/§7).
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

- When a parent is split and **all** (or a contiguous run of) its new sub-leaves get the **same `top`
  identity** in Step 2 → give them all that label. They are one population Leiden happened to fragment;
  in the output (same label = same annotation) they **merge**. Do not split them further (`split_no_gain`).
- This is also the stop rule: a split that yields one identity across its children gained nothing →
  terminalize at that identity. A split that yields ≥2 distinct identities is real → keep each.

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
whose required are all off-panel, e.g. CA1, are exempt and identified by supporting/experience) ·
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
`RS_LT2_BUT_RESOLVED` / `NEGATIVE_LEAK` (reconsider as the suggestion hints). The `suggestion` is
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
