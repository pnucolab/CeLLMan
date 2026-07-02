---
name: annotate-reference-mode
description: Reference-guided cell-type annotation — annotate a clustered scRNA/spatial h5ad TO a given reference answer-key (정답지). The reference (a set of expected cell types) restricts the Cell-Ontology wing to those terms + their KG children + ancestors; everything else is excluded (strict → out-of-reference clusters become Unknown). Identity is still judged by the SAME comprehensive graph-descent as annotate-mode, and CeLLMan's strength is preserved: when a reference term's signal is MIXED, it subclusters into that term's KG children (finer than the answer-key). Round-driven (init → subcluster* → finalize); the caller collects the user's Yes/No between rounds. Reuses annotate-mode's helper.py and wing expert agents UNCHANGED — only adds a prep step (ontology pruning) and a postcheck step (reference diagnostics).
---

# Annotate-reference-mode — graph-descent inside a reference-restricted wing

**This mode = annotate-mode + a reference-restricted ontology.** You do NOT reimplement the judgment.
The entire reasoning discipline — the comprehensive synthesis (§4), profile fields (§5), the graph-descent
decision algorithm (§7), merge/resolution (§8), validation (§9) — is **exactly annotate-mode's `SKILL.md`**.
Read it; it is your reasoning. This file only documents the **two thin wrappers** that turn a reference
"answer-key" into a restriction, plus the diagnostics that report how well you recovered it.

The single architectural idea: **helper.py consumes one `_ontology_index.json`.** If that index is pruned to
the reference subtree, every helper behavior (graph_evidence, identity_scores, subcluster mixture detection,
merge, finalize) becomes reference-constrained *for free* — no change to helper.py or the expert agents.

## When this mode applies
The user HAS a reference / answer-key — a set of expected cell types (e.g. a curated cell-type list, or an
already-annotated reference h5ad's annotation column) — and wants the input annotated **to those types**,
while still letting CeLLMan **subdivide a mixed reference term into its finer children** (the de-novo strength).
If there is no reference, use **annotate-mode** instead.

## What the reference does (and does NOT do)
- **Term-restriction ONLY.** The reference supplies a *set of cell-type terms*. Markers stay 100% from the KG
  — the reference's expression is NOT used to derive markers (no label-transfer / marker-transfer).
- **Children are included.** For every reference term, its KG **descendants** are kept too — so when a
  cluster matching a reference term shows ≥2 children's markers (a mixture), the existing helper machinery
  (`DF_MULTI_SUBTYPE` / `SHOULD_SPLIT_MIXTURE` / `bimodal_subtypes` / strong-mixture override) splits it and
  you label each fragment at the **child** level (finer than the answer-key). This is the point of the mode.
- **Ancestors are included** (up to the lineage root) — required so helper's lineage transmitter-gate,
  region-grouping, and path-to-root keep working. They are kept but tagged `ancestor` (see ABOVE_REFERENCE).
- **Strict.** Reference-outside KG terms are pruned away, so a cluster that matches none of them resolves to
  the lineage **class** (if its lineage root is in the kept ancestors) or **Unknown** — never a non-reference
  specific type. Out-of-reference cells are surfaced honestly (ABOVE_REFERENCE / Unknown), not force-fit.

## Roles (same as annotate-mode)
| Actor | Responsibility |
|---|---|
| **expert agent = you** | the SAME wing expert as annotate-mode (e.g. `mouse-whole_brain-expert`). It reads `{out}/_ontology_index.json` — now the *restricted* index — and judges by graph-descent. **No new agent.** |
| **caller** | runs the prep step, drives the rounds, asks the user Yes/No between rounds, runs the postcheck/coverage. |
| **helper.py** | UNCHANGED. Compute + validation + final write. Receives the restricted index via `--ontology`. |
| **ref_prep.py / ref_postcheck.py** | NEW companion scripts (this mode only). Prune the ontology; produce reference diagnostics. They never touch helper.py or the expert agents. |

## Resolve the scripts (once)
- `{HELPER}` = `find . -path '*Agentic-Annotation/scripts/helper.py'` (annotate-mode helper, unchanged).
- `{PREP}`   = `find . -path '*Agentic-Annotation/scripts/ref_prep.py'`
- `{POST}`   = `find . -path '*Agentic-Annotation/scripts/ref_postcheck.py'`
- `{python}` = the user venv. `{output_dir}` = `{base_out}/annotation_out` (same hard rule as annotate-mode).
- Server: pass `--server-url` (or set `$CELLONTO_SERVER_URL`); same deployed Cell-Ontology server as annotate-mode.

## The flow (round-driven; only Step 0 and the postcheck differ from annotate-mode)

### Step 0 — PREP: build the reference-restricted index (NEW; run ONCE before init)
The reference is either a **label/term_id list file** (`.json`/`.csv`/`.tsv`) or a **reference h5ad column**.
```bash
mkdir -p {output_dir}
# (a) reference = a list of expected cell-type labels (or term_ids)
{python} {PREP} --species {species} --organ {organ} --server-url {server} \
    --reference {ref_labels.json} \
    --out-index {output_dir}/restricted_ontology_index.json \
    --out-report {output_dir}/_reference_report.json
# (b) reference = an already-annotated reference h5ad's column (labels extracted; expression NOT used)
{python} {PREP} --species {species} --organ {organ} --server-url {server} \
    --reference-h5ad {ref.h5ad} --reference-col {cell_type} \
    --out-index {output_dir}/restricted_ontology_index.json \
    --out-report {output_dir}/_reference_report.json
```
- Read `_reference_report.json`. **`unmatched`** = reference labels with NO KG term → tell the caller: these
  types are not in the wing (the wing may need build-mode for them); they will NOT be annotatable. Do not guess.
- If `n_matched == 0` the prep aborts (wrong species/organ, or labels don't match canonical wing labels).
- Optional `--alias aliases.json` (`{reference_label: kg_label_or_id}`) to bridge naming differences.

### Step 1 — INIT (helper, UNCHANGED — just feed the restricted index via `--ontology`)
```bash
{python} {HELPER} init --h5ad {input_h5ad} --cluster-col {cluster_col} \
    --out {output_dir} --ontology {output_dir}/restricted_ontology_index.json
#   NOTE: --species/--organ are NOT needed here — the local --ontology bypasses the server fetch.
#   helper copies the restricted index to {output_dir}/_ontology_index.json, which the expert agent reads.
#   --disease "<name>" still applies if the input is a disease/cancer sample.
```
Then **judge exactly as annotate-mode** (read `cluster_gene_profiles_round{k}.json` + `_ontology_index.json`;
run the §4–§7 graph-descent). Because the graph is restricted: anchors can only be reference terms or their
children; descent that can't reach a reference term stops at a kept ancestor class.

### Step 2 — SUBCLUSTER (helper, UNCHANGED)
```bash
{python} {HELPER} subcluster --out {output_dir} --paths '["2","11"]' --resolution '{"2":0.5,"11":1.0}' [--gpu 0]
```
A reference term flagged `needs_subclustering` (its children's markers co-present) splits into those children
— **this is how CeLLMan annotates finer than the answer-key.** Same Leiden-only mechanism, same merge (§8).

### Step 3 — CHECK (helper, UNCHANGED) → then POSTCHECK (NEW)
```bash
{python} {HELPER} check --out {output_dir} --decisions {output_dir}/_annotate_decisions.json
{python} {POST} --out {output_dir} --mode check      # ← appends reference violations
```
`ref_postcheck --mode check` adds two advisory violation types and **merges them additively into
`_annotate_violations.json`** (helper's own entries are preserved), so you read them in the same place:
- **`ABOVE_REFERENCE`** (warn): a `celltype` landed on an `ancestor` term (coarser than the answer-key) — the
  data didn't resolve to a reference type. If a reference (anchor) child's own exclusive markers ARE on, re-judge
  to that child; if the cluster genuinely is an out-of-reference lineage member, keeping the class is honest
  (it signals "a cell outside your answer-key"). Your call, per annotate-mode §7.
- **`OUT_OF_REFERENCE`** (warn): a hand-written `term_id` not in the restricted index (should not happen; helper's
  `BAD_TERM_ID` also catches it) → relabel to a reference term or Unknown.
Re-decide flagged clusters and re-run check/postcheck until clean (same loop as annotate-mode §9).

### Step 4 — FINALIZE (helper, UNCHANGED) → then COVERAGE (NEW)
```bash
{python} {HELPER} finalize --out {output_dir} --decisions {output_dir}/_annotate_decisions.json [--accept-large]
{python} {POST} --out {output_dir} --mode coverage    # ← reference recovery report
```
`ref_postcheck --mode coverage` writes `_reference_coverage.json` — the **answer-key recovery** report:
`hit` (labeled at the reference term) · `refined_to_child` (split finer than the reference — CeLLMan's gain) ·
`above_reference` (stopped at a coarser class) · `missing_anchors` (reference types absent in the data) ·
`unknown`/`background` (out-of-reference). `recovery_rate = anchors_assigned / reference_anchors`. This is the
direct diagnostic for **claim 3 (scRNA recovery)**: how much of the curated reference CeLLMan reproduced, and
where it went deeper.

## What you do NOT change vs annotate-mode
Profile reading, the comprehensive synthesis, the graph-descent algorithm (§4–§7), the split/merge logic (§8),
the helper validations (§9), the decision JSON schema and label hygiene (§10), the self-check (§11), the report
to the caller (§12), the forbidden list (§13), the worked examples (§14). **All identical.** The ONLY deltas
are: (0) prep prunes the ontology, (3/4) postcheck adds reference diagnostics. Everything else is annotate-mode.

## Forbidden (in addition to annotate-mode §13)
- Editing `helper.py` or the wing expert agents — this mode is **purely additive** (prep + postcheck only).
- Using the reference's *expression* to override KG markers — reference supplies the **term set only** (markers
  stay from the KG). If a panel mismatch makes a KG marker off-panel, fall back per annotate-mode §4 (experience/
  supporting), not to reference-derived markers.
- Force-fitting a cluster onto the nearest reference term when its own specific markers are off (annotate-mode
  §4 lesson 3 / `spec_pass`). A non-reference cluster → class (ABOVE_REFERENCE) or Unknown, never a fake hit.
- Inventing a label outside the restricted index; writing the h5ad (only finalize); asking the user (only caller).
