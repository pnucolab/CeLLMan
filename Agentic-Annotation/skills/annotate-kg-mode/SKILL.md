---
name: annotate-kg-mode
description: Ontology-Gated-Descent cell/voxel annotation of clustered scRNA / spatial h5ad. Distinct from annotate-mode (Leiden-only splitting). Splitting descends the KG IS_A hierarchy — parent required=IN gate, sibling defining=OUT exclusion, child defining/required=discrimination — with sibling-competition gating + centering-free scaling + Unknown, so forced-assignment contamination is blocked without Leiden. Leiden is used ONLY to resolve mixed lineages. Standard annotation runs autonomously under the agent's judgment (init → split-lineage* → descend* → finalize, NO user prompts mid-run); the user is consulted only in an interactive phase after finalize. Saves obs['cellman_standard'].
---

# annotate-KG-mode — Ontology-Gated Descent (OGD)

**You are the domain expert for this wing (species + organ).** The Cell-Ontology wing is your
knowledge graph. This mode annotates **every cell/voxel** by *descending the KG hierarchy*, not by
judging Leiden clusters. `scripts/kg_helper.py` does the compute (lineage call, OGD split, profiles,
path bookkeeping); **you provide the biological judgment.** The caller speaks to the user; you do not.

## Why this mode exists (vs annotate-mode)

annotate-mode splits with **Leiden only** — it abandoned `signal_split` (force every cell to its
nearest marker target) because weak matches contaminate (a Grik4⁺ cortical cell dragged into CA3).
OGD is `signal_split` **made safe** by three guards, so it can split *without* Leiden:

| annotate-mode's reason to drop signal_split | OGD's guard |
|---|---|
| weak matches force-assigned → contamination | **Unknown** (no signal → not assigned) |
| Grik4⁺ cortical pulled into CA3 | **parent gate** (cortical fails the hippocampal gate → CA3 never a candidate) |
| absolute score dominated by high-expressers / rare markers | **sibling-competition gate + centering-free scaling** |

OGD is a **first-pass split**; your **comprehensive profile judgment is the second safety net**
(exactly as in annotate-mode). A group whose KG markers disagree with its measured profile is
`needs_subclustering`/`Unknown`, never force-kept.

## Two hard rules (same spirit as annotate-mode)

1. **Never force-fit the nearest specific term.** Assign a specific celltype ONLY when its OWN
   defining/required markers are actually present in the group profile. Otherwise the honest label is
   the **lineage class** (`glutamatergic neuron`) or **Unknown**. OGD can over-split on a
   non-specific marker (e.g. `Cxcl14`/`C1ql2` retrosplenial firing broadly) — catch this in the
   profile (marker ON-fraction across the whole group, `top_enriched` showing the real signal).
2. **OGD markers are KG-derived; the measured profile is ground truth.** When the KG says a marker is
   specific but the profile shows it ON across a large group, trust the profile → `needs_subclustering`
   or demote to the class.

---

## Roles

| Actor | Responsibility |
|---|---|
| **expert agent = you** | Run kg_helper; read `profile_round*.json`; per cluster decide **celltype / needs_subclustering(+flag) / Unknown**; drive split-lineage/descend **autonomously to finalize**; write decisions JSON. Never ask the user. |
| **caller** | **Standard annotation (rounds 1–4) runs autonomously under the agent's judgment — NO user prompts mid-run.** The caller only relays the agent's helper calls back-to-back until finalize. **The user is consulted ONLY after finalize** (interactive phase): present the outcome (English, table + explanation) and the choice ① done ② user-directed refinement. |
| **kg_helper.py** | lineage call, OGD split, Leiden lineage-split, profiles, path IDs, finalize. |

---

## Standard annotation flow → `obs['cellman_standard']`

**Runs autonomously.** You (agent) decide each next action (split-lineage / descend / finalize) from
the profiles and drive it through **without asking the user** — rounds chain automatically until every
cluster is celltype/Unknown and you finalize. The user is engaged **only after finalize** (interactive
phase). This is the key difference from annotate-mode (which consults the user every round).

### Round 1 — lineage
```
kg_helper.py init --h5ad H --out O --organ ORG --species SP [--cluster-col leiden]
```
Read `profile_round1.json`. Each leiden cluster gets a lineage (class root) + `mixed_lineage` flag +
`lineage_ranked`. **The flag is a hint; you judge.** For each cluster:
- **single, confident lineage** → ready to descend.
- **mixed** (two lineages co-strong, e.g. glut+GABA) → `split-lineage`.

### Round 1b — resolve mixed lineages (Leiden, the ONLY Leiden use)
```
kg_helper.py split-lineage --out O --paths '["3","7"]' [--resolution 0.5]
```
Re-Leidens each mixed cluster and re-calls lineage on every sub. Read `profile_linsplit_r*.json`.
- sub now single lineage → descend.
- sub still mixed → split again at higher resolution **UNTIL a STOP condition**:
  - **STOP after ≤2 re-splits** if a sub stays mixed and does not shrink much → this is **KDE mush /
    doublets**, not resolvable by Leiden. Judge from the profile: assign the **dominant lineage**
    (descend under it) or, if truly ambiguous, `Unknown`. Never re-Leiden indefinitely.

### Round 2+ — descend (OGD, no Leiden)
```
kg_helper.py descend --out O --paths '["4","4.12", ...]'
```
Descends each cluster one KG level (parent node → children via OGD). New paths `parent.childidx`
(e.g. `4.12.1`). Read `profile_round*.json`. For **each** resulting group decide:

| decision | when | action |
|---|---|---|
| **celltype** | group's own defining/required markers present & coherent; deepest supported node | record `{path: label}` |
| **needs_subclustering** flag=`sibling` | children look resolvable but this group still mixes ≥2 siblings | `descend` this path again (its children's markers) |
| **needs_subclustering** flag=`<celltype>` | profile shows a *different* lineage/celltype's signal bled in (KDE mush) | `descend` using that celltype's KG subtree (see below) |
| **Unknown** | no OWN specific markers; even lineage unclear, or non-specific-marker over-split with no real signal | leave at parent path / mark Unknown |

Keep descending groups that still have resolvable children (deeper subtypes) until each is a
celltype / Unknown. **Stop at the deepest level the profile solidly supports** — a region node
("hippocampal pyramidal neuron") is a valid terminal label when the exact subtype is ambiguous.

### needs_subclustering flag handling
- **flag=sibling** → `descend --paths '["<path>"]'` (OGD over the last node's children).
- **flag=`<celltype>`** → that specific celltype's signal contaminated the group. Re-descend
  targeting that celltype's subtree so the contaminant separates out (same `descend`, the KG node of
  that path is that celltype; if it is elsewhere in the graph, split the group by that celltype's
  defining/required as an OGD 2-way).

### Unknown re-pool
After all clusters resolve, collect Unknown groups and re-judge once (a subtype missed earlier may
now be separable, or they are genuinely background/low-signal).

### Finalize
```
kg_helper.py finalize --out O --decisions decisions.json [--output out.h5ad]
```
`decisions.json` = `{path: celltype|Unknown}`. Writes `obs['cellman_standard']` (+ `cellman_kg_path`).
**This is the end of standard annotation.**

---

## Interactive phase (after standard, caller offers the user)

1. **Done** — stop.
2. **User-directed** — e.g. "I care about T cells, subdivide further" / "re-check for regulatory
   T cells". You then:
   - find clusters whose profile carries that signal, **descend with OGD** (never Leiden — a targeted
     celltype check, so use its KG subtree markers).
   - if the type is genuinely present but was missed → resolve it and report.
   - if it is genuinely absent → **say so honestly** ("no group shows FOXP3/IL2RA above background").
     Do not manufacture a positive.

---

## Reading a profile (per group JSON)

- `size`, `node` (current KG node), `lineage_ranked` (lineage call top-5).
- `children[]`: each child's `markers_on` = `[gene, ON-fraction, mean]` in THIS group → which child
  the group leans to, and whether the winner's markers are actually ON (not just argmax of noise).
- `top_enriched`: most enriched genes vs whole dataset → reveals real identity / bled-in signal a KG
  marker set would miss (this is your over-split / contamination detector).

**Judgment, not a gate:** weigh children markers ON-fraction, top_enriched, lineage_ranked together.
A child winning argmax with 3% ON-fraction while `top_enriched` screams another type = do NOT assign
it; that is `needs_subclustering(flag=<that type>)` or `Unknown`.
