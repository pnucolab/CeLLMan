---
name: annotate-reference-mode
description: Reference-guided cell-type annotation — annotate a clustered scRNA/spatial h5ad TO a user-supplied answer-key (a set of expected cell types). The answer-key restricts the LABEL SPACE only; identity is still judged by annotate-mode's one REAL/DEEPEST/FLOOR test (R1–R8) against the FULL Cell-Ontology graph, so the discriminative bar is unchanged. DEFAULT is strict — a label may only be a reference term, and when no reference term is REAL the honest answer is Unknown (an out-of-reference class/region floor is NOT allowed). With --reference-scope refine the reference terms' KG descendants become labelable too, so a mixed reference term can be resolved finer than the answer-key. Writes obs['cellman_annotation'] plus a reference-recovery report.
---

# annotate-reference-mode — annotate-mode inside a reference-restricted label space

**This mode = annotate-mode + a restricted label space. You do NOT reimplement the judgment.**
The whole reasoning discipline — the principle, rules **R1–R8**, the profile fields, the decision algorithm
(§5), descend-and-see (§6), the finalize gates (§7), validation (§9), the decision JSON (§10), the self-check
(§11) and the forbidden list (§12) — is **exactly `skills/annotate-mode/SKILL.md`**. Read it; it is your
reasoning. This file documents only the **three deltas**.

Everything runs through the same `scripts/helper.py`; there is no second backend and no companion script.

> **NOT ENSEMBLED.** annotate-mode's default 3-judge consensus (its SKILL §8b) does **not** apply here.
> The output is written in the ANSWER-KEY's vocabulary (§5), not the KG's, so a consensus over the KG
> hierarchy would read those labels as unknown terms and collapse the run to Unknown —
> `helper.py consensus-round` refuses a reference run by design (`ERROR:REFERENCE_RUN`). A reference run
> is **one** `annotate-expert`; `annotate-lead` returns `REFERENCE_NOT_ENSEMBLED` if it is handed one.

---

## 1. What the reference does (and does NOT do)

- **Term-restriction ONLY.** The reference supplies a *set of cell-type terms*. **Markers always come from the
  KG** — the reference's expression is never read, never used to derive markers. This is not label-transfer.
- **The GRAPH is not pruned.** `share` / `region_share` / `pan_disc`, the lineage roots, `path_to_root`,
  `region_of` and `restrict_root` keep using **every** term in the wing. Only the set of terms that may
  become a **label** is narrowed.
  > **Why this matters (do not "optimize" it away).** Pruning the ontology index to the reference subtree is
  > the obvious implementation and it is **wrong**: with a small answer-key no gene reaches the `pan_n=12`
  > pan bar (so `PAN` empties) and `region_share` collapses toward 1, so **every marker starts looking
  > exclusive** — R2/R3 stop discriminating and force-fit *inside* the answer-key becomes EASIER, which is the
  > opposite of the mode's purpose. The engine therefore restricts candidates, not statistics.
- **Ancestors stay reachable but are not labelable** unless the user listed them. A leaf that is clearly
  `leukocyte` but matches no reference leukocyte is **Unknown**, not `leukocyte` — unless `leukocyte` is
  itself in the answer-key.

## 2. The two modes

| | `strict` (**default**) | `refine` (`--reference-scope refine`) |
|---|---|---|
| labelable | the matched reference terms **only** | those terms **∪ their KG descendants** |
| deeper than the answer-key | never | yes — a mixed reference term descends into its children |
| nothing REAL | **Unknown** | **Unknown** |

Pick `strict` when the answer-key defines the vocabulary (benchmarking, comparing against an existing
annotation). Pick `refine` to show that the data resolves finer than the answer-key.

## 3. R-ref — the only new rule

Replace R4-DEEPEST / R8-FLOOR **targets** with this; every other rule is unchanged:

> Judge identity exactly as in annotate-mode (R1–R3 against the full graph). Then:
> 1. the DEEPEST **REAL** term that is **ASSIGNABLE** → that is the label;
> 2. else the DEEPEST **ASSIGNABLE ancestor** of the best REAL term, if it too is REAL → that label;
> 3. else **Unknown**.
>
> **A class / region FLOOR is only permitted when that class or region node is itself in the answer-key.**
> **No answer-key term fits, no label**: an out-of-reference floor is a violation, not an honest floor.

**R3 is NOT relaxed.** A reference term still has to earn its label: its OWN low-share markers ENRICHED,
negatives clean. *"It is the only reference term left"* is **never** evidence. The narrower candidate set makes
force-fit tempting — resist it; Unknown is the designed outcome, and the coverage report exists to make an
honest Unknown visible rather than embarrassing.

**Descend (R6) changes meaning in `strict`.** Descending *below* a reference term gains nothing (its children
cannot be labelled). Descend only to separate **≥2 different reference terms** that were clustered together.
The engine already enforces this — `must_descend` counts only ASSIGNABLE children, so a leaf resting at a
reference term is not blocked at finalize.

## 3b. What the caller gathers (and what selects this mode)

Everything annotate-mode's §3 asks for (`input_h5ad`, `cluster_col`, tissue → module, optional
`output` / `use_rep` / `disease`), **plus the reference**:

| input | required | note |
|---|---|---|
| `reference` *(label list .json/.csv/.tsv)* **or** `reference h5ad` + `reference column` | yes | one of the two |
| `reference scope` | no | `strict` (default) / `refine` — how deep below the answer-key a label may go. It does **not** select the mode; the reference input already did. |
| `reference alias` | when the reference's names differ from the KG's | `{reference_label: kg_label_or_term_id}` |

**The USER declares the mode by INVOKING THIS SKILL rather than `/annotate-mode`** — the choice of skill
IS the declaration. There is no `mode` field for them to fill in: what they write is the ordinary prompt
plus the reference lines.

**THE CALLER turns that invocation into `mode : reference` when it spawns the agent.** The agent is a
separate process with its own system prompt and does **not** inherit the skill the caller loaded, so
without this the switch rests entirely on the agent noticing the reference fields. The caller sets it
whenever this SKILL was invoked, or whenever it passes any reference input.

The reference input **also** switches the mode on its own (`agents/annotate-expert.md`, *Mode
selection*, ORs the two signals), and `init` stamps the answer-key into `{out}/_ontology_index.json` so
every later round stays restricted. Two independent signals, because each covers the other's blind spot:

| what the caller sends the AGENT | agent runs | why |
|---|---|---|
| `mode : reference` **+** reference input | reference-mode | the intended case |
| reference input only (declaration forgotten) | reference-mode | the input still switches it; the agent flags the omission in its result |
| `mode : reference` only (no reference input) | **STOPS** → `REFERENCE_INPUT_MISSING` | an answer-key cannot be invented — the caller must get it from the user |
| neither | annotate-mode | correct |

So the USER's prompt is just the ordinary one plus the reference lines:

```
/annotate-reference-mode Annotate the h5ad below using the annotate-expert agent.

- input            : /path/mouse_kidney.h5ad
- tissue           : mouse, kidney
- virtual env      : /path/.venv
- cluster column   : leiden
- reference h5ad   : /path/ref.h5ad
- reference column : celltype_plot
- reference scope  : strict                    # strict (default) | refine — label-space depth
- reference alias  : /path/alias.json          # only if the names differ from the KG's
- note             : spatial KDE vector field — --adaptive-floor, descend at --resolution 0.2
```

## 3c. Resolving the reference's NAMES to KG terms — `ref-resolve` (run BEFORE `init`)

A user's labels will rarely be the KG's exact strings, and **writing the alias is YOUR job, not theirs.**
Ask for full names rather than abbreviations (`Proximal tubule S1` beats `PTS1`), but expect a mismatch
anyway and resolve it yourself:

```bash
{python} {HELPER} ref-resolve --out {out} --species {species} --organ {organ} \
    --reference-h5ad {ref.h5ad} --reference-col {col}      # or --reference {list file}
#  -> {out}/_reference_resolution.json
```

For every reference label this gives: an **exact/normalised name match** if there is one, else **ranked KG
candidates scored by MARKER AGREEMENT** — how many of each term's OWN required/defining markers are among
the genes enriched for that label *inside the reference dataset* — with the matching genes, the label's
top enriched genes, and a `confident` flag plus `confirm_because`.

> ### THE FIREWALL — read this before you touch the reference's expression
> **Celltype judgment is KG-ONLY.** The reference narrows the CANDIDATE SET and nothing else. Its
> expression is read in **`ref-resolve` and nowhere else**, to answer **exactly one** question — *which KG
> celltype is this label?* — and the only thing that leaves that step is a `{reference_label: term_id}`
> mapping. A gene measured in the reference must **never** become evidence for a cluster in the TARGET
> data: no marker transfer, no score transfer, no centroid. That is label **matching**, not label
> **transfer**. `init` opens only the mapping; nothing downstream can open the reference h5ad.

**How to use the output — you decide, the user only arbitrates what is genuinely lossy:**
- `confident` → take it.
- flagged (`confirm_because` names why: close runner-up / weak evidence / parent–child / siblings) → **judge
  it yourself from `enriched_in_reference` and your module knowledge.** `Krt19, Upk1b, Msln` is urothelium
  whatever the scorer ranked first; a scorer that puts `epithelial cell` on top there is simply outranked by
  a broad marker, and you should say so.
- Escalate to the caller **only** when the choice is genuinely the user's — the reference label is
  **coarser than the KG and lumps distinct terms** (`Per-SMC` = pericyte + smooth muscle → their common
  parent `mural cell`, or one of them?), or **granularity is a real choice** (`TAL` → `thick ascending limb
  cell` or `cortical thick ascending limb cell`?), or **no KG term exists** (`Inj_PT`). Return:

  ```
  REFERENCE_CONFIRM_REQUIRED
  labels: [{reference_label, proposed, alternatives[], why, enriched_in_reference[]}]
  ```

  The caller asks the user **once, for all of them together** (the same barrier as MODULE RESOLUTION), then
  re-spawns you with the answers.

Then write the accepted mapping to an alias JSON and pass it to `init --reference-alias`. A label with no
KG term stays unmapped — say so plainly, because **its cells will all become Unknown**.

**Calibration to expect** (measured on a 20-label mouse-kidney reference): 13/19 proposals agreed with a
careful hand-built alias, 4 were marked `confident`, and **none of the `confident` ones was wrong** — the
flag is tuned so errors surface as "confirm", never as a silent mapping. Treat an unflagged proposal as
trustworthy and a flagged one as a question, not as a failure.

**Don't want to run `ref-resolve`?** `init` alone still reports what matched by name in
`{out}/_reference_report.json`; the alias only has to cover its `unmatched` list.

## 4. Commands (the only delta is on `init`)

```bash
# (a) answer-key = a list file (.json / .csv / .tsv)
{python} {HELPER} init --h5ad {input_h5ad} --cluster-col {cluster_col} --out {out} \
    --species {species} --organ {organ} \
    --reference {ref_labels.json} [--reference-scope strict|refine] [--reference-alias {aliases.json}]

# (b) answer-key = an already-annotated reference h5ad's obs column (LABELS only — expression is never read)
{python} {HELPER} init ... --reference-h5ad {ref.h5ad} --reference-col {cell_type}
```

The restriction is stamped into `{out}/_ontology_index.json` as a `reference` block, so **every later round
(`descend` / `subcluster` / `check` / `finalize`) inherits it automatically** — do not repeat the flags.
Everything else (`descend`, `subcluster`, `check`, `finalize`) is **identical to annotate-mode**.

**Read `{out}/_reference_report.json` before judging.** `unmatched` = reference labels with **no term in this
module**: those types can never be annotated and their cells will land in Unknown. Report them to the caller
verbatim (the wing may need `build-mode`); never substitute a "close" term. `--reference-alias`
(`{reference_label: kg_label_or_term_id}`) bridges naming differences.

## 5. Validation + output deltas

- `check` adds **`OUT_OF_REFERENCE`** (**error** — blocks finalize): a decision naming a term outside the
  answer-key. Fix it to a reference term whose OWN markers are ENRICHED, or to Unknown.
- **The OUTPUT is written in the ANSWER-KEY's vocabulary, not the KG's.** The engine reasons in KG terms
  (that is what carries the markers and the hierarchy), but the label the user gets back is their own
  reference string: `obs['cellman_annotation']`, the CSV `label` column and the rationale all read
  `Immune`, never the KG's canonical `leukocyte`. The KG name is preserved in the CSV's `kg_label`
  column. Only terms the user actually listed are translated — in `refine` mode a DESCENDANT of a
  reference term is deeper than the answer-key, so the answer-key has no name for it and it keeps its KG
  label. `Unknown` / `Background` are unchanged.
- `finalize` additionally writes **`reference_coverage.json`**: `hit` (labelled at a reference term) ·
  `refined_to_child` (deeper than the answer-key — `refine` only) · `missing_anchors` (reference types no leaf
  resolved to) · `unknown_cells` / `background_cells` · `out_of_reference` · `recovery_rate`.
  This is the direct diagnostic for **how much of the answer-key was reproduced** and where the data went
  deeper — the measurement behind the scRNA-recovery claim.

## 6. Forbidden (in addition to annotate-mode §12)

- Labelling with a term outside the answer-key, or emitting an **out-of-reference class/region floor** instead
  of Unknown.
- Picking the "closest remaining" reference term when its OWN markers are OFF (R3 is unchanged — a narrowed
  candidate set is not evidence).
- Using the reference's **expression** to override KG markers (term set only).
- Pruning the ontology index to the reference subtree (see §1 — it breaks R2/R3).
