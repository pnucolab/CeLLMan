# CeLLMan

**AI agentic workflow for an automatic cell-type annotation based on knowledge graph of genes**

One specialized agent per organism and organ mines PubMed into a knowledge graph of cell-type marker
genes and infers cell identity from mRNA signals alone. It runs on any clustered `h5ad` — scRNA-seq
or spatial — and writes `obs['cellman_annotation']`.

CeLLMan is the agentic annotation component of
[**SSAF-T**](https://github.com/pnucolab/SSAF-T) (Spatial Segmentation-free Agentic Framework -
Transcriptomics), where it labels the vectors sampled from a segmentation-free spatial vector field.
This repository is that component on its own, and needs no spatial data to be useful.

---

## Installation

```bash
git clone https://github.com/pnucolab/CeLLMan.git
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r CeLLMan/requirements.txt
```

Then register the skill and the expert agent with your LLM host.

```bash
# claude code
mkdir -p .claude/skills .claude/agents
cp -r CeLLMan/skills/annotate-mode .claude/skills/
cp CeLLMan/agents/annotate-expert.md .claude/agents/

# the read-only KG lookup as native tools — the path the skill takes by default
claude mcp add cellman-kg \
  --env CELLMAN_PYTHON="$(pwd)/.venv/bin/python" \
  -- python3 "$(pwd)/CeLLMan/mcp/cellman_kg_mcp.py"
```

```bash
# codex
mkdir -p .agents/skills .codex/agents
cp -r CeLLMan/skills/annotate-mode .agents/skills/
cp CeLLMan/agents/annotate-expert.toml .codex/agents/
printf '\n[sandbox_workspace_write]\nnetwork_access = true\n' >> .codex/config.toml

cat >> ~/.codex/config.toml <<EOF

[mcp_servers.cellman-kg]
command = "python3"
args = ["$(pwd)/CeLLMan/mcp/cellman_kg_mcp.py"]
env = { CELLMAN_PYTHON = "$(pwd)/.venv/bin/python" }
EOF
```

The agent locates the backend by name, so keep the clone directory called `CeLLMan`:
`find . -path '*CeLLMan/scripts/helper.py'`.

`cellman-kg` is how the KG module lookup is reached **by default** — see [`mcp/README.md`](mcp/README.md)
for what it exposes and how to verify it. **Skipping it breaks nothing**: it only wraps
`helper.py resolve-module` / `helper.py modules`, and the skill falls back to those commands whenever the
tools are absent. Same code, same answer, one extra shell call.

---

## Knowledge Graph coverage

The Knowledge Graph holds one module per `(species, organ)`. Each module carries that organ's own cell
types; the common-lineage backbone (T/B/myeloid, endothelium, fibroblast …) is injected into every
module on top of them, so it is not listed here.

Give the tissue in any wording you like: an exact module name runs straight away, and anything else
(`mammary gland`, `cerebellum`, `gut`) is turned into a question before annotation starts — see
[`CALLER_PROMPT_GUIDE.md`](CALLER_PROMPT_GUIDE.md). Ask for several modules at once and they are
served together.

The table below is a snapshot. The live list — and what any wording resolves to — comes from the
server:

```bash
{python} CeLLMan/scripts/helper.py modules --species mouse
{python} CeLLMan/scripts/helper.py resolve-module --species mouse --tissue "mammary gland"
```

### Human

| | | | |
|---|---|---|---|
| `adipose_tissue` | `bladder` | `blood` | `bone` |
| `bone_marrow` | `brain` | `breast` | `connective_tissue` |
| `ear` | `embryo` | `eye` | `fallopian_tube` |
| `heart` | `joint` | `kidney` | `large_intestine` |
| `liver` | `lung` | `lymph_node` | `mouth` |
| `muscle` | `ovary` | `pancreas` | `peripheral_nervous_system` |
| `pituitary_gland` | `placenta` | `prostate` | `salivary_gland` |
| `skin` | `small_intestine` | `spinal_cord` | `spleen` |
| `stomach` | `testis` | `thymus` | `tongue` |
| `tonsil` | `tooth` | `trachea` | `uterus` |
| `vasculature` |  |  |  |

### Mouse

| | | | |
|---|---|---|---|
| `adipose_tissue` | `bladder` | `blood` | `bone` |
| `bone_marrow` | `brain` | `breast` | `connective_tissue` |
| `ear` | `embryo` | `eye` | `fallopian_tube` |
| `heart` | `joint` | `kidney` | `large_intestine` |
| `liver` | `lung` | `lymph_node` | `mouth` |
| `muscle` | `nasal_cavity` | `ovary` | `pancreas` |
| `peripheral_nervous_system` | `pituitary_gland` | `placenta` | `prostate` |
| `salivary_gland` | `skin` | `small_intestine` | `spinal_cord` |
| `spleen` | `stomach` | `testis` | `thymus` |
| `tongue` | `tooth` | `trachea` | `uterus` |
| `vasculature` |  |  |  |


---

## Documentation

Full documentation is in preparation and will be published on Read the Docs.

Until then, `notebooks/` holds a worked example:

| notebook | modality |
|---|---|
| [`notebooks/TS_lung.ipynb`](notebooks/TS_lung.ipynb) | scRNA-seq |

For spatial transcriptomics — array-based and imaging-based worked examples, where CeLLMan annotates
a segmentation-free vector field — see [SSAF-T](https://github.com/pnucolab/SSAF-T).

For how to phrase the prompt that starts an annotation run, see
[`CALLER_PROMPT_GUIDE.md`](CALLER_PROMPT_GUIDE.md).

---

## License

BSD 3-Clause. See [LICENSE](LICENSE).
