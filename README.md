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
```

```bash
# codex
mkdir -p .agents/skills .codex/agents
cp -r CeLLMan/skills/annotate-mode .agents/skills/
cp CeLLMan/agents/annotate-expert.toml .codex/agents/
printf '\n[sandbox_workspace_write]\nnetwork_access = true\n' >> .codex/config.toml
```

The agent locates the backend by name, so keep the clone directory called `CeLLMan`:
`find . -path '*CeLLMan/scripts/helper.py'`.

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
