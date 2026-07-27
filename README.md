# CeLLMan

**AI agentic workflow for automatic cell-type annotation, grounded in a knowledge graph of genes.**

Cell-type annotation is normally the manual bottleneck: you cluster, then you stare at marker
genes and name each cluster by hand. CeLLMan replaces that step with an **agentic annotator** —
your own LLM, reasoning against a curated **Cell-Ontology knowledge graph** (celltype → marker
genes, hierarchy, confusion pairs, negative markers), applying one explicit identity test per
cluster and refusing to force-fit a label it cannot support.

Two things ship here:

| | what | needs |
|---|---|---|
| **`Agentic-Annotation/`** | the annotator — skills + expert agents, driven by your LLM host | any clustered `.h5ad` |
| **`cellman.spatial`** | segmentation-free spatial cell-type mapping (SSAM-derived KDE vector field), multi-platform | spatial transcriptomics |

They compose, but neither requires the other. If you already have clusters, go straight to
[Annotate](#annotate-your-clusters).

```
[raw spatial transcriptomics]                    [any scRNA / spatial h5ad]
   ↓ Otsu foreground → KDE vector field                    ↓
   ↓ downsample → leiden                                   ↓
   └────────────►  AGENTIC ANNOTATION (your LLM + knowledge graph)  ◄────┘
                            ↓
                   obs['cellman_annotation']
                            ↓
              full field → per-voxel correlation → [cell-type map]
```

> **Before release:** replace `<ORG>` (GitHub org) below with the real value. The ontology server
> (`https://cellman.pnucolab.com`) is already the built-in default in
> `Agentic-Annotation/scripts/helper.py`, so users need not set it.

---

## Install

```bash
git clone https://github.com/<ORG>/CeLLMan.git
cd CeLLMan

python -m venv .venv && source .venv/bin/activate     # or: uv venv .venv --python 3.11
pip install -e .                                       # CPU
```

GPU acceleration (KDE, rapids leiden, correlation) needs CUDA-specific wheels:

```bash
pip install -e '.[gpu]' --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

The C++ extension (`cellman.spatial.utils`) is compiled by meson — you need a C++17 compiler
with OpenMP. `Agentic-Annotation/` is a plain directory in this repo, **not** a submodule; the
clone above already contains it.

### Register the skills with your LLM host

The annotator is markdown + one Python script, so it is not tied to a framework. For Claude Code:

```bash
cp -r Agentic-Annotation/skills/* ~/.claude/skills/
```

Any other LLM works too — point it at `Agentic-Annotation/skills/annotate-mode/SKILL.md`
and tell it to follow that file.

### Ontology server

The annotator fetches your tissue's ontology (a compact *slim-index*, 80–190 KB per module) from a
hosted, **read-only** server at run time — so labels always reflect the current knowledge graph.
The URL (`https://cellman.pnucolab.com`) is built into the helper; you need not set anything.
Override it only if you run your own server:

```bash
export CELLONTO_SERVER_URL=https://your-server
```

Check what is available: `curl https://cellman.pnucolab.com/modules` — one entry per
(species, organ) *module*.

---

## Annotate your clusters

### 1. Check your h5ad

Three requirements. The annotator reads `adata.X` directly and cannot detect violations for you:

| | requirement | why |
|---|---|---|
| `adata.X` | **log-normalised** (not raw counts) | marker-axis statistics assume it; raw counts silently produce wrong labels |
| `adata.var_names` | **gene symbols** (`EPCAM`, `Snap25`) — not Ensembl IDs | markers are matched by symbol; Ensembl IDs match nothing |
| `adata.obs` | a **cluster column** (e.g. `leiden`) | annotation is per cluster |

Species case does not matter — matching is case-insensitive.

### 2. Mint the expert for your tissue

One expert agent per module, generated on demand. Ask your LLM:

> use build-expert to make the human-kidney expert agent

It verifies the module exists on the server, then writes `human-kidney-expert.md` to
`~/.claude/agents/` (pass a different directory if you want it scoped to one project). The
expert is usable immediately — no install step, no restart.

### 3. Annotate

> annotate `analysis/MyData/clusters.h5ad` with `human-kidney-expert`.
> params: species=human, organ=kidney, cluster_col=leiden,
> output_dir=analysis/MyData/annotation_out, python=.venv/bin/python, gpu_device=0

The run is **autonomous** — the expert judges every cluster, splits the ambiguous ones, and
finalises without interrupting you. You are consulted afterwards, in a free-form phase where
you can question or override any label. The full parameter block and what the caller does with
it is in [`Agentic-Annotation/CALLER_PROMPT_GUIDE.md`](Agentic-Annotation/CALLER_PROMPT_GUIDE.md).

**Output:** `{input}_annotated.h5ad` — your input, copied whole, plus
`obs['cellman_annotation']`. A written rationale per cluster (markers checked, alternatives
rejected, grounding PMIDs) lands in the output directory.

### How it decides

One test, applied at every level: a label is **REAL** only when that term's *own* markers are
**enriched** (elevated against the whole dataset, not merely detected) **and discriminative**
(its own low-degree markers, not shared/pan/ambient ones). The expert assigns the **deepest**
term that is REAL, **floors** honestly to the parent class when none is, and never force-fits.

Mixtures are not pre-detected. When a cluster is not one clear identity it is **descended** —
re-clustered on the marker axis of that term's ontology children, then re-judged. Whatever
distinct identities emerge are the answer: one → label it, two or more → split and recurse,
none → floor. A rare minority population, a balanced doublet, and a cross-region mixture all
separate through the same call.

---

## Spatial cell-type mapping

Segmentation-free: no cell boundaries are ever drawn. Transcripts become a KDE vector field,
the field is clustered, clusters are annotated (above), and every voxel is then correlated
against the annotated signatures to produce a cell-type map.

Supported: **Xenium**, **MERSCOPE**, **Visium HD**, **Stereo-seq (STOmics)**, **StrataMap**.

```python
import cellman.spatial as cms

ds       = cms.SSAMDataset(STORE)                 # zarr store: vf / clusters / maps
analysis = cms.SSAMAnalysis(ds, ncores=30, verbose=True)

adata = analysis.downsample_vector(
    chunked=True, sdata_dir=CROPS,
    n_samples=10, block_um=10, bandwidth=2.5, sampling_distance=1.0,
    image_key="morphology_mip", gpu_device="cuda:1",
)
```

Imaging platforms are tiled first so the field fits in memory:

```bash
python cellman/spatial/scripts/slice_spatialdata.py \
    --sdata-path spatialdata.zarr --tile-size 10000 --output-dir crops \
    --image-key morphology_mip --keep-points transcripts \
    --drop-labels --drop-shapes --drop-tables
```

Cluster the downsampled vectors (rapids on GPU, scanpy on CPU), write the `.h5ad`, annotate it
as above, then map the full field back. **The four notebooks in [`notebooks/`](notebooks/) are
the authoritative worked examples** — one per platform, end to end:

| notebook | platform |
|---|---|
| `xenium_kidney.ipynb` | Xenium |
| `merfish_liver.ipynb` | MERSCOPE |
| `visiumhd_small_intestine.ipynb` | Visium HD |
| `stereoseq_brain.ipynb` | Stereo-seq |

Count-based platforms (Visium HD / Stereo-seq / StrataMap) skip the tiling step and use the
grid path instead — `cms.GridDataset`, `cms.load_visiumhd`, `cms.load_stereoseq`,
`cms.load_strata`.

---

## What you do not need

- **No neo4j, no ontology build, no literature mining.** The knowledge graph is server-side and
  read-only to you.
- **No ontology data in this repo.** It is fetched at run time, so it is always current.
- **No hand-labelling**, and no marker list of your own.

---

## Layout

```
CeLLMan/
├── pyproject.toml            package "cellman" (meson-python build)
├── meson.build               compiles the C++ extension → cellman.spatial.utils
├── cellman/
│   └── spatial/              import cellman.spatial as cms
│       ├── _analysis.py        SSAMAnalysis — downsample_vector / compute_vf / map_celltypes
│       ├── _dataset.py         SSAMDataset — zarr store I/O + plotting
│       ├── _griddataset.py     GridDataset — count/bin platforms
│       ├── _io/                load_visiumhd / load_stereoseq / load_strata
│       ├── _count.py           count-matrix ingestion (blur path)
│       ├── _panel.py           gene-panel selection from ontology markers
│       ├── _kde_gpu.py         GPU KDE primitives
│       ├── _mapping_gpu.py     GPU KDE + Pearson correlation
│       ├── _core/              C++ extension sources
│       └── scripts/            slice_spatialdata / merge_clusters / build_mapping_inputs
├── Agentic-Annotation/       the annotator (not a submodule)
│   ├── skills/
│   │   ├── annotate-mode/   the annotation method
│   │   └── build-expert/       mints a {species}-{organ} expert on demand
│   ├── agents/                 example experts (mint your own with build-expert)
│   ├── scripts/helper.py    the compute backend — init / descend / finalize
│   └── config/                 server URL template
├── notebooks/                worked examples, one per platform
└── docs/
```

Key API: `cellman.spatial.SSAMDataset`, `cellman.spatial.SSAMAnalysis`
(`downsample_vector`, `compute_vf`, `cluster_vectors`, `map_celltypes`,
`init_map_celltypes` + `map_celltypes_partial` for distributed mapping),
`cellman.spatial.GridDataset`.
