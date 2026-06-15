# unnamed

> Package/name is a placeholder and will be finalized before release.

SSAM-based, **segmentation-free**, multi-platform spatial **cell-type mapping**
workflow (Xenium / MERSCOPE / VisiumHD / STOmics). It turns raw spatial
transcriptomics into a tissue **cell-type map**, with an **agentic,
ontology-guided annotation** step in the middle.

```
[raw spatial transcriptomics]
   → Otsu foreground → downsample vectors (KDE)
   → leiden clustering
   → AGENTIC ANNOTATION  (LLM + ontology server)      ← the differentiator
   → cluster merge
   → full vector field → per-voxel correlation → [cell-type map]
```

The annotation step lives in the **`Agentic-Annotation`** submodule and is driven
by **your own LLM** (Claude Code, Codex, …) against a hosted, read-only
**ontology server**. You never run a database or build an ontology yourself.

---

## 1. Get it (with the submodule)

```bash
git clone --recurse-submodules https://github.com/<org>/unnamed.git
cd unnamed
# already cloned without submodules?
git submodule update --init
```
On GitHub the `Agentic-Annotation/` entry links to its own repository; the flag
above pulls it in.

## 2. Install

```bash
python -m venv .venv && source .venv/bin/activate      # (or: uv venv .venv --python 3.11)

# CPU (subclustering falls back to scanpy)
pip install -e .

# GPU acceleration (KDE + rapids leiden + correlation) — CUDA-specific wheels:
pip install -e '.[gpu]' --index-strategy unsafe-best-match \
  --extra-index-url https://pypi.nvidia.com \
  --extra-index-url https://download.pytorch.org/whl/cu128
```
Base install pulls `spatialdata-io`, `scanpy`, `anndata`, `opencv`, etc.
The C++ extension is compiled via meson (needs a C++17 compiler + OpenMP).

## 3. Point at the ontology server

The annotation step fetches your tissue's ontology (a compact *slim-index*) from
a hosted server at run time. Set its URL (provided by the maintainers):

```bash
export CELLONTO_SERVER_URL=https://<ontology-server>
```

---

## 4. Run the workflow on your data

### (1) Load → SpatialData (zarr)
```python
import spatialdata_io
sdata = spatialdata_io.xenium("<raw_dir>")     # or merscope(...) / visium_hd(...) / stomics(...)
sdata.write("dataset/MyData/spatialdata.zarr")
```

### (2) Tile-slice
```bash
python scripts/slice_spatialdata.py \
    --sdata-path dataset/MyData/spatialdata.zarr \
    --tile-size 10000 --output-dir dataset/MyData/crops
```

### (3) Downsample + (4) cluster
```python
import unnamed
ds = unnamed.SSAMDataset("analysis/MyData")
an = unnamed.SSAMAnalysis(ds, ncores=30, verbose=True)

an.downsample_vector(chunked=True, sdata_dir="dataset/MyData/crops",
                     n_samples=10, block_um=10, bandwidth=2.5,
                     sampling_distance=1.0, gpu_device="cuda:0")
an.cluster_vectors()        # → analysis/MyData/ssam_clusters.h5ad
                            #   obs['leiden'] (clusters) + obs['vf_norm'] (local density)
```

### (5) ⭐ Agentic annotation  — *the core step*

The cluster→cell-type labeling is done by **your LLM** following the
`annotate-mode` skill, using the ontology server. You don't hand-label clusters.

**a. Make the expert for your tissue** (if not already shipped). The two seed
examples are `mouse:whole_brain` and `human:lung`; generate others from the
server manifest:
```bash
python Agentic-Annotation/tools/generate_annotate_experts.py \
    --server $CELLONTO_SERVER_URL --wing mouse:whole_brain
```

**b. Tell your LLM (the caller) to annotate.** Two ways:

- *Any LLM / framework (no setup):* point it at the skill file —
  > "Follow `Agentic-Annotation/skills/annotate-mode/SKILL.md` and annotate with
  > `mouse-whole_brain-expert`. params: species=mouse, organ=whole_brain,
  > input_h5ad=analysis/MyData/ssam_clusters.h5ad, cluster_col=leiden,
  > disease=normal, python=.venv/bin/python, gpu_device=0,
  > server_url=$CELLONTO_SERVER_URL"

- *Claude Code slash-command (optional convenience):* register once, then use
  `/annotate-mode`:
  ```bash
  bash Agentic-Annotation/install_claude.sh        # links skills/agents into .claude/
  ```

**What happens (round-driven):**
1. `helper.py init` fetches your tissue's ontology slim-index from the server and
   profiles each cluster.
2. The expert (your LLM) makes a 4-way call per cluster —
   *celltype / needs_subclustering / Background / Unknown*.
3. Between rounds the caller asks **you**: "subcluster these? at what leiden
   resolution?" — you decide.
4. `finalize` writes **`ssam_clusters_annotated.h5ad`** (your input + a new
   `obs['celltype_annotation']`) and `expert_judgment_annotation.csv`.

The annotation helper is `Agentic-Annotation/scripts/helper.py`
(`init` / `subcluster` / `finalize`); the same venv runs it (GPU rapids or CPU
scanpy automatically).

### (6) Merge + (7) map
```bash
python scripts/merge_clusters.py --store-dir analysis/MyData
```
```python
an.map_celltypes_gpu(sdata, gpu_device="cuda:1")   # full field → per-voxel cell-type map
```

---

## What you do NOT need

- **No neo4j, no ontology build, no literature mining** — all server-side. You
  use a read-only URL only.
- **No ontology data in this repo** — it is fetched at run time, always current.
- Optional spatial **domain** annotation (`neighbor composition → region`) is the
  twin skill `Agentic-Annotation/skills/domain-mode/SKILL.md`.

---

## Package internals

```
unnamed/
├── pyproject.toml         build-system (meson-python) + dependencies
├── meson.build            compiles c/utils.cpp -> unnamed.utils
├── c/utils.cpp            C++ extension (KDE, correlation, normalization)
├── scripts/               workflow helpers (slice / merge / load_platforms)
├── Agentic-Annotation/    annotation step (submodule): skills, expert agents, helper.py
└── unnamed/
    ├── _analysis.py       SSAMAnalysis (downsample / cluster / map_celltypes)
    ├── _dataset.py        SSAMDataset (zarr-store I/O + plotting; stores vf_norm)
    ├── _kde_gpu.py        GPU KDE primitives (downsample)
    └── _mapping_gpu.py    GPU KDE + Pearson correlation (map_celltypes_gpu)
```

Key API: `unnamed.SSAMDataset`, `unnamed.SSAMAnalysis`
(`downsample_vector`, `cluster_vectors`, `map_celltypes_gpu` /
`map_celltypes_cpu`, `init_map_celltypes` + `map_celltypes_partial` for
distributed mapping).
