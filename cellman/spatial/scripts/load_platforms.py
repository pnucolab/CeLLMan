"""Unified SpatialData loaders for 4 spatial-transcriptomics platforms.

All loaders return a SpatialData object with a uniform schema so the SSAM
pipeline (run_kde / cluster_vectors / map_celltypes_he) can run unchanged:

    sdata.images["morphology"]    : (c, y, x) staining/morphology image
    sdata.points["transcripts"]   : DataFrame with columns x, y, gene
    sdata.tables["cells"]         : AnnData (cell × gene)         [skipped for VisiumHD]
    sdata.shapes["cells"]         : GeoDataFrame cell polygons    [skipped for VisiumHD]

For STOmics and VisiumHD the platforms do not provide molecule-resolved
transcripts, so synthetic transcript points are produced by selecting HVGs
(default 3000) from the bin/cell AnnData and repeating each spot's (x, y) by
its UMI count per HVG gene.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
from geopandas import GeoDataFrame
from scipy.sparse import issparse
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
# from spatialdata_io import merscope, stereoseq, visium_hd, xenium
import spatialdata_io

DEFAULT_HVG_N = 3000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hvg_unroll(adata: ad.AnnData, hvg_n: int) -> pd.DataFrame:
    """Select top-`hvg_n` HVGs (seurat_v3, raw counts) and unroll the cell ×
    gene count matrix into a long DataFrame of synthetic transcripts.

    Returns a DataFrame with columns ``x``, ``y``, ``gene`` (one row per UMI).
    """
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required for unroll.")

    n_genes = adata.n_vars
    if n_genes > hvg_n:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=hvg_n, flavor="seurat_v3", subset=False
        )
        hvg_mask = adata.var["highly_variable"].to_numpy()
        adata = adata[:, hvg_mask].copy()

    X = adata.X
    if issparse(X):
        coo = X.tocoo()
        rows, cols, data = coo.row, coo.col, coo.data
    else:
        rows, cols = np.nonzero(X)
        data = np.asarray(X[rows, cols]).ravel()

    data = data.astype(np.int64)
    keep = data > 0
    rows, cols, data = rows[keep], cols[keep], data[keep]

    xy = np.asarray(adata.obsm["spatial"])[rows]
    genes = adata.var_names.to_numpy()[cols]

    x = np.repeat(xy[:, 0], data)
    y = np.repeat(xy[:, 1], data)
    g = np.repeat(genes, data)

    df = pd.DataFrame({"x": x.astype(np.float32),
                       "y": y.astype(np.float32),
                       "gene": pd.Categorical(g)})
    return df


def _wrap_points(df: pd.DataFrame) -> dd.DataFrame:
    """Wrap a long-format transcripts DataFrame into a SpatialData Points
    element (`x`, `y`, `gene`)."""
    return PointsModel.parse(df, coordinates={"x": "x", "y": "y"},
                             feature_key="gene")


def _circles_from_centroids(adata: ad.AnnData, default_radius: float = 5.0) -> GeoDataFrame:
    """Build a Shapes element of cell circles from `adata.obsm['spatial']`.

    Radius is derived from `adata.obs['area']` (or `cell_area`) if present,
    otherwise a fixed `default_radius` is used.
    """
    xy = np.asarray(adata.obsm["spatial"])
    area_col = next((c for c in ("area", "cell_area", "Area") if c in adata.obs), None)
    if area_col is not None:
        radii = np.sqrt(adata.obs[area_col].to_numpy() / np.pi)
    else:
        radii = np.full(adata.n_obs, default_radius, dtype=np.float32)

    geometry = [Point(x, y) for x, y in xy]
    gdf = GeoDataFrame({"radius": radii}, geometry=geometry)
    gdf.index = adata.obs_names
    return ShapesModel.parse(gdf)


def _read_image_2d(path: Path) -> Image2DModel:
    """Read a single 2D TIFF as Image2DModel with shape (c, y, x)."""
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]            # (1, y, x)
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] not in (3, 4):
        arr = np.transpose(arr, (2, 0, 1))    # HWC -> CHW
    elif arr.ndim != 3:
        raise ValueError(f"Unsupported image shape {arr.shape} at {path}")
    return Image2DModel.parse(arr, dims=("c", "y", "x"))


# ---------------------------------------------------------------------------
# Xenium
# ---------------------------------------------------------------------------

def load_xenium(path: str | Path, **kwargs) -> SpatialData:
    """Load a 10x Xenium output directory.

    Thin delegate to :func:`spatialdata_io.xenium` — directory in, SpatialData
    out. Extra ``kwargs`` are forwarded to ``spatialdata_io.xenium``.
    """
    return spatialdata_io.xenium(path, **kwargs)


# ---------------------------------------------------------------------------
# MERSCOPE / MERFISH
# ---------------------------------------------------------------------------

def load_merscope(path: str | Path, **kwargs) -> SpatialData:
    """Load a Vizgen MERSCOPE output directory.

    Thin delegate to :func:`spatialdata_io.merscope` — directory in,
    SpatialData out. Extra ``kwargs`` are forwarded.
    """
    return spatialdata_io.merscope(path, **kwargs)


# ---------------------------------------------------------------------------
# STOmics (Stereo-seq)
# ---------------------------------------------------------------------------

def _find_one(path: Path, patterns: list[str], required: bool = True) -> Optional[Path]:
    """Return the first file in `path` matching any glob pattern."""
    for p in patterns:
        hits = sorted(path.glob(p))
        if hits:
            return hits[0]
    if required:
        raise FileNotFoundError(
            f"No file matching {patterns!r} found in {path}."
        )
    return None


def load_stomics(path: str | Path, hvg_n: int = DEFAULT_HVG_N) -> SpatialData:
    """Load a STOmics (Stereo-seq) sample directory.

    Expected files in the directory (matched by glob):
        *.h5ad           : cell-bin AnnData with obsm['spatial']
        *.tif / *.tiff   : ssDNA staining image
        (anything else is ignored)

    HVGs (`hvg_n`) are selected from the AnnData and unrolled to synthetic
    transcripts located at cell centroids.
    """
    path = Path(path)
    h5ad_path = _find_one(path, ["*.h5ad"])
    tif_path = _find_one(path, ["*ssDNA*.tif", "*staining*.tif", "*.tif", "*.tiff"])

    adata = ad.read_h5ad(h5ad_path)
    if "spatial" not in adata.obsm:
        raise ValueError(
            f"{h5ad_path} has no obsm['spatial']; cannot place transcripts."
        )

    df = _hvg_unroll(adata, hvg_n=hvg_n)

    images = {"morphology": _read_image_2d(tif_path)}
    points = {"transcripts": _wrap_points(df)}

    # Table
    adata.obs["region"] = "cells"
    adata.obs["region"] = adata.obs["region"].astype("category")
    adata.obs["instance_id"] = np.arange(adata.n_obs)
    table = TableModel.parse(
        adata, region="cells", region_key="region", instance_key="instance_id"
    )
    tables = {"cells": table}

    shapes = {"cells": _circles_from_centroids(adata)}

    return SpatialData(images=images, points=points, tables=tables, shapes=shapes)


# ---------------------------------------------------------------------------
# VisiumHD
# ---------------------------------------------------------------------------

def load_visium_hd(
    path: str | Path,
    hvg_n: int = DEFAULT_HVG_N,
    bin_um: int = 2,
) -> SpatialData:
    """Load a 10x Visium HD `outs/` directory.

    The 2µm spot AnnData (default) is used as the unit for HVG selection and
    transcript unrolling. Per user spec, table & shapes are NOT populated.
    """
    sdata = spatialdata_io.visium_hd(path, bin_size=[bin_um], filtered_counts_file=True,
                      load_all_images=False)

    # Image
    images = {}
    for prefer in ("hires_image", "fullres_image", "lowres_image", "cytassist_image"):
        match = [k for k in sdata.images if prefer in k]
        if match:
            images["morphology"] = sdata.images[match[0]]
            break
    if "morphology" not in images:
        if len(sdata.images) == 0:
            raise RuntimeError(f"No image found in VisiumHD output at {path}.")
        first = next(iter(sdata.images))
        images["morphology"] = sdata.images[first]

    # Get 2µm bin AnnData
    suffix = f"square_{bin_um:03d}um"
    table_keys = [k for k in sdata.tables if suffix in k]
    if not table_keys:
        raise RuntimeError(
            f"No {bin_um}µm bin table found. Available: {list(sdata.tables.keys())}"
        )
    adata = sdata.tables[table_keys[0]].copy()

    # spatialdata_io.visium_hd stores spot coords in obsm['spatial']
    if "spatial" not in adata.obsm:
        # Fallback: derive centroids from associated shapes element
        shape_keys = [k for k in sdata.shapes if suffix in k]
        if not shape_keys:
            raise ValueError("Cannot find spatial coordinates for VisiumHD bins.")
        gdf = sdata.shapes[shape_keys[0]]
        centroids = gdf.geometry.centroid
        adata.obsm["spatial"] = np.column_stack(
            [centroids.x.to_numpy(), centroids.y.to_numpy()]
        )

    df = _hvg_unroll(adata, hvg_n=hvg_n)
    points = {"transcripts": _wrap_points(df)}

    # Per spec: skip tables & shapes for VisiumHD
    return SpatialData(images=images, points=points, tables={}, shapes={})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

LOADERS = {
    "xenium":   load_xenium,
    "merscope": load_merscope,
    "merfish":  load_merscope,
    "stomics":  load_stomics,
    "visium_hd": load_visium_hd,
}


def load(platform: str, path: str | Path, **kwargs) -> SpatialData:
    """Dispatch by platform name (case-insensitive)."""
    key = platform.lower().replace("-", "_")
    if key not in LOADERS:
        raise ValueError(
            f"Unknown platform {platform!r}. "
            f"Supported: {sorted(LOADERS.keys())}"
        )
    return LOADERS[key](path, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a spatial dataset into unified SpatialData.")
    parser.add_argument("platform", choices=sorted(LOADERS.keys()))
    parser.add_argument("path", type=Path)
    parser.add_argument("--hvg-n", type=int, default=DEFAULT_HVG_N)
    parser.add_argument("--out", type=Path, help="Optional zarr write path.")
    args = parser.parse_args()

    kw = {}
    if args.platform in ("stomics", "visium_hd"):
        kw["hvg_n"] = args.hvg_n
    sdata = load(args.platform, args.path, **kw)
    print(sdata)
    if args.out is not None:
        sdata.write(args.out, overwrite=True)
        print(f"Wrote {args.out}")
