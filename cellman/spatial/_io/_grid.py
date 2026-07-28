"""Shared rasterization helpers for count-based loaders."""

from __future__ import annotations

from typing import Optional, Tuple

import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix, issparse


def aggregate_to_grid(adata: ad.AnnData, bin_um: float,
                      origin: Optional[Tuple[float, float]] = None) -> ad.AnnData:
    """Sum point/bin counts onto a regular ``bin_um`` µm lattice.

    Every count row is assigned to the square bin containing its
    ``obsm['spatial']`` (x, y) µm coordinate, using a shared global ``origin``
    so that grids at different bin sizes (coarse HVG vs fine vector field) stay
    aligned. Returns a bin-level AnnData with ``obs['array_row'/'array_col']``
    and ``obsm['spatial'] = (col, row)``.

    :param adata: Point/bin-level counts with ``obsm['spatial']`` in µm.
    :type adata: anndata.AnnData
    :param bin_um: Lattice bin size in µm.
    :type bin_um: float
    :param origin: Shared ``(x0, y0)`` µm origin. Defaults to ``spatial.min(0)``.
    :type origin: tuple of float, optional
    """
    sp = np.asarray(adata.obsm["spatial"])
    if origin is None:
        x0, y0 = float(sp[:, 0].min()), float(sp[:, 1].min())
    else:
        x0, y0 = float(origin[0]), float(origin[1])

    col = np.floor((sp[:, 0] - x0) / bin_um).astype(np.int64)
    row = np.floor((sp[:, 1] - y0) / bin_um).astype(np.int64)
    n_col = int(col.max()) + 1
    bid = row * n_col + col
    uniq, inv = np.unique(bid, return_inverse=True)

    S = csr_matrix((np.ones(len(inv), np.float32), (inv, np.arange(len(inv)))),
                   shape=(len(uniq), adata.n_obs))
    X = adata.X.tocsr() if issparse(adata.X) else adata.X
    binX = (S @ X).tocsr() if issparse(X) else csr_matrix(S @ X)

    out = ad.AnnData(X=binX, var=adata.var.copy())
    out.obs["array_row"] = (uniq // n_col).astype(np.int64)
    out.obs["array_col"] = (uniq % n_col).astype(np.int64)
    out.obsm["spatial"] = np.stack(
        [out.obs["array_col"].to_numpy(), out.obs["array_row"].to_numpy()],
        axis=1).astype(np.float32)
    out.uns["bin_size_um"] = float(bin_um)
    out.uns["grid_origin_um"] = (x0, y0)
    return out


def read_square_bin_gem(gem_path: str, bin_size: int = 1) -> ad.AnnData:
    """Read a Stereo-seq bin1 GEM (gzipped TSV: ``geneID x y MIDCounts``) into a
    square-bin AnnData. Coordinates are rebased to origin (empty margin removed).

    ``bin_size`` is in DNB units (1 DNB = 0.5 µm); ``uns['bin_size_um']`` is set
    accordingly. ``obsm['spatial']`` is stored as integer (col, row) bin indices;
    physical µm = index × ``bin_size_um``.

    :param gem_path: Path to the gzipped GEM TSV file.
    :type gem_path: str
    :param bin_size: Square-bin size in DNB units (1 DNB = 0.5 µm).
    :type bin_size: int
    """
    import pandas as pd

    df = pd.read_csv(gem_path, sep="\t",
                     dtype={"x": np.int32, "y": np.int32, "MIDCounts": np.int32})
    gcol = "geneID" if "geneID" in df.columns else df.columns[0]
    ccol = "MIDCounts" if "MIDCounts" in df.columns else df.columns[-1]
    gidx, names = pd.factorize(df[gcol], sort=True)
    names = np.asarray(names)
    x = df["x"].to_numpy().astype(np.int32, copy=False)
    y = df["y"].to_numpy().astype(np.int32, copy=False)
    x = x - x.min(); y = y - y.min()
    nbx = int(x.max() // bin_size) + 1
    bid = (y // bin_size).astype(np.int64) * nbx + (x // bin_size)
    uniq, inv = np.unique(bid, return_inverse=True)

    M = csr_matrix((df[ccol].to_numpy().astype(np.float32), (inv, gidx)),
                   shape=(len(uniq), len(names)))
    a = ad.AnnData(X=M)
    a.var_names = names
    a.obs["array_col"] = (uniq % nbx).astype(np.int64)
    a.obs["array_row"] = (uniq // nbx).astype(np.int64)
    a.obsm["spatial"] = np.column_stack([uniq % nbx, uniq // nbx]).astype(np.float32) * bin_size
    a.uns["bin_size_um"] = bin_size * 0.5
    return a


def fit_affine_lstsq(array_col, array_row, px_col, px_row) -> np.ndarray:
    """Least-squares affine ``(col, row, 1) -> (px_x, px_y)``, return inverse (2,3).

    Used by platforms that give a per-bin lattice-to-pixel correspondence table
    (e.g. Visium HD ``tissue_positions``: array_col/array_row to pxl_col/pxl_row).
    Fit the correspondence on the SAME lattice the vector field uses.

    :param array_col: Lattice column indices of the correspondence points.
    :type array_col: array-like
    :param array_row: Lattice row indices of the correspondence points.
    :type array_row: array-like
    :param px_col: Pixel x (column) coordinates of the correspondence points.
    :type px_col: array-like
    :param px_row: Pixel y (row) coordinates of the correspondence points.
    :type px_row: array-like
    """
    A = np.c_[np.asarray(array_col), np.asarray(array_row), np.ones(len(array_col))]
    cx, *_ = np.linalg.lstsq(A, np.asarray(px_col), rcond=None)
    cy, *_ = np.linalg.lstsq(A, np.asarray(px_row), rcond=None)
    fwd = np.array([cx, cy])
    inv = np.linalg.inv(np.vstack([fwd, [0, 0, 1]]))[:2, :3]
    return inv
