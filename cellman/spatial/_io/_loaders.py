"""Platform loaders for count-based spatial data → :class:`spatialdata.SpatialData`.

Each loader reads a platform's raw output and returns a uniform SpatialData
(assembled by :meth:`GridDataset.to_spatialdata`): ``tables['table']`` = the count
matrix (``obsm['spatial']`` in µm, region-less, GridDataset metadata in
``uns['cellman']``) and ``images['staining']`` = the morphology image carrying an
``affine_px_to_um`` (image pixel → world µm) Affine transform — the single
cross-platform coordinate contract.

Only the raw read, the image source, the affine convention, and the species differ
between platforms; everything downstream (panel → compute_vf → downsample → cluster
→ map) is shared and consumes the lightweight :meth:`GridDataset.from_sdata` view.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from .._griddataset import GridDataset
from ._grid import fit_affine_lstsq, read_square_bin_gem


# ---------------------------------------------------------------------------
# Stereo-seq (STOmics)
# ---------------------------------------------------------------------------
# dihedral group as (transpose, flip_u, flip_v); maps ssDNA norm (u,v) → GEM norm (u',v')
_STEREOSEQ_ORIENTS = {
    "identity": (False, False, False), "fliplr": (False, True, False),
    "flipud":   (False, False, True),  "rot180": (False, True, True),
    "transpose": (True, False, False), "rot90":  (True, False, True),
    "rot270":    (True, True, False),  "transpose_flip": (True, True, True),
}


def _register_ssdna_to_lattice(a, img, bin_um, *, reg_um=20.0, ssdna_ds=32,
                               orientation=None):
    """Robust ssDNA→lattice registration for Stereo-seq.

    Fixes three fragilities of the old bbox+mirror heuristic:
      * tissue bbox via **Otsu + largest connected component** (not ``img > 0``),
        so out-of-tissue speckle no longer inflates the bbox / scale;
      * the axis convention is **auto-detected** over all 8 dihedral orientations
        (transpose × 4 mirrors), not a hardcoded transpose + 4-mirror search;
      * orientation is chosen by correlating the **brain crop** intensity, robust
        to the bright chip frame and to near-symmetric sections.

    Maps the ssDNA brain bounding box onto the GEM brain bounding box (pure
    scale + translation, plus the chosen dihedral) → ``affine_px_to_um``. Pass
    ``orientation`` (a key of ``_STEREOSEQ_ORIENTS``) to force one.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    def largest_blob(mask):
        lbl, n = ndi.label(mask)
        if n == 0:
            return mask
        cnt = np.bincount(lbl.ravel()); cnt[0] = 0
        return ndi.binary_fill_holes(lbl == cnt.argmax())

    # ---- GEM count-density on a coarse lattice grid → brain bbox (col,row) ----
    C = a.obs["array_col"].to_numpy(); R = a.obs["array_row"].to_numpy()
    w = np.asarray(a.X.sum(1)).ravel().astype(np.float64)
    c_min, r_min = int(C.min()), int(R.min())
    reg_bins = max(1, int(round(reg_um / bin_um)))
    cc = (C - c_min) // reg_bins; rr = (R - r_min) // reg_bins
    D = np.zeros((int(rr.max()) + 1, int(cc.max()) + 1), np.float64)
    np.add.at(D, (rr, cc), w)
    gmask = largest_blob(D > threshold_otsu(D[D > 0]))
    gys, gxs = np.where(gmask)
    gcol0 = c_min + int(gxs.min()) * reg_bins; gcol1 = c_min + int(gxs.max()) * reg_bins
    grow0 = r_min + int(gys.min()) * reg_bins; grow1 = r_min + int(gys.max()) * reg_bins
    Dc = D[gys.min():gys.max() + 1, gxs.min():gxs.max() + 1]
    Dz = (Dc - Dc.mean()) / (Dc.std() + 1e-9)
    gh, gw = Dc.shape

    # ---- ssDNA brain bbox (px), Otsu + largest CC (dilated to bridge sparse nuclei) ----
    DS = int(ssdna_ds)
    sm = img[::DS, ::DS].astype(np.float64)
    smask = largest_blob(ndi.binary_dilation(sm > threshold_otsu(sm[sm > 0]), iterations=2))
    sysr, sxsr = np.where(smask)
    ipx0 = int(sxsr.min()) * DS; ipx1 = int(sxsr.max()) * DS
    ipy0 = int(sysr.min()) * DS; ipy1 = int(sysr.max()) * DS
    Sc = sm[sysr.min():sysr.max() + 1, sxsr.min():sxsr.max() + 1]
    Sh, Sw = Sc.shape

    # ---- pick orientation by resampling ssDNA into the GEM brain grid + correlating ----
    def score(transpose, fu, fv):
        vg, ug = np.mgrid[0:gh, 0:gw].astype(np.float64)
        ug /= max(gw - 1, 1); vg /= max(gh - 1, 1)
        if transpose:
            uu, vv = vg, ug
        else:
            uu, vv = ug, vg
        us = (1 - uu) if fu else uu
        vs = (1 - vv) if fv else vv
        warp = Sc[np.clip((vs * (Sh - 1)).astype(int), 0, Sh - 1),
                  np.clip((us * (Sw - 1)).astype(int), 0, Sw - 1)]
        wz = (warp - warp.mean()) / (warp.std() + 1e-9)
        return float((Dz * wz).mean())

    if orientation is not None and orientation in _STEREOSEQ_ORIENTS:
        best_name = orientation
    else:
        best_name = max(_STEREOSEQ_ORIENTS, key=lambda k: score(*_STEREOSEQ_ORIENTS[k]))
    transpose, fu, fv = _STEREOSEQ_ORIENTS[best_name]

    # ---- build affine from the 4 brain-bbox corner correspondences ----
    px_c, px_r, col, row = [], [], [], []
    for u in (0, 1):
        for v in (0, 1):
            px_c.append(ipx0 + u * (ipx1 - ipx0)); px_r.append(ipy0 + v * (ipy1 - ipy0))
            uu = (1 - u) if fu else u; vv = (1 - v) if fv else v
            if transpose:
                uu, vv = vv, uu
            col.append(gcol0 + uu * (gcol1 - gcol0)); row.append(grow0 + vv * (grow1 - grow0))
    inv = fit_affine_lstsq(col, row, px_c, px_r)      # px → (col,row) lattice
    affine_px_to_um = inv * bin_um                    # px → world µm
    det = abs(affine_px_to_um[0, 0] * affine_px_to_um[1, 1]
              - affine_px_to_um[0, 1] * affine_px_to_um[1, 0])
    um_per_px = float(np.sqrt(det)) if det > 0 else bin_um
    return affine_px_to_um, um_per_px, best_name


def load_stereoseq(gem_path: str | Path, ssdna_tif: str | Path, *,
                   species: str = "mouse", organ: str = "whole_brain",
                   bin_size: int = 1, flip_downsample: int = 32,
                   orientation: Optional[str] = None,
                   affine_px_to_um=None) -> GridDataset:
    """Load a Stereo-seq bin1 GEM + ssDNA image.

    The GEM is read to a square-bin AnnData (native 0.5 µm DNB units) and
    ``obsm['spatial']`` is converted to µm. The ssDNA→lattice affine is resolved
    robustly by :func:`_register_ssdna_to_lattice` — Otsu brain bounding boxes on
    both the GEM count density and the ssDNA, then the dihedral orientation whose
    brain crop best correlates (all 8 candidates, no hardcoded transpose).

    Parameters
    ----------
    orientation : str, optional
        Force a dihedral orientation (key of ``_STEREOSEQ_ORIENTS``: ``'identity'``,
        ``'fliplr'``, ``'flipud'``, ``'rot180'``, ``'transpose'``, ``'rot90'``,
        ``'rot270'``, ``'transpose_flip'``). Default ``None`` → auto-detect.
    affine_px_to_um : (2, 3) array, optional
        Fully bypass registration and use this pixel→µm affine directly.
    """
    a = read_square_bin_gem(str(gem_path), bin_size=bin_size)
    bin_um = float(a.uns["bin_size_um"])
    a.obsm["spatial"] = a.obsm["spatial"].astype(np.float64) * bin_um

    img = tifffile.imread(str(ssdna_tif))
    if img.ndim == 3:
        img = img[0]

    if affine_px_to_um is not None:
        affine_px_to_um = np.asarray(affine_px_to_um, float)
        det = abs(affine_px_to_um[0, 0] * affine_px_to_um[1, 1]
                  - affine_px_to_um[0, 1] * affine_px_to_um[1, 0])
        um_per_px = float(np.sqrt(det)) if det > 0 else bin_um
        orient_name = "manual"
    else:
        affine_px_to_um, um_per_px, orient_name = _register_ssdna_to_lattice(
            a, img, bin_um, ssdna_ds=flip_downsample, orientation=orientation)

    return GridDataset(
        raw=a, image=img, um_per_px=um_per_px, species=species, organ=organ,
        platform="stereoseq", affine_px_to_um=affine_px_to_um,
        he_revert=False,   # ssDNA: bright tissue on dark → threshold directly
        meta={"bin_size_um": bin_um, "orientation": orient_name},
    ).to_spatialdata(copy=False)   # `a` is local to this call → skip the ~107s copy


# ---------------------------------------------------------------------------
# 10x Visium HD
# ---------------------------------------------------------------------------
def load_visiumhd(path: str | Path, *, species: str = "mouse",
                  organ: str = "small_intestine", bin_um: int = 2,
                  fullres_image_file: Optional[str] = None) -> GridDataset:
    """Load a 10x Visium HD ``outs/`` directory at the ``bin_um`` µm bin.

    Uses ``spatialdata_io.visium_hd`` for the counts + fullres H&E, and fits a
    lattice→pixel affine from the bin's ``tissue_positions`` (array_col/row →
    pxl_col/row) on the SAME lattice the vector field uses.
    """
    import pandas as pd
    import spatialdata_io

    path = Path(path)
    kw = {}
    if fullres_image_file is not None:
        kw["fullres_image_file"] = fullres_image_file
    sdata = spatialdata_io.visium_hd(str(path), bin_size=bin_um,
                                     load_all_images=True, **kw)

    suffix = f"square_{bin_um:03d}um"
    tkey = next((k for k in sdata.tables if suffix in k), None)
    if tkey is None:
        raise RuntimeError(f"no {bin_um}µm table; have {list(sdata.tables)}")
    adata = sdata.tables[tkey].copy()
    adata.var_names_make_unique()

    # bin lattice + in-tissue from tissue_positions parquet
    pos_path = path / "binned_outputs" / suffix / "spatial" / "tissue_positions.parquet"
    pos = pd.read_parquet(pos_path).set_index("barcode")
    common = adata.obs_names.intersection(pos.index)
    adata = adata[common].copy()
    pos = pos.loc[common]
    for c in ("array_row", "array_col", "in_tissue",
              "pxl_col_in_fullres", "pxl_row_in_fullres"):
        if c in pos.columns:
            adata.obs[c] = pos[c].to_numpy()
    if "in_tissue" in adata.obs:
        adata = adata[adata.obs["in_tissue"].astype(bool)].copy()

    # spatial (µm) = bin index × bin_um
    adata.obsm["spatial"] = np.stack(
        [adata.obs["array_col"].to_numpy(), adata.obs["array_row"].to_numpy()],
        axis=1).astype(np.float64) * bin_um

    # fullres H&E image (c,y,x) → (y,x,3)
    imgkey = next((k for k in sdata.images if "full" in k), next(iter(sdata.images)))
    img_el = sdata.images[imgkey]
    # fullres is frequently loaded as a MULTISCALE image (DataTree of scale0..N),
    # whose ``.values`` is a dict-method (→ 0-d asarray). Take the highest-res
    # scale ('scale0', var 'image'); single-scale images are a plain DataArray.
    from xarray import DataArray as _DataArray
    if not isinstance(img_el, _DataArray):
        _keys = list(img_el.keys())
        _node = img_el["scale0"] if "scale0" in _keys else img_el[_keys[0]]
        _vars = list(_node.data_vars) if hasattr(_node, "data_vars") else list(_node.keys())
        img_el = _node["image"] if "image" in _vars else _node[_vars[0]]
    arr = np.asarray(img_el.values)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)

    # affine: lattice(col,row) -> fullres px, fit on this bin's positions
    ptab = pos[pos.get("in_tissue", 1) == 1] if "in_tissue" in pos else pos
    affine_inv = fit_affine_lstsq(ptab["array_col"], ptab["array_row"],
                                  ptab["pxl_col_in_fullres"], ptab["pxl_row_in_fullres"])
    affine_px_to_um = affine_inv * float(bin_um)

    # µm/px from scalefactors
    um_per_px = 1.0
    sf = path / "binned_outputs" / suffix / "spatial" / "scalefactors_json.json"
    if sf.exists():
        import json
        um_per_px = float(json.load(open(sf)).get("microns_per_pixel", 1.0))

    return GridDataset(
        raw=adata, image=arr, um_per_px=um_per_px, species=species, organ=organ,
        platform="visiumhd", affine_px_to_um=affine_px_to_um, he_revert=True,
        meta={"bin_um": bin_um},
    ).to_spatialdata()


# ---------------------------------------------------------------------------
# NovaSeq-X StrataMap
# ---------------------------------------------------------------------------
def load_strata(h5ad_path: str | Path, he_ome_tiff: str | Path, *,
                species: str = "human", organ: str = "breast",
                he_level: int = 2, um_per_px_full: float = 0.260) -> GridDataset:
    """Load a StrataMap NovaSeq-X sample: point-barcode ``.h5ad`` + registered H&E.

    Barcode coordinates are already in ``obsm['spatial']`` (µm) if the h5ad was
    prepared that way; otherwise pass an h5ad whose ``obs_names`` encode
    ``…:X:Y:…`` in nm. Gene symbols are promoted to ``var_names``. The H&E axes
    are transposed relative to barcode µm (image row → x, image col → y).
    """
    import scanpy as sc

    adata = sc.read_h5ad(str(h5ad_path))
    if "gene_symbols" in adata.var:
        adata.var["ensembl"] = adata.var_names.to_numpy()
        adata.var_names = adata.var["gene_symbols"].astype(str).to_numpy()
        adata.var_names_make_unique()
    if "spatial" not in adata.obsm:
        xy = np.array([b.split(":")[1:3] for b in adata.obs_names],
                      dtype=np.float64) / 1000.0    # nm → µm
        adata.obsm["spatial"] = xy

    # read H&E pyramid level → grayscale + its µm/px
    with tifffile.TiffFile(str(he_ome_tiff)) as tif:
        base = tif.series[0]
        full_h = base.levels[0].shape[0]
        arr = base.levels[he_level].asarray()
        m = re.search(r'PhysicalSizeX="([\d.]+)"', tif.ome_metadata or "")
    full_um_per_px = float(m.group(1)) / 1000.0 if m else um_per_px_full
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    arr = arr[..., :3] if arr.ndim == 3 else arr
    he_um_per_px = full_um_per_px * (full_h / arr.shape[0])

    # StrataMap transpose: image row (px_y) → world x ; image col (px_x) → world y
    # world_x = px_y * upp ; world_y = px_x * upp
    affine_px_to_um = np.array([[0., he_um_per_px, 0.],
                                [he_um_per_px, 0., 0.]])

    return GridDataset(
        raw=adata, image=arr, um_per_px=he_um_per_px, species=species, organ=organ,
        platform="strata", affine_px_to_um=affine_px_to_um, he_revert=True,
        meta={"he_level": he_level},
    ).to_spatialdata()
