"""Canonical intermediate for **count-matrix** spatial platforms.

Transcript platforms (Xenium / MERSCOPE / CosMx) flow through SpatialData points
and the KDE pipeline unchanged. Count platforms (Stereo-seq / Visium HD /
NovaSeq-X StrataMap) do **not** provide molecule-resolved transcripts; they give a
cell/bin x gene count matrix on a spatial lattice. Rather than unroll counts into
synthetic transcripts (memory blow-up on dense data — Stereo-seq bin1 ~1e8 UMIs),
the count path rasterizes counts onto a grid and GPU-blurs them into a
KDE-equivalent vector field (see :meth:`Analysis.compute_vf`).

``GridDataset`` is the lightweight hand-off between a platform loader
(:mod:`cellman.spatial._io`) and that backend. It deliberately does **not** wrap a
SpatialData object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from anndata import AnnData


class _RawView:
    """Lightweight AnnData-duck for a count ``GridDataset.raw`` — exposes only the
    surface aggregate_to_grid / compute_vf / GridDataset touch (``X``,
    ``obsm['spatial']``, ``var``, ``var_names``, ``n_obs``, ``n_vars``, ``shape``,
    column-slicing ``[:, genes]``). Avoids constructing a full AnnData over a
    10^8-row count table, whose ``obs_names`` string-index materialization alone
    costs ~144s (the real bottleneck of ``spatialdata``/``anndata`` read_zarr on
    huge count stores). Read-only view (not writable back to SpatialData)."""
    def __init__(self, X, spatial, var):
        self.X = X
        self.obsm = {"spatial": spatial}
        self.var = var

    @property
    def var_names(self):
        return self.var.index

    @property
    def n_obs(self):
        return self.X.shape[0]

    @property
    def n_vars(self):
        return self.X.shape[1]

    @property
    def shape(self):
        return self.X.shape

    def copy(self):
        return self

    def __getitem__(self, key):
        cols = key[1] if isinstance(key, tuple) else slice(None)
        if isinstance(cols, slice):
            return self
        idx = self.var.index.get_indexer(list(cols))
        idx = idx[idx >= 0]
        return _RawView(self.X[:, idx].tocsr(), self.obsm["spatial"], self.var.iloc[idx])



def _jsonify(obj):
    """Recursively coerce a metadata value into something the AnnData zarr writer
    accepts in ``uns`` (tuples → lists, numpy scalars → python scalars). The count
    loaders stash e.g. ``meta['flip'] = (0, 1)`` — a tuple anndata cannot encode."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


@dataclass
class GridDataset:
    """Platform-agnostic container for a count-based spatial sample.

    Attributes
    ----------
    raw : AnnData
        Point/bin-level counts. ``obsm['spatial']`` holds x,y in **µm**;
        ``var_names`` are gene symbols. This is the unit HVG selection and
        rasterization operate on (via :func:`cellman.spatial._io.aggregate_to_grid`).
    image : numpy.ndarray
        Morphology / staining image used for tissue-mask voxel sampling. Either
        ``(c, y, x)`` or ``(y, x)``. ssDNA (Stereo-seq), H&E (Visium HD /
        StrataMap), etc.
    um_per_px : float
        Physical scale of ``image`` (µm per pixel), at the resolution stored.
    species : str
    organ : str
        Cell-Ontology module name (e.g. ``'whole_brain'``, ``'breast'``). Used to
        fetch marker genes for the gene panel.
    platform : str
        ``'stereoseq'`` | ``'visiumhd'`` | ``'strata'`` (informational).
    origin : tuple(float, float)
        Shared global µm origin ``(x0, y0)`` = ``raw.obsm['spatial'].min(0)``.
        Every lattice (coarse HVG grid, fine vector-field grid, voxel mapping)
        must use this same origin so they align.
    affine_px_to_um : numpy.ndarray, optional
        ``(2, 3)`` affine mapping image pixel ``(px_x=col, px_y=row, 1)`` -> world
        ``(µm_x, µm_y)``. This is the ONE cross-platform coordinate contract: each
        loader folds its own convention (StrataMap axis transpose, Stereo-seq
        mirror-flip, Visium HD lstsq lattice fit) into this matrix, so
        :meth:`Analysis.downsample_vector` samples voxels identically for all.
        World µm are converted to grid bins at compute time via ``origin`` +
        ``bin_um``. ``None`` ⇒ fall back to a plain ``um_per_px`` scale.
    he_revert : bool
        Tissue-mask polarity for Otsu: ``True`` = bright-background/dark-tissue
        (brightfield H&E, invert before threshold); ``False`` = dark-background/
        bright-tissue (fluorescence / ssDNA, threshold directly).
    """

    raw: AnnData
    image: np.ndarray
    um_per_px: float
    species: str
    organ: str
    platform: str = "unknown"
    origin: Tuple[float, float] = (0.0, 0.0)
    affine_px_to_um: Optional[np.ndarray] = None
    he_revert: bool = True
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if "spatial" not in self.raw.obsm:
            raise ValueError("GridDataset.raw needs obsm['spatial'] (x,y in µm).")
        sp = np.asarray(self.raw.obsm["spatial"])
        if self.origin == (0.0, 0.0):
            self.origin = (float(sp[:, 0].min()), float(sp[:, 1].min()))

    @property
    def n_genes(self) -> int:
        return int(self.raw.n_vars)

    def subset_genes(self, genes) -> "GridDataset":
        """Return a copy of this dataset restricted to ``genes`` (a list of
        symbols present in ``raw.var_names``), preserving all metadata.

        :param genes: Gene symbols to keep; those absent from ``raw.var_names``
            are ignored.
        :type genes: list of str
        """
        keep = [g for g in genes if g in set(self.raw.var_names)]
        sub = self.raw[:, keep].copy()
        return GridDataset(
            raw=sub, image=self.image, um_per_px=self.um_per_px,
            species=self.species, organ=self.organ, platform=self.platform,
            origin=self.origin, affine_px_to_um=self.affine_px_to_um,
            he_revert=self.he_revert, meta=dict(self.meta),
        )

    def to_spatialdata(self, *, table_key: str = "table",
                       image_key: str = "staining", copy: bool = True):
        """Assemble a :class:`spatialdata.SpatialData` from this count sample.

        * ``tables[table_key]`` is the count AnnData (region-less), with all
          GridDataset metadata stashed in ``uns['cellman']`` so it round-trips.
        * ``images[image_key]`` is the staining image ``(c, y, x)`` with an
          **Affine** ``px`` to ``world µm`` transform built from
          ``affine_px_to_um`` (falls back to an isotropic ``um_per_px`` Scale).
          This is the single cross-platform coordinate contract, now expressed
          natively.

        The count pipeline (:meth:`Analysis.compute_vf`,
        :meth:`~Analysis.downsample_vector`) threads this object through;
        :meth:`from_sdata` reconstructs the lightweight view the backend needs.

        :param table_key: Key under which the count table is stored in the
            resulting SpatialData ``tables``.
        :type table_key: str
        :param image_key: Key under which the staining image is stored in the
            resulting SpatialData ``images``.
        :type image_key: str
        :param copy: If ``True`` duplicate ``raw`` before adding metadata; pass
            ``False`` to mutate ``raw`` in place and skip the copy.
        :type copy: bool
        """
        import spatialdata as sd
        from spatialdata.models import Image2DModel, TableModel
        from spatialdata.transformations import Affine, Scale

        img = np.asarray(self.image)
        orig_ndim = int(img.ndim)
        if img.ndim == 2:
            cyx, nchan = img[None, :, :], 1
        elif img.shape[-1] in (3, 4):
            cyx = np.moveaxis(img[..., :3], -1, 0); nchan = cyx.shape[0]
        elif img.shape[0] in (3, 4):
            cyx = img[:3]; nchan = cyx.shape[0]
        else:
            cyx = img.reshape(-1, *img.shape[-2:])[:1]; nchan = 1

        if self.affine_px_to_um is not None:
            M = np.asarray(self.affine_px_to_um, float)
            mat = np.array([[M[0, 0], M[0, 1], M[0, 2]],
                            [M[1, 0], M[1, 1], M[1, 2]],
                            [0.0, 0.0, 1.0]])
            tf = Affine(mat, input_axes=("x", "y"), output_axes=("x", "y"))
        else:
            tf = Scale([self.um_per_px, self.um_per_px], axes=("x", "y"))
        image = Image2DModel.parse(cyx, dims=("c", "y", "x"),
                                   transformations={"global": tf})

        tab = self.raw.copy() if copy else self.raw
        tab.uns["cellman"] = _jsonify({
            "species": self.species, "organ": self.organ, "platform": self.platform,
            "um_per_px": float(self.um_per_px),
            "affine_px_to_um": (None if self.affine_px_to_um is None
                                else np.asarray(self.affine_px_to_um, float).tolist()),
            "origin": [float(self.origin[0]), float(self.origin[1])],
            "he_revert": bool(self.he_revert),
            "image_ndim": orig_ndim, "image_nchan": int(nchan),
            "image_key": image_key, "meta": dict(self.meta),
        })
        table = TableModel.parse(tab)
        return sd.SpatialData(tables={table_key: table}, images={image_key: image})

    @classmethod
    def from_sdata(cls, sdata, *, table_key=None, image_key=None) -> "GridDataset":
        """Reconstruct a GridDataset view from a count SpatialData built by
        :meth:`to_spatialdata` (reads the count table + ``uns['cellman']`` +
        staining image). The heavy backend consumes this view directly, so the
        blur / lookup internals stay array-based and unchanged.

        :param sdata: Count SpatialData built by :meth:`to_spatialdata`.
        :type sdata: spatialdata.SpatialData
        :param table_key: Count-table key; defaults to the table carrying
            ``uns['cellman']``.
        :type table_key: str, optional
        :param image_key: Staining-image key; defaults to the ``image_key``
            stored in the metadata.
        :type image_key: str, optional
        """
        if table_key is None:
            table_key = next((k for k, t in sdata.tables.items()
                              if "cellman" in t.uns), next(iter(sdata.tables)))
        tab = sdata.tables[table_key]
        meta = dict(tab.uns.get("cellman", {}))
        ik = image_key or meta.get("image_key", "staining")
        if ik not in sdata.images:
            ik = next(iter(sdata.images))
        arr = np.asarray(sdata.images[ik].data)
        nchan = int(meta.get("image_nchan", arr.shape[0] if arr.ndim == 3 else 1))
        if arr.ndim == 3 and nchan == 1:
            image = arr[0]
        elif arr.ndim == 3:
            image = np.moveaxis(arr[:3], 0, -1)
        else:
            image = arr
        aff = meta.get("affine_px_to_um")
        return cls(
            raw=tab, image=image, um_per_px=float(meta.get("um_per_px", 1.0)),
            species=meta.get("species", "unknown"), organ=meta.get("organ", "unknown"),
            platform=meta.get("platform", "unknown"),
            origin=tuple(meta.get("origin", (0.0, 0.0))),
            affine_px_to_um=(None if aff is None else np.asarray(aff, float)),
            he_revert=bool(meta.get("he_revert", True)),
            meta=dict(meta.get("meta", {})),
        )

    @classmethod
    def from_sdata_zarr(cls, sdata_path, *, table_key="table",
                        image_key=None) -> "GridDataset":
        """FAST count-SpatialData reader (read-only). Reads the store zarr DIRECTLY
        (X CSR + ``obsm/spatial`` via zarr; ``var``/``uns`` via
        ``anndata.read_elem``) and wraps counts in :class:`_RawView`, bypassing
        ``spatialdata.read_zarr`` / ``anndata.read_zarr`` whose ~144s cost on huge
        (10^8-row) count tables is dominated by materializing 10^8 string
        ``obs_names`` (never used by the count backend). ~25-30x faster; the
        resulting GridDataset is consumed identically by aggregate_to_grid /
        compute_vf. Use in place of ``sd.read_zarr(path)`` + ``from_sdata(sdata)``
        for the heavy count path (no ``to_spatialdata`` on the result).

        :param sdata_path: Filesystem path to the SpatialData zarr store.
        :type sdata_path: str or pathlib.Path
        :param table_key: Count-table key within the store.
        :type table_key: str
        :param image_key: Staining-image key; defaults to the ``image_key``
            stored in the metadata.
        :type image_key: str, optional
        """
        import zarr
        from scipy.sparse import csr_matrix
        try:
            from anndata.io import read_elem
        except Exception:  # noqa: BLE001
            from anndata.experimental import read_elem
        tpath = f"{sdata_path}/tables/{table_key}"
        tg = zarr.open_group(tpath, mode="r")
        Xg = zarr.open_group(f"{tpath}/X", mode="r")
        indptr = Xg["indptr"][:]; indices = Xg["indices"][:]; data = Xg["data"][:]
        var = read_elem(tg["var"])
        X = csr_matrix((data, indices, indptr),
                       shape=(len(indptr) - 1, var.shape[0]))
        spatial = np.asarray(zarr.open(f"{tpath}/obsm/spatial", mode="r")[:])
        meta = dict(read_elem(tg["uns"]).get("cellman", {}))
        raw = _RawView(X, spatial, var)
        ik = image_key or meta.get("image_key", "staining")
        ipath = f"{sdata_path}/images/{ik}"
        ig = zarr.open_group(ipath, mode="r")

        def _find(g, pre=""):
            for k in g.keys():
                full = f"{pre}/{k}" if pre else k
                node = g[k]
                if hasattr(node, "shape"):
                    return f"{ipath}/{full}"
                s = _find(node, full)
                if s:
                    return s
            return None
        arr = np.asarray(zarr.open(_find(ig), mode="r")[:])
        nchan = int(meta.get("image_nchan", arr.shape[0] if arr.ndim == 3 else 1))
        if arr.ndim == 3 and nchan == 1:
            image = arr[0]
        elif arr.ndim == 3:
            image = np.moveaxis(arr[:3], 0, -1)
        else:
            image = arr
        aff = meta.get("affine_px_to_um")
        return cls(
            raw=raw, image=image, um_per_px=float(meta.get("um_per_px", 1.0)),
            species=meta.get("species", "unknown"),
            organ=meta.get("organ", "unknown"),
            platform=meta.get("platform", "unknown"),
            origin=tuple(meta.get("origin", (0.0, 0.0))),
            affine_px_to_um=(None if aff is None else np.asarray(aff, float)),
            he_revert=bool(meta.get("he_revert", True)),
            meta=dict(meta.get("meta", {})),
        )
