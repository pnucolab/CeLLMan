"""Count-based spatial loaders and rasterization for cellman.spatial.

Loaders return a :class:`spatialdata.SpatialData` with ``tables['table']`` (the
count matrix, region-less, + ``uns['cellman']`` metadata) and ``images['staining']``
(the morphology image with an ``px → µm`` Affine transform). The SSAM count backend
(:meth:`SSAMAnalysis.compute_vf` etc.) adapts it via
:meth:`cellman.spatial.GridDataset.from_sdata`. Transcript platforms
(Xenium/MERSCOPE/CosMx) flow through their own SpatialData (points + images); the
final cell-type map is written back into either as ``labels['cellman_celltype_map']``
+ ``images['cellman_celltype_map_rgb']`` + ``tables['cellman_celltype']``.
"""

from ._grid import aggregate_to_grid, fit_affine_lstsq, read_square_bin_gem
from ._loaders import load_stereoseq, load_strata, load_visiumhd

LOADERS = {
    "stereoseq": load_stereoseq,
    "stomics": load_stereoseq,
    "visiumhd": load_visiumhd,
    "visium_hd": load_visiumhd,
    "strata": load_strata,
    "stratamap": load_strata,
    "novaseq": load_strata,
}


def load(platform: str, *args, **kwargs):
    """Dispatch to a count-platform loader by name (case-insensitive)."""
    key = platform.lower().replace("-", "_")
    if key not in LOADERS:
        raise ValueError(f"unknown count platform {platform!r}; "
                         f"supported: {sorted(set(LOADERS))}")
    return LOADERS[key](*args, **kwargs)


__all__ = [
    "aggregate_to_grid", "read_square_bin_gem", "fit_affine_lstsq",
    "load_stereoseq", "load_visiumhd", "load_strata", "load", "LOADERS",
]
