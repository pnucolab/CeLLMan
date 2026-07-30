"""Build ``map_celltypes``' inputs (``cluster_labels`` + ``centroids``) from an
annotated h5ad.

``map_celltypes`` calls :func:`build_centroids_from_h5ad` itself when the store has no
``centroids`` / ``cluster_labels`` (or when an ``h5ad`` is passed explicitly). The
annotated h5ad — row-aligned with the store's ``normalized_vectors`` — is grouped into
celltypes purely by the label string in ``obs[column]``:

  * a named celltype       -> one class per distinct label (different label = different
                              celltype; state-suffixed names stay separate).
  * ``Unknown``            -> grouped by the ORIGINAL cluster (``obs[leiden_column]``):
                              each source cluster with Unknown vectors becomes its own
                              class ``unknown1``, ``unknown2``, ... So unresolved
                              populations are kept separable and are *mapped* (they are
                              real classes for both correlation and GNB), not dropped.
  * ``Background`` / ``*subclustering`` -> dropped to -1 (non-cell / never-finalized).

``centroids[k]`` is the mean of the row-L2-normalized member vectors (= Pearson-
correlation space). Only ``cluster_labels`` + ``centroids`` are written; the store's
``normalized_vectors`` is never modified, so this is non-destructive and idempotent.
"""
import os

import anndata as ad
import numpy as np
import pandas as pd

DROP_LABELS = {'Background', 'needs_subclustering', 'unresolved_subclustering'}
UNKNOWN_LABEL = 'Unknown'


def _qualitative_pool(n):
    """A pool of ``>= n`` maximally distinct hex colours (best effort).

    Celltypes are a NOMINAL variable, so the palette must be qualitative:
    sampling a continuous map (``'jet'``) hands neighbouring ids nearly the same
    colour, and with tens of classes the map becomes unreadable. Sources, in
    order of size, each already optimized for categorical separation:

    * ``n <= 20``  — matplotlib ``tab20``, de-paired so consecutive ids get the
      saturated variants first (``tab20`` is stored as dark/light pairs of the
      same hue, which would give adjacent ids near-identical colours).
    * ``n <= 28``  — scanpy's ``default_28`` (Zeileis). Same palette scanpy uses
      for a categorical ``obs`` column, so the celltype-map and any
      ``sc.pl.spatial`` of the annotated h5ad agree.
    * ``n <= 102`` — scanpy's ``default_102`` (Godsnot).
    * beyond that  — ``default_102`` cycled with a lightness shift per cycle, so
      colours repeat in family but never exactly.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex, to_rgb

    if n <= 20:
        tab = plt.get_cmap("tab20").colors
        return [to_hex(tab[i]) for i in list(range(0, 20, 2)) + list(range(1, 20, 2))]
    try:
        from scanpy.plotting import palettes as _pal
        pool = list(_pal.default_28) if n <= 28 else list(_pal.default_102)
    except Exception:  # noqa: BLE001 — scanpy optional at this layer
        pool = [to_hex(c) for name in ("tab20", "tab20b", "tab20c")
                for c in plt.get_cmap(name).colors]
    if n <= len(pool):
        return pool
    out = []
    for cycle in range((n + len(pool) - 1) // len(pool)):
        f = 1.0 - 0.25 * (cycle % 3)
        out += [to_hex(tuple(min(1.0, max(0.0, ch * f)) for ch in to_rgb(c)))
                for c in pool]
    return out


def celltype_colors(n_celltypes, cmap=None):
    """``n_celltypes`` hex colours, index-aligned with the celltype ids.

    The palette is fixed HERE, once, against the number of classes the mapping
    actually uses (``centroids.shape[0]``) — not against ``celltype_maps.max()``,
    which only counts the classes that won at least one voxel. Everything
    downstream (the labels raster, its RGB rendering, the annotated table's
    ``uns['cellman_annotation_colors']``, any legend) reads this one list, so the
    same celltype cannot come out in different colours in different views.

    :param n_celltypes: number of celltype classes (``centroids.shape[0]``).
    :type n_celltypes: int
    :param cmap: ``None`` (default) picks a QUALITATIVE palette sized to the
        class count (see :func:`_qualitative_pool`). Pass a matplotlib colormap
        name to sample it continuously instead (e.g. ``'jet'``) — legible only
        for a handful of classes.
    :type cmap: str or matplotlib.colors.Colormap or None
    """
    n = int(n_celltypes)
    if cmap is None:
        return _qualitative_pool(n)[:n]
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    cm = plt.get_cmap(cmap)
    return [to_hex(cm(i / max(n - 1, 1))) for i in range(n)]


def compute_centroids(nv, labels, n_clusters, chunk=20000):
    """Per-cluster mean of the row-L2-normalized member vectors.

    :param nv: the store's ``normalized_vectors`` (L, g); read in chunks.
    :type nv: array-like
    :param labels: per-vector cluster index (L,), -1 = dropped.
    :type labels: numpy.ndarray
    :param n_clusters: number of clusters (rows of the returned centroids).
    :type n_clusters: int
    :param chunk: rows per chunk when accumulating (default 20000). Accumulation is in
        float64, so the result is independent of the chunk size.
    :type chunk: int
    :returns: ``(centroids (k, g) float32, counts (k,) int64)``.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    n_genes = nv.shape[1]
    acc = np.zeros((n_clusters, n_genes), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for start in range(0, nv.shape[0], chunk):
        stop = min(start + chunk, nv.shape[0])
        lab = labels[start:stop]
        sel = lab >= 0
        if not sel.any():
            continue
        block = np.asarray(nv[start:stop])[sel].astype(np.float64, copy=False)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        block = block / np.where(norms == 0, 1.0, norms)
        np.add.at(acc, lab[sel], block)
        np.add.at(counts, lab[sel], 1)
    if (counts == 0).any():
        raise ValueError(f'empty cluster(s): {np.flatnonzero(counts == 0).tolist()}')
    return (acc / counts[:, None]).astype(np.float32), counts


def _build_units(celltype, leiden):
    """Group vectors into an ordered list of celltype units by label string.

    Named labels -> one unit each. ``Unknown`` -> one unit per source cluster
    (``obs[leiden_column]``), named ``unknown1..N`` in ascending-cluster order.
    ``DROP_LABELS`` -> excluded (their vectors stay -1).

    :param celltype: per-vector label string (L,).
    :type celltype: numpy.ndarray
    :param leiden: per-vector original cluster id (L,); required only to split Unknown.
    :type leiden: numpy.ndarray
    :returns: list of unit dicts (``label / mask / is_unknown / source_cluster``),
        ordered by original cluster then label.
    :rtype: list(dict)
    """
    units = []
    named = sorted(lab for lab in set(celltype.tolist())
                   if lab not in DROP_LABELS and lab != UNKNOWN_LABEL)
    for lab in named:
        mask = celltype == lab
        units.append(dict(label=lab, mask=mask, is_unknown=False,
                          source_cluster=-1, _sort=int(leiden[mask].min())))

    umask = celltype == UNKNOWN_LABEL
    if umask.any():
        for j, lc in enumerate(sorted({int(v) for v in leiden[umask]}), 1):
            units.append(dict(label=f'unknown{j}', mask=umask & (leiden == lc),
                              is_unknown=True, source_cluster=int(lc), _sort=int(lc)))

    units.sort(key=lambda u: (u['_sort'], u['label']))
    return units


def build_centroids_from_h5ad(z, h5ad_path, *, column='cellman_annotation',
                              leiden_column='leiden', chunk=20000,
                              store_dir=None, out_csv_name='celltype_index.csv',
                              cmap=None):
    """Group an annotated h5ad into celltypes and write ``cluster_labels`` +
    ``centroids`` (+ ``celltype_names`` and the colour palette) into the open
    zarr group ``z``.

    Grouping is by the label string in ``obs[column]`` (see the module docstring):
    named labels are one class each, ``Unknown`` is split by original cluster into
    ``unknown1..N``, and Background / unfinalized labels drop to -1. This is what
    :meth:`Analysis.map_celltypes` invokes when the store lacks the inputs.

    :param z: open zarr group (mode ``r+``); must hold ``normalized_vectors`` and is
        where ``cluster_labels`` / ``centroids`` are (over)written.
    :type z: zarr.hierarchy.Group
    :param h5ad_path: annotated h5ad, row-aligned with ``normalized_vectors``.
    :type h5ad_path: str
    :param column: obs column with the final celltype label.
    :type column: str
    :param leiden_column: obs column with the original cluster id, used to split
        ``Unknown`` (required only when Unknown vectors are present).
    :type leiden_column: str
    :param chunk: rows per chunk when accumulating centroids.
    :type chunk: int
    :param store_dir: if given, the celltype-index CSV is written here.
    :type store_dir: str or None
    :param out_csv_name: name of the celltype-index CSV.
    :type out_csv_name: str
    :param cmap: palette used to fix the per-celltype colours (see
        :func:`celltype_colors`); ``None`` = qualitative, sized to the class
        count. Stored in ``z.attrs['cellman_celltype_colors']``.
    :type cmap: str or matplotlib.colors.Colormap or None
    :returns: the celltype-index DataFrame (id -> label / is_unknown / source_cluster /
        n_vectors).
    :rtype: pandas.DataFrame
    """
    if 'normalized_vectors' not in z:
        raise ValueError('store has no normalized_vectors; run downsample_vector() first')
    nv = z['normalized_vectors']
    n_total = nv.shape[0]

    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(f'annotated h5ad not found: {h5ad_path}')
    adata = ad.read_h5ad(h5ad_path, backed='r')
    if adata.n_obs != n_total:
        raise ValueError(f'h5ad rows ({adata.n_obs}) != normalized_vectors rows ({n_total}); '
                         'the annotated h5ad must be row-aligned with the store')
    obs = adata.obs.copy()
    adata.file.close()
    if column not in obs.columns:
        raise ValueError(f'column {column!r} not in obs (have: {list(obs.columns)})')

    celltype = obs[column].astype(str).values
    has_unknown = bool((celltype == UNKNOWN_LABEL).any())
    if leiden_column in obs.columns:
        leiden = pd.to_numeric(obs[leiden_column].astype(str), errors='coerce') \
                   .fillna(-1).astype(np.int64).values
    elif has_unknown:
        raise ValueError(
            f'obs has no {leiden_column!r} column but contains Unknown vectors; Unknown '
            'is split into unknown1..N by the original cluster, so that column is required.')
    else:
        leiden = np.zeros(n_total, dtype=np.int64)

    units = _build_units(celltype, leiden)
    if not units:
        raise ValueError(f'no callable celltypes in obs[{column!r}] '
                         '(only Background/unfinalized labels?)')

    labels = np.full(n_total, -1, dtype=np.int64)
    print(f'[map_celltypes] building centroids from {os.path.basename(h5ad_path)}: '
          f'{len(units)} celltypes, {n_total} vectors')
    for new_id, u in enumerate(units):
        labels[u['mask']] = new_id
        tag = 'unknown' if u['is_unknown'] else 'celltype'
        print(f'    id={new_id:3d}  {u["label"]:<32}  [{tag}]  ({int(u["mask"].sum())} vectors)')
    n_labeled = int((labels >= 0).sum())
    print(f'[map_celltypes] labeled {n_labeled}/{n_total} (dropped -1: {n_total - n_labeled})')

    centroids, counts = compute_centroids(nv, labels, len(units), chunk)

    def overwrite(name, arr):
        if name in z:
            del z[name]
        z[name] = arr

    names = [u['label'] for u in units]
    colors = celltype_colors(len(names), cmap=cmap)
    overwrite('cluster_labels', labels)
    overwrite('centroids', centroids)
    overwrite('celltype_names', np.asarray(names))
    z.attrs['cellman_celltype_colors'] = colors
    print(f'[map_celltypes] wrote cluster_labels {labels.shape} + centroids {centroids.shape} '
          f'+ celltype_names ({len(names)}) + colors (normalized_vectors untouched)')

    df = pd.DataFrame([
        dict(cluster=i, celltype=u['label'], is_unknown=u['is_unknown'],
             source_cluster=u['source_cluster'], n_vectors=int(counts[i]))
        for i, u in enumerate(units)])
    if store_dir:
        out_csv = os.path.join(store_dir, out_csv_name)
        df.to_csv(out_csv, index=False)
        print(f'[map_celltypes] wrote {out_csv}')
    return df
