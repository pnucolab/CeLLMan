"""Build the celltype-mapping inputs directly from an annotated h5ad.

Replaces the clustering-time centroids + post-hoc `merge_clusters.py` round trip:
`cluster_vectors()` no longer needs to produce centroids at all, because the
centroids merge_clusters wrote were never a combination of the pre-merge ones --
they were recomputed from scratch out of `normalized_vectors`. So we compute them
ONCE here, after annotation, straight from the final celltype grouping.

Writes into the SSAMDataset zarr store exactly what `map_celltypes` (correlation)
and `fit_celltype_tsvm` read:

    cluster_labels           (L,)    per-vector celltype index, -1 = not callable
    filtered_cluster_labels  (L,)    same array (kept for API compatibility)
    centroids                (k, g)  mean of the row-L2-normalized member vectors
    centroids_stdev          (k, g)  stdev of the same
    celltype_index.csv               id -> label / term_id / n_vectors / review

Unlike merge_clusters.py this is NON-DESTRUCTIVE and IDEMPOTENT: `normalized_vectors`
and `downsampled_vectors` are never sliced or rewritten. Dropped vectors are marked
with the store's existing -1 sentinel, and `fit_celltype_tsvm` filters them out.
Re-run it as often as you like (e.g. after re-annotating).

Grouping (identical to merge_clusters.py):

  - DROP (-1)          : decision/label in {Unknown, Background, needs_subclustering,
                         unresolved_subclustering}.
  - MERGE by term_id   : decision=='celltype' AND subclass_resolved AND term_id.
  - KEEP-SEPARATE      : decision=='celltype' but class-final / unresolved subclass --
                         each cluster_path leaf stays its own celltype (review=True),
                         so distinct unresolved populations are not fused.
  - --legacy           : group purely by the label string in --column (needs no
                         cluster_path / decision metadata).

Usage:
    .venv/bin/python scripts/build_mapping_inputs.py \
        --store-dir <DATASET> --annotate-out <DATASET>/annotation_out \
        --column cellman_annotation [--legacy] [--dry-run]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import zarr


DROP_LABELS = {'Background', 'Unknown', 'needs_subclustering', 'unresolved_subclustering'}
DROP_DECISIONS = DROP_LABELS


def _as_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ('true', '1', 'yes')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--store-dir', required=True,
                   help='SSAMDataset zarr root (holds normalized_vectors, vf_norm, ...).')
    p.add_argument('--annotate-out', default=None,
                   help='Annotate-mode output dir (default: <store-dir>/annotation_out). '
                        'Source of the annotated h5ad and, when obs lacks them, of '
                        'cluster_path / decision metadata.')
    p.add_argument('--h5ad-name', default=None,
                   help='Annotated h5ad filename; default = auto-discover *_annotated.h5ad.')
    p.add_argument('--column', default='cellman_annotation',
                   help='obs column with the final celltype label (default: %(default)s). '
                        'Use cellman_free_annotation for the interactive-phase result '
                        '(pair it with --legacy: decision metadata describes the standard labels).')
    p.add_argument('--leiden-column', default='leiden',
                   help='obs column with the pre-merge cluster id, used only to order the '
                        'output celltype ids (default: %(default)s).')
    p.add_argument('--out-csv-name', default='celltype_index.csv')
    p.add_argument('--legacy', action='store_true',
                   help='Group purely by the --column label string.')
    p.add_argument('--chunk', type=int, default=20000,
                   help='Rows per chunk when accumulating centroids (default: %(default)s).')
    p.add_argument('--dry-run', action='store_true',
                   help='Print the grouping plan; write nothing.')
    return p.parse_args()


def _load_annotate_json(annotate_out, name):
    import tarfile
    loose = os.path.join(annotate_out, name)
    if os.path.isfile(loose):
        with open(loose) as fh:
            return json.load(fh)
    for tarname in ('annotation_artifacts.tar.gz', '.annotate_trace.tar.gz'):
        tar = os.path.join(annotate_out, tarname)
        if os.path.isfile(tar):
            with tarfile.open(tar) as tf:
                members = [m for m in tf.getnames() if m.endswith(name)]
                if members:
                    return json.load(tf.extractfile(members[0]))
    sys.exit(f'[error] {name} not found in {annotate_out}')


def _find_h5ad(search_dirs, h5ad_name):
    import glob as _glob
    if h5ad_name:
        for d in search_dirs:
            cand = os.path.join(d, h5ad_name)
            if os.path.isfile(cand):
                return cand
    for d in search_dirs:
        hits = sorted(_glob.glob(os.path.join(d, '*_annotated.h5ad')))
        if len(hits) == 1:
            print(f'[build-mapping] auto-discovered annotated h5ad: {os.path.basename(hits[0])}')
            return hits[0]
        if len(hits) > 1:
            sys.exit(f'[error] multiple *_annotated.h5ad in {d}: '
                     f'{[os.path.basename(h) for h in hits]}. Disambiguate with --h5ad-name.')
    sys.exit(f'[error] annotated h5ad not found in: {search_dirs}')


def build_units(obs, n_total, *, column, legacy, annotate_out, leiden):
    """-> (units, drop_reason) where units = ordered list of celltype groups.

    Each unit: dict(label, term_id, review, mask, cps). `mask` is a boolean
    per-vector membership array; unassigned vectors get label -1.
    """
    celltype = obs[column].astype(str).values

    has_dec = all(c in obs.columns for c in ('decision', 'subclass_resolved', 'celltype_term_id'))
    if legacy or not (has_dec or annotate_out):
        print('[build-mapping] mode: LEGACY (group by label string)')
        keep = ~np.isin(celltype, list(DROP_LABELS))
        units = []
        for lab in sorted(set(celltype[keep].tolist())):
            mask = keep & (celltype == lab)
            units.append(dict(label=lab, term_id='', review=False, mask=mask, cps=[],
                              _sort=int(leiden[mask].min())))
        units.sort(key=lambda u: (u['_sort'], u['label']))
        return units, celltype

    # ---- decision-aware: needs per-cell cluster_path + decision metadata ----
    if 'cluster_path' in obs.columns:
        cluster_path = obs['cluster_path'].astype(str).values
    else:
        st = _load_annotate_json(annotate_out, '_annotate_state.json')
        cps = st.get('cluster_paths')
        if cps is None:
            sys.exit('[error] _annotate_state.json has no "cluster_paths"')
        if len(cps) != n_total:
            sys.exit(f'[error] state cluster_paths ({len(cps)}) != n_vectors ({n_total})')
        cluster_path = np.array([str(x) for x in cps], dtype=object)
        print(f'[build-mapping] cluster_path recovered from annotate-out '
              f'({len(set(cluster_path.tolist()))} leaves)')

    if has_dec:
        decision = obs['decision'].astype(str).values
        subclass_resolved = np.array([_as_bool(v) for v in obs['subclass_resolved'].values])
        term_id = obs['celltype_term_id'].astype(str).values
        print('[build-mapping] mode: DECISION-AWARE (from h5ad obs)')
    else:
        dec_map = _load_annotate_json(annotate_out, '_annotate_decisions.json')
        missing = sorted({cp for cp in cluster_path.tolist() if cp not in dec_map})
        if missing:
            sys.exit(f'[error] {len(missing)} cluster_path(s) absent from _annotate_decisions.json: '
                     f'{missing[:10]}')
        decision = np.array([str(dec_map[cp].get('decision', 'celltype')) for cp in cluster_path])
        subclass_resolved = np.array([_as_bool(dec_map[cp].get('subclass_resolved', True))
                                      for cp in cluster_path])
        term_id = np.array([str(dec_map[cp].get('term_id', '') or '') for cp in cluster_path])
        print('[build-mapping] mode: DECISION-AWARE (recovered from annotate-out decisions)')

    cp_info = {}
    for i in range(n_total):
        cp = cluster_path[i]
        rec = dict(ct=celltype[i], dec=decision[i], sr=bool(subclass_resolved[i]),
                   tid=term_id[i] if term_id[i] not in ('', 'nan', 'None') else '')
        if cp in cp_info and cp_info[cp] != rec:
            sys.exit(f'[error] cluster_path {cp} has inconsistent annotation: {cp_info[cp]} vs {rec}')
        cp_info[cp] = rec
    print(f'[build-mapping] {len(cp_info)} pre-merge clusters (cluster_path leaves)')

    merge_units = defaultdict(list)
    keep_units, excluded = [], []
    for cp, r in cp_info.items():
        if r['ct'] in DROP_LABELS or r['dec'] in DROP_DECISIONS:
            excluded.append((cp, r['ct'], f'drop:{r["dec"]}'))
        elif r['dec'] == 'celltype' and r['sr'] and r['tid']:
            merge_units[r['tid']].append(cp)
        elif r['dec'] == 'celltype':
            keep_units.append(cp)
        else:
            excluded.append((cp, r['ct'], f'drop:unexpected_decision:{r["dec"]}'))

    def leiden_of(cp):
        try:
            return int(str(cp).split('.')[0])
        except ValueError:
            return 1 << 30

    units = []
    for tid, cps in merge_units.items():
        labs = {cp_info[cp]['ct'] for cp in cps}
        label = sorted(labs, key=lambda s: (len(s), s))[0]
        if len(labs) > 1:
            print(f'[warn] term_id {tid} has multiple labels {labs}; using {label!r}')
        units.append(dict(label=label, term_id=tid, review=False,
                          mask=np.isin(cluster_path, cps),
                          cps=sorted(cps, key=lambda s: (leiden_of(s), s)),
                          _sort=min(leiden_of(cp) for cp in cps)))
    for cp in keep_units:
        units.append(dict(label=cp_info[cp]['ct'], term_id=cp_info[cp]['tid'], review=True,
                          mask=(cluster_path == cp), cps=[cp], _sort=leiden_of(cp)))
    units.sort(key=lambda u: (u['_sort'], u['label']))

    print(f'[build-mapping] excluded leaves ({len(excluded)}):')
    for cp, ct, why in sorted(excluded):
        print(f'    {cp:>10}  ({ct})  [{why}]')
    if keep_units:
        print('[build-mapping] class-final kept SEPARATE (review):')
        for u in units:
            if u['review']:
                print(f'    {u["cps"][0]:>10}  {u["label"]}  (term_id={u["term_id"] or "-"})')
    return units, celltype


def compute_centroids(nv, labels, n_clusters, chunk):
    """Streaming mean/stdev of the row-L2-normalized member vectors, per cluster.

    Chunked so the (L, g) matrix is never fully resident; accumulates in float64,
    which also makes the result independent of the chunk size.
    """
    n_genes = nv.shape[1]
    acc = np.zeros((n_clusters, n_genes), dtype=np.float64)
    acc_sq = np.zeros((n_clusters, n_genes), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for start in range(0, nv.shape[0], chunk):
        stop = min(start + chunk, nv.shape[0])
        lab = labels[start:stop]
        sel = lab >= 0
        if not sel.any():
            continue
        block = np.asarray(nv[start:stop])[sel].astype(np.float64, copy=False)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        block = block / np.where(norms == 0, 1.0, norms)      # row L2, as merge did
        np.add.at(acc, lab[sel], block)
        np.add.at(acc_sq, lab[sel], block ** 2)
        np.add.at(counts, lab[sel], 1)
    if (counts == 0).any():
        sys.exit(f'[error] empty cluster(s): {np.flatnonzero(counts == 0).tolist()}')
    n = counts[:, None]
    mean = acc / n
    var = np.maximum(acc_sq / n - mean ** 2, 0.0)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32), counts


def main():
    args = parse_args()
    store_dir = os.path.abspath(args.store_dir)
    if not os.path.isdir(store_dir):
        sys.exit(f'[error] store dir not found: {store_dir}')
    if args.annotate_out is None:
        default_ao = os.path.join(store_dir, 'annotation_out')
        args.annotate_out = default_ao if os.path.isdir(default_ao) else None

    search_dirs = [d for d in (args.annotate_out, store_dir,
                               os.path.join(store_dir, 'annotation_out')) if d]
    h5ad_path = _find_h5ad(search_dirs, args.h5ad_name)
    out_csv = os.path.join(store_dir, args.out_csv_name)

    print(f'[build-mapping] store-dir : {store_dir}')
    print(f'[build-mapping] h5ad      : {h5ad_path}')
    print(f'[build-mapping] dry-run   : {args.dry_run}')

    z = zarr.open(store_dir, mode='r' if args.dry_run else 'r+')
    if 'normalized_vectors' not in z:
        sys.exit('[error] store has no normalized_vectors; run downsample_vector() first')
    nv = z['normalized_vectors']
    n_total = nv.shape[0]

    adata = ad.read_h5ad(h5ad_path, backed='r')
    if adata.n_obs != n_total:
        sys.exit(f'[error] h5ad rows ({adata.n_obs}) != normalized_vectors rows ({n_total}); '
                 'the annotated h5ad must be row-aligned with the store')
    obs = adata.obs.copy()
    adata.file.close()
    if args.column not in obs.columns:
        sys.exit(f'[error] --column {args.column!r} not in obs (have: {list(obs.columns)})')
    print(f'[build-mapping] label col : {args.column}   (n_vectors={n_total})')

    if args.leiden_column in obs.columns:
        # Coerced, not cast: this column is whatever the user clustered with, so it need
        # not be numeric (string cluster names, or empty for vectors the clustering left
        # out). Non-numeric values get a sentinel id so they simply sort last. Used ONLY
        # to order the output celltype ids -- never for grouping.
        leiden = pd.to_numeric(obs[args.leiden_column].astype(str), errors='coerce') \
                   .fillna(1 << 30).astype(np.int64).values
    else:
        leiden = np.zeros(n_total, dtype=int)
        print(f'[build-mapping] no {args.leiden_column!r} column; celltype ids ordered by label')

    units, _ = build_units(obs, n_total, column=args.column, legacy=args.legacy,
                           annotate_out=args.annotate_out, leiden=leiden)

    labels = np.full(n_total, -1, dtype=np.int64)
    print(f'\n[build-mapping] {len(units)} celltypes:')
    for new_id, u in enumerate(units):
        labels[u['mask']] = new_id
        tag = 'class-final' if u['review'] else 'celltype'
        print(f'    id={new_id:3d}  {u["label"]}  [{tag}]  ({int(u["mask"].sum())} vectors)')
    n_labeled = int((labels >= 0).sum())
    print(f'\n[build-mapping] labeled vectors: {n_labeled} / {n_total} '
          f'(unlabeled -1: {n_total - n_labeled})')

    print('[build-mapping] computing centroids ...')
    centroids, centroids_stdev, counts = compute_centroids(nv, labels, len(units), args.chunk)
    print(f'[build-mapping] centroids shape: {centroids.shape}')

    if args.dry_run:
        print('\n[build-mapping] --dry-run; nothing written')
        return

    def overwrite(name, arr):
        if name in z:
            del z[name]
        z[name] = arr

    overwrite('cluster_labels', labels)
    overwrite('filtered_cluster_labels', labels)
    overwrite('centroids', centroids)
    overwrite('centroids_stdev', centroids_stdev)
    print('[build-mapping] wrote cluster_labels / filtered_cluster_labels / '
          'centroids / centroids_stdev (normalized_vectors untouched)')

    df = pd.DataFrame([
        dict(cluster=i, celltype=u['label'], celltype_term_id=u['term_id'],
             review=u['review'], n_vectors=int(counts[i]),
             original_clusters=json.dumps(u['cps']), n_member_clusters=len(u['cps']))
        for i, u in enumerate(units)])
    df.to_csv(out_csv, index=False)
    print(f'[build-mapping] wrote {out_csv}')
    print(df.drop(columns=['original_clusters']).to_string(index=False))


if __name__ == '__main__':
    main()
