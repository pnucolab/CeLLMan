"""Cluster-merger: decision-aware merge of SSAM clusters.

Reads the annotated h5ad (from `{species}-{organ}-expert` annotate mode) and
groups vectors into post-merge clusters using the annotate decision metadata:

  - DROP (not a callable cell)          : decision in {Unknown, Background,
        needs_subclustering, unresolved_subclustering} or label in those.
  - MERGE (group by celltype_term_id)   : decision=='celltype' AND
        subclass_resolved==True AND non-empty celltype_term_id. Confirmed
        subclasses with the SAME term_id merge (confidence-agnostic).
  - KEEP-SEPARATE (survive, own cluster): decision=='celltype' AND
        (subclass_resolved==False OR empty term_id) = "class-final / unresolved
        subclass". These are real cells but their subtype is unresolved, so two
        leaves sharing the same CLASS term_id are NOT fused — each stays its own
        post-merge cluster (flagged review=True in the CSV). This prevents
        distinct unresolved populations (e.g. several unresolved GABAergic
        clusters) from collapsing into one.

Backward compatible: if the h5ad lacks `decision`/`subclass_resolved`/
`celltype_term_id`, falls back to the legacy behavior (group by
`cellman_annotation` string, excluding Background/Unknown/needs_subclustering).

Then assigns new contiguous integer labels, recomputes centroids/centroids_stdev,
overwrites the zarr arrays, and writes `merged_annotation.csv`.

WARNING — NOT idempotent (shrinks zarr arrays). Aborts if the CSV exists unless
--force. Use --dry-run to preview.

Store layout (current): --store-dir is the SSAMDataset zarr root, which holds the
zarr arrays at its top level and the annotate-mode output under `annotation_out/`:

    <DATASET>/cellman_spatial/            <- pass this as --store-dir
        cluster_labels/ normalized_vectors/ downsampled_vectors/ centroids/ ...
        ssam_clusters.h5ad
        annotation_out/
            ssam_clusters_annotated.h5ad          (obs: leiden, cellman_annotation)
            annotation_artifacts.tar.gz           (_annotate_state.json, _annotate_decisions.json, ...)

The annotated h5ad and the decision/cluster_path metadata are auto-resolved from
`annotation_out/` (loose files or inside annotation_artifacts.tar.gz); no extra
flags are needed. Override with --h5ad-name / --annotate-out if your layout differs.

Usage:
    .venv/bin/python scripts/merge_clusters.py \
        --store-dir <DATASET>/cellman_spatial --dry-run
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
from sklearn import preprocessing


# Labels / decisions that are NOT callable cells -> dropped from the celltype map.
DROP_LABELS = {'Background', 'Unknown', 'needs_subclustering', 'unresolved_subclustering'}
DROP_DECISIONS = {'Background', 'Unknown', 'needs_subclustering', 'unresolved_subclustering'}


def _as_bool(v):
    """Coerce an obs cell (bool / 'True' / 1 / 'true') to bool."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ('true', '1', 'yes')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--store-dir', required=True,
                   help='SSAMDataset zarr root (contains ssam_clusters_annotated.h5ad + zarr arrays).')
    p.add_argument('--h5ad-name', default='ssam_clusters_annotated.h5ad',
                   help='Annotated h5ad filename inside --store-dir (default: %(default)s).')
    p.add_argument('--out-csv-name', default='merged_annotation.csv',
                   help='Output CSV filename inside --store-dir (default: %(default)s).')
    p.add_argument('--force', action='store_true',
                   help='Run even if --out-csv-name exists. Dangerous: zarr re-merge is not idempotent.')
    p.add_argument('--dry-run', action='store_true',
                   help='Compute and print the merge plan without modifying zarr / writing CSV.')
    p.add_argument('--annotate-out', default=None,
                   help='Annotate-mode output dir. Used to recover per-cell cluster_path '
                        '(_annotate_state.json) and decision metadata (_annotate_decisions.json) '
                        'when the h5ad obs lacks them (finalize wrote only cellman_annotation). '
                        'Reads loose JSONs, or extracts them from .annotate_trace.tar.gz.')
    p.add_argument('--legacy', action='store_true',
                   help='Force LEGACY merge: group purely by cellman_annotation string '
                        '(every named celltype -> one cluster; Background/Unknown/needs_subclustering '
                        'dropped). Ignores decision/subclass_resolved metadata even if present.')
    return p.parse_args()


def _load_annotate_json(annotate_out, name):
    """Load a JSON from annotate_out, falling back to .annotate_trace.tar.gz."""
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
    sys.exit(f'[error] {name} not found in {annotate_out} '
             '(loose, or inside annotation_artifacts.tar.gz / .annotate_trace.tar.gz)')


def main():
    args = parse_args()
    store_dir = os.path.abspath(args.store_dir)
    out_csv = os.path.join(store_dir, args.out_csv_name)

    if not os.path.isdir(store_dir):
        sys.exit(f'[error] store dir not found: {store_dir}')

    # Annotate-mode writes the annotated h5ad + trace into an `annotation_out/`
    # subdir of the zarr store root (store_dir = .../cellman_spatial). Resolve the
    # h5ad at the store root first, then fall back to that subdir.
    h5ad_path = os.path.join(store_dir, args.h5ad_name)
    if not os.path.isfile(h5ad_path):
        alt = os.path.join(store_dir, 'annotation_out', args.h5ad_name)
        if os.path.isfile(alt):
            h5ad_path = alt
    if not os.path.isfile(h5ad_path):
        sys.exit(f'[error] h5ad not found: {os.path.join(store_dir, args.h5ad_name)} '
                 f'(also tried annotation_out/{args.h5ad_name})')

    # Default --annotate-out to the standard annotation_out/ subdir when present,
    # so cluster_path / decision metadata can be recovered without an explicit flag
    # (finalize writes only cellman_annotation into the h5ad obs).
    if args.annotate_out is None:
        default_ao = os.path.join(store_dir, 'annotation_out')
        if os.path.isdir(default_ao):
            args.annotate_out = default_ao
    if os.path.exists(out_csv) and not args.force and not args.dry_run:
        sys.exit(f'[error] {out_csv} already exists. Not idempotent — use --force or --dry-run.')

    print(f'[cluster-merger] store-dir : {store_dir}')
    print(f'[cluster-merger] h5ad      : {h5ad_path}')
    print(f'[cluster-merger] dry-run   : {args.dry_run}')

    adata = ad.read_h5ad(h5ad_path, backed='r')
    z = zarr.open(store_dir, mode='r' if args.dry_run else 'r+')

    zarr_cl = np.asarray(z['cluster_labels'])
    h5_cl = adata.obs['leiden'].astype(int).values
    if not np.array_equal(zarr_cl, h5_cl):
        sys.exit('[error] h5ad row order does not match zarr cluster_labels order')
    n_total = len(zarr_cl)
    print(f'[cluster-merger] alignment OK; n_vectors={n_total}')

    obs = adata.obs
    celltype = obs['cellman_annotation'].astype(str).values

    # --- per-cell cluster_path: from obs, else recovered from annotate-out state ---
    if 'cluster_path' in obs.columns:
        cluster_path = obs['cluster_path'].astype(str).values
    elif args.annotate_out:
        st = _load_annotate_json(args.annotate_out, '_annotate_state.json')
        cps = st.get('cluster_paths')
        if cps is None:
            sys.exit('[error] _annotate_state.json has no "cluster_paths"')
        if len(cps) != n_total:
            sys.exit(f'[error] state cluster_paths ({len(cps)}) != n_vectors ({n_total}); '
                     'row order/length mismatch')
        cluster_path = np.array([str(x) for x in cps], dtype=object)
        print(f'[cluster-merger] cluster_path recovered from annotate-out '
              f'({len(set(cluster_path.tolist()))} leaves)')
    else:
        sys.exit('[error] obs lacks cluster_path. Pass --annotate-out <dir> to recover it '
                 'from _annotate_state.json.')

    # --- decision metadata: from obs, else from annotate-out decisions, else legacy ---
    has_dec = ('decision' in obs.columns and 'subclass_resolved' in obs.columns
               and 'celltype_term_id' in obs.columns)
    if args.legacy:
        decision = np.array(['celltype'] * n_total)
        subclass_resolved = np.ones(n_total, dtype=bool)
        term_id = celltype  # legacy: group by label string via term_id slot
        print('[cluster-merger] mode: LEGACY (--legacy; group by cellman_annotation string, '
              'every named celltype -> one cluster)')
    elif has_dec:
        decision = obs['decision'].astype(str).values
        subclass_resolved = np.array([_as_bool(v) for v in obs['subclass_resolved'].values])
        term_id = obs['celltype_term_id'].astype(str).values
        print('[cluster-merger] mode: DECISION-AWARE (decision/subclass_resolved/celltype_term_id present)')
    elif args.annotate_out:
        dec_map = _load_annotate_json(args.annotate_out, '_annotate_decisions.json')
        missing = sorted({cp for cp in cluster_path.tolist() if cp not in dec_map})
        if missing:
            sys.exit(f'[error] {len(missing)} cluster_path(s) absent from _annotate_decisions.json: '
                     f'{missing[:10]}')
        decision = np.array([str(dec_map[cp].get('decision', 'celltype')) for cp in cluster_path])
        subclass_resolved = np.array([_as_bool(dec_map[cp].get('subclass_resolved', True))
                                      for cp in cluster_path])
        term_id = np.array([str(dec_map[cp].get('term_id', '') or '') for cp in cluster_path])
        print('[cluster-merger] mode: DECISION-AWARE (recovered from annotate-out decisions)')
    else:
        decision = np.array(['celltype'] * n_total)
        subclass_resolved = np.ones(n_total, dtype=bool)
        term_id = celltype  # legacy: group by label string via term_id slot
        print('[cluster-merger] mode: LEGACY (no decision columns; group by cellman_annotation string)')

    # --- 1. one record per cluster_path leaf (consistency-checked) ---
    cp_info = {}  # cp -> dict(ct, dec, sr, tid)
    for i in range(n_total):
        cp = cluster_path[i]
        rec = dict(ct=celltype[i], dec=decision[i], sr=bool(subclass_resolved[i]),
                   tid=term_id[i] if term_id[i] not in ('', 'nan', 'None') else '')
        if cp in cp_info:
            if cp_info[cp] != rec:
                sys.exit(f'[error] cluster_path {cp} has inconsistent annotation: {cp_info[cp]} vs {rec}')
        else:
            cp_info[cp] = rec
    n_pre = len(cp_info)
    print(f'[cluster-merger] {n_pre} pre-merge clusters (cluster_path leaves)')

    # --- 2. classify each leaf: drop / merge / keep-separate ---
    merge_units = defaultdict(list)   # term_id -> [cp...]   (subclass_resolved celltypes)
    keep_units = []                   # [cp...]              (class-final / unresolved: each its own unit)
    excluded = []                     # [(cp, ct, reason)]
    for cp, r in cp_info.items():
        ct, dec, sr, tid = r['ct'], r['dec'], r['sr'], r['tid']
        if ct in DROP_LABELS or dec in DROP_DECISIONS:
            excluded.append((cp, ct, f'drop:{dec if has_dec else ct}'))
        elif dec == 'celltype' and sr and tid:
            merge_units[tid].append(cp)               # confirmed subclass -> merge by term_id
        elif dec == 'celltype':                       # class-final / unresolved subclass -> keep separate
            keep_units.append(cp)
        else:                                         # unexpected decision -> drop conservatively
            excluded.append((cp, ct, f'drop:unexpected_decision:{dec}'))

    # --- 3. build ordered post-merge units ---
    def leiden_of(cp):
        try:
            return int(cp.split('.')[0])
        except ValueError:
            return 1 << 30

    units = []  # dict(label, term_id, decision, subclass_resolved, review, cps)
    for tid, cps in merge_units.items():
        labs = {cp_info[cp]['ct'] for cp in cps}
        label = sorted(labs, key=lambda s: (len(s), s))[0]
        if len(labs) > 1:
            print(f'[warn] term_id {tid} has multiple labels {labs}; using {label!r}')
        units.append(dict(label=label, term_id=tid, decision='celltype', subclass_resolved=True,
                          review=False, cps=sorted(cps, key=lambda s: (leiden_of(s), s)),
                          _sort=min(leiden_of(cp) for cp in cps)))
    for cp in keep_units:
        r = cp_info[cp]
        units.append(dict(label=r['ct'], term_id=r['tid'], decision='celltype',
                          subclass_resolved=False, review=True, cps=[cp], _sort=leiden_of(cp)))
    units.sort(key=lambda u: (u['_sort'], u['label']))

    n_post = len(units)
    n_merge = len(merge_units)
    n_keep = len(keep_units)
    n_excluded = len(excluded)
    print(f'\n[cluster-merger] pre={n_pre}  excluded={n_excluded}  '
          f'post={n_post} (merge-units={n_merge}, class-final keep-separate={n_keep})')
    print('[cluster-merger] excluded leaves:')
    for cp, ct, why in sorted(excluded):
        print(f'    {cp:>10}  ({ct})  [{why}]')
    if n_keep:
        print('[cluster-merger] class-final kept SEPARATE (review):')
        for u in units:
            if u['review']:
                print(f'    {u["cps"][0]:>10}  {u["label"]}  (term_id={u["term_id"] or "-"})')

    # --- 4. assign new contiguous labels ---
    new_labels = np.full(n_total, -1, dtype=np.int64)
    print('\n[cluster-merger] post-merge cluster assignments:')
    for new_id, u in enumerate(units):
        mask = np.isin(cluster_path, u['cps'])
        new_labels[mask] = new_id
        tag = 'class-final' if u['review'] else 'subclass'
        print(f'    new_id={new_id:2d}  {u["label"]}  [{tag}] '
              f'({len(u["cps"])} leaf, {int(mask.sum())} vectors)')

    surviving_mask = new_labels != -1
    n_surviving = int(surviving_mask.sum())
    n_dropped = n_total - n_surviving
    print(f'\n[cluster-merger] surviving vectors: {n_surviving} / {n_total} (dropped {n_dropped})')

    adata.file.close()

    # --- 5. slice arrays / recompute centroids ---
    print('[cluster-merger] loading normalized_vectors and downsampled_vectors ...')
    nv = np.asarray(z['normalized_vectors'])
    lm = np.asarray(z['downsampled_vectors'])

    new_cluster_labels = new_labels[surviving_mask].astype(np.int64)
    new_normalized_vectors = nv[surviving_mask, :]
    new_downsampled_vectors = lm[:, surviving_mask]
    print(f'[cluster-merger] new shapes: cluster_labels={new_cluster_labels.shape}, '
          f'normalized_vectors={new_normalized_vectors.shape}, downsampled_vectors={new_downsampled_vectors.shape}')

    print('[cluster-merger] recomputing centroids ...')
    n_clusters = n_post
    n_genes = nv.shape[1]
    centroids = np.zeros((n_clusters, n_genes), dtype=np.float32)
    centroids_stdev = np.zeros((n_clusters, n_genes), dtype=np.float32)
    for k in range(n_clusters):
        sel = new_cluster_labels == k
        cl_vecs = new_normalized_vectors[sel, :].astype(np.float32, copy=True)
        cl_vecs = preprocessing.normalize(cl_vecs, norm='l2', axis=1)
        centroids[k, :] = np.mean(cl_vecs, axis=0)
        centroids_stdev[k, :] = np.std(cl_vecs, axis=0)
    print(f'[cluster-merger] centroids shape: {centroids.shape}')

    if args.dry_run:
        print('\n[cluster-merger] --dry-run; skipping zarr write and CSV write')
    else:
        print('[cluster-merger] writing zarr arrays ...')

        def overwrite(group, name, arr):
            if name in group:
                del group[name]
            group[name] = arr

        overwrite(z, 'cluster_labels', new_cluster_labels)
        overwrite(z, 'filtered_cluster_labels', new_cluster_labels)
        overwrite(z, 'normalized_vectors', new_normalized_vectors)
        overwrite(z, 'downsampled_vectors', new_downsampled_vectors)
        overwrite(z, 'centroids', centroids)
        overwrite(z, 'centroids_stdev', centroids_stdev)
        print('[cluster-merger] zarr write complete')

        rows = []
        for new_id, u in enumerate(units):
            rows.append({
                'cluster': new_id,
                'celltype': u['label'],
                'celltype_term_id': u['term_id'],
                'decision': u['decision'],
                'subclass_resolved': u['subclass_resolved'],
                'review': u['review'],   # True = class-final kept separate; inspect before trusting as a subtype
                'original_clusters': json.dumps(u['cps']),
                'n_member_clusters': len(u['cps']),
            })
        merged_df = pd.DataFrame(rows)
        merged_df.to_csv(out_csv, index=False)
        print(f'[cluster-merger] wrote {out_csv}')
        print(merged_df.to_string(index=False))

    print('\n=== SUMMARY ===')
    print(f'pre-merge clusters (leaves):        {n_pre}')
    print(f'excluded (drop):                    {n_excluded}')
    print(f'post-merge clusters:                {n_post}  '
          f'(merged subclasses={n_merge}, class-final kept-separate={n_keep})')
    print(f'pre-merge vectors:                  {n_total}')
    print(f'post-merge vectors:                 {n_surviving} (dropped {n_dropped})')


if __name__ == '__main__':
    main()
