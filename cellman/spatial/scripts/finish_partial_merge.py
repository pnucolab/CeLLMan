"""Finish a partial cluster-merge.

If `cluster_labels` and `normalized_vectors` in the SSAMDataset zarr have
already been shrunk to post-merge form (Background/Unknown rows dropped,
re-labeled 0..N-1) — typically by the annotate-mode step — but the rest of
the merge artifacts are still pre-merge:

    cluster_labels          POST-merge   (length S = surviving rows)
    normalized_vectors      POST-merge   (length S)
    downsampled_vectors              PRE-merge    (length T = full annotated h5ad)
    centroids               PRE-merge    (n_leiden × n_genes)
    centroids_stdev         PRE-merge    (n_leiden × n_genes)
    merged_annotation.csv   missing

… this script finishes the merge:
    1. Read h5ad, build surviving mask (~Background, ~Unknown).
    2. Assert surviving count == zarr cluster_labels length.
    3. Walk h5ad surviving rows in order; verify that each zarr cluster id
       maps to a unique celltype_annotation label (alignment check).
    4. Recompute centroids/centroids_stdev from current zarr
       normalized_vectors grouped by current zarr cluster_labels (l2-normalize
       first, mirroring SSAM `_calc_centroid`).
    5. Shrink downsampled_vectors to surviving rows.
    6. Write merged_annotation.csv (post-merge id -> celltype + member
       cluster_paths).

Usage:
    .venv/bin/python scripts/finish_partial_merge.py \\
        --store-dir test/human_breast_ontology_bg_test
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


EXCLUDE_LABELS = {'Background', 'Unknown'}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--store-dir', required=True)
    p.add_argument('--h5ad-name', default='ssam_clusters_annotated.h5ad')
    p.add_argument('--out-csv-name', default='merged_annotation.csv')
    p.add_argument('--force', action='store_true',
                   help='Overwrite existing merged_annotation.csv.')
    p.add_argument('--dry-run', action='store_true',
                   help='Compute and print plan; do not modify zarr or write CSV.')
    return p.parse_args()


def main():
    args = parse_args()
    store_dir = os.path.abspath(args.store_dir)
    h5ad_path = os.path.join(store_dir, args.h5ad_name)
    out_csv = os.path.join(store_dir, args.out_csv_name)

    if not os.path.isdir(store_dir):
        sys.exit(f'[error] store dir not found: {store_dir}')
    if not os.path.isfile(h5ad_path):
        sys.exit(f'[error] h5ad not found: {h5ad_path}')
    if os.path.exists(out_csv) and not args.force and not args.dry_run:
        sys.exit(f'[error] {out_csv} already exists. Use --force to overwrite.')

    print(f'[finish-merge] store-dir : {store_dir}')
    print(f'[finish-merge] h5ad      : {h5ad_path}')
    print(f'[finish-merge] dry-run   : {args.dry_run}')

    adata = ad.read_h5ad(h5ad_path, backed='r')
    z = zarr.open(store_dir, mode='r' if args.dry_run else 'r+')

    n_total = adata.n_obs
    cl_z = np.asarray(z['cluster_labels'])
    nv_shape = z['normalized_vectors'].shape
    lm_shape = z['downsampled_vectors'].shape
    print(f'[finish-merge] h5ad rows                : {n_total}')
    print(f'[finish-merge] zarr cluster_labels      : {cl_z.shape}, '
          f'uniques={len(np.unique(cl_z))}')
    print(f'[finish-merge] zarr normalized_vectors  : {nv_shape}')
    print(f'[finish-merge] zarr downsampled_vectors          : {lm_shape}')

    celltype = adata.obs['celltype_annotation'].astype(str).values
    cluster_path = adata.obs['cluster_path'].astype(str).values
    surviving_mask = ~np.isin(celltype, list(EXCLUDE_LABELS))
    n_surviving = int(surviving_mask.sum())
    n_excluded = n_total - n_surviving
    print(f'[finish-merge] surviving (non-B/U) rows  : {n_surviving}')
    print(f'[finish-merge] excluded (B + U)          : {n_excluded}')

    if cl_z.shape[0] != n_surviving:
        sys.exit(
            f'[error] zarr cluster_labels length ({cl_z.shape[0]}) does not '
            f'match h5ad surviving rows ({n_surviving}). zarr is not in the '
            f'expected partial-merge state — abort.'
        )
    if nv_shape[0] != n_surviving:
        sys.exit(
            f'[error] zarr normalized_vectors length ({nv_shape[0]}) does not '
            f'match h5ad surviving rows ({n_surviving}). abort.'
        )
    if lm_shape[1] != n_total:
        sys.exit(
            f'[error] zarr downsampled_vectors length ({lm_shape[1]}) does not match '
            f'h5ad full row count ({n_total}). abort.'
        )

    surv_celltype = celltype[surviving_mask]
    surv_cluster_path = cluster_path[surviving_mask]
    cl_to_ct = {}
    for cid, ct in zip(cl_z, surv_celltype):
        if cid in cl_to_ct:
            if cl_to_ct[cid] != ct:
                sys.exit(
                    f'[error] zarr cluster id {cid} maps to multiple celltypes: '
                    f'{cl_to_ct[cid]} vs {ct}. row order alignment is broken.'
                )
        else:
            cl_to_ct[cid] = ct
    n_post = len(cl_to_ct)
    print(f'[finish-merge] alignment OK; {n_post} post-merge cluster(s)')
    for cid in sorted(cl_to_ct):
        print(f'    new_id={cid:2d}  ->  {cl_to_ct[cid]}')

    cp_to_ct = {}
    for cp, ct in zip(surv_cluster_path, surv_celltype):
        if cp in cp_to_ct and cp_to_ct[cp] != ct:
            sys.exit(
                f'[error] cluster_path {cp} has multiple celltype assignments: '
                f'{cp_to_ct[cp]} vs {ct}'
            )
        cp_to_ct[cp] = ct
    celltype_to_cps = defaultdict(list)
    for cp, ct in cp_to_ct.items():
        celltype_to_cps[ct].append(cp)

    print('[finish-merge] loading normalized_vectors ...')
    nv = np.asarray(z['normalized_vectors'])
    n_genes = nv.shape[1]
    centroids = np.zeros((n_post, n_genes), dtype=np.float32)
    centroids_stdev = np.zeros((n_post, n_genes), dtype=np.float32)
    for k in sorted(cl_to_ct):
        sel = cl_z == k
        cl_vecs = nv[sel, :].astype(np.float32, copy=True)
        cl_vecs = preprocessing.normalize(cl_vecs, norm='l2', axis=1)
        centroids[k, :] = np.mean(cl_vecs, axis=0)
        centroids_stdev[k, :] = np.std(cl_vecs, axis=0)
    print(f'[finish-merge] centroids shape          : {centroids.shape}')

    print('[finish-merge] loading downsampled_vectors ...')
    lm = np.asarray(z['downsampled_vectors'])
    new_downsampled_vectors = lm[:, surviving_mask]
    print(f'[finish-merge] new downsampled_vectors shape     : {new_downsampled_vectors.shape}')

    adata.file.close()

    if args.dry_run:
        print('\n[finish-merge] --dry-run; skipping zarr write and CSV write')
    else:
        print('[finish-merge] writing zarr arrays ...')

        def overwrite(group, name, arr):
            if name in group:
                del group[name]
            group[name] = arr

        overwrite(z, 'centroids', centroids)
        overwrite(z, 'centroids_stdev', centroids_stdev)
        overwrite(z, 'downsampled_vectors', new_downsampled_vectors)
        overwrite(z, 'filtered_cluster_labels', cl_z.astype(np.int64))
        print('[finish-merge] zarr write complete')

        rows = []
        for cid in sorted(cl_to_ct):
            ct = cl_to_ct[cid]
            member_cps = sorted(celltype_to_cps[ct], key=lambda s: (len(s), s))
            rows.append({
                'cluster': cid,
                'celltype': ct,
                'original_clusters': json.dumps(member_cps),
                'n_member_clusters': len(member_cps),
            })
        merged_df = pd.DataFrame(rows)
        merged_df.to_csv(out_csv, index=False)
        print(f'[finish-merge] wrote {out_csv}')
        print(merged_df.to_string(index=False))

    print('\n=== SUMMARY ===')
    print(f'h5ad total rows                 : {n_total}')
    print(f'excluded (Background + Unknown) : {n_excluded}')
    print(f'post-merge clusters             : {n_post}')
    print(f'post-merge vectors              : {n_surviving}')


if __name__ == '__main__':
    main()
