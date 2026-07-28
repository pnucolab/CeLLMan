"""Slice a SpatialData zarr into a tile grid; save each tile as a separate zarr.

Usage:
    .venv/bin/python scripts/slice_spatialdata.py \\
        --sdata-path  dataset/<DATASET>/spatialdata.zarr \\
        --tile-size   10000 \\
        --output-dir  dataset/<DATASET>/spatialdata_crops
"""
import argparse
import numpy as np
import os
import sys
import dask.array as da
import pandas as pd
import xarray as xr
import spatialdata as sd
from spatialdata import get_extent, bounding_box_query
from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel, PointsModel
from spatialdata.transformations import Affine, Identity, get_transformation


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--sdata-path', required=True,
        help='Path to the input SpatialData zarr.',
    )
    p.add_argument(
        '--tile-size', type=float, required=True,
        help='Tile size in global coordinate units.',
    )
    p.add_argument(
        '--output-dir', required=True,
        help='Output directory for r{row:02d}_c{col:02d}.zarr tiles.',
    )
    p.add_argument(
        '--min-tile-size', type=float, default=None,
        help='Minimum size for the trailing row/col (default: tile_size / 2). '
             'If the last edge segment is smaller than this it is merged into '
             'the previous tile.',
    )
    p.add_argument(
        '--coordinate-system', default='global',
        help="SpatialData coordinate system to use (default: 'global').",
    )
    p.add_argument(
        '--image-key', default=None,
        help='Staining / morphology image key to keep (e.g. H&E or polyT). '
             'If set, all other image elements are dropped from each tile. '
             'Useful when only the staining image is needed downstream '
             '(e.g. ssam.downsample_vector).',
    )
    p.add_argument(
        '--keep-points', nargs='*', default=None,
        help='Whitespace-separated points keys to keep. Default: keep all.',
    )
    p.add_argument(
        '--drop-labels', action='store_true',
        help='Drop all labels elements from each tile.',
    )
    p.add_argument(
        '--drop-shapes', action='store_true',
        help='Drop all shapes elements from each tile.',
    )
    p.add_argument(
        '--drop-tables', action='store_true',
        help='Drop all tables elements from each tile.',
    )
    return p.parse_args()


def _highest_res_dataarray(tree):
    """Return the highest-resolution DataArray from a multiscale DataTree."""
    children = list(tree.items())
    pick_name = 'scale0' if 'scale0' in dict(children) else children[0][0]
    sub = tree[pick_name] if pick_name in tree else children[0][1]
    ds = sub.ds
    var_name = list(ds.data_vars)[0]
    return ds[var_name]


def _rechunk_dataarray(xda, target_chunk=2048):
    arr = xda.data
    if not isinstance(arr, da.Array):
        return xda
    new_chunks = []
    for dim in xda.dims:
        size = xda.sizes[dim]
        if size <= 0:
            new_chunks.append(1)
            continue
        if dim in ('y', 'x'):
            new_chunks.append(min(target_chunk, size))
        elif dim in ('c', 'z'):
            new_chunks.append(size)
        else:
            new_chunks.append(size)
    rechunked = arr.rechunk(tuple(new_chunks))
    return xr.DataArray(rechunked, dims=xda.dims, coords=xda.coords, attrs=xda.attrs, name=xda.name)


def rebuild_raster(elem, kind):
    """Rebuild a raster element so it has regular chunks. Drops multiscale pyramid."""
    try:
        transformations = get_transformation(elem, get_all=True)
        if not isinstance(transformations, dict):
            transformations = {'global': transformations}
    except Exception:
        transformations = None

    if isinstance(elem, xr.DataArray):
        xda = elem
    else:
        xda = _highest_res_dataarray(elem)

    new_xda = _rechunk_dataarray(xda)
    dims = tuple(new_xda.dims)
    if kind == 'image':
        model_cls = Image3DModel if 'z' in dims else Image2DModel
    else:
        model_cls = Labels3DModel if 'z' in dims else Labels2DModel

    parsed = model_cls.parse(new_xda.data, dims=dims, transformations=transformations)
    return parsed


def rechunk_all_rasters(cropped_sdata):
    for k in list(cropped_sdata.images.keys()):
        cropped_sdata.images[k] = rebuild_raster(cropped_sdata.images[k], 'image')
    for k in list(cropped_sdata.labels.keys()):
        cropped_sdata.labels[k] = rebuild_raster(cropped_sdata.labels[k], 'label')


def strip_points_meta_transform(cropped_sdata):
    """Workaround for spatialdata io_points.py JSON serialization bug.

    `points_without_transform.to_parquet(...)` triggers a dask compute that
    reaches `pa.Table.from_pandas(partition)`. pyarrow tries to JSON-encode
    `partition.attrs` for parquet's pandas-metadata, and chokes on Identity().
    spatialdata only deletes attrs['transform'] on the wrapper copy, not on
    `_meta` and not on the partition payload baked into the dask-expr graph.

    Fix: rebuild the dask DataFrame by computing it to pandas, clearing all
    attrs, and re-parsing through PointsModel with the same transformation.
    Tile-cropped points should already fit in memory, so compute() is OK.
    """
    from spatialdata.models import PointsModel
    from spatialdata.transformations import get_transformation

    for k, p in list(cropped_sdata.points.items()):
        try:
            tx = get_transformation(p, get_all=True)
            if not isinstance(tx, dict):
                tx = {'global': tx}
        except Exception:
            tx = None

        spatialdata_attrs = p.attrs.get('spatialdata_attrs', {}) if hasattr(p, 'attrs') else {}
        feature_key = spatialdata_attrs.get('feature_key')

        pdf = p.compute()
        pdf.attrs = {}
        pdf = pdf.reset_index(drop=True)

        kw = {'coordinates': {'x': 'x', 'y': 'y'}, 'transformations': tx}
        if feature_key is not None and feature_key in pdf.columns:
            kw['feature_key'] = feature_key
        cropped_sdata.points[k] = PointsModel.parse(pdf, **kw)


def points_count(p):
    try:
        n = int(p.shape[0].compute()) if hasattr(p.shape[0], 'compute') else int(p.shape[0])
        return n
    except Exception:
        try:
            return len(p)
        except Exception:
            return 0


def slim_cropped(cropped, image_key=None, keep_points=None,
                 drop_labels=False, drop_shapes=False, drop_tables=False):
    """Drop unused elements from a cropped SpatialData in-place.

    Used to reduce per-tile zarr footprint when the downstream consumer
    only needs a subset of elements (e.g. images[image_key] + points for
    ssam.downsample_vector).
    """
    if drop_labels:
        for k in list(cropped.labels.keys()):
            del cropped.labels[k]
    if drop_shapes:
        for k in list(cropped.shapes.keys()):
            del cropped.shapes[k]
    if drop_tables:
        for k in list(cropped.tables.keys()):
            del cropped.tables[k]
    if image_key is not None:
        if image_key not in cropped.images:
            raise KeyError(
                f"--image-key '{image_key}' not found after crop. "
                f"Available images: {list(cropped.images.keys())}"
            )
        for k in list(cropped.images.keys()):
            if k != image_key:
                del cropped.images[k]
    if keep_points is not None:
        keep_set = set(keep_points)
        for k in list(cropped.points.keys()):
            if k not in keep_set:
                del cropped.points[k]
    return cropped


def has_biological_data(cropped):
    """A tile is meaningful only if it contains transcripts or cells (a table)."""
    if cropped is None:
        return False
    for k, p in cropped.points.items():
        if points_count(p) > 0:
            return True
    for k, t in cropped.tables.items():
        if t.n_obs > 0:
            return True
    return False


def normalize_points_indices(sdata):
    """Rewrite every points dask DataFrame with a clean global RangeIndex.

    Two problems we need to fix up-front, before `get_extent` or
    `bounding_box_query`:

    1. Duplicate labels.
       Vizgen/Merscope `detected_transcripts` parquet is read as a
       multi-partition dask DataFrame whose per-partition RangeIndex(0..N_i)
       gives duplicate labels across partitions. `spatialdata.transform()`
       does `pd.Series(values, index=transformed.index)`, which materializes
       that index and trips `ValueError: cannot reindex on an axis with
       duplicate labels`. Both get_extent and bounding_box_query go through
       transform.

    2. 3D-data / 2D-transform mismatch.
       The source has an `x, y, z` column trio but the on-disk transformation
       is 2D. `PointsModel.parse` reports "Column z will be ignored", but the
       column survives, and `get_axes_names(DaskDataFrame)` later scans
       columns and reports 3D — `bounding_box_query` then trips case 3
       (data_dim=3, transform_dim=2, unsupported). Dropping `z` aligns the
       data dimensionality with the transformation.

    We materialize the dataframe to pandas once (Merscope transcripts is
    typically a few hundred MB — fits in RAM), drop `z`, reset_index, clear
    stale attrs, and re-parse through PointsModel. Computing also forces the
    `gene` categorical to known categories, silencing the perf warning that
    PointsModel.parse otherwise emits on a partially-known categorical.
    """
    def _project_to_2d(t):
        if isinstance(t, Identity):
            return Identity()
        try:
            m2d = t.to_affine_matrix(input_axes=('x', 'y'), output_axes=('x', 'y'))
            return Affine(m2d, input_axes=('x', 'y'), output_axes=('x', 'y'))
        except Exception:
            return Identity()

    for k, p in list(sdata.points.items()):
        try:
            tx = get_transformation(p, get_all=True)
            if not isinstance(tx, dict):
                tx = {'global': tx}
        except Exception:
            tx = None

        if tx is not None:
            tx = {cs: _project_to_2d(t) for cs, t in tx.items()}

        sdata_attrs = p.attrs.get('spatialdata_attrs', {}) if hasattr(p, 'attrs') else {}
        feature_key = sdata_attrs.get('feature_key')

        pdf = p.compute()
        drop_cols = [c for c in ('z',) if c in pdf.columns]
        if drop_cols:
            pdf = pdf.drop(columns=drop_cols)
        pdf = pdf.reset_index(drop=True)
        pdf.attrs = {}

        kw = {'coordinates': {'x': 'x', 'y': 'y'}, 'transformations': tx}
        if feature_key is not None and feature_key in pdf.columns:
            kw['feature_key'] = feature_key
        sdata.points[k] = PointsModel.parse(pdf, **kw)


def main():
    args = parse_args()
    SDATA_PATH = args.sdata_path
    TILE_SIZE = float(args.tile_size)
    OUTPUT_DIR = args.output_dir
    MIN_TILE_SIZE = (
        float(args.min_tile_size) if args.min_tile_size is not None else TILE_SIZE / 2.0
    )
    COORDINATE_SYSTEM = args.coordinate_system

    print(f'SDATA_PATH    : {SDATA_PATH}')
    print(f'TILE_SIZE     : {TILE_SIZE}')
    print(f'OUTPUT_DIR    : {OUTPUT_DIR}')
    print(f'MIN_TILE_SIZE : {MIN_TILE_SIZE}')
    print(f'COORD_SYSTEM  : {COORDINATE_SYSTEM}\n')

    sdata = sd.read_zarr(SDATA_PATH)

    print('SpatialData loaded:')
    print(f'  images : {list(sdata.images.keys())}')
    print(f'  labels : {list(sdata.labels.keys())}')
    print(f'  points : {list(sdata.points.keys())}')
    print(f'  shapes : {list(sdata.shapes.keys())}')
    print(f'  tables : {list(sdata.tables.keys())}')

    normalize_points_indices(sdata)

    try:
        extent = get_extent(sdata, coordinate_system=COORDINATE_SYSTEM)
    except ValueError as e:
        if 'duplicate labels' not in str(e):
            raise
        print(f'  WARN: get_extent failed on points ({e}); '
              f'recomputing from images/labels/shapes only.', flush=True)
        extent = get_extent(
            sdata, coordinate_system=COORDINATE_SYSTEM, has_points=False
        )
    x_min_global, x_max_global = extent['x']
    y_min_global, y_max_global = extent['y']

    x_edges = np.arange(x_min_global, x_max_global, TILE_SIZE).tolist() + [x_max_global]
    y_edges = np.arange(y_min_global, y_max_global, TILE_SIZE).tolist() + [y_max_global]

    if len(x_edges) > 2 and (x_edges[-1] - x_edges[-2]) < MIN_TILE_SIZE:
        x_edges.pop(-2)
    if len(y_edges) > 2 and (y_edges[-1] - y_edges[-2]) < MIN_TILE_SIZE:
        y_edges.pop(-2)

    N_COLS = len(x_edges) - 1
    N_ROWS = len(y_edges) - 1

    print(f'\nGlobal extent: x=[{x_min_global:.2f}, {x_max_global:.2f}], y=[{y_min_global:.2f}, {y_max_global:.2f}]')
    print(f'x_edges: {[f"{v:.1f}" for v in x_edges]}')
    print(f'y_edges: {[f"{v:.1f}" for v in y_edges]}')
    print(f'Grid: {N_ROWS} rows x {N_COLS} cols = {N_ROWS * N_COLS} tiles\n')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    saved = 0
    skipped = 0
    saved_files = []

    for row in range(N_ROWS):
        for col in range(N_COLS):
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            out_path = os.path.join(OUTPUT_DIR, f'r{row:02d}_c{col:02d}.zarr')
            print(f'[{row},{col}] x=[{x0:.2f},{x1:.2f}], y=[{y0:.2f},{y1:.2f}] -> {out_path}', flush=True)

            try:
                cropped = bounding_box_query(
                    sdata,
                    axes=('x', 'y'),
                    min_coordinate=[x0, y0],
                    max_coordinate=[x1, y1],
                    target_coordinate_system=COORDINATE_SYSTEM,
                )
            except Exception as e:
                print(f'  ERROR during bounding_box_query: {type(e).__name__}: {e}', flush=True)
                raise

            if not has_biological_data(cropped):
                npts = sum(points_count(p) for p in cropped.points.values()) if cropped is not None else 0
                ntab = sum(t.n_obs for t in cropped.tables.values()) if cropped is not None else 0
                print(f'  -> empty (transcripts={npts}, cells={ntab}), skipping', flush=True)
                skipped += 1
                continue

            slim_cropped(
                cropped,
                image_key=args.image_key,
                keep_points=args.keep_points,
                drop_labels=args.drop_labels,
                drop_shapes=args.drop_shapes,
                drop_tables=args.drop_tables,
            )
            rechunk_all_rasters(cropped)
            strip_points_meta_transform(cropped)

            try:
                cropped.write(out_path, overwrite=True)
            except Exception as e:
                print(f'  ERROR during write: {type(e).__name__}: {e}', flush=True)
                raise

            try:
                written = sd.read_zarr(out_path)
                npts = sum(points_count(p) for p in written.points.values())
                ncells = sum(t.n_obs for t in written.tables.values())
                print(f'  -> saved (images={list(written.images.keys())}, labels={list(written.labels.keys())}, '
                      f'points={list(written.points.keys())}, shapes={list(written.shapes.keys())}, '
                      f'tables={list(written.tables.keys())}, transcripts={npts}, cells={ncells})', flush=True)
            except Exception as e:
                print(f'  -> saved but failed to verify: {e}', flush=True)

            saved += 1
            saved_files.append(os.path.basename(out_path))

    print(f'\nDone.')
    print(f'  Total grid cells : {N_ROWS * N_COLS}')
    print(f'  Tiles saved      : {saved}')
    print(f'  Tiles skipped    : {skipped}')
    print(f'  Output directory : {OUTPUT_DIR}')
    print(f'\nSaved files:')
    for f in saved_files:
        print(f'  {f}')


if __name__ == '__main__':
    main()
