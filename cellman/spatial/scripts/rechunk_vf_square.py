#!/usr/bin/env python
"""Re-chunk an existing ``vf_normalized`` field to small SQUARE spatial tiles.

Why: the blur field is stored ``(n_G, n_R, n_C)`` and zarr decodes whole chunks.
A legacy full-width chunk ``(n_G, chunk_h, n_C)`` forces a sparse voxel gather
(``downsample_vector`` / ``select_vectors_from_image``) to decompress an entire
row band (hundreds of MB) to read a handful of voxels — and since sampled voxels
touch every row band, effectively the whole field is decoded. Re-chunking to
square tiles ``(n_G, tile, tile)`` lets the tile-aware reader decode only the
tiles that actually contain a sampled voxel, skipping every empty tile.

This is a one-time conversion: it reads the old field ONCE (streaming full-width
row bands, so each old chunk is decoded exactly once) and writes a new aligned
array, then swaps it in. The gene axis stays full (every voxel needs all genes);
only the two spatial axes are re-tiled.

Usage
-----
    python rechunk_vf_square.py <zarr_store_dir> [--tile 32] [--band-mult 8]
                                [--array vf_normalized] [--dry-run] [--keep-backup]

<zarr_store_dir> is the SSAM/count store directory (the one holding vf_normalized),
e.g. ``.../stereoseq_brain`` — NOT the SpatialData .zarr.
"""
import argparse
import os
import shutil
import sys
import time

import numpy as np
import zarr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store", help="zarr store dir holding vf_normalized")
    ap.add_argument("--tile", type=int, default=32,
                    help="square store chunk side in bins (default 32)")
    ap.add_argument("--band-mult", type=int, default=8,
                    help="read band height = band_mult * tile rows/pass (RAM vs #passes)")
    ap.add_argument("--array", default="vf_normalized",
                    help="array name to rechunk (default vf_normalized)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (RAM/passes) and exit")
    ap.add_argument("--keep-backup", action="store_true",
                    help="keep the old array as <array>__old instead of deleting it")
    args = ap.parse_args()

    store_path = os.path.abspath(args.store)
    zg = zarr.open(store_path, mode="a")
    if args.array not in zg:
        sys.exit(f"'{args.array}' not found in {store_path} (keys: {list(zg.keys())})")
    src = zg[args.array]
    n_G, n_R, n_C = src.shape
    cur = tuple(int(c) for c in src.chunks)
    tile = int(args.tile)
    print(f"{args.array}: shape {(n_G, n_R, n_C)}, current chunks {cur}, dtype {src.dtype}")

    if cur[1] == tile and cur[2] == tile:
        print("already square-tiled at this size; nothing to do.")
        return

    band_h = max(tile, int(args.band_mult) * tile)
    band_h -= band_h % tile
    band_bytes = n_G * band_h * n_C * 4
    n_pass = -(-n_R // band_h)
    print(f"-> new chunks {(n_G, tile, tile)}  |  read band {band_h} rows "
          f"(~{band_bytes/1e9:.1f} GB/pass, {n_pass} passes)")
    if args.dry_run:
        return
    if band_bytes > 400e9:
        sys.exit(f"band {band_bytes/1e9:.0f} GB too large; lower --band-mult")

    tmp = f"{args.array}__sq_tmp"
    if tmp in zg:
        del zg[tmp]
    dst = zg.create_array(tmp, shape=(n_G, n_R, n_C),
                          chunks=(n_G, tile, tile), dtype=src.dtype, fill_value=0.0)
    for k, v in dict(src.attrs).items():
        dst.attrs[k] = v

    t0 = time.perf_counter()
    for r0 in range(0, n_R, band_h):
        r1 = min(r0 + band_h, n_R)
        dst[:, r0:r1, :] = np.asarray(src[:, r0:r1, :])
        el = time.perf_counter() - t0
        print(f"  rows {r0:>6}:{r1:<6}  {el:6.0f}s", flush=True)

    old_dir = os.path.join(store_path, args.array)
    tmp_dir = os.path.join(store_path, tmp)
    bak_dir = os.path.join(store_path, f"{args.array}__old")
    if os.path.exists(bak_dir):
        shutil.rmtree(bak_dir)
    os.rename(old_dir, bak_dir)
    shutil.move(tmp_dir, old_dir)

    chk = zarr.open(store_path, mode="r")[args.array]
    assert tuple(int(c) for c in chk.chunks) == (n_G, tile, tile), \
        f"verify failed: chunks {chk.chunks}"
    if args.keep_backup:
        print(f"backup kept at {bak_dir}")
    else:
        shutil.rmtree(bak_dir)
    print(f"done: {args.array} rechunked to {(n_G, tile, tile)} "
          f"in {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
