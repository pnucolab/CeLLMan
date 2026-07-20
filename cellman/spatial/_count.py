"""Count-data vector-field backend for cellman.spatial.

For count platforms the KDE-equivalent vector field is built by **rasterizing bin
counts onto a grid and GPU Gaussian-blurring** it (:func:`compute_vf_blur`),
persisting only ``vf_normalized`` + ``vf_norm`` to the SSAM zarr store. Voxels are
then sampled from the tissue image and *looked up* in that field
(:func:`select_vectors_from_image`) — the count analogue of the transcript path's
KDE-at-sampled-points. Cluster centroids are mapped over the field by correlation
(:func:`map_corr`); the TSVM mapper lives in :mod:`cellman.spatial._tsvm`.

All functions take the ``SSAMAnalysis`` instance so they can read/write
``analysis.dataset.zarr_group`` and honour ``analysis.verbose``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from scipy.sparse import issparse

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(x, **k):
        return x


# ---------------------------------------------------------------------------
# GPU separable Gaussian blur
# ---------------------------------------------------------------------------
def _gaussian_kernel_1d(sigma_bin, device, prune_coefficient=4.3):
    import torch
    # kernel truncation radius = prune_coefficient · σ (in bins). SSAM's default
    # is 4.3 (radius ≈ FWTM ≈ cell diameter); 3.0 ≈ 99.7% mass, tighter/faster.
    radius = int(np.ceil(prune_coefficient * sigma_bin))
    k = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-k ** 2 / (2 * sigma_bin ** 2)); g /= g.sum()
    return g.view(1, 1, 1, -1), g.view(1, 1, -1, 1), radius


def _iter_spatial_chunks(n_R, chunk_h, radius):
    for r0 in range(0, n_R, chunk_h):
        r1 = min(r0 + chunk_h, n_R)
        yield r0, r1, max(0, r0 - radius), min(n_R, r1 + radius)


def _rasterize_blur_tile_gpu(Xt, Rt, Ct, rp0, rp1, cp0, cp1, n_G,
                             kx_g, ky_g, radius, device, dtype):
    """Scatter one (row band × col band) TILE of pre-sliced bins onto a GPU tensor
    and separable-Gaussian-blur it.

    ``Xt``/``Rt``/``Ct`` are ALREADY the bins of this tile (raster row in
    ``[rp0, rp1)`` AND raster col in ``[cp0, cp1)`` — the caller slices the
    column-sorted band with ``searchsorted``, so no boolean re-scan happens here).
    Only the nonzero coo inside the tile is transferred (no dense CPU array), and
    the conv runs in ``dtype`` (fp32). ``kx_g``/``ky_g`` are the per-gene expanded
    kernels, built ONCE by the caller (constant across tiles).

    The tile carries a ``±radius`` halo on every side (the caller sets ``rp0/rp1``
    and ``cp0/cp1`` accordingly), so the truncated separable Gaussian is EXACT for
    the interior output rows/cols the caller keeps — the halo rows/cols absorb the
    zero-padded edge effect and are discarded. Bounds GPU memory to
    ``n_G × (band_h + 2r) × (band_w + 2r)`` regardless of the full raster size."""
    import torch
    import torch.nn.functional as F
    h = rp1 - rp0
    w = cp1 - cp0
    raster = torch.zeros((n_G, h, w), device=device, dtype=dtype)       # GPU zeros (no CPU dense)
    if Xt.shape[0] > 0:
        sub = Xt.tocoo()
        sub_R = (Rt - rp0)[sub.row]
        sub_C = (Ct - cp0)[sub.row]
        g = torch.as_tensor(sub.col, device=device, dtype=torch.long)
        r = torch.as_tensor(sub_R, device=device, dtype=torch.long)
        c = torch.as_tensor(sub_C, device=device, dtype=torch.long)
        v = torch.as_tensor(sub.data, device=device, dtype=dtype)
        raster.index_put_((g, r, c), v, accumulate=True)               # transfer only nnz
    t = raster.unsqueeze(0)
    t = F.conv2d(t, ky_g, padding=(radius, 0), groups=n_G)
    t = F.conv2d(t, kx_g, padding=(0, radius), groups=n_G)
    return t.squeeze(0)


def _l1_normalize_log1p(blurred_gpu, scale=10.0, eps=1e-12):
    import torch
    vf_norm = blurred_gpu.sum(dim=0)
    blurred_gpu.div_(torch.clamp(vf_norm, min=eps).unsqueeze(0)).mul_(scale)
    torch.log1p_(blurred_gpu)
    return blurred_gpu, vf_norm


def _tensorstore_array(group_dir, name, shape, chunks):
    """Open a store array as a tensorstore zarr-v3 array whose metadata MIRRORS
    the zarr-python writer EXACTLY (float32, bytes+zstd level 0, ``default``
    '/'-separated chunk keys, fill 0.0) so the on-disk store is read-back-identical
    and any zarr-v3 reader (zarr-python, the downstream sparse gather) sees no
    difference. Works for the 3D count field ``(n_G, R, C)`` and the 4D transcript
    field ``(n_G, vx, vy, vz)`` alike.

    Routing the large ``vf_normalized`` write through tensorstore's C++
    multithreaded async pipeline is ~4.7× faster than the single-event-loop
    zarr-python writer (~2.6× end-to-end since async writes overlap the next tile's
    GPU work) — the write is the dominant cost. Values are bit-identical (same zstd
    codec; lossless). Returns ``None`` if tensorstore is not installed, the store is
    not a local directory, or ``CELLMAN_NO_TENSORSTORE`` is set — the caller then
    falls back to the zarr-python writer.
    """
    if group_dir is None:
        return None
    if os.environ.get("CELLMAN_NO_TENSORSTORE"):       # explicit opt-out → zarr writer
        return None
    try:
        import tensorstore as ts
    except Exception:  # noqa: BLE001
        return None
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": os.path.join(group_dir, name)},
        "metadata": {
            "shape": [int(s) for s in shape],
            "data_type": "float32",
            "chunk_grid": {"name": "regular",
                           "configuration": {"chunk_shape": [int(c) for c in chunks]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": 0.0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}},
                       {"name": "zstd", "configuration": {"level": 0, "checksum": False}}],
        },
    }
    return ts.open(spec, create=True, delete_existing=True).result()


def _tensorstore_vf(group_dir, n_G, out_R, out_C, sh, sw):
    """Back-compat thin wrapper: the 3D count ``vf_normalized`` writer."""
    return _tensorstore_array(group_dir, "vf_normalized",
                              (n_G, out_R, out_C), (n_G, sh, sw))


def compute_vf_blur(analysis, bin_adata, *, sigma_bin: float = 1.0,
                    chunk_h: int = 64, chunk_w: int = 512, bin_size_um: float = 2.0,
                    sampling_distance: float = None, prune_coefficient: float = 4.3,
                    size_after_normalization: float = 10.0,
                    store_chunk=32,
                    device: str = "cuda") -> None:
    """Rasterize ``bin_adata`` counts and GPU-blur into ``vf_normalized`` (zarr).

    Two independent resolutions:
      * ``bin_size_um`` — the **blur raster** grid the Gaussian is computed on
        (finer = more accurate blur; σ in bins = ``sigma_bin``, bandwidth =
        ``sigma_bin * bin_size_um`` µm).
      * ``sampling_distance`` — the **output vector-field** grid spacing. The
        blurred field is point-sampled (decimated) by ``stride =
        round(sampling_distance / bin_size_um)`` → a coarser vf than the raster
        (fewer voxels on disk) while the blur itself stays raster-accurate.
        Defaults to ``bin_size_um`` (stride 1 = output == raster).

    Writes ``vf_normalized`` ``(n_G, out_R, out_C)`` and ``vf_norm``
    ``(out_R, out_C)`` at ``sampling_distance`` spacing (+ metadata,
    ``backend='blur'``). Only the normalized field is persisted (never the raw 4D vf).

    ``store_chunk`` decouples the on-disk chunk shape of ``vf_normalized`` from the
    GPU compute tile (``chunk_h`` × ``chunk_w``). The compute tile stays large for
    GPU efficiency; the store is written in small SQUARE spatial tiles (gene axis
    always full) so a sparse voxel gather (``select_vectors_from_image`` /
    ``downsample_vector``) decodes only the tiles that actually contain a sampled
    voxel instead of whole full-width row bands. Give an int (square side) or a
    ``(h, w)`` tuple; ``None`` reverts to the legacy store == compute tile. For
    alignment (no read-modify-write on interior writes) ``chunk_h`` and ``chunk_w``
    should be integer multiples of the store tile (the defaults 64/512 vs 32 are).
    """
    import torch
    zg = analysis.dataset.zarr_group
    ds = analysis.dataset

    R = bin_adata.obs["array_row"].to_numpy()
    C = bin_adata.obs["array_col"].to_numpy()
    n_R, n_C = int(R.max() + 1), int(C.max() + 1)       # raster (blur) grid
    n_G = bin_adata.n_vars
    X = bin_adata.X.tocsr() if issparse(bin_adata.X) else bin_adata.X

    if sampling_distance is None:
        sampling_distance = bin_size_um
    stride = max(1, int(round(float(sampling_distance) / float(bin_size_um))))
    sd_um = float(bin_size_um) * stride                  # true output spacing (µm)
    bw_um = float(sigma_bin * bin_size_um)
    out_R = (n_R + stride - 1) // stride                 # output vf grid
    out_C = (n_C + stride - 1) // stride
    kx, ky, radius = _gaussian_kernel_1d(sigma_bin, device, prune_coefficient)
    # Blur always runs in fp32 (full precision).
    blur_dtype = torch.float32
    chunk_w = max(1, int(chunk_w))
    # sort bins by raster row ONCE → O(log n) contiguous row-band selection per tile
    # (replaces the O(n_bins) boolean scan the untiled path did inside every chunk).
    row_order = np.argsort(R, kind="stable")
    R_s = R[row_order]; C_s = C[row_order]

    for k in ("vf_normalized", "vf_norm", "genes", "kde_computed", "vf_params",
              "coordinate_offset", "coordinate_bounds", "original_coordinate_bounds"):
        if k in zg:
            del zg[k]
    # Store chunk decoupled from the GPU compute tile. Small SQUARE spatial tiles
    # (gene axis full) let a sparse voxel gather decode only occupied tiles instead
    # of whole full-width row bands; the compute tile stays large (below). Interior
    # writes span whole store chunks (chunk_h/chunk_w are multiples of the store tile)
    # so no read-modify-write. fill_value=0 → unwritten (empty) tiles read back as 0.
    if store_chunk is None:
        sh_store, sw_store = chunk_h, min(chunk_w, out_C)          # legacy: store == compute tile
    elif isinstance(store_chunk, (tuple, list)):
        sh_store, sw_store = int(store_chunk[0]), int(store_chunk[1])
    else:
        sh_store = sw_store = int(store_chunk)
    sh_store = max(1, min(sh_store, out_R))
    sw_store = max(1, min(sw_store, out_C))
    # vf_normalized = the large field and the write bottleneck. Prefer tensorstore's
    # C++ async writer (bit-identical output, same zarr-v3 metadata); fall back to
    # the zarr-python writer if tensorstore is absent or the store is not local.
    _store_root = getattr(zg.store, "root", None)
    _group_dir = (os.path.join(str(_store_root), zg.path) if zg.path else str(_store_root)) \
        if _store_root is not None else None
    vf_ts = _tensorstore_vf(_group_dir, n_G, out_R, out_C, sh_store, sw_store) \
        if _group_dir is not None else None
    vf_z = None
    if vf_ts is None:
        vf_z = zg.create_array("vf_normalized", shape=(n_G, out_R, out_C),
                               chunks=(n_G, sh_store, sw_store), dtype="f4", fill_value=0.0)
    _vf_inflight = []                              # bounded in-flight async writes
    # vf_norm is 2D, small, and read whole by the gather — keep it compute-tile chunked.
    col_chunk = min(chunk_w, out_C)
    norm_z = zg.create_array("vf_norm", shape=(out_R, out_C),
                             chunks=(chunk_h, col_chunk), dtype="f4", fill_value=0.0)
    coord_bnd = np.array([0.0, 0.0, 0.0, out_C * sd_um, out_R * sd_um, 0.0], dtype="f8")
    zg["genes"] = np.asarray(list(bin_adata.var_names))
    zg["kde_computed"] = np.zeros(n_G, dtype=bool)
    zg["vf_params"] = np.array([sd_um, bw_um], dtype="f8")
    zg["coordinate_offset"] = np.zeros(3, "f8")
    zg["coordinate_bounds"] = coord_bnd
    zg["original_coordinate_bounds"] = coord_bnd.copy()
    zg.attrs["sigma_bin"] = float(sigma_bin)
    zg.attrs["bin_size_um"] = float(bin_size_um)          # raster spacing
    zg.attrs["sampling_distance"] = sd_um                 # output vf spacing
    zg.attrs["stride"] = int(stride)
    zg.attrs["prune_coefficient"] = float(prune_coefficient)
    zg.attrs["size_after_normalization"] = float(size_after_normalization)
    zg.attrs["backend"] = "blur"

    ds.genes = list(bin_adata.var_names)
    ds.coordinate_offset = [0.0, 0.0, 0.0]
    ds.coordinate_bounds = coord_bnd.tolist()
    ds.original_coordinate_bounds = coord_bnd.tolist()
    ds.sampling_distance = sd_um
    ds.shape = (out_C, out_R, 1)

    # iterate OUTPUT-row chunks; blur the corresponding fine raster rows (+halo),
    # then decimate rows (stride-spaced) and columns (out_cols).
    # 2D tiling: iterate OUTPUT-row chunks × OUTPUT-col chunks. Each tile blurs the
    # corresponding fine raster rows/cols (+ ±radius halo), then decimates to the
    # output grid. Memory is bounded to n_G × (chunk_h*stride + 2r) × (chunk_w*stride
    # + 2r), independent of the full raster width (the untiled path held the full n_C).
    # per-gene expanded kernels are CONSTANT across every tile — build ONCE (was
    # re-expanded+contiguous'd inside _rasterize on every tile: n_tiles wasted copies).
    kx_g = kx.to(blur_dtype).expand(n_G, 1, 1, -1).contiguous()
    ky_g = ky.to(blur_dtype).expand(n_G, 1, -1, 1).contiguous()
    _cuda = str(device).startswith("cuda")

    o_starts = list(range(0, out_R, chunk_h))
    if analysis.verbose:
        o_starts = tqdm(o_starts, desc="compute_vf (blur, tiled)")
    for o0 in o_starts:
        o1 = min(out_R, o0 + chunk_h)
        frows = np.arange(o0, o1) * stride               # global raster rows for these output rows
        frows = frows[frows < n_R]
        if len(frows) == 0:
            continue
        rp0 = max(0, int(frows[0]) - radius)
        rp1 = min(n_R, int(frows[-1]) + 1 + radius)
        # select this row band's bins ONCE via the sorted-row index (contiguous slice)
        lo = int(np.searchsorted(R_s, rp0, side="left"))
        hi = int(np.searchsorted(R_s, rp1, side="left"))
        if hi <= lo:
            continue                                     # empty band → vf stays 0 (fill_value)
        pos = row_order[lo:hi]
        Rb = R_s[lo:hi]; Cb = C_s[lo:hi]; Xb = X[pos]
        # sort this band by raster COLUMN once → each col tile is a contiguous slice
        # via searchsorted (replaces the O(band_nnz) np.where boolean re-scan the
        # untiled inner loop did once PER col tile, i.e. out_C/chunk_w times/band).
        col_order = np.argsort(Cb, kind="stable")
        Cb_c = Cb[col_order]; Rb_c = Rb[col_order]; Xb_c = Xb[col_order]
        lr = torch.as_tensor(frows - rp0, device=device, dtype=torch.long)
        for oc0 in range(0, out_C, chunk_w):
            oc1 = min(out_C, oc0 + chunk_w)
            ocols = np.arange(oc0, oc1) * stride          # global raster cols for these output cols
            ocols = ocols[ocols < n_C]
            if len(ocols) == 0:
                continue
            cp0 = max(0, int(ocols[0]) - radius)
            cp1 = min(n_C, int(ocols[-1]) + 1 + radius)
            clo = int(np.searchsorted(Cb_c, cp0, side="left"))
            chi = int(np.searchsorted(Cb_c, cp1, side="left"))
            if chi <= clo:
                continue                                  # empty tile → vf stays 0 (fill_value)
            Xt = Xb_c[clo:chi]; Rt = Rb_c[clo:chi]; Ct = Cb_c[clo:chi]
            # GPU scatter-rasterize + separable blur of the (row band × col band) tile.
            blurred = _rasterize_blur_tile_gpu(Xt, Rt, Ct, rp0, rp1, cp0, cp1, n_G,
                                               kx_g, ky_g, radius, device, blur_dtype)
            lc = torch.as_tensor(ocols - cp0, device=device, dtype=torch.long)
            # decimate rows+cols, then normalize in fp32 (protect the per-voxel gene sum).
            chunk = blurred.index_select(1, lr).index_select(2, lc).float().contiguous()
            normalized, vfn = _l1_normalize_log1p(chunk, size_after_normalization)
            nn = normalized.cpu().numpy()
            if vf_ts is not None:                  # async C++ write (overlaps next tile)
                _vf_inflight.append(
                    vf_ts[:, o0:o0 + len(frows), oc0:oc0 + len(ocols)].write(nn))
                while len(_vf_inflight) > 8:        # bound memory: drain oldest
                    _vf_inflight.pop(0).result()
            else:
                vf_z[:, o0:o0 + len(frows), oc0:oc0 + len(ocols)] = nn
            norm_z[o0:o0 + len(frows), oc0:oc0 + len(ocols)] = vfn.cpu().numpy()
            del blurred, chunk, normalized
    for _f in _vf_inflight:                         # flush remaining async writes
        _f.result()
    # NOTE: torch.cuda.empty_cache() is intentionally NOT called per tile — the
    # tiling already bounds GPU memory and same-shape tiles are reused by the caching
    # allocator; releasing to the driver every tile forced a sync + re-request (slow).
    if _cuda:
        torch.cuda.empty_cache()
    zg["kde_computed"][:] = True
    if analysis.verbose:
        print(f"[compute_vf] blur raster {(n_R, n_C)}@{bin_size_um}µm → vf {(n_G, out_R, out_C)}@{sd_um}µm "
              f"(stride {stride}, compute tile {chunk_h}×{chunk_w}, store chunk {sh_store}×{sw_store}, "
              f"radius {radius}, bandwidth {bw_um}µm, fp32)")


# ---------------------------------------------------------------------------
# Voxel selection from the tissue image (count downsample)
# ---------------------------------------------------------------------------
def _image_gray(image, revert):
    arr = np.asarray(image)
    if arr.ndim == 3:
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        arr = arr[..., :3]
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        gray = arr.astype(np.float64)
    return (255.0 - gray) if revert else gray.astype(np.float64)


def select_vectors_from_image(analysis, grid, *, bin_size_um: float,
                              n_samples_per_block: int = 3, block_um: float = 10.0,
                              otsu_threshold: Optional[float] = None,
                              norm_threshold_mult: float = 2.0,
                              vf_norm_percentile: Optional[float] = 25.0,
                              lookup_chunk_h: int = 64, seed: int = 0,
                              lookup_n_workers: Optional[int] = None,
                              lookup_ram_gb: float = 32.0) -> np.ndarray:
    """Sample voxels from the tissue image (Otsu mask) and look them up in the
    precomputed ``vf_normalized`` field → ``normalized_vectors`` (count downsample).

    Pixels are mapped to world µm via ``grid.affine_px_to_um`` (folds each
    platform's transpose/flip), then to grid bins via ``grid.origin`` + ``bin_size_um``.
    """
    from skimage.filters import threshold_otsu
    zg = analysis.dataset.zarr_group
    ds = analysis.dataset
    vf_z, norm_z = zg["vf_normalized"], zg["vf_norm"]
    n_G, n_R, n_C = vf_z.shape
    sigma_bin = float(zg.attrs.get("sigma_bin", 1.0))
    X0, Y0 = grid.origin

    # ---- A. Otsu mask + area-proportional pixel sampling ----
    he = _image_gray(grid.image, grid.he_revert)
    upp = grid.um_per_px
    if otsu_threshold is None:
        otsu_threshold = float(threshold_otsu(he))
    mask = he > otsu_threshold
    ys, xs = np.where(mask)                                   # ys=row(px_y), xs=col(px_x)
    n_mask = len(ys)
    block_px = (block_um / upp) ** 2
    n_sel = max(int(n_mask / block_px * n_samples_per_block), 1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(n_mask, min(n_sel, n_mask), replace=False)
    px_x = xs[pick].astype(np.float64); px_y = ys[pick].astype(np.float64)

    # ---- A'. pixel → world µm (affine) → bin (C, R) ----
    M = grid.affine_px_to_um
    if M is not None:
        M = np.asarray(M)
        x_um = M[0, 0] * px_x + M[0, 1] * px_y + M[0, 2]
        y_um = M[1, 0] * px_x + M[1, 1] * px_y + M[1, 2]
    else:
        x_um = px_x * upp; y_um = px_y * upp
    C_i = np.floor((x_um - X0) / bin_size_um).astype(np.int64)
    R_i = np.floor((y_um - Y0) / bin_size_um).astype(np.int64)

    # ---- B. bounds + vf_norm threshold filter ----
    norm_full = norm_z[:]
    if vf_norm_percentile is not None:
        nz = norm_full[norm_full > 0]
        thr = float(np.percentile(nz, vf_norm_percentile)) if nz.size else 0.0
    else:
        thr = norm_threshold_mult / (np.sqrt(2 * np.pi) * sigma_bin) ** 2
    in_b = (R_i >= 0) & (R_i < n_R) & (C_i >= 0) & (C_i < n_C)
    keep = in_b.copy()
    keep[in_b] &= norm_full[R_i[in_b], C_i[in_b]] > thr
    R_sel, C_sel = R_i[keep], C_i[keep]
    order = np.argsort(R_sel)
    R_sel, C_sel = R_sel[order], C_sel[order]

    # ---- C. lookup vf_normalized at selected voxels (parallel over occupied tiles) ----
    # zarr decodes exactly the chunks that overlap a request, so we group the sampled
    # voxels by the field's OWN storage chunk grid (vf_z.chunks) and read only the
    # chunks that actually contain a voxel — every empty tile is skipped entirely. For
    # a square-tiled field (store_chunk in compute_vf) this is the win: a sparse gather
    # touches only ~occupied tiles instead of full-width row bands. For a legacy
    # full-width (n_G, h, n_C) field each tile IS a full row band, so this gracefully
    # reduces to the previous row-band behavior. Each voxel lands in exactly one chunk
    # → tiles write DISJOINT rows of `nv`, so they decode concurrently without locking.
    # A "tile" is exactly one storage chunk — reading less than a chunk still decodes
    # the whole chunk, so we never sub-split it (that would re-decode the same chunk).
    # lookup_chunk_h is retained for API compat but no longer sizes the read; RAM is
    # bounded by capping the number of concurrent tile decodes (lookup_ram_gb).
    from concurrent.futures import ThreadPoolExecutor
    nv = np.zeros((len(R_sel), n_G), dtype=np.float32)
    tile_h = int(vf_z.chunks[1]); tile_w = int(vf_z.chunks[2])
    n_col_tiles = (n_C + tile_w - 1) // tile_w
    tr = R_sel // tile_h
    tc = C_sel // tile_w
    tile_key = tr.astype(np.int64) * n_col_tiles + tc
    torder = np.argsort(tile_key, kind="stable")
    key_o = tile_key[torder]
    uniq = np.unique(key_o)
    starts = np.searchsorted(key_o, uniq)
    ends = np.r_[starts[1:], len(key_o)]

    # one decoded tile = n_G * tile_h * tile_w * 4 bytes; cap concurrency to the budget.
    tile_bytes = n_G * tile_h * tile_w * 4
    ram_cap = max(1, int(float(lookup_ram_gb) * (1024 ** 3) / max(1, tile_bytes)))
    n_workers = int(lookup_n_workers if lookup_n_workers is not None
                    else getattr(analysis, "ncores", 1) or 1)
    n_workers = max(1, min(n_workers, len(uniq), ram_cap))

    def _lookup_tile(i):
        s, e = int(starts[i]), int(ends[i])
        key = int(uniq[i])
        rt, ct = key // n_col_tiles, key % n_col_tiles
        r0, r1 = rt * tile_h, min(rt * tile_h + tile_h, n_R)
        c0, c1 = ct * tile_w, min(ct * tile_w + tile_w, n_C)
        block = np.asarray(vf_z[:, r0:r1, c0:c1])
        idx = torder[s:e]
        nv[idx] = block[:, R_sel[idx] - r0, C_sel[idx] - c0].T
        return e - s

    try:
        import numcodecs
        _prev_blosc = numcodecs.blosc.get_nthreads()
        numcodecs.blosc.set_nthreads(1)
    except Exception:                                     # noqa: BLE001
        numcodecs, _prev_blosc = None, None
    try:
        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                it = ex.map(_lookup_tile, range(len(uniq)))
                if analysis.verbose:
                    it = tqdm(it, total=len(uniq),
                              desc=f"downsample_vector (lookup, {len(uniq)} tiles, {n_workers}w)")
                for _ in it:
                    pass
        else:
            rng_it = range(len(uniq))
            it = tqdm(rng_it, desc="downsample_vector (lookup)") if analysis.verbose else rng_it
            for i in it:
                _lookup_tile(i)
    finally:
        if numcodecs is not None and _prev_blosc is not None:
            numcodecs.blosc.set_nthreads(_prev_blosc)

    gx = C_sel.astype(np.int64)
    gy = R_sel.astype(np.int64)
    gz = np.zeros(len(C_sel), np.int64)
    dvn = norm_full[R_sel, C_sel].astype(np.float32)
    gpx = (X0 + (C_sel + 0.5) * bin_size_um).astype(np.float64)
    gpy = (Y0 + (R_sel + 0.5) * bin_size_um).astype(np.float64)
    for k in ("normalized_vectors", "downsampled_vectors",
              "downsampled_vf_norm", "downsampled_global_xy"):
        if k in zg:
            del zg[k]
    zg["normalized_vectors"] = nv
    zg["downsampled_vectors"] = np.stack([gx, gy, gz], axis=0)
    zg["downsampled_vf_norm"] = dvn
    zg["downsampled_global_xy"] = np.stack([gpx, gpy], axis=0)
    ds.normalized_vectors = nv
    ds.downsampled_vectors = (gx, gy, gz)
    ds.downsampled_vf_norm = dvn
    ds.downsampled_global_xy = np.stack([gpx, gpy], axis=0)
    zg.attrs["select_n_kept"] = int(len(R_sel))
    zg.attrs["select_otsu_threshold"] = float(otsu_threshold)
    zg.attrs["select_he_um_per_px"] = float(upp)
    zg.attrs["select_vf_norm_threshold"] = float(thr)
    if analysis.verbose:
        print(f"[downsample] otsu={otsu_threshold:.4g} mask_px={n_mask} "
              f"sampled={n_sel} kept(vf_norm>{thr:.4g})={len(R_sel)}")
    return nv


# ---------------------------------------------------------------------------
# Mappers over the precomputed field (count path)
# ---------------------------------------------------------------------------
def _write_maps(zg, ds, M, Rc, sd, *, overwrite=True, set_ds_attrs=True):
    cm_3d = M.T[:, :, None].astype(np.int32)
    mc_3d = Rc.T[:, :, None].astype(np.float32)
    vx, vy, vz = cm_3d.shape
    if overwrite:
        for nm in ("celltype_maps", "max_correlations"):
            if nm in zg:
                del zg[nm]
    zg["celltype_maps"] = cm_3d
    zg["max_correlations"] = mc_3d
    if set_ds_attrs:
        import dask.array as da
        ds.shape = (vx, vy, vz)
        ds.x_scale = 1.0
        ds.y_scale = 1.0
        ds.max_probabilities = None
        ds.original_coordinate_bounds = [[0.0, vx * sd], [0.0, vy * sd], [0.0, 0.0]]
        ds.celltype_maps = cm_3d
        ds.max_correlations = mc_3d
    return cm_3d, mc_3d


def map_corr(analysis, centroids=None, *, vf_norm_threshold=None, chunk_h=128,
             device="cuda", overwrite=True):
    """Map the blur field to celltypes by centroid Pearson correlation (no KDE).

    Reads ``vf_normalized`` chunk-by-chunk, mean-centers + L2-normalizes each
    voxel and each centroid, assigns the max-correlation cluster. Output layout
    matches the transcript KDE path so downstream filter/plot work unchanged.
    """
    import torch
    zg = analysis.dataset.zarr_group
    ds = analysis.dataset
    vf_z = zg["vf_normalized"]
    n_G, n_R, n_C = vf_z.shape
    norm = np.asarray(zg["vf_norm"][:])
    sd = float(getattr(ds, "sampling_distance", zg.attrs.get("bin_size_um", 1.0)))

    if centroids is None:
        if "centroids" not in zg:
            raise ValueError("centroids missing — run cluster_vectors first.")
        centroids = zg["centroids"][:]
    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.shape[1] != n_G:
        raise ValueError(f"centroids gene dim {centroids.shape[1]} != vf {n_G}")

    if vf_norm_threshold is None:
        vf_norm_threshold = zg.attrs.get("select_vf_norm_threshold", None)
        if vf_norm_threshold is None:
            nz = norm[norm > 0]
            vf_norm_threshold = float(np.percentile(nz, 25)) if nz.size else 0.0
    vf_norm_threshold = float(vf_norm_threshold)

    c = torch.as_tensor(centroids, dtype=torch.float32, device=device)
    c = c - c.mean(dim=1, keepdim=True)
    c = c / torch.clamp(torch.linalg.norm(c, dim=1, keepdim=True), min=1e-12)

    M = np.full((n_R, n_C), -1, dtype=np.int32)
    Rc = np.full((n_R, n_C), -1.0, dtype=np.float32)
    it = range(0, n_R, chunk_h)
    if analysis.verbose:
        it = tqdm(list(it), desc="map_celltypes (corr)")
    for r0 in it:
        r1 = min(r0 + chunk_h, n_R)
        h = r1 - r0
        sig = norm[r0:r1] > vf_norm_threshold
        if not sig.any():
            continue
        block = np.asarray(vf_z[:, r0:r1, :]).reshape(n_G, h * n_C).T
        sig_flat = sig.reshape(-1)
        d = torch.as_tensor(block[sig_flat], device=device)
        d = d - d.mean(dim=1, keepdim=True)
        d = d / torch.clamp(torch.linalg.norm(d, dim=1, keepdim=True), min=1e-12)
        corr = d @ c.T
        best_corr, best_idx = corr.max(dim=1)
        pos = np.where(sig_flat)[0]
        m_flat = M[r0:r1].reshape(-1); rc_flat = Rc[r0:r1].reshape(-1)
        m_flat[pos] = best_idx.cpu().numpy().astype(np.int32)
        rc_flat[pos] = best_corr.cpu().numpy().astype(np.float32)
        M[r0:r1] = m_flat.reshape(h, n_C); Rc[r0:r1] = rc_flat.reshape(h, n_C)
        del block, d, corr, best_corr, best_idx
        torch.cuda.empty_cache()
    if analysis.verbose:
        print(f"[map_corr] assigned {int((M >= 0).sum())} voxels (thr={vf_norm_threshold:.4g})")
    return _write_maps(zg, ds, M, Rc, sd, overwrite=overwrite)
