"""GPU helpers for ``SSAMAnalysis.map_celltypes_gpu``.

Reuses the Triton KDE kernel + cell-list builders from
``ssam._kde_gpu`` and adds:

* ``_kde_setup_celllist``  — build the cell-list once per outer chunk.
* ``_kde_run_subchunk``    — run KDE for one sub-chunk of query points
                             using a pre-built cell-list (no rebuild).
* ``_pearson_corr_gpu``    — (k, g) × (n, g) → (k, n) Pearson correlation
                             entirely on GPU via torch.matmul.

These primitives let ``map_celltypes_gpu`` stream chunks like
``map_celltypes_he`` while staying on GPU through the correlation step.

Numerical equivalence vs the C++ ``map_celltypes_he`` path:
- KDE row-L1 relative diff <2e-6 (Triton kernel inherits the same
  bound documented in ``_kde_gpu``).
- Pearson correlation absolute diff <1e-5 (cuBLAS GEMM vs AVX512 FMA).
- Final ``celltype_maps`` may flip labels at decision boundaries
  (≪0.1% of voxels in well-separated cluster cases).
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
    from ._kde_gpu import (
        _kde_cell_kernel,
        _kde_cell_kernel_3d,
        _build_cell_list,
        _build_cell_list_3d,
    )
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


def _kde_setup_celllist(loc_x_t, loc_y_t, loc_z_t, gi_t,
                        chunk_q_xmin, chunk_q_ymin, chunk_q_zmin,
                        chunk_q_xmax, chunk_q_ymax, chunk_q_zmax,
                        sampling_distance, bandwidth, prune_coefficient,
                        dim):
    """Build a cell-list spanning the chunk's query bbox + transcripts.

    Returns a state dict consumed by :func:`_kde_run_subchunk`, or
    ``None`` when there are no transcripts (caller treats the result
    as all-zero KDE).
    """
    if not _HAS_TRITON:
        raise ImportError("_kde_setup_celllist requires torch + triton.")
    if loc_x_t.shape[0] == 0:
        return None

    Rw = bandwidth * prune_coefficient
    inv_sd = 1.0 / sampling_distance

    x_min = float(min(loc_x_t.min().item(), chunk_q_xmin)) - Rw - sampling_distance
    y_min = float(min(loc_y_t.min().item(), chunk_q_ymin)) - Rw - sampling_distance
    x_max = float(max(loc_x_t.max().item(), chunk_q_xmax)) + Rw + sampling_distance
    y_max = float(max(loc_y_t.max().item(), chunk_q_ymax)) + Rw + sampling_distance
    W = int(math.ceil((x_max - x_min) * inv_sd)) + 1
    H = int(math.ceil((y_max - y_min) * inv_sd)) + 1

    state = {
        'dim': int(dim),
        'sampling_distance': float(sampling_distance),
        'bandwidth': float(bandwidth),
        'prune_coefficient': float(prune_coefficient),
        'x_min': x_min, 'y_min': y_min, 'W': W, 'H': H,
    }

    if dim == 2:
        sx, sy, sg, cell_start = _build_cell_list(
            loc_x_t, loc_y_t, gi_t, sampling_distance, x_min, y_min, W, H,
        )
        state.update(sx=sx, sy=sy, sg=sg, cell_start=cell_start)
    else:
        z_min = float(min(loc_z_t.min().item(), chunk_q_zmin)) - Rw - sampling_distance
        z_max = float(max(loc_z_t.max().item(), chunk_q_zmax)) + Rw + sampling_distance
        D = int(math.ceil((z_max - z_min) * inv_sd)) + 1
        sx, sy, sz, sg, cell_start = _build_cell_list_3d(
            loc_x_t, loc_y_t, loc_z_t, gi_t, sampling_distance,
            x_min, y_min, z_min, W, H, D,
        )
        state.update(sx=sx, sy=sy, sz=sz, sg=sg, cell_start=cell_start,
                     z_min=z_min, D=D)
    return state


def _kde_run_subchunk(state, qx, qy, qz, n_genes, device):
    """Run KDE for a sub-chunk of query points using ``state``.

    Returns a torch.float32 tensor of shape ``(N, n_genes)`` on
    ``device``.
    """
    if not _HAS_TRITON:
        raise ImportError("_kde_run_subchunk requires torch + triton.")

    qx_g = torch.as_tensor(qx, dtype=torch.float64, device=device).contiguous()
    N = int(qx_g.shape[0])
    if N == 0 or state is None:
        return torch.zeros((N, n_genes), dtype=torch.float32, device=device)

    qy_g = torch.as_tensor(qy, dtype=torch.float64, device=device).contiguous()

    bw = state['bandwidth']
    pc = state['prune_coefficient']
    sd = state['sampling_distance']
    Rw = bw * pc
    Rw2 = Rw * Rw
    two_h2 = 2.0 * bw * bw
    norm_const = (2.0 * math.pi * bw * bw) ** (0.5 * state['dim'])
    inv_norm = 1.0 / norm_const
    inv_sd = 1.0 / sd
    Rp = int(math.floor(Rw / sd + 1e-12))

    out = torch.zeros((N, n_genes), dtype=torch.float32, device=device)

    if state['dim'] == 2:
        with torch.cuda.device(qx_g.device):
            _kde_cell_kernel[(N,)](
                qx_g, qy_g,
                state['sx'], state['sy'], state['sg'], state['cell_start'],
                out,
                N, n_genes, state['W'], state['H'], Rp,
                state['x_min'], state['y_min'],
                inv_sd, two_h2, inv_norm, Rw2,
                BLOCK_T=32,
            )
    else:
        qz_g = torch.as_tensor(qz, dtype=torch.float64, device=device).contiguous()
        with torch.cuda.device(qx_g.device):
            _kde_cell_kernel_3d[(N,)](
                qx_g, qy_g, qz_g,
                state['sx'], state['sy'], state['sz'], state['sg'], state['cell_start'],
                out,
                N, n_genes, state['W'], state['H'], state['D'], Rp,
                state['x_min'], state['y_min'], state['z_min'],
                inv_sd, two_h2, inv_norm, Rw2,
                BLOCK_T=32,
            )
    return out


def _pearson_corr_gpu(centroids_t, data_t):
    """Pearson correlation between (k, g) centroids and (n, g) rows.

    Returns a torch tensor of shape ``(k, n)`` on the same device.
    Both inputs are assumed float32; the result is float32.
    """
    if not _HAS_TRITON:
        raise ImportError("_pearson_corr_gpu requires torch.")
    c = centroids_t - centroids_t.mean(dim=1, keepdim=True)
    d = data_t - data_t.mean(dim=1, keepdim=True)
    cn = torch.clamp(torch.linalg.norm(c, dim=1), min=1e-12)
    dn = torch.clamp(torch.linalg.norm(d, dim=1), min=1e-12)
    return (c @ d.T) / (cn[:, None] * dn[None, :])


def _pearson_max_gpu(centroids_t, data_t):
    """Shared mapping kernel: best-centroid label + Pearson score per row.

    ``centroids_t`` (k, g) and ``data_t`` (n, g) → ``(best_idx (n,), best_corr
    (n,))`` on the same device. This is the single consumer used by BOTH the
    count path (field read from the stored ``vf_normalized``) and the transcript
    path (KDE field streamed chunk-by-chunk) — identical Pearson-max so results
    do not depend on how the field was produced. Returns int32 idx, float32 corr.
    """
    corr = _pearson_corr_gpu(centroids_t, data_t)
    best_corr, best_idx = corr.max(dim=0)
    return best_idx.to(torch.int32), best_corr.to(torch.float32)
