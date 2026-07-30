"""TSVM-based celltype mapping for CeLLMan spatial.

Alternative to ``map_celltypes_correlation`` (centroid-Pearson template matching)
that replaces the mapping step with a multiclass Transductive SVM. The default
algorithm is a classic transductive SVM solved by LABEL SWITCHING over the whole
signal field, with an RBF kernel; ``map_celltypes(method='tsvm')`` uses it.

Why not plain centroid-correlation
----------------------------------
Correlation assigns each voxel the celltype whose single centroid it most
resembles (cosine on mean-centered profiles). It learns no discriminative weights.
A margin classifier learns what separates celltypes and yields a per-voxel
confidence (the top-class softmax probability) for optional filtering.

Kernel
------
RBF by default (``kernel='rbf'``), realized as an explicit Random Fourier Feature
map on the PCA-50 coords, so the model stays a compact (k, D) linear machine that
streams over ~1e7 voxels (an exact kernel SVM would need K(field, support-vectors)
per apply). ``kernel='linear'`` restores the un-lifted behaviour. Full-gene mode
(``n_components=None``) is linear-only.

Algorithm (``algo='labelswitch'``, the default)
-----------------------------------------------
Classic label-switching transductive SVM (Joachims-style), realized for a large
field: the WHOLE signal field is the transductive pool, every voxel carries a hard
tentative label, and those labels are switched each round under a class-balancing
constraint while the unlabeled weight anneals 0 -> 1. Block-coordinate descent:

  W-step  : fix the tentative labels y~, minimize the Crammer-Singer multiclass
            hinge over labeled + unlabeled(y~) + L2.
  y~-step : fix (W, b), re-score the whole field and re-assign every label under
            the balancing constraint (per-class dual variables) -- the label switch.

Practical realizations of the textbook algorithm: multiclass via Crammer-Singer,
RBF via RFF, the QP sub-problem via Adam SGD, and the balancing constraint via
per-class dual variables. ``algo='soft'`` selects the earlier S3VM relaxation
(``train_tsvm``: soft entropy/balancing penalties on a fixed sample, no label
switching), kept for comparison.

Pipeline
--------
1. labeled   = ``normalized_vectors`` (downsampled, annotated) + ``cluster_labels``
               (celltype index 0..k-1; row i of ``centroids``).
2. unlabeled = every field voxel (``vf_normalized``) whose ``vf_norm`` is above the
               training floor; the whole field is the transductive pool.
3. PCA (fit on labeled) -> d dims (default 50), cached over the field once so
   training and apply reuse the projected field.
4. Train by label switching; init from projected centroids (near the correlation
   solution).
5. Inductive apply: classify every voxel with ``vf_norm > 0``;
   the top-class softmax probability goes to ``max_probabilities``. Output zarr
   layout mirrors the correlation path (``celltype_maps`` + a per-voxel score
   volume) so ``filter_celltypemaps`` / ``plot_celltypes_map`` keep working -- the
   score lands in ``max_probabilities`` (not ``max_correlations``), so
   ``filter_celltypemaps`` thresholds it with ``min_p`` rather than ``min_r``.
   Abstention at map time is OFF by default; thresholding weak voxels is
   ``filter_celltypemaps``' job.
"""

import numpy as np
import torch
import dask.array as da
from tqdm.auto import tqdm


def _resolve_device(device):
    """Resolve a requested torch device, falling back to CPU when CUDA is unavailable.

    A ``'cuda'`` request on a machine without CUDA is downgraded to ``'cpu'``; an
    explicit ``'cpu'`` is honored.
    """
    dev = str(device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[tsvm] CUDA unavailable; running on device='cpu' (much slower — "
              "consider map_celltypes(method='corr', force_cpu=True) for CPU).")
        return "cpu"
    return dev


def _empty_cache(device):
    """``torch.cuda.empty_cache()`` only on a CUDA device (a no-op on CPU)."""
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _row_normalize(Xt):
    """Per-voxel: mean-center across genes, then L2-normalize. (n, g) -> (n, g).

    This is EXACTLY the preprocessing the centroid-correlation path applies inside
    ``_pearson_corr_gpu`` (row mean-center + divide by row L2 norm). Without it the
    PCA/SVM keys on each voxel's signal *magnitude* (local transcript density,
    which tracks tissue structure) instead of its expression *direction* (cell-type
    identity) -- the cause of the field collapsing onto one class. Applying it to
    BOTH labeled vectors and field voxels puts the TSVM in the same magnitude- and
    shift-invariant space as the correlation baseline.
    """
    Xc = Xt - Xt.mean(dim=1, keepdim=True)
    return Xc / torch.clamp(Xc.norm(dim=1, keepdim=True), min=1e-12)


def fit_pca(X, n_components=50, device="cuda", seed=0):
    """Fit PCA via low-rank SVD on GPU. Returns (mean (g,), comps (d,g)) tensors.

    ``torch.pca_lowrank`` uses randomized SVD; we seed the RNG so the basis (and
    therefore the field-PCA cache fingerprint) is reproducible across runs.

    PCA is fit in the per-voxel-normalized space (see ``_row_normalize``) so the
    latent axes describe expression *shape*, not signal magnitude.
    """
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
    Xt = _row_normalize(Xt)
    mean = Xt.mean(dim=0)
    Xc = Xt - mean
    q = min(Xc.shape[1], n_components + 10)
    _, _, V = torch.pca_lowrank(Xc, q=q, center=False)
    comps = V[:, :n_components].T.contiguous()
    sign = torch.sign(comps[torch.arange(comps.shape[0]), comps.abs().argmax(dim=1)])
    comps = comps * sign.unsqueeze(1)
    del Xt, Xc
    _empty_cache(device)
    return mean, comps


def project(X, mean, comps):
    """Project (n, g) -> (n, d). X may be np or torch; result torch on comps.device.

    Row-normalizes each voxel first (same space the PCA basis was fit in).
    """
    if torch.is_tensor(X):
        Xt = X.to(comps.device).float()
    else:
        Xt = torch.as_tensor(np.asarray(X, dtype=np.float32), device=comps.device)
    return (_row_normalize(Xt) - mean) @ comps.T


def rff_gamma_median(Z, n_sample=4096, seed=0):
    """Median heuristic for the RBF bandwidth: gamma = 1 / (2 * median_dist^2).

    Computed on a subsample of the LABELED PCA coords, i.e. in exactly the space
    the classifier sees. Scale matters far more for RBF than for a linear model
    (the kernel is distance-based), so this must never be a hardcoded constant.
    """
    n = Z.shape[0]
    if n > n_sample:
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[:n_sample].to(Z.device)
        S = Z[idx]
    else:
        S = Z
    m = S.shape[0]
    D = torch.cdist(S.float(), S.float())
    iu = torch.triu_indices(m, m, offset=1, device=D.device)
    med = float(D[iu[0], iu[1]].median())
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return 1.0 / (2.0 * med * med)


def rff_params(d, n_features, gamma, seed=0, device="cuda"):
    """Draw the RFF projection. Returns (Omega (D, d), beta (D,)) on ``device``.

    Sampled from a CPU generator so the feature map is reproducible across CUDA
    devices (matching the persisted-PCA-basis policy).
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    Omega = torch.randn(n_features, d, generator=g) * float(np.sqrt(2.0 * gamma))
    beta = torch.rand(n_features, generator=g) * float(2.0 * np.pi)
    return Omega.to(device).float(), beta.to(device).float()


def rff_apply(Z, Omega, beta):
    """(n, d) -> (n, D) RBF feature map. z(x) = sqrt(2/D) cos(Omega x + beta)."""
    return float(np.sqrt(2.0 / Omega.shape[0])) * torch.cos(Z @ Omega.T + beta)


def sample_unlabeled_field(ds, mean, comps, *, vf_norm_threshold,
                           n_target=200_000, chunk_h=128, device="cuda", seed=0):
    """Randomly subsample field voxels above threshold and return their PCA coords.

    WARNING: reading ``vf_normalized`` densely is heavy (each 128-row chunk
    decompresses to ~12 GB here; a full scan is hundreds of GB). To stay cheap
    this reads row-chunks **in shuffled order and stops as soon as n_target
    voxels are collected** -- typically only a handful of chunks. Use this only
    when you want genuine field voxels as the unlabeled pool; otherwise sample
    from the PCA cache via ``fit_celltype_tsvm(unlabeled_source='cache')`` (the
    default) for a zero-scan fit.
    """
    zg = ds.zarr_group
    vf_z = zg["vf_normalized"]
    n_G, n_R, n_C = vf_z.shape
    norm = np.asarray(zg["vf_norm"][:])
    rng = np.random.default_rng(seed)
    starts = list(range(0, n_R, chunk_h))
    rng.shuffle(starts)

    Z_chunks, n_have = [], 0
    pbar = tqdm(starts, desc="sample_unlabeled")
    for r0 in pbar:
        r1 = min(r0 + chunk_h, n_R)
        h = r1 - r0
        sig = norm[r0:r1] > vf_norm_threshold
        if not sig.any():
            continue
        sig_flat = sig.reshape(-1)
        block = np.asarray(vf_z[:, r0:r1, :]).reshape(n_G, h * n_C)
        gb = torch.as_tensor(block, device=device)
        sm = torch.as_tensor(sig_flat, device=device)
        Z = (_row_normalize(gb.T[sm]) - mean) @ comps.T
        Z_chunks.append(Z)
        n_have += Z.shape[0]
        del block, gb, sm
        _empty_cache(device)
        pbar.set_postfix(have=n_have)
        if n_have >= n_target:
            break
    if not Z_chunks:
        return torch.empty(0, comps.shape[0], device=device)
    Zu = torch.cat(Z_chunks, dim=0)
    if Zu.shape[0] > n_target:
        idx = torch.as_tensor(rng.choice(Zu.shape[0], n_target, replace=False),
                              device=Zu.device)
        Zu = Zu[idx]
    return Zu


def sample_unlabeled_from_cache(ds, *, key="tsvm_field_pca", vf_norm_threshold,
                                n_target=200_000, device="cuda", seed=0):
    """Sample real field voxels (signal only) from the PCA cache -- instant, no
    field decompression. Returns (n_target, d) PCA coords. This is the proper
    transductive pool: the actual distribution the map will be applied to.
    """
    zg = ds.zarr_group
    cache = zg[key]
    d = cache.shape[0]
    norm = np.asarray(zg["vf_norm"][:]).reshape(-1)
    sig_idx = np.where(norm > vf_norm_threshold)[0]
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(sig_idx, min(n_target, sig_idx.size), replace=False))
    Zf = np.asarray(cache[:]).reshape(d, -1)
    Zu = Zf[:, take].T
    return torch.as_tensor(np.ascontiguousarray(Zu), device=device)


def train_tsvm(Zl, yl, k, Zu=None, *, W_init=None,
               lam=1e-4, ent=0.3, balance=1.0,
               anneal_steps=4, inner_iters=200, lr=0.1, device="cuda",
               verbose=True):
    """Train a linear multiclass TSVM on GPU.

    "Linear" is in the FEATURE SPACE it is handed: with ``kernel='rbf'`` the caller
    passes RFF-lifted coords (see ``rff_apply``), which makes this exact solver an
    RBF-kernel SVM. Nothing here changes between the two kernels.

    Objective (f(z) = W z + b, k classes):
        mean_i max(0, 1 + max_{c != y_i} f_c(z_i) - f_{y_i}(z_i))   [Crammer-Singer hinge]
      + 0.5*lam*||W||^2
      + w_u * ent     * mean_{j in U} H(softmax(f(z_j)))            [low-density separation]
      + w_u * balance * KL( mean_{j in U} softmax(f(z_j)) || prior )[class-balancing]
    prior = labeled class frequencies. w_u is annealed 0 -> 1 over ``anneal_steps``
    (non-convexity remedy). With Zu=None this is a supervised CS linear SVM.

    Returns W (k, d), b (k,) detached tensors on ``device``.
    """
    Zl = Zl.to(device).float()
    yl = torch.as_tensor(np.asarray(yl), device=device).long()
    L, d = Zl.shape
    ar = torch.arange(L, device=device)

    prior = torch.bincount(yl, minlength=k).float()
    prior = prior / prior.sum()

    if W_init is not None:
        W = W_init.clone().to(device).float().requires_grad_(True)
    else:
        W = torch.zeros(k, d, device=device, requires_grad=True)
    b = torch.zeros(k, device=device, requires_grad=True)

    has_u = Zu is not None and Zu.shape[0] > 0
    if has_u:
        Zu = Zu.to(device).float()

    opt = torch.optim.Adam([W, b], lr=lr)
    schedule = np.linspace(0.0, 1.0, anneal_steps) if has_u else np.array([0.0])
    for si, w_u in enumerate(schedule):
        for it in range(inner_iters):
            opt.zero_grad()
            F = Zl @ W.T + b
            true_s = F.gather(1, yl[:, None]).squeeze(1)
            Fm = F.clone()
            Fm[ar, yl] = -1e9
            loss = torch.relu(1.0 + Fm.max(dim=1).values - true_s).mean()
            loss = loss + 0.5 * lam * (W * W).sum()
            if has_u and w_u > 0:
                Fu = Zu @ W.T + b
                pu = torch.softmax(Fu, dim=1)
                loss = loss + w_u * ent * (-(pu * torch.log(pu + 1e-9)).sum(dim=1)).mean()
                mp = pu.mean(dim=0)
                loss = loss + w_u * balance * (mp * (torch.log(mp + 1e-9)
                                                     - torch.log(prior + 1e-9))).sum()
            loss.backward()
            opt.step()
        if verbose:
            print(f"  anneal {si+1}/{len(schedule)}  w_u={w_u:.2f}  loss={loss.item():.4f}")
    return W.detach(), b.detach()


@torch.no_grad()
def _balanced_assign(S, prior, iters=40, lr=0.5):
    """Assign each row of ``S`` (n,k scores) a label = argmax(S + delta), choosing the
    per-class offsets ``delta`` (Lagrange multipliers of the balancing constraint) so
    the class proportions match ``prior``. Dual ascent: raise delta for under-
    represented classes. Returns (labels (n,), delta (k,))."""
    k = S.shape[1]
    n = S.shape[0]
    delta = torch.zeros(k, device=S.device, dtype=torch.float32)
    lab = None
    for _ in range(iters):
        lab = (S + delta).argmax(1)
        counts = torch.bincount(lab, minlength=k).float() / n
        delta -= lr * (counts - prior)
    return lab, delta


@torch.no_grad()
def _field_scores_into(Zu_pca, omega, beta, W, b, out, chunk=1_000_000):
    """Score every unlabeled point into ``out`` (n,k) in place. ``Zu_pca`` are PCA
    coords on GPU; the RFF lift is applied per chunk so the (n,D) matrix is never
    resident (it would be D/d times the field). chunk=1M keeps the transient RFF
    tensors (matmul + cos, ~2*chunk*D*4 bytes) small enough for a 24 GB card."""
    for s in range(0, Zu_pca.shape[0], chunk):
        e = min(s + chunk, Zu_pca.shape[0])
        Zr = rff_apply(Zu_pca[s:e], omega, beta) if omega is not None else Zu_pca[s:e]
        out[s:e] = (Zr @ W.T + b).to(out.dtype)
        del Zr


def _train_labelswitch(Zl_r, yl, k, Zu_pca, omega, beta, *, W_init=None, lam=1e-4,
                       Cu=1.0, anneal_steps=4, switch_rounds=4, inner=40, lr=0.1,
                       u_batch=1_000_000, switch_tol=0.005, assign_iters=40,
                       device="cuda", seed=0, verbose=True):
    """Classic label-switching multiclass TSVM.

    Zl_r    : labeled features, ALREADY RFF-lifted (L, D)
    Zu_pca  : unlabeled PCA coords (n_u, d) on GPU; lifted to RFF per chunk here
    omega,beta : RFF params (None -> linear kernel, lift is identity)

    Block-coordinate descent, C_u annealed 0->Cu:
      W-step  : fix tentative labels y~, minimize labeled hinge + C_u*unlabeled hinge(y~)
                + L2 (SGD minibatch on the fixed y~).
      y~-step : fix (W,b), re-score the WHOLE field and re-assign every label under the
                balancing constraint (the label switch).
    """
    yl = torch.as_tensor(np.asarray(yl), device=device).long()
    D = Zl_r.shape[1]
    ar = torch.arange(Zl_r.shape[0], device=device)
    prior = torch.bincount(yl, minlength=k).float()
    prior = prior / prior.sum()
    n_u = Zu_pca.shape[0]

    W = (W_init.clone().to(device).float() if W_init is not None
         else torch.zeros(k, D, device=device)).requires_grad_(True)
    b = torch.zeros(k, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    def _hinge(Zr, yy):
        F = Zr @ W.T + b
        true_s = F.gather(1, yy[:, None]).squeeze(1)
        Fm = F.clone()
        Fm[torch.arange(F.shape[0], device=device), yy] = -1e9
        return torch.relu(1.0 + Fm.max(1).values - true_s).mean()

    for _ in range(300):
        opt.zero_grad()
        (_hinge(Zl_r, yl) + 0.5 * lam * (W * W).sum()).backward()
        opt.step()

    # fp16 halves the (n_u, k) score buffer on GPU; CPU fp16 is unsupported for many
    # ops and yields no memory win, so the CPU path uses fp32.
    score_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    S = torch.empty(n_u, k, device=device, dtype=score_dtype)
    _field_scores_into(Zu_pca, omega, beta, W.detach(), b.detach(), S)
    ytil, _ = _balanced_assign(S.float(), prior, iters=assign_iters)

    gen = torch.Generator(device=device).manual_seed(int(seed))
    for ai, frac in enumerate(np.linspace(0.0, 1.0, anneal_steps)):
        cu = Cu * frac
        for r in range(switch_rounds):
            for _ in range(inner):
                opt.zero_grad()
                loss = _hinge(Zl_r, yl) + 0.5 * lam * (W * W).sum()
                if cu > 0:
                    idx = torch.randint(0, n_u, (min(u_batch, n_u),), generator=gen, device=device)
                    Zr = rff_apply(Zu_pca[idx], omega, beta) if omega is not None else Zu_pca[idx]
                    loss = loss + cu * _hinge(Zr, ytil[idx])
                loss.backward()
                opt.step()
            _field_scores_into(Zu_pca, omega, beta, W.detach(), b.detach(), S)
            ytil_new, _ = _balanced_assign(S.float(), prior, iters=assign_iters)
            switched = float((ytil_new != ytil).float().mean())
            ytil = ytil_new
            if verbose:
                with torch.no_grad():
                    tr = float(((Zl_r @ W.T + b).argmax(1) == yl).float().mean())
                print(f"  [ls] anneal {ai+1}/{anneal_steps} Cu={cu:.2f} round {r+1}: "
                      f"switched={100*switched:.2f}%  train_acc={tr:.4f}")
            if switched < switch_tol:
                break
    del S
    _empty_cache(device)
    return W.detach(), b.detach()


def load_unlabeled_from_cache(ds, *, key="tsvm_field_pca", vf_norm_threshold, device="cuda"):
    """Load the FULL transductive pool (every signal voxel) from the PCA cache as
    (n_u, d) PCA coords on ``device``. Unlike ``sample_unlabeled_from_cache`` this
    takes ALL voxels above the floor — the whole field is the transductive set."""
    zg = ds.zarr_group
    cache = zg[key]
    d = cache.shape[0]
    norm = np.asarray(zg["vf_norm"][:]).reshape(-1)
    sig = norm > vf_norm_threshold
    Zf = np.asarray(cache[:]).reshape(d, -1)[:, sig].T
    return torch.as_tensor(np.ascontiguousarray(Zf), device=device)


def fit_celltype_tsvm(ds, *, n_components=50, n_unlabeled=200_000,
                      vf_norm_threshold=None, lam=1e-4, ent=0.0, balance=2.0,
                      anneal_steps=4, inner_iters=200, lr=0.1,
                      device="cuda", seed=0, use_centroid_init=True,
                      use_unlabeled=True, unlabeled_source="cache",
                      abstain_q=0.0, chunk_h=128,
                      kernel="rbf", rff_dim=1024, gamma=None, rff_seed=0,
                      algo="labelswitch", Cu=1.0, switch_rounds=4, ls_inner=40,
                      train_floor="principled", max_fit_cells=200_000):
    """Fit PCA + multiclass TSVM from the store's labeled vectors and field pool.

    Returns a model dict and caches it on ``ds._tsvm_model``.

    Algorithm (``algo``)
    --------------------
    * ``'labelswitch'`` (DEFAULT) — the classic transductive SVM by LABEL SWITCHING:
      the WHOLE signal field is the transductive pool, every voxel carries a HARD
      tentative label, and those labels are switched each round under Joachims'
      balancing constraint while ``C_u`` anneals 0->1 (see ``_train_labelswitch``).
      This is the faithful transductive algorithm.
    * ``'soft'`` — the earlier S3VM relaxation (``train_tsvm``): soft entropy/balancing
      penalties on a fixed ``n_unlabeled`` sample, no tentative labels. Kept for
      comparison; not transductive in the label-switching sense.

    ``train_floor`` sets the vf_norm floor for the TRAINING unlabeled set. Near-zero
    voxels become arbitrary unit vectors after row-normalization, and label switching
    would then push boundaries at random, so they are excluded from TRAINING (apply
    still classifies every vf_norm>0 voxel). ``'principled'`` = 2/(sqrt(2pi)*sigma)^2
    (>= two transcripts' worth of blurred mass); ``'store'`` = the stored
    select_vf_norm_threshold; or a float.

    :param ds: Dataset/store exposing ``zarr_group`` with labeled vectors, cluster
        labels, centroids and the field; the fitted model is cached on ``ds._tsvm_model``.
    :type ds: Dataset
    :param n_components: PCA latent dim (default 50). ``None`` selects full-gene mode:
        no PCA, a linear classifier on the row-normalized full-gene vectors
        (``kernel='rbf'`` is not supported there). PCA-50 + linear caps training fidelity
        at ~0.88 (it drops low-variance class-discriminative directions); full genes
        restore it to ~0.95 but the field can no longer be pre-projected into a small
        cache, so mapping streams the field per apply.
    :type n_components: int or None
    :param n_unlabeled: Size of the unlabeled sample for the ``'soft'`` algo (ignored by
        ``'labelswitch'``, which uses the whole field).
    :type n_unlabeled: int
    :param vf_norm_threshold: vf_norm floor recorded on the model; ``None`` reads the
        stored select/map threshold from the group attrs.
    :type vf_norm_threshold: float or None
    :param lam: L2 regularization weight on the classifier weights.
    :type lam: float
    :param ent: Entropy-minimization weight for the ``'soft'`` algo.
    :type ent: float
    :param balance: Class-balancing weight for the ``'soft'`` algo.
    :type balance: float
    :param anneal_steps: Number of unlabeled-weight (or ``C_u``) annealing steps.
    :type anneal_steps: int
    :param inner_iters: Inner optimization iterations per anneal step (``'soft'`` algo and
        full-gene cross-entropy schedule).
    :type inner_iters: int
    :param lr: Adam learning rate for the classifier optimization.
    :type lr: float
    :param device: Torch device for fitting.
    :type device: str
    :param seed: RNG seed for PCA and unlabeled sampling.
    :type seed: int
    :param use_centroid_init: Initialize the classifier weights from the projected
        centroids (a correlation-like warm start) when True.
    :type use_centroid_init: bool
    :param use_unlabeled: When False, fit the supervised (no-transduction) variant.
    :type use_unlabeled: bool
    :param unlabeled_source: ``'cache'`` (default) samples real field voxels from the PCA
        cache (instant, builds it once if missing); ``'field'`` samples fresh voxels by
        decompressing the field (slow).
    :type unlabeled_source: str
    :param abstain_q: Abstention strength stored on the model; a field voxel is left
        Unknown when its top-class confidence is below the ``abstain_q`` quantile of
        labeled confidences.
    :type abstain_q: float
    :param chunk_h: Row-chunk height used when building the field PCA cache.
    :type chunk_h: int
    :param kernel: ``'rbf'`` (default) lifts the PCA coords through an explicit Random
        Fourier Feature map, making the Crammer-Singer machine a kernel SVM while every
        downstream step (training, transduction, streaming apply, abstention, the field
        PCA cache) stays the same shape of computation; ``'linear'`` restores the plain
        linear machine. Full-gene mode is linear-only.
    :type kernel: str
    :param rff_dim: Number of random Fourier features D (default 1024). Higher is closer
        to the exact RBF kernel; training memory is O(L*D), the apply cost is hidden by
        field I/O.
    :type rff_dim: int
    :param gamma: RBF bandwidth in ``exp(-gamma||x-y||^2)``. ``None`` uses the median
        heuristic on the labeled PCA coords.
    :type gamma: float or None
    :param rff_seed: Seed for the RFF draw; the map is reproducible across devices.
    :type rff_seed: int
    :param algo: ``'labelswitch'`` (default, faithful transductive) or ``'soft'`` (S3VM
        relaxation). See the Algorithm section above.
    :type algo: str
    :param Cu: Final unlabeled-hinge weight for the ``'labelswitch'`` algo (annealed 0->Cu).
    :type Cu: float
    :param switch_rounds: Label-switch rounds per anneal step (``'labelswitch'`` algo).
    :type switch_rounds: int
    :param ls_inner: Inner W-step SGD iterations per switch round (``'labelswitch'`` algo).
    :type ls_inner: int
    :param train_floor: vf_norm floor for the TRAINING unlabeled set: ``'principled'``,
        ``'store'`` or a float (see above).
    :type train_floor: str or float
    :param max_fit_cells: Cap on labeled vectors used for FITTING (PCA basis + margin
        training); a per-class stratified, seeded subsample is taken above this. The whole
        labeled matrix is put on the GPU and row-normalized (several transient copies), so a
        large store (e.g. 650k x 5k genes) OOMs here before the field is touched. centroids /
        cluster_labels are unaffected (computed from all vectors), so the map is unchanged in
        expectation. ``None`` uses every labeled vector.
    :type max_fit_cells: int or None
    """
    device = _resolve_device(device)
    kernel = str(kernel).lower()
    if kernel not in ("linear", "rbf"):
        raise ValueError("kernel must be 'rbf' or 'linear'")
    if kernel == "rbf" and n_components is None:
        raise ValueError(
            "kernel='rbf' requires PCA mode (n_components=int, e.g. 50). Full-gene mode "
            "is linear-only: lifting thousands of genes through RFF would defeat the "
            "point (full-gene already reaches the kNN ceiling, and its field cannot be "
            "cached). Use n_components=50 with kernel='rbf'.")
    zg = ds.zarr_group
    X = np.asarray(zg["normalized_vectors"][:])
    y = np.asarray(zg["cluster_labels"][:])
    keep = np.asarray(y) >= 0
    if not keep.all():
        X, y = X[keep], np.asarray(y)[keep]
        print(f"[fit] dropped {int((~keep).sum())} unlabeled (-1) vectors")

    # Cap the labeled set used for FITTING (PCA basis + margin training). The whole
    # labeled matrix is put on the GPU and row-normalized (3 transient copies in
    # fit_pca / project); for large stores (e.g. 650k x 5k = ~13 GB each) that alone
    # OOMs before the field is ever touched. The classifier does not need every
    # vector — a per-class stratified subsample gives the same basis/margin. centroids
    # / cluster_labels stay computed from ALL vectors (in the store), so the map is
    # unaffected. Deterministic (seeded).
    if max_fit_cells is not None and len(y) > max_fit_cells:
        rng = np.random.default_rng(seed)
        take = []
        for c in np.unique(y):
            ci = np.flatnonzero(y == c)
            n_c = max(1, int(round(len(ci) * max_fit_cells / len(y))))
            take.append(rng.choice(ci, min(n_c, len(ci)), replace=False))
        sub = np.sort(np.concatenate(take))
        print(f"[fit] subsampled labeled {len(y)} -> {len(sub)} for fitting "
              f"(max_fit_cells={max_fit_cells}; stratified, seeded)")
        X, y = X[sub], y[sub]

    k = int(zg["centroids"].shape[0])
    if vf_norm_threshold is None:
        vf_norm_threshold = float(zg.attrs.get("select_vf_norm_threshold",
                                  zg.attrs.get("map_vf_norm_threshold", 0.0)))

    if n_components is None:
        print(f"[fit] labeled={X.shape}  k={k}  FULL-GENE (no PCA)  thr={vf_norm_threshold:.4f}")
        Zl = _row_normalize(torch.as_tensor(X, device=device).float())
        yt = torch.as_tensor(np.asarray(y), device=device).long()
        d = Zl.shape[1]
        if use_centroid_init:
            cents = _row_normalize(torch.as_tensor(np.asarray(zg["centroids"][:]),
                                                   device=device).float())
            W = (cents / torch.clamp(cents.norm(dim=1, keepdim=True), min=1e-12) * 0.1
                 ).clone().requires_grad_(True)
        else:
            W = torch.zeros(k, d, device=device, requires_grad=True)
        b = torch.zeros(k, device=device, requires_grad=True)
        ce_lr, ce_iters = 0.05, max(inner_iters * anneal_steps, 1500)
        opt = torch.optim.Adam([W, b], lr=ce_lr)
        for it in range(ce_iters):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(Zl @ W.T + b, yt).backward()
            opt.step()
        W, b = W.detach(), b.detach()
        train_acc = float(((Zl @ W.T + b).argmax(1) == yt).float().mean())
        print(f"[fit] full-gene cross-entropy: {ce_iters} iters  train-acc={train_acc:.3f}")
        print(f"[fit] abstain_q={abstain_q} (Unknown = most-ambiguous {abstain_q:.0%} of field)")
        model = dict(W=W, b=b, mean=None, comps=None, k=k,
                     vf_norm_threshold=float(vf_norm_threshold), n_components=None,
                     abstain_q=float(abstain_q))
        ds._tsvm_model = model
        return model

    print(f"[fit] labeled={X.shape}  k={k}  pca={n_components}  thr={vf_norm_threshold:.4f}")
    mkey, ckey = "tsvm_pca_mean", "tsvm_pca_comps"
    PCA_ROWNORM_VER = 2
    have = (ckey in zg and tuple(zg[ckey].shape) == (n_components, X.shape[1])
            and int(zg.attrs.get("tsvm_pca_rownorm_ver", 0)) == PCA_ROWNORM_VER)
    if have:
        mean = torch.as_tensor(np.asarray(zg[mkey][:]), device=device)
        comps = torch.as_tensor(np.asarray(zg[ckey][:]), device=device)
        print("[fit] loaded persisted PCA basis (row-normalized)")
    else:
        mean, comps = fit_pca(X, n_components=n_components, device=device, seed=seed)
        for kk in (mkey, ckey):
            if kk in zg:
                del zg[kk]
        zg[mkey] = mean.cpu().numpy().astype(np.float32)
        zg[ckey] = comps.cpu().numpy().astype(np.float32)
        zg.attrs["tsvm_pca_rownorm_ver"] = PCA_ROWNORM_VER
        print("[fit] computed & persisted PCA basis (row-normalized)")
    Zl = project(X, mean, comps)

    sigma_bin = float(zg.attrs.get("sigma_bin", 1.0))
    if train_floor == "principled":
        floor = 2.0 / (np.sqrt(2 * np.pi) * sigma_bin) ** 2
    elif train_floor == "store":
        floor = float(vf_norm_threshold)
    else:
        floor = float(train_floor)

    Zu = None
    if use_unlabeled:
        build_field_pca_cache(ds, dict(mean=mean, comps=comps), chunk_h=chunk_h, device=device)
        _empty_cache(device)
        if algo == "labelswitch":
            Zu = load_unlabeled_from_cache(ds, vf_norm_threshold=floor, device=device)
            print(f"[fit] transductive pool = FULL field {tuple(Zu.shape)} "
                  f"(train-floor={floor:.4f})")
        elif unlabeled_source == "field":
            Zu = sample_unlabeled_field(ds, mean, comps, vf_norm_threshold=floor,
                                        n_target=n_unlabeled, device=device, seed=seed)
            print(f"[fit] unlabeled sample={tuple(Zu.shape)} source=field")
        else:
            Zu = sample_unlabeled_from_cache(ds, vf_norm_threshold=floor,
                                             n_target=n_unlabeled, device=device, seed=seed)
            print(f"[fit] unlabeled sample={tuple(Zu.shape)} source=cache")

    omega = beta_rff = None
    if kernel == "rbf":
        if gamma is None:
            gamma = rff_gamma_median(Zl, seed=rff_seed)
            print(f"[fit] gamma (median heuristic) = {gamma:.6g}")
        omega, beta_rff = rff_params(Zl.shape[1], rff_dim, gamma, seed=rff_seed, device=device)
        for kk, arr in (("tsvm_rff_omega", omega), ("tsvm_rff_beta", beta_rff)):
            if kk in zg:
                del zg[kk]
            zg[kk] = arr.cpu().numpy().astype(np.float32)
        zg.attrs["tsvm_kernel"] = "rbf"
        zg.attrs["tsvm_rff_gamma"] = float(gamma)
        zg.attrs["tsvm_rff_dim"] = int(rff_dim)
        zg.attrs["tsvm_rff_seed"] = int(rff_seed)
        Zl = rff_apply(Zl, omega, beta_rff)
        if Zu is not None and algo == "soft":
            Zu = rff_apply(Zu, omega, beta_rff)
        print(f"[fit] RBF via RFF: d={omega.shape[1]} -> D={rff_dim}  labeled={tuple(Zl.shape)}")
    else:
        zg.attrs["tsvm_kernel"] = "linear"

    W_init = None
    if use_centroid_init:
        cents = np.asarray(zg["centroids"][:])
        Wc = project(cents, mean, comps)
        if kernel == "rbf":
            Wc = rff_apply(Wc, omega, beta_rff)
        Wc = Wc / torch.clamp(Wc.norm(dim=1, keepdim=True), min=1e-12)
        W_init = Wc * 0.1

    if algo == "labelswitch":
        W, b = _train_labelswitch(Zl, y, k, Zu, omega, beta_rff, W_init=W_init, lam=lam,
                                  Cu=Cu, anneal_steps=anneal_steps, switch_rounds=switch_rounds,
                                  inner=ls_inner, lr=lr, device=device, seed=seed)
    elif algo == "soft":
        W, b = train_tsvm(Zl, y, k, Zu=Zu, W_init=W_init, lam=lam, ent=ent,
                          balance=balance, anneal_steps=anneal_steps,
                          inner_iters=inner_iters, lr=lr, device=device)
    else:
        raise ValueError("algo must be 'labelswitch' or 'soft'")

    with torch.no_grad():
        tr = float(((Zl @ W.T + b).argmax(1) == torch.as_tensor(y, device=device).long()).float().mean())
    print(f"[fit] algo={algo}  train_acc={tr:.4f}  abstain_q={abstain_q} "
          f"({'off' if abstain_q <= 0 else 'on'} at map time; filtering is filter_celltypemaps' job)")
    model = dict(W=W, b=b, mean=mean, comps=comps, k=k,
                 vf_norm_threshold=float(vf_norm_threshold), n_components=n_components,
                 abstain_q=float(abstain_q), algo=algo,
                 apply_floor=0.0,
                 kernel=kernel, rff_omega=omega, rff_beta=beta_rff,
                 gamma=(float(gamma) if kernel == "rbf" else None),
                 rff_dim=(int(rff_dim) if kernel == "rbf" else None))
    ds._tsvm_model = model
    return model


def _pca_fingerprint(comps):
    """Cheap stable id for a PCA basis (n_components + checksum)."""
    c = comps.detach().float()
    return f"{c.shape[0]}x{c.shape[1]}:{float(c.abs().sum()):.4f}"


def build_field_pca_cache(ds, model=None, *, key="tsvm_field_pca",
                          chunk_h=128, device="cuda", overwrite=False):
    """Project the whole field to PCA once and store it as ``key`` (d, n_R, n_C).

    The projected field (~d/n_G smaller, e.g. 50 vs 3808 -> ~3.9 GB here) depends
    ONLY on the PCA basis, which is fixed across classifier tuning (ent/balance/
    tau/W). So this expensive full-field read happens once; ``map_celltypes_tsvm``
    then reuses the cache and runs in seconds.

    Per chunk: read (n_G, h, n_C) -> GPU -> project ALL voxels on GPU (no slow CPU
    transpose/gather) -> write (d, h, n_C). Returns the fingerprint of the basis.
    """
    if model is None:
        model = getattr(ds, "_tsvm_model", None)
    mean, comps = model["mean"].to(device), model["comps"].to(device)
    d = comps.shape[0]
    fp = _pca_fingerprint(comps)

    zg = ds.zarr_group
    vf_z = zg["vf_normalized"]
    n_G, n_R, n_C = vf_z.shape

    if (not overwrite) and key in zg and zg.attrs.get(f"{key}_fp") == fp:
        print(f"[cache] reuse {key} (fp match)")
        return fp

    if key in zg:
        del zg[key]
    cache = zg.create_array(key, shape=(d, n_R, n_C),
                            chunks=(d, chunk_h, n_C), dtype="f4")
    for r0 in tqdm(list(range(0, n_R, chunk_h)), desc="build_field_pca_cache"):
        r1 = min(r0 + chunk_h, n_R)
        h = r1 - r0
        block = np.asarray(vf_z[:, r0:r1, :]).reshape(n_G, h * n_C)
        gb = torch.as_tensor(block, device=device)
        gb -= gb.mean(dim=0, keepdim=True)
        gb /= torch.clamp(gb.norm(dim=0, keepdim=True), min=1e-12)
        Z = (comps @ gb).T - (comps @ mean)
        Zc = Z.T.reshape(d, h, n_C).contiguous().cpu().numpy()
        cache[:, r0:r1, :] = Zc
        del block, gb, Z, Zc
        _empty_cache(device)
    zg.attrs[f"{key}_fp"] = fp
    print(f"[cache] wrote {key} {(d, n_R, n_C)} fp={fp}")
    return fp


def _signal_col_runs(col_any, cw, max_gap_chunks=2):
    """Chunk-aligned column ranges covering every True in ``col_any``.

    ``cw`` = the field's column-chunk width; ranges start/end on chunk boundaries so
    a partially-covered chunk is decoded once, not twice. Ranges separated by at most
    ``max_gap_chunks`` empty chunks are merged: decoding a couple of empty chunks is
    cheaper than paying per-read overhead for many fragments.
    """
    col_any = np.asarray(col_any, dtype=bool)
    n = col_any.size
    cw = max(int(cw), 1)
    n_ch = (n + cw - 1) // cw
    padded = np.zeros(n_ch * cw, dtype=bool)
    padded[:n] = col_any
    ch_any = padded.reshape(n_ch, cw).any(axis=1)
    idx = np.flatnonzero(ch_any)
    if idx.size == 0:
        return []
    runs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev - 1 <= max_gap_chunks:
            prev = i
            continue
        runs.append((start * cw, min((prev + 1) * cw, n)))
        start = prev = i
    runs.append((start * cw, min((prev + 1) * cw, n)))
    return runs


def map_celltypes_tsvm(ds, model=None, *, tau_conf=None, chunk_h=128, device="cuda",
                       overwrite=True, write_prefix="", set_ds_attrs=True,
                       use_cache=True, cache_key="tsvm_field_pca", tau=None):
    """Apply a fitted TSVM model over the full field and write celltype maps.

    Output zarr layout matches ``map_celltypes_correlation``:
      - ``{prefix}celltype_maps``     int32 (vx=col, vy=row, vz=1), -1 = background/Unknown
      - ``{prefix}max_probabilities`` f4    (top-class softmax confidence in [0,1];
        ``filter_celltypemaps`` thresholds it with ``min_p``, not ``min_r``)
    A voxel is Unknown (-1) when its confidence < ``tau_conf`` (abstention).

    Speed: with ``use_cache=True`` the field is PCA-projected once into a small
    cache (``cache_key``); re-mapping then reads only the cache and classifies on
    GPU in seconds.

    :param ds: Dataset/store whose ``zarr_group`` holds the field; results are written
        back into it (and onto ``ds`` when ``set_ds_attrs``).
    :type ds: Dataset
    :param model: Fitted model dict from :func:`fit_celltype_tsvm`; ``None`` uses the one
        cached on ``ds._tsvm_model``.
    :type model: dict or None
    :param tau_conf: Confidence threshold for abstention; ``None`` uses the fit-time
        calibrated value, a float overrides, 0 keeps every signal voxel.
    :type tau_conf: float or None
    :param chunk_h: Row-chunk height for streaming the field / cache.
    :type chunk_h: int
    :param device: Torch device for scoring.
    :type device: str
    :param overwrite: Delete any existing output arrays before writing.
    :type overwrite: bool
    :param write_prefix: Prefix for the output keys; ``''`` overwrites the standard keys
        (downstream-compatible), ``'tsvm_'`` keeps both the correlation and TSVM maps.
    :type write_prefix: str
    :param set_ds_attrs: When True (and ``write_prefix`` is ``''``), also set the maps and
        shape attributes on ``ds``.
    :type set_ds_attrs: bool
    :param use_cache: Reuse/build the small PCA field cache for a fast re-map (ignored in
        full-gene mode, which always streams).
    :type use_cache: bool
    :param cache_key: Zarr key of the PCA field cache.
    :type cache_key: str
    :param tau: Deprecated alias, ignored.
    :type tau: float or None
    """
    device = _resolve_device(device)
    if model is None:
        model = getattr(ds, "_tsvm_model", None)
        if model is None:
            raise ValueError("no model: pass model= or run fit_celltype_tsvm first.")
    W, b = model["W"].to(device), model["b"].to(device)
    full_gene = model.get("comps") is None
    mean = None if full_gene else model["mean"].to(device)
    comps = None if full_gene else model["comps"].to(device)
    k = model["k"]
    thr = float(model.get("apply_floor", model["vf_norm_threshold"]))
    abstain_q = float(model.get("abstain_q", 0.0))
    omega = model.get("rff_omega")
    beta_rff = model.get("rff_beta")
    is_rbf = omega is not None and beta_rff is not None
    if is_rbf:
        omega, beta_rff = omega.to(device), beta_rff.to(device)
        if full_gene:
            raise ValueError("kernel='rbf' is not supported in full-gene mode")
        print(f"[map] kernel=rbf (RFF D={omega.shape[0]}, gamma={model.get('gamma')})")

    def _score(Z):
        """(n, d) PCA coords -> (n, k) logits, lifting through RFF when kernel='rbf'."""
        return (rff_apply(Z, omega, beta_rff) if is_rbf else Z) @ W.T + b

    zg = ds.zarr_group
    n_G, n_R, n_C = zg["vf_normalized"].shape
    norm = np.asarray(zg["vf_norm"][:])
    sd = float(getattr(ds, "sampling_distance", zg.attrs.get("bin_size_um", 1.0)))
    sig_mask = torch.as_tensor(norm > thr, device=device).reshape(-1)
    if full_gene:
        use_cache = False

    def _classify(Z_all, score_chunk=1_000_000):
        """Z_all (n_R*n_C, d) on GPU -> (M, conf, tc) ; abstain by field quantile.

        Scored in ROW-CHUNKS: with kernel='rbf' the lifted (N, D) matrix would be
        D/d times the cached field (~40 GB at N=1e7, D=1024), so the lift happens
        per chunk and only the (N,) conf/best survive. Numerically identical to the
        one-shot form for the linear path.
        """
        N = Z_all.shape[0]
        conf = torch.empty(N, device=Z_all.device, dtype=torch.float32)
        best = torch.empty(N, device=Z_all.device, dtype=torch.int32)
        for s in range(0, N, score_chunk):
            e = min(s + score_chunk, N)
            p = torch.softmax(_score(Z_all[s:e]), dim=1)
            c, bi = p.max(dim=1)
            conf[s:e] = c
            best[s:e] = bi.to(torch.int32)
            del p, c, bi
        if tau_conf is not None:
            tc = float(tau_conf)
        elif abstain_q <= 0:
            tc = -1.0
        else:
            sc = conf[sig_mask]
            if sc.numel() > 2_000_000:
                sc = sc[torch.randperm(sc.numel(), device=sc.device)[:2_000_000]]
            tc = float(torch.quantile(sc, abstain_q).item())
        keep = sig_mask & (conf >= tc)
        best = torch.where(keep, best, torch.full_like(best, -1))
        conf = torch.where(sig_mask, conf, torch.full_like(conf, -1.0))
        return (best.cpu().numpy().reshape(n_R, n_C),
                conf.cpu().numpy().astype(np.float32).reshape(n_R, n_C), tc)

    if full_gene:
        vf_z = zg["vf_normalized"]
        M  = np.full((n_R, n_C), -1, dtype=np.int32)
        Mg = np.full((n_R, n_C), -1.0, dtype=np.float32)
        cw = int(vf_z.chunks[2]) or 1
        for r0 in tqdm(list(range(0, n_R, chunk_h)), desc="map_celltypes_tsvm[full]"):
            r1 = min(r0 + chunk_h, n_R); h = r1 - r0
            sig = norm[r0:r1] > thr
            if not sig.any():
                continue
            for c0, c1 in _signal_col_runs(sig.any(axis=0), cw):
                w = c1 - c0
                block = np.asarray(vf_z[:, r0:r1, c0:c1]).reshape(n_G, h * w)
                gb = torch.as_tensor(block, device=device)
                gb -= gb.mean(dim=0, keepdim=True)
                gb /= torch.clamp(gb.norm(dim=0, keepdim=True), min=1e-12)
                S = (W @ gb).T + b
                p = torch.softmax(S, dim=1)
                conf, best = p.max(dim=1)
                sub = sig[:, c0:c1]
                bn = best.to(torch.int32).cpu().numpy().reshape(h, w)
                cn = conf.cpu().numpy().astype(np.float32).reshape(h, w)
                m_sub = M[r0:r1, c0:c1]; m_sub[sub] = bn[sub]; M[r0:r1, c0:c1] = m_sub
                g_sub = Mg[r0:r1, c0:c1]; g_sub[sub] = cn[sub]; Mg[r0:r1, c0:c1] = g_sub
                del block, gb, S, p, conf, best
            _empty_cache(device)
        if tau_conf is not None:
            tc = float(tau_conf)
        else:
            sc = Mg[Mg >= 0.0]
            if sc.size > 2_000_000:
                sc = np.random.default_rng(0).choice(sc, 2_000_000, replace=False)
            tc = float(np.quantile(sc, abstain_q)) if sc.size else 0.0
        M[(Mg >= 0.0) & (Mg < tc)] = -1
    elif use_cache:
        build_field_pca_cache(ds, model, key=cache_key, chunk_h=chunk_h, device=device)
        cache = zg[cache_key]
        d = cache.shape[0]
        M = np.full(n_R * n_C, -1, dtype=np.int32)
        Mg = np.full(n_R * n_C, -1.0, dtype=np.float32)
        for r0 in tqdm(list(range(0, n_R, chunk_h)), desc="map_celltypes_tsvm[cache]"):
            r1 = min(r0 + chunk_h, n_R)
            Zc = np.ascontiguousarray(np.asarray(cache[:, r0:r1, :]).reshape(d, -1).T)
            Z = torch.as_tensor(Zc, device=device)
            p = torch.softmax(_score(Z), dim=1)
            c, bi = p.max(dim=1)
            off = r0 * n_C
            M[off:off + Z.shape[0]] = bi.to(torch.int32).cpu().numpy()
            Mg[off:off + Z.shape[0]] = c.cpu().numpy().astype(np.float32)
            del Zc, Z, p, c, bi
            _empty_cache(device)
        M = M.reshape(n_R, n_C); Mg = Mg.reshape(n_R, n_C)
        sig = norm > thr
        M[~sig] = -1; Mg[~sig] = -1.0
        if tau_conf is not None:
            tc = float(tau_conf)
        elif abstain_q <= 0:
            tc = -1.0
        else:
            sc = Mg[sig]
            if sc.size > 2_000_000:
                sc = np.random.default_rng(0).choice(sc, 2_000_000, replace=False)
            tc = float(np.quantile(sc, abstain_q)) if sc.size else -1.0
        M[(Mg >= 0.0) & (Mg < tc)] = -1
    else:
        tc = float(tau_conf) if tau_conf is not None else 0.0
        vf_z = zg["vf_normalized"]
        d = comps.shape[0]
        M = np.full((n_R, n_C), -1, dtype=np.int32)
        Mg = np.full((n_R, n_C), -1.0, dtype=np.float32)
        for r0 in tqdm(list(range(0, n_R, chunk_h)), desc="map_celltypes_tsvm"):
            r1 = min(r0 + chunk_h, n_R)
            h = r1 - r0
            sig = norm[r0:r1] > thr
            if not sig.any():
                continue
            block = np.asarray(vf_z[:, r0:r1, :]).reshape(n_G, h * n_C)
            gb = torch.as_tensor(block, device=device)
            sm = torch.as_tensor(sig.reshape(-1), device=device)
            Z = (_row_normalize(gb.T[sm]) - mean) @ comps.T
            p = torch.softmax(_score(Z), dim=1)
            conf, best = p.max(dim=1)
            best = best.to(torch.int32)
            best[conf < tc] = -1
            pos = np.where(sig.reshape(-1))[0]
            m_flat = M[r0:r1].reshape(-1); g_flat = Mg[r0:r1].reshape(-1)
            m_flat[pos] = best.cpu().numpy()
            g_flat[pos] = conf.cpu().numpy().astype(np.float32)
            M[r0:r1] = m_flat.reshape(h, n_C); Mg[r0:r1] = g_flat.reshape(h, n_C)
            del block, gb, sm, Z, p, conf, best
            _empty_cache(device)

    n_assigned = int((M >= 0).sum())
    print(f"[map] assigned {n_assigned} voxels (tau_conf={tc:.3f})")

    cm_3d = M.T[:, :, None].astype(np.int32)
    mp_3d = Mg.T[:, :, None].astype(np.float32)
    vx, vy, vz = cm_3d.shape
    cm_key = f"{write_prefix}celltype_maps"
    mp_key = f"{write_prefix}max_probabilities"
    mc_key = f"{write_prefix}max_correlations"
    if overwrite:
        for nm in (cm_key, mp_key, mc_key):
            if nm in zg:
                del zg[nm]
    zg[cm_key] = cm_3d
    zg[mp_key] = mp_3d
    zg.attrs["tsvm_vf_norm_threshold"] = float(thr)
    zg.attrs["tsvm_tau_conf"] = float(tc)
    zg.attrs["tsvm_n_celltypes"] = int(k)

    if set_ds_attrs and write_prefix == "":
        ds.shape = (vx, vy, vz)
        ds.x_scale = 1.0
        ds.y_scale = 1.0
        ds.max_correlations = None
        ds.original_coordinate_bounds = [[0.0, vx * sd], [0.0, vy * sd], [0.0, 0.0]]
        ds.celltype_maps = cm_3d
        ds.max_probabilities = mp_3d
        ds._vf_norm = da.from_array(norm.T[:, :, None], chunks=-1)

    return cm_3d, mp_3d


def map_tsvm(analysis, *, n_components=50, device="cuda", overwrite=True,
             fit_kwargs=None, apply_kwargs=None):
    """Fit a TSVM on the store's labeled vectors and map the whole field.

    Thin wrapper over :func:`fit_celltype_tsvm` + :func:`map_celltypes_tsvm`,
    operating on ``analysis.dataset``. Returns ``(celltype_maps, max_probabilities)``;
    the score volume is the top-class softmax probability, thresholded downstream by
    ``filter_celltypemaps(min_p=…)``.

    Default = **PCA-50 + RBF kernel** (RFF): the small cacheable field representation
    with a non-linear boundary. ``n_components=None`` selects full-gene mode, which is
    LINEAR-only and streams the field per apply -- pass
    ``fit_kwargs={'kernel': 'linear'}`` with it.

    :param analysis: Analysis object exposing ``dataset`` (the store fit and mapped here).
    :type analysis: Analysis
    :param n_components: PCA latent dim passed to :func:`fit_celltype_tsvm`; ``None``
        selects full-gene (linear) mode. May also be supplied inside ``fit_kwargs``.
    :type n_components: int or None
    :param device: Torch device for both fit and apply.
    :type device: str
    :param overwrite: Passed to :func:`map_celltypes_tsvm` to overwrite existing outputs.
    :type overwrite: bool
    :param fit_kwargs: Extra keyword arguments forwarded to :func:`fit_celltype_tsvm`.
    :type fit_kwargs: dict or None
    :param apply_kwargs: Extra keyword arguments forwarded to :func:`map_celltypes_tsvm`.
    :type apply_kwargs: dict or None
    """
    ds = analysis.dataset
    device = _resolve_device(device)
    if "vf_normalized" in ds.zarr_group and ds.zarr_group["vf_normalized"].ndim != 3:
        raise NotImplementedError(
            "map_tsvm supports the 3D count vector field only; the stored field is "
            f"{ds.zarr_group['vf_normalized'].ndim}D (transcript/imaging). Use "
            "map_celltypes(method='corr') for imaging data.")
    fit_kwargs = dict(fit_kwargs or {})
    apply_kwargs = dict(apply_kwargs or {})
    n_components = fit_kwargs.pop("n_components", n_components)
    if n_components is None:
        fit_kwargs.setdefault("kernel", "linear")
    fit_kwargs.setdefault("device", device)
    apply_kwargs.setdefault("device", device)
    apply_kwargs.setdefault("overwrite", overwrite)
    model = fit_celltype_tsvm(ds, n_components=n_components, **fit_kwargs)
    return map_celltypes_tsvm(ds, model=model, **apply_kwargs)
