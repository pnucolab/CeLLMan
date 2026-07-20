"""TSVM-based celltype mapping for CeLLMan spatial (notebook-level, no package edits).

Drop-in alternative to ``map_celltypes_correlation`` (centroid-Pearson template
matching) that replaces the mapping step with a **linear multiclass Transductive
SVM** trained in a PCA latent space, then applied inductively over the field.

Why not plain centroid-correlation
----------------------------------
Correlation assigns each voxel the celltype whose single centroid it most
resembles (cosine on mean-centered profiles). It learns no discriminative
weights and cannot abstain. A margin classifier learns what *separates*
celltypes and gives a calibrated confidence (top1-top2 margin) for abstention.

Formulation (multiclass, calibrated)
------------------------------------
We use the **Crammer-Singer multiclass SVM** margin, NOT one-vs-rest hat-loss:
OvR + the classic TSVM hat-loss ``max(0,1-|f|)`` leaves the per-class scores
uncalibrated, so the multiclass ``argmax`` collapses onto a dominant class.
Crammer-Singer optimizes a single coherent margin across all k classes.

The transductive (unlabeled-field) signal follows TSVM's **low-density
separation** principle, expressed in a multiclass-safe way:
  - entropy minimization on unlabeled posteriors  -> boundaries avoid dense
    regions (the multiclass analog of pushing the margin off the data);
  - a **class-balancing constraint** KL(mean_u softmax || labeled prior) -> the
    unlabeled marginal matches the labeled class frequencies, preventing the
    collapse the balancing constraint also guards against in classic TSVM.
Both unlabeled terms are annealed in (weight 0 -> 1), the standard remedy for
the non-convex transductive objective. With the unlabeled weight at 0 the
trainer reduces to a supervised Crammer-Singer linear SVM.

Pipeline
--------
1. labeled  = ``normalized_vectors`` (downsampled, annotated) + ``cluster_labels``
              (already merged celltype index 0..k-1; row i of ``centroids``).
2. unlabeled = a random subsample of field voxels (``vf_normalized`` where
               ``vf_norm > threshold``) — the transductive pool.
3. PCA (fit on labeled) -> d dims (default 50, matches cluster_vectors).
4. Train (Crammer-Singer + entropy + balancing, annealed). Init from projected
   centroids so the supervised start is near the correlation solution.
5. Inductive apply: stream ``vf_normalized`` row-chunk by row-chunk (exactly like
   ``map_celltypes_correlation``), project to PCA, score ``Z @ W.T + b``, take the
   argmax; the top-class softmax probability goes to ``max_probabilities`` and
   drives abstention (-> Unknown, -1) when below ``tau``. Output zarr layout
   mirrors the correlation path (``celltype_maps`` + a per-voxel score volume) so
   ``filter_celltypemaps`` / ``plot_celltypes_map`` keep working — the score lands
   in ``max_probabilities`` (not ``max_correlations``), so ``filter_celltypemaps``
   thresholds it with ``min_p`` rather than ``min_r``.

True transduction needs the test points at fit time, but the full field is too
large (here 3808 x 3081 x 6361 ~ 75 GB); we fit transductively on a field
*subsample* and apply the learned hyperplanes inductively (S3VM-induced).
"""

import numpy as np
import torch
import dask.array as da
from tqdm.auto import tqdm


# ----------------------------------------------------------------------------
# Per-voxel normalization (matches the correlation baseline's Pearson preproc)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# PCA (GPU, fit on labeled vectors)
# ----------------------------------------------------------------------------
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
    comps = V[:, :n_components].T.contiguous()          # (d, g)
    # fix per-component sign (largest-magnitude entry positive) for a stable basis
    sign = torch.sign(comps[torch.arange(comps.shape[0]), comps.abs().argmax(dim=1)])
    comps = comps * sign.unsqueeze(1)
    del Xt, Xc
    torch.cuda.empty_cache()
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


# ----------------------------------------------------------------------------
# Unlabeled field subsampling (the transductive pool)
# ----------------------------------------------------------------------------
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
    vf_z = zg["vf_normalized"]                          # (n_G, n_R, n_C)
    n_G, n_R, n_C = vf_z.shape
    norm = np.asarray(zg["vf_norm"][:])                 # (n_R, n_C)
    rng = np.random.default_rng(seed)
    starts = list(range(0, n_R, chunk_h))
    rng.shuffle(starts)                                 # spread sampling spatially

    Z_chunks, n_have = [], 0
    pbar = tqdm(starts, desc="sample_unlabeled")
    for r0 in pbar:
        r1 = min(r0 + chunk_h, n_R)
        h = r1 - r0
        sig = norm[r0:r1] > vf_norm_threshold           # (h, n_C)
        if not sig.any():
            continue
        sig_flat = sig.reshape(-1)
        block = np.asarray(vf_z[:, r0:r1, :]).reshape(n_G, h * n_C)     # (g, h*n_C)
        gb = torch.as_tensor(block, device=device)                     # filter+proj on GPU
        sm = torch.as_tensor(sig_flat, device=device)
        Z = (_row_normalize(gb.T[sm]) - mean) @ comps.T                # (n_sig, d)
        Z_chunks.append(Z)
        n_have += Z.shape[0]
        del block, gb, sm
        torch.cuda.empty_cache()
        pbar.set_postfix(have=n_have)
        if n_have >= n_target:                           # early stop
            break
    if not Z_chunks:
        return torch.empty(0, comps.shape[0], device=device)
    Zu = torch.cat(Z_chunks, dim=0)
    if Zu.shape[0] > n_target:                           # trim to target
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
    cache = zg[key]                                      # (d, n_R, n_C) already projected
    d = cache.shape[0]
    norm = np.asarray(zg["vf_norm"][:]).reshape(-1)
    sig_idx = np.where(norm > vf_norm_threshold)[0]
    rng = np.random.default_rng(seed)
    take = np.sort(rng.choice(sig_idx, min(n_target, sig_idx.size), replace=False))
    Zf = np.asarray(cache[:]).reshape(d, -1)             # (d, N)  ~few GB
    Zu = Zf[:, take].T                                   # (n_take, d)
    return torch.as_tensor(np.ascontiguousarray(Zu), device=device)


# ----------------------------------------------------------------------------
# Multiclass transductive SVM (Crammer-Singer + low-density + balancing)
# ----------------------------------------------------------------------------
def train_tsvm(Zl, yl, k, Zu=None, *, W_init=None,
               lam=1e-4, ent=0.3, balance=1.0,
               anneal_steps=4, inner_iters=200, lr=0.1, device="cuda",
               verbose=True):
    """Train a linear multiclass TSVM on GPU.

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
            F = Zl @ W.T + b                              # (L, k)
            true_s = F.gather(1, yl[:, None]).squeeze(1)
            Fm = F.clone()
            Fm[ar, yl] = -1e9                             # mask true class
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


# ----------------------------------------------------------------------------
# Top-level: fit
# ----------------------------------------------------------------------------
def fit_celltype_tsvm(ds, *, n_components=50, n_unlabeled=200_000,
                      vf_norm_threshold=None, lam=1e-4, ent=0.0, balance=2.0,
                      anneal_steps=4, inner_iters=200, lr=0.1,
                      device="cuda", seed=0, use_centroid_init=True,
                      use_unlabeled=True, unlabeled_source="cache",
                      abstain_q=0.15, chunk_h=128):
    """Fit PCA + multiclass TSVM from the store's labeled vectors and field pool.

    Returns a model dict and caches it on ``ds._tsvm_model``.

    Key design (fixes the CAF-collapse): the transductive pool is the **real
    field** (``unlabeled_source='cache'``), so the class-balancing constraint acts
    on the distribution we actually colour and pulls predicted field proportions
    toward the labeled class priors -- a single class can no longer swallow the
    field. The overconfidence-inducing entropy term is OFF by default (``ent=0``).

    Parameters
    ----------
    n_components : int or None
        PCA latent dim. ``None`` -> **full-gene mode**: no PCA, the linear classifier
        is trained on the row-normalized full-gene vectors. PCA-50 caps training
        fidelity to the labels at ~0.88 (it drops low-variance class-discriminative
        directions); full genes restore it to ~0.95. Trade-off: the field can't be
        pre-projected into a small cache, so mapping STREAMS the field per apply.
    use_unlabeled : bool
        False -> supervised (no-transduction) variant.
    unlabeled_source : {'cache', 'field'}
        ``'cache'`` (default): sample real field voxels from the PCA cache
        (instant; builds the cache once if missing). ``'field'``: sample fresh
        voxels by decompressing the field (slow).
    abstain_q : float
        Abstention strength. A field voxel is left **Unknown** when its top-class
        confidence is below the ``abstain_q`` quantile of labeled confidences
        (0.10 = medium: more ambiguous than the least-confident ~10% of labels).
    """
    zg = ds.zarr_group
    X = np.asarray(zg["normalized_vectors"][:])          # (L, g)
    y = np.asarray(zg["cluster_labels"][:])              # (L,) merged celltype idx
    # -1 = not a callable cell (Background / Unknown / unresolved, or a clustering
    # outlier). Drop those rows here so the labeled set never has to be materialized
    # as a sliced copy of normalized_vectors in the store: the supervised fit, the
    # PCA basis and the centroid init all see only real labels.
    keep = np.asarray(y) >= 0
    if not keep.all():
        X, y = X[keep], np.asarray(y)[keep]
        print(f"[fit] dropped {int((~keep).sum())} unlabeled (-1) vectors")
    k = int(zg["centroids"].shape[0])
    if vf_norm_threshold is None:
        vf_norm_threshold = float(zg.attrs.get("select_vf_norm_threshold",
                                  zg.attrs.get("map_vf_norm_threshold", 0.0)))

    # ----------------------------------------------------------------------
    # FULL-GENE mode (n_components=None): skip PCA, classify on the row-normalized
    # full-gene vectors. PCA-50 keeps high-VARIANCE axes but discards the low-
    # variance class-DISCRIMINATIVE directions a LINEAR boundary needs, capping
    # training fidelity at ~0.88; full genes restore it to ~0.95 (= the kNN / full-
    # gene-logistic ceiling). Cost: the field can't be pre-projected into a small
    # cache, so map_celltypes_tsvm STREAMS the field per apply (like correlation).
    # ----------------------------------------------------------------------
    if n_components is None:
        print(f"[fit] labeled={X.shape}  k={k}  FULL-GENE (no PCA)  thr={vf_norm_threshold:.4f}")
        Zl = _row_normalize(torch.as_tensor(X, device=device).float())   # (L, g)
        yt = torch.as_tensor(np.asarray(y), device=device).long()
        # Cross-entropy (softmax) regression, NOT the Crammer-Singer hinge: the goal
        # of full-gene mode is to FIT the CeLLMan labels faithfully, and the hinge
        # tolerates margin slack (caps ~0.92) while cross-entropy drives training
        # fidelity to the ~0.95 ceiling. The transductive balance/entropy terms
        # (anti-collapse) are not needed once the magnitude bug is fixed and the
        # full-gene classes are well separated, so they are dropped here.
        d = Zl.shape[1]
        if use_centroid_init:
            cents = _row_normalize(torch.as_tensor(np.asarray(zg["centroids"][:]),
                                                   device=device).float())
            W = (cents / torch.clamp(cents.norm(dim=1, keepdim=True), min=1e-12) * 0.1
                 ).clone().requires_grad_(True)
        else:
            W = torch.zeros(k, d, device=device, requires_grad=True)
        b = torch.zeros(k, device=device, requires_grad=True)
        # CE-specific optimizer settings (the hinge-tuned lr/lam defaults do not apply):
        # lr=0.05 + NO L2 (lam=0) reaches the ~0.95 full-gene fidelity ceiling; even a
        # tiny L2 caps it lower (it shrinks the discriminative weights we want here).
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
    # Persist the PCA basis so it is byte-identical across runs/devices: this keeps
    # the field-PCA cache fingerprint stable (pca_lowrank's randomized SVD differs
    # per CUDA device, which would otherwise force a cache rebuild every run).
    mkey, ckey = "tsvm_pca_mean", "tsvm_pca_comps"
    # The persisted basis is keyed by a normalization version: a basis written by an
    # older (non-row-normalized) build has the same shape but lives in a different
    # space, so we MUST recompute rather than silently reuse it.
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
    Zl = project(X, mean, comps)                         # (L, d)

    Zu = None
    if use_unlabeled:
        if unlabeled_source == "cache":
            model_tmp = dict(mean=mean, comps=comps)     # ensure cache exists (build once)
            build_field_pca_cache(ds, model_tmp, chunk_h=chunk_h, device=device)
            Zu = sample_unlabeled_from_cache(ds, vf_norm_threshold=vf_norm_threshold,
                                             n_target=n_unlabeled, device=device, seed=seed)
        elif unlabeled_source == "field":
            Zu = sample_unlabeled_field(ds, mean, comps, vf_norm_threshold=vf_norm_threshold,
                                        n_target=n_unlabeled, device=device, seed=seed)
        else:
            raise ValueError("unlabeled_source must be 'cache' or 'field'")
        print(f"[fit] unlabeled pool={tuple(Zu.shape)}  source={unlabeled_source}")

    W_init = None
    if use_centroid_init:
        cents = np.asarray(zg["centroids"][:])           # (k, g)
        Wc = project(cents, mean, comps)                 # (k, d)
        Wc = Wc / torch.clamp(Wc.norm(dim=1, keepdim=True), min=1e-12)
        W_init = Wc * 0.1

    W, b = train_tsvm(Zl, y, k, Zu=Zu, W_init=W_init, lam=lam, ent=ent,
                      balance=balance, anneal_steps=anneal_steps,
                      inner_iters=inner_iters, lr=lr, device=device)

    # Abstention is calibrated on the FIELD at map time (field confidences sit
    # lower than labeled ones once balancing spreads predictions); abstain_q is
    # the fraction of signal field voxels to leave Unknown.
    print(f"[fit] abstain_q={abstain_q} (Unknown = most-ambiguous {abstain_q:.0%} of field)")
    model = dict(W=W, b=b, mean=mean, comps=comps, k=k,
                 vf_norm_threshold=float(vf_norm_threshold), n_components=n_components,
                 abstain_q=float(abstain_q))
    ds._tsvm_model = model
    return model


# ----------------------------------------------------------------------------
# Field <-> PCA cache  (project the field ONCE; reuse for all classifier tuning)
# ----------------------------------------------------------------------------
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
        block = np.asarray(vf_z[:, r0:r1, :]).reshape(n_G, h * n_C)     # (n_G, h*n_C)
        gb = torch.as_tensor(block, device=device)                     # -> GPU (g, n)
        # Per-voxel normalize IN PLACE on the gene-major block (each column = one
        # voxel): mean-center across genes + L2. Avoids the (n, g) transpose copy.
        gb -= gb.mean(dim=0, keepdim=True)
        gb /= torch.clamp(gb.norm(dim=0, keepdim=True), min=1e-12)
        # Project without materializing the big (n, g) `(gb.T - mean)` tensor:
        #   (gb.T - mean) @ comps.T == (comps @ gb).T - comps @ mean   ((d, n) is small)
        Z = (comps @ gb).T - (comps @ mean)                            # (h*n_C, d) on GPU
        Zc = Z.T.reshape(d, h, n_C).contiguous().cpu().numpy()         # (d, h, n_C)
        cache[:, r0:r1, :] = Zc
        del block, gb, Z, Zc
        torch.cuda.empty_cache()
    zg.attrs[f"{key}_fp"] = fp
    print(f"[cache] wrote {key} {(d, n_R, n_C)} fp={fp}")
    return fp


# ----------------------------------------------------------------------------
# Top-level: inductive apply (mirrors map_celltypes_correlation I/O)
# ----------------------------------------------------------------------------
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
    A voxel is **Unknown (-1)** when its confidence < ``tau_conf`` (abstention).
    ``tau_conf`` defaults to the value calibrated at fit time from labeled
    confidences (``model['tau_conf']``); pass a float to override, or 0 to keep
    every signal voxel. With ``write_prefix=''`` this overwrites the standard keys
    (downstream-compatible); use ``write_prefix='tsvm_'`` to keep both maps.

    Speed: with ``use_cache=True`` the field is PCA-projected once into a small
    cache (``cache_key``); re-mapping then reads only the cache and classifies on
    GPU in seconds. (``tau`` is a deprecated alias, ignored.)
    """
    if model is None:
        model = getattr(ds, "_tsvm_model", None)
        if model is None:
            raise ValueError("no model: pass model= or run fit_celltype_tsvm first.")
    W, b = model["W"].to(device), model["b"].to(device)
    full_gene = model.get("comps") is None
    mean = None if full_gene else model["mean"].to(device)
    comps = None if full_gene else model["comps"].to(device)
    k = model["k"]
    thr = model["vf_norm_threshold"]
    abstain_q = float(model.get("abstain_q", 0.15))

    zg = ds.zarr_group
    n_G, n_R, n_C = zg["vf_normalized"].shape
    norm = np.asarray(zg["vf_norm"][:])                   # (n_R, n_C)
    sd = float(getattr(ds, "sampling_distance", zg.attrs.get("bin_size_um", 1.0)))
    sig_mask = torch.as_tensor(norm > thr, device=device).reshape(-1)   # (n_R*n_C,)
    if full_gene:
        use_cache = False                                 # no PCA cache in full-gene mode

    def _classify(Z_all):
        """Z_all (n_R*n_C, d) on GPU -> (M, conf, tc) ; abstain by field quantile."""
        p = torch.softmax(Z_all @ W.T + b, dim=1)         # (N, k)
        conf, best = p.max(dim=1)
        best = best.to(torch.int32)
        # confidence threshold = abstain_q quantile over SIGNAL field voxels
        if tau_conf is not None:
            tc = float(tau_conf)
        else:
            sc = conf[sig_mask]
            if sc.numel() > 2_000_000:                    # subsample for quantile
                sc = sc[torch.randperm(sc.numel(), device=sc.device)[:2_000_000]]
            tc = float(torch.quantile(sc, abstain_q).item())
        keep = sig_mask & (conf >= tc)
        best = torch.where(keep, best, torch.full_like(best, -1))
        conf = torch.where(sig_mask, conf, torch.full_like(conf, -1.0))
        return (best.cpu().numpy().reshape(n_R, n_C),
                conf.cpu().numpy().astype(np.float32).reshape(n_R, n_C), tc)

    if full_gene:
        # FULL-GENE: stream the field, classify on row-normalized full-gene voxels.
        # Per chunk, normalize each voxel (column) in place and score W @ gb (small
        # (k, n_cols) GEMM -> no big transpose copy). Confidences are kept for the
        # whole field, so field-quantile abstention is applied AFTER the sweep.
        vf_z = zg["vf_normalized"]
        M  = np.full((n_R, n_C), -1, dtype=np.int32)
        Mg = np.full((n_R, n_C), -1.0, dtype=np.float32)
        # The sweep is bound by zstd decompression of the field, not by the GPU, so
        # the only real lever is decompressing less: read each row block as the
        # COLUMN RUNS that actually contain signal instead of the whole row. Runs are
        # snapped out to the field's column-chunk width so no chunk is ever decoded
        # twice, and runs separated by a small gap are merged (a few large reads beat
        # many small ones). Typical tissue leaves ~half the tiles empty -> ~2x.
        cw = int(vf_z.chunks[2]) or 1
        for r0 in tqdm(list(range(0, n_R, chunk_h)), desc="map_celltypes_tsvm[full]"):
            r1 = min(r0 + chunk_h, n_R); h = r1 - r0
            sig = norm[r0:r1] > thr
            if not sig.any():
                continue
            for c0, c1 in _signal_col_runs(sig.any(axis=0), cw):
                w = c1 - c0
                block = np.asarray(vf_z[:, r0:r1, c0:c1]).reshape(n_G, h * w)  # (g, n_cols)
                gb = torch.as_tensor(block, device=device)
                gb -= gb.mean(dim=0, keepdim=True)                       # per-voxel row-normalize
                gb /= torch.clamp(gb.norm(dim=0, keepdim=True), min=1e-12)
                S = (W @ gb).T + b                                       # (n_cols, k)
                p = torch.softmax(S, dim=1)
                conf, best = p.max(dim=1)
                sub = sig[:, c0:c1]
                bn = best.to(torch.int32).cpu().numpy().reshape(h, w)
                cn = conf.cpu().numpy().astype(np.float32).reshape(h, w)
                m_sub = M[r0:r1, c0:c1]; m_sub[sub] = bn[sub]; M[r0:r1, c0:c1] = m_sub
                g_sub = Mg[r0:r1, c0:c1]; g_sub[sub] = cn[sub]; Mg[r0:r1, c0:c1] = g_sub
                del block, gb, S, p, conf, best
            torch.cuda.empty_cache()
        # field-quantile abstention (whole-field confidences now available)
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
        cache = zg[cache_key]                             # (d, n_R, n_C)
        d = cache.shape[0]
        Zf = torch.as_tensor(np.asarray(cache[:]), device=device)        # (d, n_R, n_C) ~ small
        Z_all = Zf.reshape(d, n_R * n_C).T.contiguous()  # (N, d)
        M, Mg, tc = _classify(Z_all)
        del Zf, Z_all
        torch.cuda.empty_cache()
    else:
        # streaming fallback: project on GPU per chunk (no slow CPU transpose/gather).
        # Field-quantile abstention needs the whole field at once -> here we only
        # honour an explicit tau_conf (else keep all signal voxels).
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
            Z = (_row_normalize(gb.T[sm]) - mean) @ comps.T  # (n_sig, d)
            p = torch.softmax(Z @ W.T + b, dim=1)
            conf, best = p.max(dim=1)
            best = best.to(torch.int32)
            best[conf < tc] = -1
            pos = np.where(sig.reshape(-1))[0]
            m_flat = M[r0:r1].reshape(-1); g_flat = Mg[r0:r1].reshape(-1)
            m_flat[pos] = best.cpu().numpy()
            g_flat[pos] = conf.cpu().numpy().astype(np.float32)
            M[r0:r1] = m_flat.reshape(h, n_C); Mg[r0:r1] = g_flat.reshape(h, n_C)
            del block, gb, sm, Z, p, conf, best
            torch.cuda.empty_cache()

    n_assigned = int((M >= 0).sum())
    print(f"[map] assigned {n_assigned} voxels (tau_conf={tc:.3f})")

    cm_3d = M.T[:, :, None].astype(np.int32)
    mp_3d = Mg.T[:, :, None].astype(np.float32)      # top-class SOFTMAX probability
    vx, vy, vz = cm_3d.shape
    cm_key = f"{write_prefix}celltype_maps"
    mp_key = f"{write_prefix}max_probabilities"       # softmax → the min_p filter slot
    mc_key = f"{write_prefix}max_correlations"        # stale sibling from a prior corr run
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
        # softmax confidence lives in max_probabilities so filter_celltypemaps
        # uses min_p (its probability threshold), not min_r (correlation).
        ds.max_correlations = None
        ds.original_coordinate_bounds = [[0.0, vx * sd], [0.0, vy * sd], [0.0, 0.0]]
        ds.celltype_maps = cm_3d
        ds.max_probabilities = mp_3d
        ds._vf_norm = da.from_array(norm.T[:, :, None], chunks=-1)

    return cm_3d, mp_3d


# ----------------------------------------------------------------------------
# Package wrapper: fit + apply in one call, for SSAMAnalysis.map_celltypes(method='tsvm')
# ----------------------------------------------------------------------------
def map_tsvm(analysis, *, n_components=None, device="cuda", overwrite=True,
             fit_kwargs=None, apply_kwargs=None):
    """Fit a TSVM on the store's labeled vectors and map the whole field.

    Thin wrapper over :func:`fit_celltype_tsvm` + :func:`map_celltypes_tsvm`,
    operating on ``analysis.dataset``. ``n_components=None`` = full-gene mode
    (highest label fidelity; streams the field). Returns
    ``(celltype_maps, max_probabilities)``; the score volume is the top-class
    softmax probability, thresholded downstream by ``filter_celltypemaps(min_p=…)``.
    """
    ds = analysis.dataset
    # tsvm consumes the stored 3D count field (n_G, R, C). The transcript KDE
    # field is 4D (n_G, vx, vy, vz) and the transcript regime is 'corr'-only
    # (see map_celltypes) — guard rather than silently mis-shape a 4D field.
    if "vf_normalized" in ds.zarr_group and ds.zarr_group["vf_normalized"].ndim != 3:
        raise NotImplementedError(
            "map_tsvm supports the 3D count vector field only; the stored field is "
            f"{ds.zarr_group['vf_normalized'].ndim}D (transcript/imaging). Use "
            "map_celltypes(method='corr') for imaging data.")
    fit_kwargs = dict(fit_kwargs or {})
    apply_kwargs = dict(apply_kwargs or {})
    # map_celltypes() has no n_components parameter of its own, so accept it inside
    # fit_kwargs (otherwise it collides with the explicit keyword below).
    n_components = fit_kwargs.pop("n_components", n_components)
    fit_kwargs.setdefault("device", device)
    apply_kwargs.setdefault("device", device)
    apply_kwargs.setdefault("overwrite", overwrite)
    model = fit_celltype_tsvm(ds, n_components=n_components, **fit_kwargs)
    return map_celltypes_tsvm(ds, model=model, **apply_kwargs)
