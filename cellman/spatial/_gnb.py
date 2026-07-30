"""Gaussian naive Bayes celltype mapping for cellman.spatial.

An alternative to the centroid-correlation mapper (:func:`cellman.spatial._count.map_corr`).
A :class:`~sklearn.naive_bayes.GaussianNB` (cuML on GPU, scikit-learn on CPU) is fitted on
the labeled ``normalized_vectors`` — ALL of them, streamed via ``partial_fit`` so the full
matrix is never resident — and every field voxel is classified by its posterior. Being
GENERATIVE (one Gaussian per class), each class keeps a non-empty decision region, so GNB
preserves rare cell types that the discriminative correlation argmax can squeeze out.

Both regimes are supported, mirroring the correlation mapper:

  * **count / blur** — :func:`map_gnb` reads the stored 3D ``vf_normalized`` and classifies
    each voxel (the array-based counterpart of :func:`map_corr`).
  * **transcript / imaging** — ``_map_celltypes_cpu`` / ``_map_celltypes_gpu`` fit a model
    here (on the shared-gene subset) and score each KDE chunk on the fly (the KDE
    counterpart, like the correlation transcript path).

Vectors are row-normalized (mean-center + L2) before fit and predict — the same Pearson
space the correlation mapper uses — so the two are directly comparable.
"""
import numpy as np


def _row_normalize(A, xp):
    """Per-row mean-center + L2 (the corr Pearson space); ``A`` on ``xp`` (numpy/cupy)."""
    A = A - A.mean(axis=1, keepdims=True)
    n = xp.linalg.norm(A, axis=1, keepdims=True)
    return A / xp.where(n == 0, xp.asarray(1.0, dtype=A.dtype), n)


def _resolve_backend(gpu_device, force_cpu):
    """``(GaussianNB, xp, use_gpu, device_int)`` — cuML+cupy on GPU, else scikit-learn+numpy."""
    if not force_cpu:
        try:
            import cupy as cp
            from cuml.naive_bayes import GaussianNB as _GNB
            if cp.cuda.runtime.getDeviceCount() > 0:
                dev = gpu_device
                if isinstance(dev, str):
                    dev = int(dev.split(':')[1]) if ':' in dev else 0
                return _GNB, cp, True, int(dev)
        except Exception:
            pass
    from sklearn.naive_bayes import GaussianNB as _GNB
    return _GNB, np, False, None


def fit_gnb(zg, *, gpu_device='cuda', force_cpu=False, row_normalize=True,
            fit_chunk=100_000, var_smoothing=1e-9, gene_indices=None):
    """Fit a GaussianNB on ALL labeled ``normalized_vectors`` (streamed; no subsample).

    :param zg: zarr group holding ``normalized_vectors`` (L, g) + ``cluster_labels`` (L,).
    :param gene_indices: optional column subset of the vectors — used by the transcript
        path so the classifier's genes match the KDE field's shared-gene axis.
    :returns: ``(clf, classes, backend)`` where ``classes`` are the sorted celltype ids and
        ``backend = (xp, use_gpu, device)`` for :func:`predict_gnb`.
    """
    GNB, xp, use_gpu, dev = _resolve_backend(gpu_device, force_cpu)
    nv = zg['normalized_vectors']
    y_all = np.asarray(zg['cluster_labels'][:])
    classes = np.unique(y_all[y_all >= 0])
    L = nv.shape[0]
    gi = None if gene_indices is None else np.asarray(gene_indices)

    def _run():
        clf = GNB(var_smoothing=var_smoothing)
        cls_dev = xp.asarray(classes)
        n_used = 0
        for s in range(0, L, fit_chunk):
            e = min(s + fit_chunk, L)
            yc = y_all[s:e]
            m = yc >= 0
            if not m.any():
                continue
            Xc = np.asarray(nv[s:e])[m]
            if gi is not None:
                Xc = Xc[:, gi]
            Xc = xp.asarray(Xc, dtype=xp.float32)
            if row_normalize:
                Xc = _row_normalize(Xc, xp)
            clf.partial_fit(Xc, xp.asarray(yc[m]), classes=cls_dev)
            n_used += int(m.sum())
        return clf, n_used

    if use_gpu:
        import cupy as cp
        with cp.cuda.Device(dev):
            clf, n_used = _run()
    else:
        clf, n_used = _run()
    print(f"[gnb] fit GaussianNB on {n_used} labeled vectors, {len(classes)} classes "
          f"({'GPU cuML' if use_gpu else 'CPU sklearn'}, streamed, no subsample)")
    return clf, classes, (xp, use_gpu, dev)


def predict_gnb(clf, classes, V, backend, *, row_normalize=True, batch=20_000):
    """Classify ``(n, g)`` vectors ``V`` (numpy) -> ``(labels int32 celltype ids, conf float32)``.

    ``conf`` is the top-class posterior probability. ``classes`` maps the argmax index back
    to the celltype id (both cuML and scikit-learn keep ``classes_`` sorted, matching).

    Predicted in ``batch``-row sub-chunks: ``GaussianNB.predict_proba`` builds an
    ``(n, n_genes)`` array per class, so scoring a whole KDE/field chunk (hundreds of
    thousands of voxels x thousands of genes) at once OOMs the GPU. ``batch`` bounds the
    transient; lower it if a card is small.
    """
    xp, use_gpu, dev = backend
    n = int(V.shape[0])
    if n == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)

    def _one(Vb):
        Vx = xp.asarray(np.ascontiguousarray(Vb), dtype=xp.float32)
        if row_normalize:
            Vx = _row_normalize(Vx, xp)
        proba = clf.predict_proba(Vx)
        best = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        if use_gpu:
            best, conf = best.get(), conf.get()
            del Vx, proba
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
        return np.asarray(best), np.asarray(conf, dtype=np.float32)

    def _loop():
        bs, cs = [], []
        for s in range(0, n, batch):
            b, c = _one(V[s:s + batch])
            bs.append(b); cs.append(c)
        return np.concatenate(bs), np.concatenate(cs)

    if use_gpu:
        import cupy as cp
        with cp.cuda.Device(dev):
            best, conf = _loop()
    else:
        best, conf = _loop()
    return classes[best].astype(np.int32), conf


def map_gnb(analysis, *, centroids=None, gpu_device='cuda', force_cpu=False,
            chunk_h=128, vf_norm_threshold=None, row_normalize=True,
            var_smoothing=1e-9, fit_chunk=100_000, overwrite=True):
    """Map the count/blur field to celltypes with GaussianNB (the GNB counterpart of
    :func:`map_corr`).

    Fits on the labeled ``normalized_vectors`` then classifies every ``vf_normalized`` voxel
    with ``vf_norm > threshold``. The label raster and the top-class probability are written
    with the same layout as the correlation mapper (``celltype_maps`` + the score in
    ``max_correlations``), so ``filter_celltypemaps`` / ``plot_celltypes_map`` are unchanged.

    :param centroids: accepted for signature parity with :func:`map_corr`; unused (GNB fits
        its own per-class Gaussians from the labeled vectors).
    """
    from tqdm.auto import tqdm
    from ._count import _write_maps

    zg = analysis.dataset.zarr_group
    ds = analysis.dataset
    vf = zg['vf_normalized']
    n_G, n_R, n_C = vf.shape
    norm = np.asarray(zg['vf_norm'][:])
    if vf_norm_threshold is None:
        vf_norm_threshold = zg.attrs.get('select_vf_norm_threshold', None)
        if vf_norm_threshold is None:
            nz = norm[norm > 0]
            vf_norm_threshold = float(np.percentile(nz, 25)) if nz.size else 0.0
    vf_norm_threshold = float(vf_norm_threshold)
    sd = float(getattr(ds, 'sampling_distance', None)
               or zg.attrs.get('sampling_distance', zg.attrs.get('bin_size_um', 1.0)))

    clf, classes, backend = fit_gnb(zg, gpu_device=gpu_device, force_cpu=force_cpu,
                                    row_normalize=row_normalize, fit_chunk=fit_chunk,
                                    var_smoothing=var_smoothing)
    k = int(zg['centroids'].shape[0]) if 'centroids' in zg else len(classes)

    M = np.full((n_R, n_C), -1, dtype=np.int32)
    P = np.full((n_R, n_C), -1.0, dtype=np.float32)
    it = range(0, n_R, chunk_h)
    if analysis.verbose:
        it = tqdm(list(it), desc='map_celltypes (gnb)')
    for r0 in it:
        r1 = min(r0 + chunk_h, n_R)
        sig = norm[r0:r1] > vf_norm_threshold
        if not sig.any():
            continue
        h = r1 - r0
        block = np.asarray(vf[:, r0:r1, :]).reshape(n_G, h * n_C).T
        sig_flat = sig.reshape(-1)
        labels, conf = predict_gnb(clf, classes, block[sig_flat], backend,
                                   row_normalize=row_normalize)
        pos = np.where(sig_flat)[0]
        m_flat = M[r0:r1].reshape(-1); p_flat = P[r0:r1].reshape(-1)
        m_flat[pos] = labels
        p_flat[pos] = conf
        M[r0:r1] = m_flat.reshape(h, n_C)
        P[r0:r1] = p_flat.reshape(h, n_C)

    if analysis.verbose:
        print(f"[gnb] assigned {int((M >= 0).sum())} voxels (k={k}, thr={vf_norm_threshold:.4g})")
    return _write_maps(zg, ds, M, P, sd, overwrite=overwrite)
