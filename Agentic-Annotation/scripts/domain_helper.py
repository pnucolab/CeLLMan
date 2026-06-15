#!/usr/bin/env python
"""Domain-annotation helper (companion to /domain-mode skill).

Mirrors `_annotate_helper_v2.py` but the feature is *neighbor cell-type
composition* instead of gene expression, and the leaf label is a spatial
*domain* instead of a celltype.

Pipeline (round-driven, one mode per invocation):

    build      : h5ad(spatial + celltype) -> per-point neighbor celltype
                 composition -> leiden niches -> niche_profiles_round{k}.json
    subcluster : re-leiden the composition vectors *within* each flagged niche
                 -> sub-niche profiles (parent_ctx injected)
    finalize   : read cumulative _domain_decisions.json -> broadcast niche
                 label to every point -> optional spatial majority smoothing
                 -> write obs['domain'] (+ companions) into the h5ad, emit
                 domain_annotation.csv

Feature = neighbor composition:
  * radius graph (default 30 um), Gaussian distance weighting (sigma = r/2),
    self included.
  * one-hot celltype matrix H (N x C); composition = rownorm(W @ H).
  * Unknown / Background celltypes ARE kept as composition categories
    (a niche dominated by them is itself a signal -> Background domain).
  * Hellinger transform (sqrt) before PCA/leiden (compositional data).

GPU: rapids_singlecell leiden. Pin with --gpu (sets CUDA_VISIBLE_DEVICES).
"""
import argparse, json, os, sys, time
import numpy as np


# ----------------------------------------------------------------- io helpers
def _log(*a):
    print(f"[domain_helper {time.strftime('%H:%M:%S')}]", *a, flush=True)


def _state_path(out):
    return os.path.join(out, "_domain_state.json")


def _load_state(out):
    with open(_state_path(out)) as f:
        return json.load(f)


def _save_state(out, st):
    with open(_state_path(out), "w") as f:
        json.dump(st, f, indent=1)


# ----------------------------------------------------------- composition core
def build_composition(spatial, celltypes, radius, sigma=None):
    """Return (composition NxC float32, categories list, radius_graph csr).

    composition[i, c] = Gaussian-distance-weighted fraction of celltype c among
    the radius-r spatial neighbors of point i (self included).
    """
    from sklearn.neighbors import radius_neighbors_graph
    import scipy.sparse as sp

    if sigma is None:
        sigma = radius / 2.0
    cats = sorted(map(str, np.unique(celltypes)))
    cat_idx = {c: i for i, c in enumerate(cats)}
    codes = np.array([cat_idx[str(x)] for x in celltypes], dtype=np.int32)
    N, C = len(codes), len(cats)

    _log(f"radius graph: N={N} radius={radius} sigma={sigma:.2f} C={C}")
    G = radius_neighbors_graph(spatial, radius=radius, mode="distance",
                               include_self=False, n_jobs=-1)  # csr, distances
    G = G.tocoo()
    w = np.exp(-(G.data ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    W = sp.coo_matrix((w, (G.row, G.col)), shape=(N, N)).tocsr()
    W = W + sp.identity(N, format="csr", dtype=np.float32)  # self weight 1.0

    H = sp.coo_matrix((np.ones(N, np.float32), (np.arange(N), codes)),
                      shape=(N, C)).tocsr()
    comp = np.asarray((W @ H).todense(), dtype=np.float32)
    rs = comp.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    comp /= rs
    return comp, cats, W.tocsr()


def leiden_niches(comp_sub, resolution, gpu, n_pcs=None, seed=0):
    """Hellinger -> PCA -> neighbors -> leiden. Returns int label array."""
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import anndata as ad
    import rapids_singlecell as rsc

    X = np.sqrt(np.clip(comp_sub, 0, None)).astype(np.float32)  # Hellinger
    A = ad.AnnData(X)
    npc = n_pcs or min(50, X.shape[1] - 1)
    npc = max(2, min(npc, X.shape[1] - 1, X.shape[0] - 1))
    rsc.get.anndata_to_GPU(A)
    rsc.pp.pca(A, n_comps=npc)
    rsc.pp.neighbors(A, n_neighbors=min(30, A.n_obs - 1), use_rep="X_pca")
    rsc.tl.leiden(A, resolution=float(resolution), random_state=seed)
    rsc.get.anndata_to_CPU(A)
    return A.obs["leiden"].astype(int).to_numpy()


def spatial_coherence(W, paths):
    """Per-point fraction of spatial neighbors sharing the same leaf path."""
    Wb = W.copy()
    Wb.setdiag(0)
    Wb.eliminate_zeros()
    Wb.data[:] = 1.0
    paths = np.asarray(paths)
    coh = np.zeros(len(paths), np.float32)
    indptr, indices = Wb.indptr, Wb.indices
    for i in range(len(paths)):
        nb = indices[indptr[i]:indptr[i + 1]]
        if len(nb):
            coh[i] = np.mean(paths[nb] == paths[i])
    return coh


# --------------------------------------------------------------- profiling
def build_profiles(comp, cats, leaf_arr, leaves, coh, own_codes,
                   parent_ctx=None, top=15, eps=1e-4):
    """One profile per current leaf (niche), in the domain v1 schema."""
    cats = list(cats)
    g_mean = comp.mean(0)                       # global expected composition
    unk_idx = [i for i, c in enumerate(cats) if c in ("Unknown", "Background")]
    profiles = {}
    for lf in leaves:
        m = leaf_arr == lf
        n = int(m.sum())
        if n == 0:
            continue
        cin = comp[m].mean(0)
        log2fc = np.log2((cin + eps) / (g_mean + eps))
        order_enr = np.argsort(-log2fc)
        order_dom = np.argsort(-cin)
        order_dep = np.argsort(log2fc)

        enriched = [[cats[i], round(float(cin[i]), 4), round(float(g_mean[i]), 4),
                     round(float(log2fc[i]), 3)]
                    for i in order_enr if cin[i] >= 0.02][:top]
        dominant = [[cats[i], round(float(cin[i]), 4)]
                    for i in order_dom if cin[i] >= 0.01][:10]
        depleted = [[cats[i], round(float(cin[i]), 4), round(float(g_mean[i]), 4),
                     round(float(log2fc[i]), 3)]
                    for i in order_dep if (log2fc[i] <= -1.0 and g_mean[i] >= 0.01)][:10]

        p = cin / max(cin.sum(), eps)
        entropy = float(-(p * np.log2(p + eps)).sum())
        unk_frac = float(cin[unk_idx].sum()) if unk_idx else 0.0
        # fraction of niche points whose OWN celltype is Unknown/Background
        own_unk = float(np.isin(own_codes[m],
                                [cats.index(c) for c in ("Unknown", "Background")
                                 if c in cats]).mean()) if unk_idx else 0.0

        prof = {
            "n": n,
            "enriched": enriched,
            "dominant": dominant,
            "depleted": depleted,
            "bg": {"unknown_frac": round(unk_frac, 4),
                   "own_unknown_frac": round(own_unk, 4),
                   "entropy": round(entropy, 3)},
            "spatial": {"n_points": n,
                        "coherence": round(float(coh[m].mean()), 3)},
        }
        if parent_ctx and lf in parent_ctx:
            prof["parent_ctx"] = parent_ctx[lf]
        profiles[str(lf)] = prof
    return profiles


# ------------------------------------------------------------------ cmd build
def cmd_build(args):
    import anndata as ad
    os.makedirs(args.out, exist_ok=True)
    _log(f"reading {args.h5ad} (obs + spatial only)")
    A = ad.read_h5ad(args.h5ad, backed="r")
    spatial = np.asarray(A.obsm["spatial"], dtype=np.float64)
    ct = A.obs[args.celltype_col].astype(str).to_numpy()
    A.file.close()

    comp, cats, W = build_composition(spatial, ct, args.radius)
    cat_idx = {c: i for i, c in enumerate(cats)}
    own_codes = np.array([cat_idx[c] for c in ct], dtype=np.int32)

    _log(f"leiden niches @ resolution={args.resolution}")
    niche = leiden_niches(comp, args.resolution, args.gpu)
    paths = niche.astype(str)
    _log(f"{len(np.unique(paths))} niches")

    coh = spatial_coherence(W, paths)
    leaves = sorted(np.unique(paths), key=lambda s: (len(s), s))
    profiles = build_profiles(comp, cats, paths, leaves, coh, own_codes)

    np.save(os.path.join(args.out, "_domain_comp.npy"), comp)
    np.save(os.path.join(args.out, "_domain_paths.npy"), paths)
    np.save(os.path.join(args.out, "_domain_own.npy"), own_codes)
    import scipy.sparse as sp
    sp.save_npz(os.path.join(args.out, "_domain_radius_graph.npz"), W)
    _save_state(args.out, {
        "h5ad": os.path.abspath(args.h5ad),
        "celltype_col": args.celltype_col,
        "radius": args.radius, "round": 1,
        "cats": cats, "leaves": leaves,
        "resolution_history": [args.resolution],
    })
    outp = os.path.join(args.out, "niche_profiles_round1.json")
    with open(outp, "w") as f:
        json.dump(profiles, f, indent=1)
    _log(f"wrote {outp}  ({len(profiles)} niches)")
    _log("global composition (top 10):")
    g = comp.mean(0)
    for i in np.argsort(-g)[:10]:
        _log(f"   {cats[i]:42s} {g[i]:.3f}")


# -------------------------------------------------------------- cmd subcluster
def cmd_subcluster(args):
    st = _load_state(args.out)
    comp = np.load(os.path.join(args.out, "_domain_comp.npy"))
    paths = np.load(os.path.join(args.out, "_domain_paths.npy")).astype(object)
    own = np.load(os.path.join(args.out, "_domain_own.npy"))
    import scipy.sparse as sp
    W = sp.load_npz(os.path.join(args.out, "_domain_radius_graph.npz"))
    cats = st["cats"]
    parents = json.loads(args.niches)
    rmap = json.loads(args.resolution_map) if args.resolution_map else {}

    g_mean = comp.mean(0)
    eps = 1e-4
    parent_ctx, new_leaves = {}, []
    for par in parents:
        m = paths == par
        idx = np.where(m)[0]
        if len(idx) < args.min_points:
            _log(f"niche {par}: n={len(idx)} < min_points, keep as-is")
            continue
        res = float(rmap.get(par, args.resolution))
        sub = leiden_niches(comp[idx], res, args.gpu, seed=0)
        # Degenerate-split guard: a homogeneous niche (near-zero composition
        # variance) makes the kNN graph meaningless and leiden fragments into
        # ~singletons. Such a niche is genuinely unsplittable -> keep it unsplit
        # (its round-1 needs_subclustering decision falls through to Unknown at
        # finalize). Catches e.g. "near-pure generic glutamatergic" niche 56
        # which split into 711 one-point clusters.
        n_sub = len(np.unique(sub))
        mean_sz = len(idx) / max(n_sub, 1)
        if n_sub < 2 or mean_sz < 8:
            _log(f"niche {par}: degenerate/unsplittable (n_sub={n_sub}, "
                 f"mean_size={mean_sz:.1f}) -> keep unsplit (homogeneous)")
            continue
        # parent context = parent's top enriched/dominant celltypes
        cin = comp[idx].mean(0)
        l2 = np.log2((cin + eps) / (g_mean + eps))
        parent_ctx_entry = {
            "path": par,
            "enriched": [[cats[i], round(float(cin[i]), 4), round(float(l2[i]), 3)]
                         for i in np.argsort(-l2) if cin[i] >= 0.02][:8],
            "dominant": [[cats[i], round(float(cin[i]), 4)]
                         for i in np.argsort(-cin) if cin[i] >= 0.01][:8],
        }
        for s in np.unique(sub):
            child = f"{par}.{s}"
            paths[idx[sub == s]] = child
            new_leaves.append(child)
            parent_ctx[child] = parent_ctx_entry
        _log(f"niche {par} (n={len(idx)}) -> {len(np.unique(sub))} sub @ res={res}")

    coh = spatial_coherence(W, paths)
    profiles = build_profiles(comp, cats, paths, new_leaves, coh, own,
                              parent_ctx=parent_ctx)
    np.save(os.path.join(args.out, "_domain_paths.npy"), paths.astype(str))
    st["round"] = int(st.get("round", 1)) + 1
    st["leaves"] = sorted(set(map(str, np.unique(paths))), key=lambda s: (len(s), s))
    st.setdefault("resolution_history", []).append(args.resolution)
    _save_state(args.out, st)
    k = st["round"]
    outp = os.path.join(args.out, f"niche_profiles_round{k}.json")
    with open(outp, "w") as f:
        json.dump(profiles, f, indent=1)
    _log(f"wrote {outp}  ({len(profiles)} new sub-niches)")


# ---------------------------------------------------------------- cmd finalize
def _smooth(paths_lbl, W, passes, frac):
    """Spatial majority vote. Flip a point only if a single *other* label holds
    >= frac of its spatial neighbors. Conservative: protects small real regions."""
    import scipy.sparse as sp
    Wb = W.copy(); Wb.setdiag(0); Wb.eliminate_zeros(); Wb.data[:] = 1.0
    indptr, indices = Wb.indptr, Wb.indices
    lbl = np.asarray(paths_lbl, dtype=object).copy()
    for _ in range(passes):
        new = lbl.copy()
        for i in range(len(lbl)):
            nb = indices[indptr[i]:indptr[i + 1]]
            if len(nb) < 4:
                continue
            vals, cnts = np.unique(lbl[nb], return_counts=True)
            j = cnts.argmax()
            if vals[j] != lbl[i] and cnts[j] / len(nb) >= frac:
                new[i] = vals[j]
        lbl = new
    return lbl


def cmd_finalize(args):
    import anndata as ad
    st = _load_state(args.out)
    paths = np.load(os.path.join(args.out, "_domain_paths.npy")).astype(object)
    import scipy.sparse as sp
    W = sp.load_npz(os.path.join(args.out, "_domain_radius_graph.npz"))
    with open(os.path.join(args.out, "_domain_decisions.json")) as f:
        dec = json.load(f)

    # any leaf still flagged needs_subclustering at finalize -> unresolved/Unknown
    label = np.empty(len(paths), object)
    decision = np.empty(len(paths), object)
    conf = np.empty(len(paths), object)
    keyct = np.empty(len(paths), object)
    miss = set()
    for lf in np.unique(paths):
        d = dec.get(str(lf))
        if d is None:
            miss.add(str(lf)); d = {}
        de = d.get("decision", "Unknown")
        lb = d.get("label", "Unknown")
        if de in ("needs_subclustering", "unresolved_subclustering"):
            de, lb = "unresolved_subclustering", "Unknown"
        m = paths == lf
        label[m] = lb
        decision[m] = de
        conf[m] = d.get("confidence", "")
        kc = d.get("key_celltypes", "")
        keyct[m] = ",".join(kc) if isinstance(kc, list) else str(kc)
    if miss:
        _log(f"WARNING: {len(miss)} leaves had no decision -> Unknown: {sorted(miss)[:10]}")

    if args.smooth_passes > 0:
        _log(f"spatial smoothing: {args.smooth_passes} pass(es), frac={args.smooth_frac}")
        before = label.copy()
        label = _smooth(label, W, args.smooth_passes, args.smooth_frac)
        _log(f"  smoothing changed {(before != label).sum()} / {len(label)} points")

    _log(f"loading full h5ad to write obs: {st['h5ad']}")
    A = ad.read_h5ad(st["h5ad"])
    A.obs["domain"] = list(label)
    A.obs["domain"] = A.obs["domain"].astype("category")
    A.obs["domain_decision"] = list(decision)
    A.obs["domain_confidence"] = list(conf)
    A.obs["domain_key_celltypes"] = list(keyct)
    A.obs["niche_path"] = list(map(str, paths))
    A.write(st["h5ad"])
    _log(f"wrote obs['domain'] into {st['h5ad']}")

    # csv summaries
    import pandas as pd
    rows = []
    for lf in sorted(np.unique(paths), key=lambda s: (len(str(s)), str(s))):
        d = dec.get(str(lf), {})
        m = paths == lf
        rows.append({"niche_path": lf, "n": int(m.sum()),
                     "decision": d.get("decision", "Unknown"),
                     "domain": d.get("label", "Unknown"),
                     "confidence": d.get("confidence", ""),
                     "key_celltypes": ",".join(d.get("key_celltypes", []))
                                      if isinstance(d.get("key_celltypes"), list)
                                      else d.get("key_celltypes", ""),
                     "rationale": d.get("rationale", "")})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "domain_annotation.csv"), index=False)
    _log("wrote domain_annotation.csv")
    vc = pd.Series(label).value_counts()
    _log("domain distribution:")
    for k, v in vc.items():
        _log(f"   {str(k):40s} {v}")


# ---------------------------------------------------------------- cmd refine
def cmd_refine(args):
    """Build a per-domain ADJACENCY profile from the current obs['domain'].
    For each domain: size, centroid, spatial coherence, mean neighbor celltype
    composition, and the distribution of bordering domains. The agent uses this
    (esp. `neighbor_domains`) to assign clean anatomical names, correct
    composition-inherited mislabels via spatial context (e.g. a pyramidal domain
    ringed by CA1 + Dentate gyrus is CA3, not Basolateral amygdala), tag
    domain_type, and merge duplicate labels."""
    import anndata as ad
    import scipy.sparse as sp
    from collections import Counter
    st = _load_state(args.out)
    A = ad.read_h5ad(st["h5ad"], backed="r")
    dom = A.obs["domain"].astype(str).to_numpy()
    xy = np.asarray(A.obsm["spatial"])
    A.file.close()
    comp = np.load(os.path.join(args.out, "_domain_comp.npy"))
    cats = st["cats"]
    W = sp.load_npz(os.path.join(args.out, "_domain_radius_graph.npz")).tocsr()
    W.setdiag(0); W.eliminate_zeros(); W.data[:] = 1.0

    # Vectorized adjacency via one-hot domain matrix: B = W @ Hd gives, per point,
    # the count of spatial neighbors in each domain. Per-point degree = row sum.
    labels = sorted(set(dom))
    lab_idx = {d: i for i, d in enumerate(labels)}
    codes = np.array([lab_idx[d] for d in dom], dtype=np.int32)
    K = len(labels)
    Hd = sp.coo_matrix((np.ones(len(codes), np.float32),
                        (np.arange(len(codes)), codes)), shape=(len(codes), K)).tocsr()
    B = np.asarray((W @ Hd).todense(), dtype=np.float64)   # N x K neighbor counts
    deg = B.sum(1)
    same = B[np.arange(len(codes)), codes]
    coh_pt = np.divide(same, deg, out=np.zeros_like(same), where=deg > 0)

    prof = {}
    for d in labels:
        idx = np.where(dom == d)[0]
        di = lab_idx[d]
        nb = B[idx].sum(0).copy()        # total neighbor counts per domain
        nb[di] = 0                       # drop self-domain (interior edges)
        tot = nb.sum() or 1.0
        order = np.argsort(-nb)
        neighbor_domains = [[labels[j], round(float(nb[j] / tot), 3)]
                            for j in order if nb[j] > 0][:6]
        cin = comp[idx].mean(0)
        prof[d] = {
            "n": int(len(idx)),
            "centroid": [round(float(xy[idx, 0].mean())), round(float(xy[idx, 1].mean()))],
            "coherence": round(float(coh_pt[idx].mean()), 3),
            "composition_top": [[cats[i], round(float(cin[i]), 3)]
                                for i in np.argsort(-cin)[:8] if cin[i] >= 0.02],
            "neighbor_domains": neighbor_domains,
        }
    outp = os.path.join(args.out, "domain_adjacency_profile.json")
    with open(outp, "w") as f:
        json.dump(prof, f, indent=1)
    _log(f"wrote {outp} ({len(prof)} domains)")


def cmd_refine_apply(args):
    """Apply the agent's relabel map (_domain_refine.json: {old_label:
    {label, domain_type}}) to obs['domain']; add obs['domain_type']; preserve
    the pre-refine labels in obs['domain_raw']."""
    import anndata as ad
    import pandas as pd
    st = _load_state(args.out)
    with open(os.path.join(args.out, "_domain_refine.json")) as f:
        rmap = json.load(f)
    A = ad.read_h5ad(st["h5ad"])
    old = A.obs["domain"].astype(str).to_numpy()
    new = np.array([rmap.get(d, {}).get("label", d) for d in old], object)
    dtype = np.array([rmap.get(d, {}).get("domain_type", "") for d in old], object)
    if "domain_raw" not in A.obs:
        A.obs["domain_raw"] = list(old)
    A.obs["domain"] = list(new)
    A.obs["domain"] = A.obs["domain"].astype("category")
    A.obs["domain_type"] = list(dtype)
    A.obs["domain_type"] = A.obs["domain_type"].astype("category")
    A.write(st["h5ad"])
    n_changed = int((old != new).sum())
    _log(f"refine applied: {n_changed} points relabeled; "
         f"{len(set(old))} -> {len(set(new))} domains")
    df = pd.DataFrame({"domain": new, "domain_type": dtype})
    g = df.value_counts().reset_index(name="n").sort_values("n", ascending=False)
    g.to_csv(os.path.join(args.out, "domain_annotation_refined.csv"), index=False)
    for _, r in g.iterrows():
        _log(f"   {str(r['domain']):42s} [{r['domain_type']:22s}] {r['n']}")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build")
    pb.add_argument("--h5ad", required=True)
    pb.add_argument("--out", required=True)
    pb.add_argument("--celltype-col", default="celltype_annotation")
    pb.add_argument("--radius", type=float, default=30.0)
    pb.add_argument("--resolution", type=float, required=True)
    pb.add_argument("--gpu", default="0")
    pb.set_defaults(func=cmd_build)

    ps = sub.add_parser("subcluster")
    ps.add_argument("--out", required=True)
    ps.add_argument("--niches", required=True, help="JSON list of niche paths")
    ps.add_argument("--resolution", type=float, required=True)
    ps.add_argument("--resolution-map", default=None, help="JSON {niche: res}")
    ps.add_argument("--min-points", type=int, default=50)
    ps.add_argument("--gpu", default="0")
    ps.set_defaults(func=cmd_subcluster)

    pf = sub.add_parser("finalize")
    pf.add_argument("--out", required=True)
    pf.add_argument("--smooth-passes", type=int, default=1)
    pf.add_argument("--smooth-radius", type=float, default=None)
    pf.add_argument("--smooth-frac", type=float, default=0.5)
    pf.set_defaults(func=cmd_finalize)

    pr = sub.add_parser("refine")
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=cmd_refine)

    pa = sub.add_parser("refine-apply")
    pa.add_argument("--out", required=True)
    pa.set_defaults(func=cmd_refine_apply)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
