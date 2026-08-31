#!/usr/bin/env python3
# NOTE: helper.py is a STANDALONE build — the helper.py engine is INLINED below,
#       plus the annotate-mode marker-axis split (subcluster/descend). Do NOT edit
#       helper.py expecting changes here; regenerate via scratchpad/build_helper.py.
"""CeLLMan annotate helper — in-memory recursive subclustering + ontology-aware profiling.

Design notes:
  - the cluster column is a parameter (--cluster-col; no hardcoded `leiden`)
  - ontology = **slim-index fetched from the server** (once, at init) + `--ontology` local fallback
  - subcluster backend is chosen **automatically**: GPU + rapids_singlecell -> rapids, else scanpy
  - profile `bg` -> **identity_evidence** (vf_norm density + identity signal); up/dn unbounded, hp=50
  - output = **{input}_annotated.h5ad** (full copy of the input plus only obs['cellman_annotation']
    (standard) / obs['cellman_free_annotation'] (user interaction))

Subcommands:
  init       --h5ad --cluster-col --out [--ontology-url | --ontology] [--use-rep <obsm key>|recompute]
  subcluster --out --paths '["4","11"]' --resolution 0.2 [--gpu 0] [--ontology ...]
  finalize   --out --decisions <json> [--output <path>]
"""
import argparse, json, os, sys, re, datetime
import numpy as np
import anndata as ad
from scipy.stats import rankdata
from scipy.sparse import csr_matrix

# ===================================================================== ontology
# The single Cell-Ontology server we publish (public, read-only). Users normally need to pass nothing
# and are connected here automatically.
# Precedence: --server-url flag > CELLONTO_SERVER_URL env var > this default.
# If you run your own server, point CELLONTO_SERVER_URL at it.
DEFAULT_ONTOLOGY_SERVER = "https://cellman.pnucolab.com"
def _resolve_server(server_url=None):
    return (server_url or os.environ.get("CELLONTO_SERVER_URL") or DEFAULT_ONTOLOGY_SERVER).rstrip("/")

def _dedup_parents(idx):
    """Guard against a KG build defect: a term's P (parent list) can contain the same
    parent twice (observed: 28/373 terms in mouse_whole_brain had P=['X','X']). The children index and
    subtype_groups iterate `for p in P` without dedup, so the same child is counted twice -> a term with a
    single child yields a bogus '>=2 siblings' subtype_group, and children_resolvable picks up duplicates
    too (a false split signal). Order-preserving dedup at every point where the index is built fixes all
    consumers at once (subtype_groups, _candidate_subtypes, _graph_evidence, _build_ident_maps)."""
    if idx and isinstance(idx.get("terms"), dict):
        for t in idx["terms"].values():
            P = t.get("P")
            if P and len(P) != len(dict.fromkeys(P)):
                t["P"] = list(dict.fromkeys(P))
    return idx

def _served_organs(url, species):
    """The organs the server actually serves for `species` — used to make a 404 actionable.
    Best-effort: any failure returns [] and the caller just omits the hint."""
    try:
        import urllib.request
        base = url.split("/module/")[0]
        with urllib.request.urlopen(f"{base}/modules", timeout=20) as r:
            wings = json.loads(r.read().decode()).get("wings", [])
        return sorted(w["organ"] for w in wings if w.get("species") == species)
    except Exception:
        return []


def _merge_slim_indexes(idxs, organs):
    """Union of several wings' slim-indexes into ONE index (multi-module serving).

    Requested by the user when a tissue legitimately spans wings (e.g. 'brain' + 'spinal_cord').
    Term ids are globally unique and wing-prefixed (HUMAN_BRAIN:… / HUMAN_COMMON:…), so the union is
    by id: organ-specific terms never collide, and the common-lineage terms injected into every wing
    collide on the SAME id — first wing wins (the requested organ order), which keeps that wing's
    REFINES-merged deltas. Axis payloads (cp/ca/sa/oq) are concatenated and de-duplicated."""
    species = {i.get("species") for i in idxs if i.get("species")}
    if len(species) > 1:
        sys.exit(f"[init][ERROR:MODULE_SPECIES_MISMATCH] the merged modules disagree on species ({sorted(species)}). "
                 f"All --organ entries must belong to ONE species.")
    merged = {"species": next(iter(species), None), "organ": "+".join(organs), "terms": {}}
    dup = 0
    for idx in idxs:
        for tid, t in (idx.get("terms") or {}).items():
            if tid in merged["terms"]:
                dup += 1
                continue
            merged["terms"][tid] = t
    for key in ("cp", "oq", "ca", "sa"):
        seen, out_l = set(), []
        for idx in idxs:
            for item in (idx.get(key) or []):
                k = json.dumps(item, sort_keys=True)
                if k in seen:
                    continue
                seen.add(k); out_l.append(item)
        merged[key] = out_l
    merged["n_terms"] = len(merged["terms"])
    merged["version"] = max((i.get("version") or "") for i in idxs) or None
    print(f"[ontology] merged {len(idxs)} modules ({', '.join(organs)}) -> {merged['n_terms']} terms "
          f"({dup} shared common-lineage terms de-duplicated)", flush=True)
    return merged


def _fetch_slim_index(out, ontology_url=None, ontology_file=None):
    """Obtain the slim-index and store it at {out}/_ontology_index.json.
    Precedence: server fetch (ontology_url) > local file (ontology_file) > a file that already exists.
    `ontology_url` may be a LIST of URLs — one per module — in which case the wings are merged
    into one index (multi-module serving; the user asked for several modules at once)."""
    dst = os.path.join(out, "_ontology_index.json")
    if ontology_url:
        import urllib.request, urllib.error
        urls = [ontology_url] if isinstance(ontology_url, str) else list(ontology_url)

        def _get(url):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    parts = url.split("/module/")[-1].split("/")
                    sp = parts[0] if parts else ""
                    served = _served_organs(url, sp)
                    hint = (f" The server serves these {sp} modules: {', '.join(served)}." if served else "")
                    sys.exit(f"[init][ERROR:MODULE_NOT_FOUND] the server has no such module ({url}, HTTP 404).{hint} "
                             f"Check (species, organ) or request a build of that module. -> the caller should tell the user "
                             f"'the ontology for the requested tissue (species/organ) is not on the server, so annotation cannot proceed' "
                             f"and, when a close module exists, ASK which one to use instead.")
                sys.exit(f"[init][ERROR:SERVER_HTTP_{e.code}] the ontology server returned error {e.code} ({url}).")
            except urllib.error.URLError as e:
                sys.exit(f"[init][ERROR:SERVER_UNREACHABLE] cannot reach the ontology server ({url}; {e.reason}). "
                         f"Check that the server is running and verify CELLONTO_SERVER_URL. -> the caller should tell the user "
                         f"'the ontology server is unreachable (server down or wrong address)'.")

        idxs = []
        for url in urls:
            idx = _get(url)
            if not idx or not idx.get("terms"):
                sys.exit(f"[init][ERROR:MODULE_EMPTY] the module responded but contains no terms ({url}). "
                         f"That module needs to be built. -> the caller should tell the user 'the ontology for this tissue is empty, so we cannot proceed'.")
            idxs.append(idx)
        idx = (idxs[0] if len(idxs) == 1
               else _merge_slim_indexes(idxs, [i.get("organ") or "?" for i in idxs]))
        idx = _dedup_parents(idx)                       # dedup duplicate parents in P (prevents bogus sibling groups)
        with open(dst, "w") as f: json.dump(idx, f, separators=(",", ":"))
        print(f"[ontology] fetched slim-index from server -> {dst} ({len(idx.get('terms',{}))} terms)", flush=True)
        return idx
    if ontology_file and os.path.exists(ontology_file):
        raw = json.load(open(ontology_file))
        if "terms" in raw and isinstance(raw["terms"], dict):
            idx = raw
        else:
            idx = {"terms": {}, "cp": [], "oq": []}
            for t in raw.get("terms", []):
                m = t.get("markers", {})
                idx["terms"][t["id"]] = {"L": t.get("label",""), "T": t.get("tier",""),
                    "P": t.get("is_a", []), "R": list(m.get("required", [])),
                    "S": list(m.get("supporting", []))[:6], "N": list(m.get("negative", []))[:4],
                    "D": t.get("disease_context","normal"),
                    "A": list(t.get("anchor", []) or []),
                    "MP": list(t.get("mimics", []) or [])}
        idx = _dedup_parents(idx)                       # dedup duplicate parents in P (prevents bogus sibling groups)
        with open(dst, "w") as f: json.dump(idx, f, separators=(",", ":"))
        print(f"[ontology] loaded slim-index from {ontology_file} -> {dst}", flush=True)
        return idx
    if os.path.exists(dst):
        return _dedup_parents(json.load(open(dst)))
    return None

def _load_ontology_index(out):
    p = os.path.join(out, "_ontology_index.json")
    return _dedup_parents(json.load(open(p))) if os.path.exists(p) else None

def _ontology_gene_set(idx):
    s = set()
    for t in (idx or {}).get("terms", {}).values():
        for k in ("R","S","N","DF"): s.update(t.get(k, []) or [])
    return s

def _ontology_protected(idx):
    s = set()
    for t in (idx or {}).get("terms", {}).values():
        for k in ("R","S","DF"): s.update(t.get(k, []) or [])
    return s

def _ontology_hits(genes, idx):
    panel = {g.upper() for g in genes}
    def split(ms):
        on=[m for m in (ms or []) if str(m).upper() in panel]
        off=[m for m in (ms or []) if str(m).upper() not in panel]
        return on, off
    out={}
    for tid,t in (idx or {}).get("terms",{}).items():
        R_on,R_off=split(t.get("R")); S_on,S_off=split(t.get("S")); N_on,N_off=split(t.get("N"))
        out[tid]={"label":t.get("L",""),"tier":t.get("T",""),"R_on":R_on,"R_off":R_off,
                  "S_on":S_on,"S_off":S_off,"N_on":N_on,"N_off":N_off}
    return out

# ---------- gene-panel denylist + KG inclusion (mirror cellman.spatial._panel.select_gene_panel) ----------
# The HVG set feeding PCA (1) drops technical genes and (2) puts back markers that are in the KG and in
# the data but that HVG selection missed.
_PANEL_DENY_MOUSE=("mt-","Rps","Rpl","Hba","Hbb")
_PANEL_DENY_HUMAN=("MT-","RPS","RPL","MRPL","MRPS","HBA","HBB")
_PANEL_DENY_HUMAN_RE=re.compile(r"^(ENSG|LINC|MIR|SNOR)|(-AS\d*$)|(-DT$)|(-IT\d*$)|(-OT\d*$)"
                                r"|(^RP\d+-)|(^A[CLP]\d{6})|(^Z\d{5})|(\.\d+$)")
def _panel_denier(species):
    """technical/low-info gene predicate (g->bool). Defaults to mouse when species is unset."""
    sp=(species or "").lower()
    if sp.startswith("human") or sp in ("hs","homo_sapiens"):
        return lambda g: g.startswith(_PANEL_DENY_HUMAN) or g=="MALAT1" or bool(_PANEL_DENY_HUMAN_RE.search(g))
    return lambda g: g.startswith(_PANEL_DENY_MOUSE) or bool(re.match(r"Gm\d",g)) or g.endswith("Rik")

def _panel_clean(g):
    """Normalize a marker symbol to match var_names (strip -/_ suffixes) — same as _panel._clean."""
    return str(g).split("-")[0].split("_")[0]


# ===================================================================== housekeeping
import re as _re
_HK_REGEX=[_re.compile(p,_re.I) for p in (r'^MT-',r'^MT[A-Z]{2,}\d',r'^MT[A-Z]+\d+P\d+$',
    r'^EEF\d',r'^EIF\d',r'^RPL\d',r'^RPS\d',r'^RPLP\d',r'^TUBA\d',r'^TUBB\d',
    r'^HSP90A[AB]\d?$',r'^HSPA[1-9]',r'^HSPB[1-9]')]
_HK_EXACT={g.upper() for g in ('ACTB','ACTG1','B2M','GAPDH','PKM','TPI1','MALAT1','NEAT1','XIST',
    'KCNQ1OT1','UBC','UBB','HSP90AA1','HSP90AB1','HSPA8','HSPA1A')}
def _is_hk(g):
    return g.upper() in _HK_EXACT or any(p.match(g) for p in _HK_REGEX)
def _data_driven_hk(cmeans, max_auc, genes, mean_pct=75, max_cv=0.2, max_auc_t=0.65):
    gm=cmeans.mean(axis=0); cv=cmeans.std(axis=0)/(gm+1e-8)
    thr=np.percentile(gm, mean_pct)
    mask=(gm>=thr)&(cv<max_cv)&(max_auc<max_auc_t)
    return set(genes[i] for i in np.where(mask)[0])

# ===================================================================== ontology-derived domain context
# lineage / specific / pan / sibling are derived from that module's ontology (slim-index), never from
# hardcoded per-organ tables. The cross-organ fallback below is used only when derivation is impossible
# (a hierarchy or marker set too sparse). This generalizes to any organ.
_BG_MAX_N=500; MIN_N_OUT=20
def _U(s): return str(s).upper()

_LINEAGE_FALLBACK={
    "oligodendrocyte":{"MBP","PLP1","MOG","MOBP","MAL","MAG","CNP","CLDN11"},
    "astrocyte":{"SLC1A2","AQP4","GFAP","SLC1A3","MLC1"},
    "neuron":{"SLC17A7","SLC17A6","SLC17A8","GAD1","GAD2","SLC32A1","SNAP25","SYT1","RBFOX3"},
    "microglia/myeloid":{"CX3CR1","C1QA","C1QB","C1QC","CSF1R","P2RY12","CD68","LYZ","ITGAM"},
    "endothelial":{"CLDN5","PECAM1","FLT1","VWF","CDH5"},
    "mural":{"PDGFRB","RGS5","ACTA2","NOTCH3"},
    "fibroblast":{"COL1A1","COL1A2","DCN","LUM","PDGFRA"},
    "epithelial":{"EPCAM","KRT8","KRT18","CDH1"},
    "T cell":{"CD3D","CD3E","CD2","TRAC"}, "B cell":{"CD79A","MS4A1","CD19"}}
_SPECIFIC_FALLBACK={"PMCH","HCRT","AVP","OXT","TTR","TH","TPH2","CHAT","PVALB","SST","VIP","LAMP5",
    "INS","GCG","SST","PPY","ALB","MYH6","TNNT2","SFTPC","MUC2"}
_PAN_FALLBACK={"SLC17A7","SLC17A6","SLC17A8","GAD1","GAD2","SLC32A1","SNAP25","SYT1"}

def _derive_lineages(idx):
    """lineage = a root node of the hierarchy (no parent). lineage markers = only that root node's **own**
    required set (canonical pan-LINEAGE transmitters: glut->Slc17a7/6/8, GABA->Gad1/2 & Slc32a1,
    DA->Th/Ddc/Slc6a3 ...).
    ★ Do NOT sweep up root + descendants: sweeping mixes pan-NEURONAL (Camk2a) and subtype
    markers attached to descendants into the lineage set, so it points at 'neuron' rather than at 'which
    lineage' — that is why cluster 38 was called glut via Camk2a despite Slc17a7=0. Narrowing to the
    root's own R leaves only transmitters and keeps the lineage call clean.
    ★ An R-EMPTY root is SKIPPED, and must stay skipped — do NOT fall back to its DF. An empty R is curated,
    not missing: a broad class has no marker its children all share (mouse/brain `glial cell` R=[] because
    astrocytes lack Sox10/Plp1 and oligodendrocytes lack Gfap/Aqp4), so nothing is *required*. Its DF
    ([S100b,Gfap,Sox10,Plp1]) is a union, and every one of those is pan (R/S degree>=4 = octx.pan_disc).
    Lineage strength scores by on_present (ABSOLUTE presence, not enrichment — see the on_present comment
    below), and on KDE/spatial data ambient myelin/Gfap is present in ~every voxel, so registering that DF set
    makes `glial cell` win the lineage in CORTICAL leaves too; cls_root becomes glial, the anchor loop drops
    every term outside it, and the neocortical layer subtypes (L4 IT = Rspo1/Rorb) become unreachable. Measured:
    the DF fallback silently removed L4 from a stereo-seq cortex run. The cost of skipping is real but smaller —
    the glial branch is reached via restrict_root/descend instead — and the fix belongs in the KG (give the
    broad roots a genuinely shared R, or make the transmitter classes roots), not here.
    Returns {} when there is no hierarchy (-> fallback). {lineage_label: {MARKER_UPPER}}."""
    terms=(idx or {}).get("terms",{})
    if not terms or not any(t.get("P") for t in terms.values()): return {}
    def roots_of(tid):
        stack=[tid]; seen=set(); roots=set()
        while stack:
            x=stack.pop()
            if x in seen: continue
            seen.add(x); t=terms.get(x)
            ps=[p for p in (t.get("P") or []) if p in terms and p!=x] if t else []
            (roots.add(x) if not ps else stack.extend(ps))
        return roots or {tid}
    roots=set()
    for tid in terms: roots|=roots_of(tid)
    lin={}
    for r in roots:
        rt=terms.get(r,{}); mk={m.upper() for m in (rt.get("R") or [])}
        if mk: lin[rt.get("L",r)]=mk
    return lin if len(lin)>=2 else {}

def _derive_specific(idx):
    """Markers that are required by <=2 terms (= highly specific identifiers) union DF (defining).
    Returns {} when too sparse."""
    terms=(idx or {}).get("terms",{}); cnt={}; spec=set()
    for t in terms.values():
        for m in (t.get("R") or []): cnt[m.upper()]=cnt.get(m.upper(),0)+1
        spec.update(m.upper() for m in (t.get("DF") or []))
    spec.update(m for m,c in cnt.items() if c<=2)
    return spec

def _derive_pan(idx, min_terms=4):
    """R/S markers shared by many terms (= non-discriminative pan/class markers). Used for neg-leak
    exemptions and for excluding them from lineage_present."""
    terms=(idx or {}).get("terms",{}); cnt={}
    for t in terms.values():
        for k in ("R","S"):
            for m in (t.get(k) or []): cnt[m.upper()]=cnt.get(m.upper(),0)+1
    return {m for m,c in cnt.items() if c>=min_terms}

def _sibling_groups(idx):
    """Groups of >=2 children that share a parent (= sibling subclass groups). {parent_id:[child_id...]}
    plus each child's required set."""
    terms=(idx or {}).get("terms",{}); bp={}
    for tid,t in terms.items():
        for p in (t.get("P") or []):
            if p in terms: bp.setdefault(p,[]).append(tid)
    return {p:ch for p,ch in bp.items() if len(ch)>=2}

# ★ PROCESS tier — a tissue region filled not with cell bodies but with **the parts that extend out of
#   cells** (processes). Definition: "a region that has mRNA but is not a nucleus/soma". In the nervous
#   system these are neuropil (the mesh of dendrites, axons and glial processes) and fiber tract (bundles
#   of myelinated axons); the concept **extends to non-neural tissue** as well — the retinal plexiform
#   layer, kidney podocyte foot processes, the thin extensions of lung type I pneumocytes.
#   Processes are NOT labelled by the clustering-based path: a process **shares** its positive markers
#   with the soma that made the transcripts, so a marker axis cannot separate it from that soma — only a
#   **negative axis** (nucleus-retained transcripts such as MALAT1) can. Process detection is therefore
#   not part of this release.
#   _PROC_TIERS is still needed: tier='process' terms exist in the KG, so _onto_ctx must exclude them from
#   the celltype engine (lineage/anchor). Without that exclusion neuropil is picked up as a lineage root
#   and a real oligodendrocyte is called 'fiber tract'.
#   'compartment' is still accepted as a legacy tier.
_PROC_TIERS={"process","compartment"}


def _onto_ctx(idx):
    """Domain context used by profiling/validate (derived from the ontology, with fallbacks)."""
    terms_all=(idx or {}).get("terms",{})
    # ★ PROCESS isolation: tier='process' terms (neuropil, fiber tract, ...) are not celltypes, so they are
    #   excluded from lineage/specific/pan/sibling derivation and from IDM (anchor/top) — the same intent as
    #   keeping state terms out of top. Without the exclusion neuropil / fiber tract is taken as root=lineage
    #   and a genuine oligodendrocyte (MBP+) is called the 'fiber tract' lineage. This is required because the
    #   process module remains in the KG.
    #   Process markers stay in _ontology_gene_set (the full idx) so they remain measurable. A module with no
    #   process term passes through unchanged -> no effect at all.
    _has_proc=any((t.get("T") or "") in _PROC_TIERS for t in terms_all.values())
    idx_ct=({**(idx or {}), "terms":{tid:t for tid,t in terms_all.items() if (t.get("T") or "") not in _PROC_TIERS}}
            if _has_proc else idx)
    terms=(idx_ct or {}).get("terms",{})
    return {"lineages": _derive_lineages(idx_ct) or {k:set(v) for k,v in _LINEAGE_FALLBACK.items()},
            "specific": _derive_specific(idx_ct) or set(_SPECIFIC_FALLBACK),
            "pan": _derive_pan(idx_ct) or set(_PAN_FALLBACK),
            # ★ pan_disc: the pan set used to judge discriminative markers for candidate_subtypes /
            #   children_resolvable. octx["pan"] (>=4 terms) also kills *layer-discriminating* markers such as
            #   Cux2 (4) / Rorb (5) / Fezf2 (5), so cortical layers disappear from children_resolvable and
            #   n_nonpan_present (SHOULD_SPLIT_DEEPER under-fires). This uses the **same >=12 threshold** that
            #   _build_ident_maps uses for graph_evidence — only true pan markers (Slc17a7=52 etc.) are filtered and
            #   layer markers survive as discriminators. octx["pan"] (>=4) is left alone because _validate /
            #   lineage_present depend on it.
            "pan_disc": _derive_pan(idx_ct, 12) or set(_PAN_FALLBACK),
            "siblings": _sibling_groups(idx_ct),
            "terms": terms,
            "cp": (idx or {}).get("cp") or [],   # confusion pairs from slim-index → surfaced by _graph_evidence
            "ca": (idx or {}).get("ca") or [],   # context axes (zonation/layer) → surfaced by _context_axis_eval (§4.3)
            "sa": (idx or {}).get("sa") or [],   # state axes (base-agnostic cell states) → surfaced by _state_axis_eval
            "term_req": {tid:{m.upper() for m in (t.get("R") or [])} for tid,t in terms.items()}}

def _context_axis_eval(lineage_label, ca, fon, lf, pin, gidx_U, lean_ratio=1.15, sig_floor=0.03, extra_labels=None):
    """★ Context-axis subtyping (§4.3) — DETERMINISTIC engine support. When a lineage's subtypes are not
    separated by cell-intrinsic markers but by *spatial position* (zonation/layer), score the per-pole
    context_markers of the ca axis the module declares (= markers of the dominant neighbouring cell type;
    ambient in segmented data) and propose dominant pole -> subtype. The lineage is already fixed by
    graph_evidence from intrinsic markers (never changed here) — poles apply only to the subtype axis.
    Off-panel context markers are skipped.
    ★ WEAK-LEAN: zonation is a continuum, so the signal is weak — demanding a strong lead
    would drop most leaves to the bare class. Therefore (a) the score is graded (a sum of fon with no hard
    ON threshold, so faint ambient still counts), and (b) if one pole leads clearly (>=lean_ratio x) even
    while *weak*, that pole is assigned (pole_confidence=low; >=1.5x -> medium). If both are genuinely
    present and near-tied, the result is mixed (split when this is a parent). Because the axis is a bounded
    same-lineage axis, the worst case is an adjacent zone (not a fatal misassignment)."""
    if not ca: return None
    # ★ region/anchor-aware matching: in liver, lineage = celltype ("Hepatocyte") so the lineage label is
    #   enough; in brain, lineage = a transmitter class ("glutamatergic neuron") and is far too broad. So
    #   applies_to is also matched against the dominant_region / anchor labels (extra_labels), applying the
    #   axis only to the right region/celltype (thalamic core/matrix, CA1 deep/superficial).
    cand={str(lineage_label).strip().lower()} if lineage_label else set()
    for x in (extra_labels or []):
        if x: cand.add(str(x).strip().lower())
    if not cand: return None
    def lvl(g):
        i=gidx_U.get(str(g).upper())
        if i is None: return None
        return float(max(fon[i],0.0))
    out=[]
    for ax in ca:
        # ★ lineage gate: when an axis declares a lineage, it applies only if the cluster's transmitter lineage
        #   matches. This blocks a glut-pyramidal axis such as CA1 radial from firing on a GABAergic interneuron
        #   cluster (matched via anchor/path because its parent is CA1) and mislabelling it "CA1 deep pyramidal".
        #   An axis with no declared lineage has no gate (liver compatible).
        ax_lin=ax.get("lineage")
        if ax_lin and lineage_label and str(ax_lin).strip().lower()!=str(lineage_label).strip().lower(): continue
        appl=[str(x).strip().lower() for x in (ax.get("applies_to") or [])]
        matched=[a for a in appl if a in cand]
        if not matched: continue
        mkey=matched[0]
        poles=[]
        for p in (ax.get("poles") or []):
            on_g=[]; score=0.0; n_onpanel=0
            for g in (p.get("context_markers") or []):
                v=lvl(g)
                if v is None: continue
                n_onpanel+=1; score+=v
                if v>=sig_floor: on_g.append(str(g))
            poles.append({"pole":p.get("pole"),"score":round(score,3),"on_context_markers":on_g,
                          "n_onpanel":n_onpanel,
                          "subtype":next((v2 for k,v2 in (p.get("subtypes") or {}).items()
                                          if str(k).strip().lower()==mkey), None)})
        ranked=sorted(poles, key=lambda p:-p["score"])
        present=[p for p in ranked if p["score"]>=sig_floor]
        top=present[0]["score"] if present else 0.0
        second=present[1]["score"] if len(present)>1 else 0.0
        dom=None; conf=None
        # ★ MEASURABILITY + POSITIVE-EVIDENCE gate (fixes assignment-by-exclusion, e.g. kidney mTAL):
        #   assign a pole ONLY if (a) the winner has >=1 of its OWN on-panel context markers ON
        #   (on_context_markers non-empty, not just a faint sub-threshold sum), AND (b) EVERY pole of the
        #   axis is measurable on this panel (n_onpanel>0). If ANY pole's context markers are entirely
        #   off-panel (mTAL medulla: Cldn10/Ptgs2 off-panel) the axis cannot discriminate the zones -> the
        #   winner would lead only because a competitor is silent = assignment by exclusion -> withhold the
        #   pole so the leaf floors to its lineage class. (Conservative for a >2-pole axis with an off-panel
        #   middle pole: the whole axis is withheld -> honest parent-class floor, never a forced zone.)
        all_measurable = all(p["n_onpanel"]>0 for p in poles)
        if top>0 and top>=lean_ratio*max(second,1e-9) and present[0]["on_context_markers"] and all_measurable:
            dom=present[0]; conf=("medium" if top>=1.5*max(second,1e-9) else "low")
        # ★ mixed = the second pole is *substantially* present (>=0.4x) -> for a large parent, subcluster to
        #   separate both poles so the minority pole is not buried. A leaf can have a dominant_pole and still be
        #   mixed (the parent leans one way but the opposite pole is considerable). The lean labels sub-leaves
        #   after a split; mixed is the split trigger (§4.3). When the second pole is negligible (pure-ish),
        #   mixed=False -> label by the lean directly.
        both_substantial=(len(present)>=2 and second>=0.4*max(top,1e-9))
        out.append({"axis":ax.get("name"),"poles":poles,
                    "poles_present":[p["pole"] for p in present],
                    "mixed":both_substantial,
                    "dominant_pole":(dom["pole"] if dom else None),
                    "suggested_subtype":(dom["subtype"] if dom else None),
                    "pole_confidence":conf,
                    "pmids":ax.get("pmids") or []})
    return out or None

def _state_axis_eval(sa, fon, lf, pin, gidx_U, frac=0.30, lf_on=1.5, pin_min=0.25, min_onpanel=3, coverage=0.80):
    """★ STATE-AXIS eval — base-agnostic cell states (proliferating / interferon-stimulated / hypoxic …).
    Unlike `_context_axis_eval` (pole competition for a subtype), each State is judged INDEPENDENTLY and any
    number can be active: a State is ACTIVE when a COVERAGE fraction of its OWN on-panel program markers are
    **ENRICHED** — elevated vs the whole dataset (fon≥frac OR fold≥lf_on & detected≥pin_min), NOT mere
    detection. State markers are co-regulated program markers (MKI67=proliferation …), curated base-agnostically
    in the KG; a real program lights up (nearly) ALL of them, so the engine requires ≥ceil(coverage*n_onpanel)
    ENRICHED and skips states with <min_onpanel measurable markers (a co-regulated pair = 1 signal, not 2). Ranked by summed
    enrichment. States with an IDENTICAL marker signature (EMT ≡ mesenchymal) are de-duplicated (keep one).
    Returns [{name, markers_on[], n_on, n_onpanel, score, pmids}] (highest first); [] if none active.
    ★ EVIDENCE ONLY — standard annotation does NOT append states. This surfaces the state_axis so the
    INTERACTIVE phase (§8) can subdivide by state on user request ("split the tumor by state"); the finalize
    label is base-only unless the agent sets dec["states"] there. Coverage=0.80 (≥80% of on-panel program
    markers ENRICHED) per the redesign; a real co-regulated program lights up (nearly) all its markers."""
    if not sa: return []
    out=[]; seen_sig=set()
    for st in sa:
        mks=st.get("markers") or []
        sig=frozenset(str(g).upper() for g in mks)                 # dedup identical signatures (EMT≡mesenchymal)
        if sig in seen_sig: continue
        on=[]; score=0.0; n_onpanel=0
        for g in mks:
            i=gidx_U.get(str(g).upper())
            if i is None: continue                                 # off-panel — not measured, skip
            n_onpanel+=1
            enr=(fon[i]>=frac) or (lf[i]>=lf_on and pin[i]>=pin_min)   # ENRICHED (R1), not mere detection
            if enr: on.append(str(g)); score+=float(max(lf[i],0.0))
        # ★ COVERAGE gate: a coherent program lights up (nearly) ALL its on-panel markers.
        #   Co-regulated markers (FOS/JUN/JUNB…) rise together, so the old "any ≥2" = 1 signal and let
        #   artifacts through; instead require ≥ceil(coverage*n_onpanel) of the MEASURABLE markers ENRICHED,
        #   and skip states with <min_onpanel measurable markers (too few to confirm a program). Panel-based,
        #   NOT KG marker count. See [[state-append-coverage-gate]].
        _need = int(coverage*n_onpanel)
        if coverage*n_onpanel - _need > 1e-9: _need += 1           # ceil
        if n_onpanel>=min_onpanel and len(on)>=_need:              # eligible (≥3 measurable) AND coverage met
            seen_sig.add(sig)
            out.append({"name":st.get("name"),"markers_on":on,"n_on":len(on),"n_onpanel":n_onpanel,
                        "score":round(score,3),"pmids":st.get("pmids") or []})
    out.sort(key=lambda x:-x["score"])
    return out

def _candidate_subtypes(terms, ogt, up_rows, octx, parent_up_auc=None, region_share=None):
    """Rank ontology terms whose *defining markers are actually visible* in this cluster as an evidence
    scorecard (side candidates), plus whether each candidate's one-level children are resolvable here
    (children_resolvable) — used to decide whether to split further (lookahead below).
    Quality filter: a term backed only by pan (non-discriminative) markers is not admitted, so only real
    subtypes such as CA1/CA3 survive, and only a few of them.
    Everything is derived from the ontology at runtime -> organ-agnostic, zero hardcoding. Non-binding
    (the Step 5 gate is final).
    ★ Parent pooling (parent_up_auc): prevents a homogeneous subtype from disappearing under over-split.
    In a child's `up` (= vs siblings) layer/relay markers flatten to ~0.5, but if they were enriched in the
    parent (= vs the whole brain) that identity is inherited by every child. So a term's *own* required
    markers are evaluated with parent enrichment included (inherit=True).
    Parent pooling is NOT used for children_resolvable (= would splitting separate them?), because a
    homogeneous marker does not separate under a split."""
    # ★ Uses pan_disc (>=12) — the same criterion as graph_evidence (_build_ident_maps). With octx["pan"]
    #   (>=4), layer discriminators such as Cux2/Rorb/Fezf2 are killed as pan, n_nonpan_present becomes 0 and
    #   SHOULD_SPLIT_DEEPER under-fires.
    pan=octx.get("pan_disc") or octx.get("pan") or set()
    up_auc={_U(r[0]):r[1] for r in (up_rows or [])}
    p_auc={_U(g):a for g,a in (parent_up_auc or {}).items()}
    ogt_u={_U(g):v for g,v in (ogt or {}).items()}
    children={}
    for tid,t in (terms or {}).items():
        for p in (t.get("P") or []):
            if p in terms: children.setdefault(p,[]).append(tid)
    def ev(g, inherit):
        """(present_bool, auc, on_panel_bool, is_pan_bool). With inherit=True, parent enrichment also counts."""
        # ★ The discriminative / non-discriminative (isp) test matches graph_evidence's anchors: when
        #   region_share is available, "non-discriminative" means "spread over more than 2 regions"
        #   (region_share > SPEC_REGION_MAX = 2). This stops the over-firing caused by pan>=12 failing to filter
        #   broad markers (PCP4 etc.), while layer markers (Rorb/Cux2/Fezf2: region_share=1) survive as
        #   discriminators. Falls back to the pan_disc (>=12) set when region_share is unavailable (no IDM).
        gu=_U(g); isp=(region_share.get(gu,99)>2) if region_share else (gu in pan)
        if gu in up_auc and up_auc[gu]>=0.6: return (True, round(up_auc[gu],3), True, isp)
        if inherit and p_auc.get(gu,0)>=0.6:
            return (True, round(p_auc[gu],3), True, isp)
        e=ogt_u.get(gu)
        if e is None: return (False, None, True, isp)
        if e.get("panel") is False: return (None, None, False, isp)
        a=e.get("auc")
        return ((a is not None and a>=0.5), (round(a,3) if a is not None else None), True, isp)
    def term_evidence(tid, inherit=True):
        R=(terms.get(tid) or {}).get("R") or []
        rec=[]; npres=0; nnonpan=0; noff=0
        for g in R:
            p,a,panel,isp=ev(g, inherit)
            rec.append([g, a, bool(p), bool(panel), bool(isp)])  # [gene,auc,present,on_panel,is_pan]
            if p:
                npres+=1
                if not isp: nnonpan+=1
            if panel is False: noff+=1
        return rec,npres,nnonpan,noff
    def disc_markers(tid):
        """Of all this term's markers (R u S u DF), those that are non-pan, on-panel and strong (AUC>=0.55),
        deduplicated by gene.
        Even when required is off-panel (Wfs1 for CA1), a supporting marker (Fibcd1) is picked up — the key
        to deriving signal_split targets automatically without dropping types. A type that really exists
        shows *several* discriminative markers; a spurious one carried by a single cross-listed marker
        (e.g. Cerebellar Golgi in cortex via Neurod6) has only one and falls in the ranking below."""
        tt=terms.get(tid) or {}; d={}
        for g in (tt.get("R") or [])+(tt.get("S") or [])+(tt.get("DF") or []):
            pres,a,onpanel,isp=ev(g, True)
            if onpanel and not isp and a is not None and a>=0.55:
                gu=_U(g)
                if gu not in d or a>d[gu]: d[gu]=a
        return sorted(d.items(), key=lambda x:-x[1])
    def required_ok(tid):
        """Check for an 'own (non-pan) required badge'. If this term has any non-pan required marker that is
        on-panel, at least one of them must be present (AUC>=0.55) for this to return True. If they are all
        off-panel, or there is no non-pan required marker, return True (the term is identified by supporting
        markers — CA1: Wfs1/Mpped1 are off-panel). For a term like UBC whose on-panel required marker
        (Eomes) does not fire, return False -> excluded as a target."""
        onp=[]
        for g in ((terms.get(tid) or {}).get("R") or []):
            pres,a,onpanel,isp=ev(g, True)
            if onpanel and not isp:
                onp.append(a is not None and a>=0.55)
        return (not onp) or any(onp)
    out=[]; sugg=[]
    for tid,t in (terms or {}).items():
        if not (t.get("R")): continue
        dms=disc_markers(tid)
        if dms and required_ok(tid):
            sugg.append({"term_id":tid,"label":t.get("L",""),"n_disc":len(dms),
                         "top_auc":dms[0][1],"disc_markers":[g for g,a in dms[:5]]})
        rec,npres,nnonpan,noff=term_evidence(tid, inherit=True)
        # quality filter: >=1 non-pan required, or >=2 required present (pan-only is excluded)
        if not (nnonpan>=1 or npres>=2): continue
        # child lookahead (one level): how many children have a non-pan required marker present, and which
        # inherit=False — whether a further split would separate them is judged from signal *inside this leaf*
        # only (parent inheritance excluded).
        kids=[]
        for kt in children.get(tid,[]):
            krec,knp,knn,kno=term_evidence(kt, inherit=False)
            if knn>=1 or knp>=2:
                kids.append({"label":(terms.get(kt) or {}).get("L",""),"n_req_present":knp,"n_nonpan_present":knn})
        out.append({"term_id":tid,"label":t.get("L",""),"tier":t.get("T",""),
                    "n_req_present":npres,"n_nonpan_present":nnonpan,"n_req":len(t.get("R")),"n_req_offpanel":noff,
                    "required":rec,"children_resolvable":kids})
    out.sort(key=lambda d:(-d["n_nonpan_present"], -d["n_req_present"], -max((r[1] or 0) for r in d["required"])))
    # signal_targets_suggested: propose types with strong own discriminative markers as automatic targets.
    # Ranked by (number of discriminative markers, best AUC) descending -> a real type with several markers
    # (CA1 Fibcd1+Spink8, CA3 Grik4+Npy2r+Nectin3, DG Prox1+Dsp+Neurod1) ranks high, while a spurious one
    # carried by a single cross-listed marker ranks low.
    # CA1/CA3/DG surface through supporting markers even when their required markers are off-panel, so the
    # agent cannot miss them.
    signal_targets=sorted(sugg, key=lambda d:(-d["n_disc"], -d["top_auc"]))[:15]
    return out, signal_targets

# ===================================================================== absolute identity (Leiden+absolute paradigm)
_TIERD={"class":0,"subclass":2,"supertype":2,"cluster":3,"state":1,"":0}
def _build_ident_maps(octx, genes, pan_n=12):
    """Build the ontology -> identity scoring index (once). Per term: indices of non-pan on-panel
    discriminative markers and negative markers, tier, lineage marker indices, lineage -> class term, and a
    lineage-root map. Everything is keyed on global gene indices (organ-agnostic).
    ★ The pan threshold used here (pan_n, default 12) is computed separately and is higher than
    octx['pan'] (>=4 terms, used for signal_split). At >=4, layer-discriminating markers such as Cux2 (4),
    Rorb (5) and Fezf2 (5) are removed like housekeeping genes and cortical layers stop being detected.
    At >=12 only true pan markers are filtered (Slc17a7=39, Gad1=139, Calb1=46, Foxp2=18) and layer
    markers survive as discriminative markers."""
    terms=octx.get("terms",{}) or {}; LIN=octx.get("lineages") or {}
    gUp={_U(g):i for i,g in enumerate(genes)}
    # ★ Marker-name suffix normalization: 15% of KG markers carry a context suffix (Drd1_strict, Foxp2_high,
    #   Tac1_subset, Slc17a6_neuronal, Pou4f1_PAG, Aromatase_Cyp19a1, Sf1_Nr5a1 ...) and fail to match panel
    #   symbols, so the intended marker is silently dead. They are re-matched on the base symbol:
    #   full -> first token (gene_suffix) -> last token (alias_gene: Aromatase_Cyp19a1 -> Cyp19a1).
    #   ★ IMPORTANT: normalization is applied ONLY to NEGATIVE, DF and lineage markers; DISC / PAN /
    #   region_share stay RAW. Restoring base symbols in disc (Foxp2_high x31 -> Foxp2) inflates region_share
    #   and demotes verified exclusive/defining anchors (Fezf2 for L5 ET, Drd2 for D2-MSN, Foxp2 for L6 CT) to
    #   non-exclusive, causing a broad regression. The real damage came from dead NEGATIVE markers (eSPN's
    #   Drd1_strict/Drd2_strict were dead, so eSPN was never vetoed on D1/D2 and occupied every striatal leaf),
    #   so restoring negatives alone is sufficient. Normalizing DF only improves df_offpanel accuracy and the
    #   DF override, and leaves region_share untouched (separate map). Losing the nuance is harmless because
    #   the engine's on_enr/auc thresholds already handle expression strength.
    def _gidx(g):
        gu=_U(g)
        if gu in gUp: return gUp[gu]
        if "_" in str(g):
            for cand in (str(g).split("_")[0], str(g).split("_")[-1]):
                cu=_U(cand)
                if cu in gUp: return gUp[cu]
        return None
    _cnt={}
    for _t in terms.values():
        for _k in ("R","S"):
            for _m in (_t.get(_k) or []): _cnt[_m.upper()]=_cnt.get(_m.upper(),0)+1
    PAN={_m for _m,_c in _cnt.items() if _c>=pan_n}
    term_disc={}; term_neg={}; term_neg_soft={}; term_tier={}; term_lab={}; term_def={}; term_core={}
    term_dz={}                                    # tid -> disease_context ('normal' by default). Used by _is_state_term to trust the KG
    for tid,t in terms.items():
        term_tier[tid]=t.get("T",""); term_lab[tid]=t.get("L",tid); dc=[]; seen=set()
        term_dz[tid]=(t.get("D") or "normal")
        for g in (t.get("R") or [])+(t.get("S") or [])+(t.get("DF") or []):
            gu=_U(g)
            if gu in PAN: continue
            j=gUp.get(gu)
            if j is not None and j not in seen: seen.add(j); dc.append(j)
        if dc: term_disc[tid]=dc
        # ★ CORE identity markers = R u DF (defining/required), SEPARATE from supporting (S). Used by the
        #   regime-C gate (_graph_evidence): the KG distinguishes HAS_DEFINING/REQUIRED (own identity) from
        #   HAS_SUPPORTING (co-expressed, non-exclusive), but disc merges the two and counts supporting markers as
        #   exclusive -> a specific anchor could be established from 1-2 supporting markers even when the core was
        #   measured (on-panel) but off (df_absent): CA2 Pfkp/Sv2b, OB-mitral Robo3, FC Notch2. Matching is RAW
        #   (same criterion as disc, PAN excluded) so share and normalization are unaffected.
        _core=set()
        for g in (t.get("R") or [])+(t.get("DF") or []):
            gu=_U(g)
            if gu in PAN: continue
            j=gUp.get(gu)
            if j is not None: _core.add(j)
        if _core: term_core[tid]=_core
        term_neg[tid]=[j for j in (_gidx(g) for g in (t.get("N") or [])) if j is not None]
        # ★ A "_low" graded negative is SOFT: "Pcp4_low" means "this gene should be *relatively* low", not that
        #   it is absent — yet _gidx normalizes it to the base symbol (Pcp4), so a broadly expressed gene at
        #   auc>=0.85 becomes a strong negative and hard-vetoes a genuine identity (MHb has excl_on=3 from
        #   Chrna3/Gpr151/Avil, but ambient Pcp4 at auc 0.861 rejected it outright).
        #   "_low" is therefore excluded from the hard veto (not counted in neg_strong; it stays in neg_on, where
        #   one is tolerated when excl_on>=2, §4.1). "_high" (Foxp2_high etc.) means "must not be high", so the
        #   auc>=0.85 veto is correct there and stays hard.
        _soft=set(_gidx(g) for g in (t.get("N") or []) if str(g).lower().endswith("_low"))
        term_neg_soft[tid]={j for j in _soft if j is not None}
        # DEFINING markers (a single one is sufficient) — kept separately for the DF-override.
        dfi=[j for j in (_gidx(g) for g in (t.get("DF") or [])) if j is not None]
        if dfi: term_def[tid]=dfi
    lin_idx={}; lin_class={}
    for lx,gs in LIN.items():
        idxs=[j for j in (_gidx(g) for g in gs) if j is not None]
        if idxs: lin_idx[lx]=idxs
    for tid,t in terms.items():
        lx=t.get("L","")
        if lx in LIN and lx not in lin_class: lin_class[lx]=tid
    def root(tid):
        x=tid; seen=set()
        while x and x not in seen:
            seen.add(x); ps=[p for p in (terms.get(x,{}).get("P") or []) if p in terms]
            if not ps: return x
            x=ps[0]
        return x
    share={}                                            # marker specificity: #terms whose disc lists each gene
    for dc in term_disc.values():
        for g in dc: share[g]=share.get(g,0)+1
    # ★ HIERARCHY-AWARE specificity (region_share): a marker's true specificity is how many
    #   distinct REGION-GROUPS (depth-1 children of a class) it spans — NOT how many terms list it. A region
    #   master TF (Tcf7l2→thalamus, Satb2→neocortex) is shared across that region's OWN subtypes (high `share`)
    #   yet appears under only ~1 region-group ⇒ region_share≈1 ⇒ it IS exclusive AT THE REGION LEVEL. This is
    #   what lets a region node anchor (thalamic via Tcf7l2) instead of under-calling to the bare class. A pan
    #   marker (Slc17a7) spans every region (high region_share) ⇒ not exclusive; a broad marker on many regions
    #   (Calb2) likewise. `region_of` = the ancestor on the path-to-class whose PARENT is a class-tier term.
    def region_of(tid):
        cur=tid; seen={tid}
        while True:
            ps=[p for p in (terms.get(cur,{}).get("P") or []) if p in terms]
            if not ps: return cur
            par=ps[0]
            if term_tier.get(par)=="class": return cur          # cur is a depth-1 region group
            if par in seen: return cur
            seen.add(par); cur=par
    _rmap={tid:region_of(tid) for tid in terms}
    region_share={}
    for tid,dc in term_disc.items():
        r=_rmap.get(tid,tid)
        for g in dc: region_share.setdefault(g,set()).add(r)
    region_share={g:len(rs) for g,rs in region_share.items()}
    return {"disc":term_disc,"neg":term_neg,"neg_soft":term_neg_soft,"tier":term_tier,"lab":term_lab,"share":share,
            "dz":term_dz,
            "region_share":region_share,"defn":term_def,"core":term_core,"region_of":_rmap,
            "lin_idx":lin_idx,"lin_class":lin_class,"root":{tid:root(tid) for tid in terms},
            "genes":list(genes), "P":{tid:[p for p in (t.get("P") or []) if p in terms] for tid,t in terms.items()}}

# ★ ANATOMICAL REGION-PRESENCE GATE (F-B) — blocks force-fits in the brain module.
#   Background: on low-sensitivity spatial data (Stereo-seq), a cluster with weak identity gets force-fit
#   to a rare nucleus of a **region that is not even in the section** (cerebellum, pons) because a
#   promiscuous region-exclusive marker in the KG switches ON from noise (e.g. 749 thalamic neurons called
#   cerebellar unipolar brush cells). KG region markers cannot establish absence (they are promiscuous —
#   cerebellar GABA scores higher than thalamic, which would regress the normal blocking), so a set of
#   **curated anatomical-region markers** (below) is used once to decide presence at the dataset level, and
#   region-groups of absent regions are removed from the dominant / anchor candidates.
#   The test is at the coarse anatomical level, so weak-marker subtypes (SNr = midbrain GABAergic; the
#   midbrain union includes Th/Slc6a3) are not unfairly blocked. In a non-brain module no region-group
#   label matches and the gate is inactive (no effect).
#   Validation (stereoseq_brain): PRESENT hippocampus 11% / cortex 7.9 / thalamus 4.3 / striatum 3.1 /
#   midbrain 2.5 / amygdala 2.2 vs ABSENT olfactory_bulb 0.026 / pons-medulla 0.024 / cerebellum 0.007
#   (present_thr=0.002 blocks only unambiguous absence).
BRAIN_ANAT_MARKERS = {
    "cerebellum":     ["Gabra6", "Cbln3", "Pcp2", "Eomes"],
    "pons/medulla":   ["Pou4f1", "Lmx1b", "Phox2b", "Phox2a", "Hoxb5", "Tlx3", "Barhl1"],
    "olfactory_bulb": ["Tbx21", "Sall3"],
    "thalamus":       ["Tcf7l2", "Prkcd", "Tnnt1", "Ntng1", "Ramp3"],
    "hypothalamus":   ["Otp", "Sim1", "Pomc", "Hcrt", "Pmch", "Agrp"],
    "midbrain":       ["Tfap2b", "Pax7", "Otx2", "Pax5", "Gata3", "Th", "Slc6a3", "Ddc"],
    "cortex":         ["Slc30a3", "Cux2", "Rorb", "Fezf2", "Satb2"],
    "hippocampus":    ["Spink8", "Fibcd1", "Hpca", "Prox1", "C1ql2", "Neurod6"],
    "striatum":       ["Ppp1r1b", "Drd1", "Drd2", "Adora2a", "Penk", "Gpr88"],
    "amygdala":       ["Lypd1", "C1ql3", "Cyp26b1"],
    "septum":         ["Zic1", "Zic4", "Trpc4"],
}
# (keyword, anat) — substring match against region-group labels, first match top-down
# (hypothalam must come before thalamic!)
GROUP_TO_ANAT = [
    ("hypothalam", "hypothalamus"), ("thalamic", "thalamus"), ("cerebell", "cerebellum"),
    ("ponto", "pons/medulla"), ("pontine", "pons/medulla"), ("medullary", "pons/medulla"),
    ("olfactory bulb", "olfactory_bulb"), ("midbrain", "midbrain"),
    ("neocort", "cortex"), ("allocortical", "cortex"), ("olfactory-cortex", "cortex"),
    ("piriform", "cortex"), ("entorhinal", "cortex"), ("retrosplenial", "cortex"),
    ("hippocampal", "hippocampus"), ("dentate", "hippocampus"), ("subicul", "hippocampus"),
    ("fasciola", "hippocampus"),
    ("striatal", "striatum"), ("amygdal", "amygdala"), ("septal", "septum"),
]
REGION_PRESENT_THR = 0.002   # a region is absent when its dataset presence rate (fraction of cells co-expressing >=2 anatomical markers) is below this

def _absent_region_groups(X, genes, octx, gmu, gsd, thr=REGION_PRESENT_THR):
    """Computed once over the full data (at init). Returns (absent_rg set[region-group term-id],
    present dict[anat->frac]). A subcluster sees only a subset and cannot compute this itself, so the init
    result is persisted and reused."""
    if gmu is None or gsd is None:
        return set(), {}
    genes = list(genes); gi = {g: i for i, g in enumerate(genes)}
    gmu = np.asarray(gmu).ravel(); gsd = np.asarray(gsd).ravel()
    Xa = np.asarray(X.todense() if hasattr(X, "todense") else X)
    HI = (Xa >= (gmu + gsd))
    present = {}
    for anat, mks in BRAIN_ANAT_MARKERS.items():
        idxs = [gi[g] for g in mks if g in gi]
        present[anat] = float((HI[:, idxs].sum(1) >= 2).mean()) if len(idxs) >= 2 else None
    IDM = _build_ident_maps(octx, np.asarray(genes))
    region_of = IDM["region_of"]; lab = IDM["lab"]; tier = IDM["tier"]
    absent = set()
    for rg in set(region_of.values()):
        if tier.get(rg) == "class":
            continue
        L = (lab.get(rg, rg) or "").lower()
        anat = next((an for kw, an in GROUP_TO_ANAT if kw in L), None)
        if anat is not None and present.get(anat) is not None and present[anat] < thr:
            absent.add(rg)
    return absent, present


def _graph_evidence(fon, lf, pin, IDM, octx, frac=0.30, lf_on=1.5, pin_min=0.25, SPEC_REGION_MAX=2,
                    mean=None, auc=None, auc_hi=0.85, auc_pos=0.65, anchor_floor_ratio=0.5, floor_pct=90, restrict_root=None,
                    absent_rg=None, pin_floor=None, fon_floor=None):
    """★ GRAPH-DESCENT EVIDENCE (not a verdict) — the redesigned identity substrate.
    The agent is the expert; the KG it built is its memory. helper provides EVIDENCE, the agent reasons.
    Anchors = specific terms whose OWN **exclusive** markers are ON and negatives clean, ranked by exclusive
    support, each with its ancestor `path` so the agent can STOP at a region when the subtype is ambiguous.
    **Exclusivity is HIERARCHY-AWARE (`region_share`):** a marker counts as exclusive for a term
    when it spans ≤SPEC_REGION_MAX distinct region-groups — so a region master TF (Tcf7l2→thalamus, shared
    across thalamic subtypes but absent from cortex/hippocampus) IS exclusive at the region level and lets the
    region node anchor; a pan/broad marker (Slc17a7, Calb2) spans many regions and is excluded. scRNA-
    compatible (no spatial): a leaf descends to CA1 only if CA1's exclusive Fibcd1/Spink8 are ON, to the
    thalamic region only if Tcf7l2/Gbx2 are ON — not because it shares a leaky marker."""
    disc=IDM["disc"]; neg=IDM["neg"]; neg_soft=IDM.get("neg_soft",{}); defn=IDM["defn"]; core_map=IDM.get("core",{})
    share=IDM.get("region_share") or IDM["share"]          # hierarchy-aware degree (fallback to term-degree)
    lab=IDM["lab"]; tier=IDM["tier"]; genes=IDM.get("genes"); P=IDM.get("P",{})
    lin_idx=IDM["lin_idx"]; lin_class=IDM["lin_class"]; terms=octx.get("terms",{})
    region_of=IDM.get("region_of",{})
    # ★ TWO predicates (region-coherent redesign, replaces the absolute noise-floor gate):
    #   on_present — ABSOLUTE PRESENCE (detected across the cluster), used for LINEAGE. A pan-within-lineage
    #     transmitter (Slc17a7 in cortex: pin~1) shows little FOLD vs the transmitter-rich global background, so
    #     enrichment would wrongly miss it → lineage=None → whole descent dies. Presence fixes it.
    #   on_enr — ENRICHMENT (vs-rest), used for region scoring + subtype anchors, so a genuine LOW-ABUNDANCE
    #     marker (Fezf2→L5, mean ~0.003 in 8µm) survives. NO absolute magnitude floor: cross-region false anchors
    #     (CA2 in a thalamic cluster) are blocked STRUCTURALLY by region-gated descent below — the floor was a
    #     blunt proxy that also killed real low-abundance layer markers (Fezf2 0.003 < spurious Amigo2 0.005).
    # TWO enrichment bars: on_enr (STRICT, detection-based) for REGION selection + lineage — a cross-region leaky
    #   marker (Nefm in L5 ET also marks auditory pons) must NOT inflate the wrong region. on_anchor (LOOSE, adds
    #   an auc branch) for anchor markers WITHIN the already-chosen region — catches a real layer TF that is
    #   strongly vs-rest enriched yet LOW-DETECTION (Rorb/Rspo1→L4: auc 0.9 but mean 0.01, pin<0.25 in 8µm). Auc
    #   is safe here ONLY because region-gating has already excluded the cross-region high-auc blips.
    def on_enr(g): return fon[g]>=frac or (lf[g]>=lf_on and pin[g]>=pin_min)
    def on_anchor(g): return on_enr(g) or (auc is not None and auc[g]>=auc_hi)
    # ★ LINEAGE adaptive floor: on_present is the substrate for the lineage call. When adaptive_floor is ON,
    #   pin_floor applies here too (region-agnostic — lineage is decided before region), so a rare lineage's
    #   discriminative required marker (NK GNLY, pin~0.24 < the fixed 0.25 under 15/6-cell dropout) still
    #   surfaces as codominant. SAFE: a codominant identity still requires disc_on (a discriminative marker),
    #   so a leaky pan marker that clears pin_floor cannot create a false identity — it is rejected as a hijack.
    #   pin_floor itself is min(fixed, gpin x F) plus an Nmin guard, so it never lowers the floor for common
    #   ambient markers (only for rare-specific ones). None (off) = unchanged.
    def on_present(g): return (pin[g]>=pin_min or (pin_floor is not None and pin[g]>=pin_floor[g])) and (mean is None or mean[g]>0)
    def gname(g): return str(genes[g]) if (genes is not None and 0<=g<len(genes)) else str(g)
    children={}
    for tid,ps in P.items():
        for p in ps: children.setdefault(p,[]).append(tid)
    def ev(tid):   # anchor evidence — uses LOOSE on_anchor (auc branch) since region-gating already excludes cross-region blips
        dc=disc.get(tid,[])
        excl=[g for g in dc if share.get(g,99)<=SPEC_REGION_MAX]  # exclusive: spans ≤N region-groups (hierarchy-aware)
        # ★ POSITIVE-anchor bar within a GATED region: low-copy layer/subtype markers (Rorb/Rspo1 -> L4,
        #   Cux2 -> L2/3, Fezf2 -> L5) are RANK-enriched yet fall below both the absolute on_enr floor AND
        #   auc_hi=0.85 on sparse platforms (Stereo-seq: a real L4 leaf shows Rorb auc~0.68, Rspo1~0.72,
        #   mean~0.01). With only the strict bar NO child subtype passes => anchors empty => the region node
        #   surfaces via fallback => the whole cortex collapses to a generic 'neocortical glutamatergic neuron'
        #   even though the layer signal is there. FIX: when region_gated (cross-region blips are already
        #   structurally excluded and the container node is out of the competition), POSITIVE markers
        #   (excl/df/n_on) use a looser auc_pos bar so a quiet-but-enriched layer marker anchors its layer.
        #   NEGATIVES stay STRICT (on_anchor, 0.85) — a weak negative must never veto a term; region scoring
        #   (onm) also stays strict.
        # ★ DATA-DERIVED detection floor (opt-in) — positive-anchor rescue for sparse platforms.
        #   A FIXED pin_min/frac is an absolute fraction that is unreachable under heavy dropout (0.5 um vector
        #   field: the L4 defining marker Rspo1 has pin~0.19 < 0.25 and is dropped even though lf~1.94 is
        #   genuinely specific). pin_floor/fon_floor are per-gene REFERENCE-FREE floors = min(fixed, gpin[g] x F)
        #   / min(fixed, ghi[g] x F) with an Nmin cell-count guard folded in (computed in _profile_leaves).
        #   Applied ONLY here (positive anchors) and ONLY when region_gated — negatives (on_anchor) and region
        #   scoring stay STRICT. on_present (lineage) also uses pin_floor, region-agnostically, to recover rare
        #   lineages; that is safe because a codominant identity still requires disc_on (see on_present).
        #   None (default/off) => the predicate is identical to before (zero change for existing modules).
        #   Recovers Rspo1 -> L4; Rorb/Etv1 stay off (low fold + high global detection). The auc bars are
        #   unchanged (already rank-normalized).
        def on_pos(g): return on_anchor(g) or (region_gated and auc is not None and auc[g]>=auc_pos) \
            or (region_gated and pin_floor is not None and lf[g]>=lf_on and pin[g]>=pin_floor[g]) \
            or (region_gated and fon_floor is not None and fon[g]>=fon_floor[g])
        excl_on=[g for g in excl if on_pos(g)]
        _core_set=core_map.get(tid,())
        excl_on_core=sum(1 for g in excl_on if g in _core_set)
        ng=[g for g in neg.get(tid,()) if on_anchor(g) and gname(g).upper() not in _cross_tx]  # negative-marker leak (cross-lineage transmitter ambient excluded)
        _soft=neg_soft.get(tid,())  # a "_low" graded hint is excluded from the hard veto (a relative-expression hint is not absolute absence)
        ng_strong=[g for g in ng if auc is not None and auc[g]>=auc_hi and g not in _soft]  # a strong negative (rank-enriched, not ambient) — genuine counter-evidence
        df=[g for g in defn.get(tid,()) if on_pos(g)]
        # ★ CORE-absent (a generalization of the required veto): the term has >=1 core marker (R u DF, pan
        #   excluded, on-panel) yet neither an exclusive core marker nor a defining marker is ON -> there is no
        #   identity evidence at all. df_absent (DEFINING only) could not stop a term that has no DF and only
        #   required markers (immature granule: R=Dcx..., DF=[]) from being force-fit through a supporting marker
        #   (Neurod1); this generalizes it.
        #   ★ len(df)==0 is required: without it, a genuine df-anchored type whose non-exclusive defining marker
        #   is ON (L5 ET, L6 CT, L6b, MH-LH, Sncg+) would be wrongly blocked. A defining marker that is ON is a
        #   legitimate anchor.
        core_absent=bool(len(_core_set)>=1 and excl_on_core==0 and len(df)==0)
        n_on=sum(1 for g in dc if on_pos(g))
        supported=bool((len(excl_on)>=1 or (len(excl)==0 and n_on>=2)) and len(ng)==0)
        # ★ DF-absent guard: defn[tid] is built by _build_ident_maps against the panel (adata.var_names) and
        #   therefore holds only ON-PANEL defining markers (off-panel DFs are excluded from the start, so cases
        #   like CA1's Wfs1 are exempt). A defining marker is the strongest grade — "one is enough" — so the
        #   converse holds: if such a marker was measured (on-panel) and none is ON (df==[]), the type is very
        #   unlikely => force-fit. Such a term is demoted (the primary sort key below) and excluded from the
        #   verdict even if it anchored on supporting markers alone. The auc>=0.85 branch treats a
        #   quiet-but-enriched DF as ON (cl50 Grik4 auc 0.92), which exempts it. Terms with no DF (most of them)
        #   are unaffected.
        df_onpanel=defn.get(tid,())
        df_absent=bool(len(df_onpanel)>=1 and len(df)==0)
        # ★ off-panel-defining guard: a different category from df_absent (on-panel but OFF, possibly quiet -> a
        #   weak tiebreak). If a term has defining markers in the KG (terms[tid].DF is non-empty) but they are
        #   *all off-panel* (absent from the panel, so defn[tid] is empty and df_onpanel==()), that type's own
        #   identity can **never** be confirmed with this panel (it is not quiet — it was never measured). Such a
        #   term is easy to force-fit from broad / false-exclusive supporting markers (Kisspeptin's Prlr/Tac2,
        #   since Kiss1 is off-panel). Unlike df_absent this never catches cortical layers (their defining markers
        #   Cux2/Fezf2/Foxp2/Rspo1/Car3 are all on-panel). Strong independent exclusive evidence (excl_on>=2) makes
        #   the type identifiable without a defining marker and is exempt; excl_on<2 (a single, possibly
        #   false-exclusive marker) is a panel-blind force-fit.
        df_kg=terms.get(tid,{}).get("DF") or []
        df_offpanel=bool(len(df_kg)>=1 and len(df_onpanel)==0)
        return {"term_id":tid,"label":lab.get(tid,tid),"tier":tier.get(tid,""),
                "excl_on":len(excl_on),"excl_n":len(excl),"excl_markers_on":[gname(g) for g in excl_on][:8],
                "neg_on":len(ng),"neg_strong":len(ng_strong),"neg_markers_on":[gname(g) for g in ng][:6],
                "df_on":len(df),"df_absent":df_absent,"core_absent":core_absent,"excl_on_core":excl_on_core,"df_onpanel":[gname(g) for g in df_onpanel][:6],
                "df_offpanel":df_offpanel,
                "n_on":n_on,"n_disc":len(dc),"supported":supported}
    def lvl(g): return mean[g] if mean is not None else fon[g]   # expression level (fon fallback when mean absent)
    def path_to_root(tid):
        chain=[tid]; seen={tid}
        while True:
            nxt=next((x for x in P.get(chain[-1],[]) if x not in seen), None)
            if nxt is None: break
            chain.append(nxt); seen.add(nxt)
        return [lab.get(x,x) for x in chain[::-1]]
    # ★ PARENT-CONSTRAINED descent: restrict_root = the parent's resolved term (its dominant
    #   region-group, or its lineage class) when THIS leaf is a SUBCLUSTER. The child may then only be that term
    #   or its KG DESCENDANTS — a subcluster of a neocortical cluster can ONLY be a neocortical subtype, never
    #   pontine/cerebellar (the 8µm force-fit bug: cl41 cortex fragments → "pontine"). THIS is why the KG is a
    #   graph: the parent's position bounds the child's search. With restrict_root set, transmitter lineage
    #   detection is SKIPPED (the lineage is already known from the parent).
    def _under(t, anc):
        x=t; seen=set()
        while x is not None and x not in seen:
            if x==anc: return True
            seen.add(x); ps=P.get(x,[]); x=ps[0] if ps else None
        return False
    NEURONAL_TX={"SLC17A7","SLC17A6","SLC17A8","GAD1","GAD2","SLC32A1","SLC6A5","TH","DBH","DDC","SLC6A3",
                 "SLC6A4","SLC6A2","TPH2","CHAT","SLC18A3","SLC5A7","HDC"}
    def _is_neur(ln): return any(gname(g).upper() in NEURONAL_TX for g in lin_idx.get(ln,[]))
    _neur_present=any(_is_neur(ln) and any(on_present(g) for g in idxs) for ln,idxs in lin_idx.items())
    # ★ Non-neuronal contamination detection (ENRICHMENT based): when a non-neuronal class lineage
    #   (oligo = Plp1/Mbp/Mag, astro = Aqp4/Slc1a2, micro = C1qa/Csf1r, VLMC = Col1a1/Dcn ...) has >=2 markers
    #   ENRICHED (on_anchor = vs-rest auc>=0.85) and **no neuronal transmitter is enriched**, the leaf is a
    #   genuine non-neuronal fragment. The test is ENRICHMENT, not presence (ambient neuropil), so a
    #   fiber-tract / vascular / microglial contamination fragment embedded in neuronal territory is released
    #   to glia even while it carries ambient Snap25/Gad (this releases the parent constraint and the neuronal
    #   priority; without it ~19k glial cells were mislabelled as striatal/midbrain GABAergic).
    #   ★ A real neuron is protected: its glial signal is ambient (non-enriched, auc<0.85) so _nn_enr<2
    #   (cl9 L6b: up = Cplx3/Nxph3 enriched, Plp1/Mbp ambient and not enriched). When a transmitter IS
    #   enriched (_neur_enr>0) the leaf is a real neuron and is never released.
    # ★ The neuronal-priority exception is ENRICHMENT-symmetric: a non-neuronal lineage with **>=2 markers
    #   ENRICHED** (on_anchor) while neuronal transmitters are **not enriched** (merely present as ambient) is
    #   genuinely non-neuronal. Requiring `not _neur_present` instead (i.e. no neuronal transmitter detected at
    #   all) failed to rescue microglia embedded in neuropil — ambient neuronal signal is always present — and
    #   they were misassigned to GABAergic/neuronal types (cl24). Using enriched rather than present keeps real
    #   neurons (transmitter enriched) protected while letting microglia with >=2 enriched specific markers out.
    #   The glia>=2 requirement blocks single-marker noise.
    _neur_enriched=any(_is_neur(ln) and any(on_anchor(g) for g in idxs) for ln,idxs in lin_idx.items())
    _nn_on=any((not _is_neur(ln)) and ln in lin_class and sum(1 for g in idxs if on_anchor(g))>=2 for ln,idxs in lin_idx.items())
    # ★ COMPARATIVE non-neuronal dominance: on dense/KDE spatial data (Stereo-seq) glial identity
    #   markers (Mbp/Plp1/Slc1a2) have a HIGH ambient baseline — detected in ~ALL cells — so a REAL oligo/astro
    #   cluster's markers cap at auc~0.75-0.78, BELOW on_anchor's absolute 0.85 ⇒ _nn_on fails ⇒ glia excluded
    #   from the lineage pool ⇒ the cluster is mislabeled to the ambient neuronal transmitter's lineage
    #   (oligodendrocyte→glutamatergic). That poisons descend (restrict_root=glut → glial subtypes are never
    #   searched, so oligo/astro never subdivide). FIX: when the STRICT test is inconclusive for BOTH sides
    #   (no neuronal transmitter on_anchor AND no glia ≥2 on_anchor), decide by RANK — a non-neuronal lineage
    #   is dominant when ≥2 of its markers out-rank the best neuronal transmitter by a clear margin
    #   (auc ≥ max(_LIN_CMP_MIN, best_neur_auc+_LIN_CMP_MARGIN)). Comparative (beats the runner-up, not an
    #   absolute floor) + ≥2 markers + only in the both-weak regime + the wide noise→signal gap (0.5→0.78)
    #   keeps false positives out; a real neuron (transmitter on_anchor) is untouched by `not _neur_enriched`.
    _LIN_CMP_MARGIN=0.08; _LIN_CMP_MIN=0.65
    _nn_auc_dom=False
    if auc is not None and not _neur_enriched and not _nn_on:
        _best_neur_auc=max((auc[g] for ln,idxs in lin_idx.items() if _is_neur(ln) for g in idxs), default=0.0)
        _lin_bar=max(_LIN_CMP_MIN, _best_neur_auc+_LIN_CMP_MARGIN)
        _nn_auc_dom=any((not _is_neur(ln)) and ln in lin_class and sum(1 for g in idxs if auc[g]>=_lin_bar)>=2
                        for ln,idxs in lin_idx.items())
    _nonneur_dominant=bool((_nn_on or _nn_auc_dom) and not _neur_enriched)
    allow=None; lineage_ambiguous=False; codom=[]; lineage_repick=None; overlay_lineages=[]; lineage_unresolved=False; unresolved_to=None
    if restrict_root and restrict_root in terms and not _nonneur_dominant:
        allow={t for t in terms if _under(t, restrict_root)}
        cls_root=IDM["root"].get(restrict_root,restrict_root); cls=cls_root; lin_tx_sym=set()
    else:
        # LINEAGE (transmitter HARD gate): prefer the lineage whose transmitter is ENRICHED (rejects ambient
        #   myelin Mbp ⇒ not oligodendrocyte), else the strongest PRESENT transmitter by MEAN (cortical Slc17a7
        #   is everywhere ⇒ low fold but high mean ⇒ glut). NEURONAL PRIORITY: a detected neuronal transmitter
        #   excludes glial lineages — BUT yields when a non-neuronal lineage is ENRICHED-dominant (_nonneur_dominant).
        neur_priority=_neur_present and not _nonneur_dominant
        lin_pool={ln:idxs for ln,idxs in lin_idx.items() if (not neur_priority) or _is_neur(ln)}
        strength={ln:max((lvl(g) for g in idxs if on_present(g)), default=0.0) for ln,idxs in lin_pool.items()}
        strength={ln:s for ln,s in strength.items() if s>0}
        if not strength or max(strength, key=lambda k:strength[k]) not in lin_class:
            return {"lineage":None,"anchors":[],"verdict":"no_lineage","cp":[],
                    "lineage_ambiguous":False,"codominant_lineages":[],"dominant_region":[]}
        best=max(strength.items(), key=lambda kv:kv[1]); LIN_RATIO=0.5
        codom=sorted([ln for ln,s in strength.items() if s>=LIN_RATIO*best[1]], key=lambda ln:-strength[ln])
        lineage_ambiguous=len(codom)>=2
        # ★ NON-IDENTITY LINEAGE OVERLAY (generalized). CASE: a cluster's discriminative identity is lineage B
        #   (B's discriminative required markers are ON), yet the program of another lineage A is present too. If
        #   A's presence rests *only on non-discriminative required markers* (pan_disc = R+S shared by >=12 terms)
        #   and A has 0 discriminative (not in pan_disc) required markers ON, then A is not a competing identity
        #   but a program overlaid on B (an activation/transition state such as EndMT/EMT) or ambient signal.
        #   Lineage strength is "the max over single required markers", so one non-discriminative marker of A
        #   scoring high through broad sharing can hijack the lineage. This is independent of lineage and gene
        #   names: it is defined purely by the structure "discriminative ON-required = 0", using only KG
        #   marker-sharing statistics (octx.pan_disc, the same criterion as _build). Evidence, not verdict: it is
        #   surfaced as overlay_lineages / lineage_repick.
        _PAN=octx.get("pan_disc") or set()
        def _disc_on(ln): return [g for g in lin_idx.get(ln,[]) if on_present(g) and gname(g).upper() not in _PAN]
        # (1) REPICK action: an overlay lineage (disc_on = {}) holds the raw max strength -> re-designate best to
        #   a codominant competitor with >=2 discriminative ON-required markers (>=2 blocks a re-designation
        #   caused by a single ambient-bleed marker). Self-protection: a real SMC has MYH11/CNN1 (not in
        #   pan_disc) ON -> disc_on>=1 -> it is never demoted; with no lineage_ambiguous this never fires.
        if lineage_ambiguous and not _disc_on(best[0]):
            _alts=[ln for ln in codom if ln!=best[0] and len(_disc_on(ln))>=2]
            if _alts:
                _new=max(_alts, key=lambda ln:(len(_disc_on(ln)), strength[ln]))
                lineage_repick={"from":lab.get(lin_class.get(best[0]),best[0]),"to":lab.get(lin_class.get(_new),_new),
                                "winner_on":[gname(g) for g in lin_idx.get(best[0],[]) if on_present(g)],
                                "alt_disc_on":[gname(g) for g in _disc_on(_new)],
                                "reason":"codominant non-specific hijack: winner held only by pan_disc(broad-shared) required marker(s); deferred to discriminative-corroborated codominant lineage"}
                best=(_new, strength[_new])
        cls=lin_class[best[0]]; cls_root=IDM["root"].get(cls,cls)
        # (2) OVERLAY surface: among codominant present lineages, one that is not the identity (best) and has 0
        #   discriminative ON-required markers is a program / ambient signal, not an identity. This stops the
        #   agent from switching identity to a high-magnitude program (cl8 EC -> myofibroblast).
        overlay_lineages=[{"lineage":lab.get(lin_class.get(ln),ln),"present":[gname(g) for g in lin_idx.get(ln,[]) if on_present(g)]}
                          for ln in codom if ln!=best[0] and not _disc_on(ln)]
        # (3) INDISCRIMINABLE LINEAGES -> common ancestor / Unknown. CASE: every on_present required marker of
        #   the best lineage is *shared with another codominant lineage* (PTPRC is in both myeloid.R and
        #   lymphocyte.R and is the top signal of both), so best has 0 own (codominant-private) discriminative
        #   markers = there is no evidence that separates the codominants -> neither can be chosen confidently.
        #   RULE: label the deepest common ancestor class (Unknown when there is none).
        #   ★ This is not pan_disc — the test is codominant-private, so brain transmitters (Slc17a7 is in glut
        #   only, absent from GABA = private) never trigger it and a genuine glut/GABA doublet stays protected
        #   (the split path is preserved). Skipped when repick already found the identity B
        #   (lineage_repick != None) — cl8 never reaches here.
        if lineage_ambiguous and lineage_repick is None:
            _oth_best=set().union(*[{gname(g).upper() for g in lin_idx.get(o,[])} for o in codom if o!=best[0]])
            _priv_best=[g for g in lin_idx.get(best[0],[]) if on_present(g) and gname(g).upper() not in _oth_best]
            if not _priv_best:
                lineage_unresolved=True
                def _anc(tid):
                    ch=[tid]; seen={tid}
                    while True:
                        nx=next((x for x in P.get(ch[-1],[]) if x not in seen),None)
                        if nx is None: break
                        ch.append(nx); seen.add(nx)
                    return ch
                _common=set.intersection(*[set(_anc(lin_class.get(ln,ln))) for ln in codom])
                for x in _anc(lin_class.get(best[0],best[0])):
                    if x in _common: unresolved_to=lab.get(x,x); break
        # ★ dual-transmitter cross-root reachability: MHb (cholinergic-glutamatergic) has root=glut, so a leaf
        #   like cl28 with the glut transmitter OFF and cholinergic ON is detected as the cholinergic lineage and
        #   MHb becomes unreachable. The transmitter symbols of the detected lineage are collected, and the anchor
        #   loop below admits cross-root terms that carry them in their own R (MHb holds Chat/Slc18a3) as
        #   region-gate bypass candidates, recovering co-releasing neurons.
        lin_tx_sym={gname(g).upper() for g in lin_idx.get(best[0],[])}
    # ★ Ignore cross-lineage transmitter negatives: the lineage gate has already fixed glut/GABA/etc., so
    #   counting another lineage's transmitter as a negative (pan-glut Slc17a7/6/8 on a GABAergic leaf, Gad1/2
    #   on a glut leaf) always leaks under 8 um ambient bleed and unfairly vetoes the real type — this was the
    #   direct cause of D1/D2-MSN failing on every striatal leaf through the pan-glut negative while eSPN (a
    #   false negative) survived. A leaf's own lineage transmitter is kept as genuine negative evidence.
    #   _cross_tx is therefore excluded from the ng computation in ev().
    _leaf_lins=[ln for ln,tid in lin_class.items() if IDM["root"].get(tid,tid)==cls_root]
    _own_tx={gname(g).upper() for ln in _leaf_lins for g in lin_idx.get(ln,[]) if gname(g).upper() in NEURONAL_TX}
    _cross_tx={x for x in NEURONAL_TX if x not in _own_tx}
    # ★ REGION-COHERENT DESCENT WITHIN the lineage: among the lineage class's region-groups (depth-1 children
    #   — neocortical / thalamic / hippocampal pyramidal ...), pick the DOMINANT by expression-weighted
    #   (relative, not an absolute floor) sum over its subtree's ENRICHED EXCLUSIVE markers; >=2 concordant
    #   markers are required so a lone broad gene (Stmn1) cannot raise a region. ONLY that region's subtypes
    #   are anchor candidates => cross-region blips (CA2 in a thalamic glut cluster) are structurally excluded
    #   WITHOUT a magnitude floor, so low-abundance layer markers (Fezf2 -> L5) survive. Gene-based,
    #   spatial-free.
    # Region scoring runs first over **all** regions of the lineage (independent of allow) — it is needed for
    # the fragment-escape decision.
    rg_terms={}
    for t in terms:
        if IDM["root"].get(t)==cls_root: rg_terms.setdefault(region_of.get(t,t),[]).append(t)
    _absent=absent_rg or set()
    region_groups_all=[rg for rg in children.get(cls,[]) if tier.get(rg)!="class" and rg not in _absent]
    rg_score={}; rg_hasdf={}
    for rg in region_groups_all:
        gg=set(disc.get(rg,())); rdf=set(defn.get(rg,()))
        for t in rg_terms.get(rg,()): gg|=set(disc.get(t,())); rdf|=set(defn.get(t,()))
        gg=[g for g in gg if share.get(g,99)<=SPEC_REGION_MAX]
        # RC3: region-exclusive markers count via on_anchor (auc≥0.85 branch), so a quiet-but-enriched
        #   region marker (Fezf2→neocortical, auc 0.90 yet detected<25%) resolves the region in sparse 2µm data —
        #   else a deep-cortical (L5) leaf loses neocortical to hippocampal (shared Slc17a7/Pcp4) and is mislabeled.
        #   gg is already region-exclusive (share≤SPEC_REGION_MAX) and ≥2 concordant required ⇒ no lone-blip region.
        onm=[g for g in gg if on_anchor(g)]
        if len(onm)>=2:
            rg_score[rg]=sum(lvl(g) for g in onm)
            rg_hasdf[rg]=any(on_anchor(g) for g in rdf if share.get(g,99)<=SPEC_REGION_MAX)
    # ★ COMPARATIVE region establishment: the strict pass above needs ≥2 region-exclusive markers
    #   at on_anchor (auc≥0.85 OR fon≥0.30). On dense KDE spatial data (Stereo-seq) a region master TF
    #   (Satb2→neocortical) caps at auc~0.70 and the layer markers are DILUTED at the mixed-cluster level
    #   (Cux2/Rorb/Fezf2 each high only in their own layer's subset) ⇒ NO region scores ⇒ dominant_region empty
    #   ⇒ the region_gated auc_pos(0.65) layer-marker rescue never fires ⇒ the whole cortex collapses to the
    #   generic region/class and can never be descended into layers. FIX (mirror of the lineage comparative
    #   above): ONLY when the strict pass found NO region, run a RANK pass at a looser auc bar (_REG_CMP_AUC)
    #   and accept a dominant region ONLY when ONE region out-ranks every other by a clear ratio
    #   (_REG_CMP_RATIO) — Satb2 0.70 ≫ hippocampal Fibcd1 0.43 / thalamic Tcf7l2 0.40. Still needs ≥2
    #   region-exclusive markers (no lone-blip region) and never fires on a tie (a genuine cross-region mix
    #   stays dominant-empty → descend). Strict pass unchanged ⇒ clean/targeted (auc>0.85) data identical.
    if not rg_score and auc is not None:
        # _REG_CMP_RATIO=1.3: the winner must out-rank the runner-up 1.3×, so a lone-region cluster (cortical:
        #   only neocortical has ≥2 markers; ratio→∞) and a clear winner both establish. A cross-region-ambiguous
        #   cluster whose winner is only ~1.5× (basolateral amygdala: hippocampal 3.9 vs neocortical 2.5) is NOT
        #   ratio-separable from a genuine cortical cluster (cl26/29/36 sit at 1.3-1.7 too), so it may get an
        #   adjacent-region context — but that is SELF-CORRECTING: the region-node fallback needs that region's OWN
        #   exclusive/DF markers ON (Spink8/Fibcd1 for hippocampal), which an amygdalar cluster lacks ⇒ anchors
        #   stay empty ⇒ class_or_unknown ⇒ NO confident wrong label (the agent reads its own markers, floors to
        #   class). Chosen to MAXIMIZE layer recovery (keeps cortical cl26/29/36); raise toward 2.0 to be more
        #   conservative at the cost of dropping those. Strict pass unchanged ⇒ clean/targeted (auc>0.85) identical.
        _REG_CMP_AUC=0.60; _REG_CMP_RATIO=1.3
        _rank={}
        for rg in region_groups_all:
            gg=set(disc.get(rg,()))
            for t in rg_terms.get(rg,()): gg|=set(disc.get(t,()))
            gg=[g for g in gg if share.get(g,99)<=SPEC_REGION_MAX and auc[g]>=_REG_CMP_AUC]
            if len(gg)>=2: _rank[rg]=sum(auc[g] for g in gg)          # rank-weighted (auc, comparative)
        if _rank:
            _r1=max(_rank, key=lambda r:_rank[r]); _top=_rank[_r1]
            _second=max((s for r,s in _rank.items() if r!=_r1), default=0.0)
            if _top>=_REG_CMP_RATIO*max(_second,1e-9):               # ONE region clearly wins → establish it
                rg_score={_r1:_top}
                _rdf=set(defn.get(_r1,()))
                for t in rg_terms.get(_r1,()): _rdf|=set(defn.get(t,()))
                rg_hasdf[_r1]=any(auc[g]>=_REG_CMP_AUC for g in _rdf if share.get(g,99)<=SPEC_REGION_MAX)
    # ★ FRAGMENT-ESCAPE: a subcluster is constrained to the parent region (restrict_root -> allow), but a
    #   region **outside** that constraint scores higher AND has a **defining marker ON** — that means a
    #   minority which was mixed into the parent has been isolated with a different-region identity
    #   (stomics cl6: a cortical majority plus a 19% BLA minority -> the split isolates the BLA blob). The
    #   parent constraint is released so the child can resolve to its own identity. The defining gate plus the
    #   score lead blocks a leaky cross-region force-fit (a single leaky marker satisfies neither >=2
    #   on_anchor nor a defining marker). The same applies when the parent never fixed a region
    #   (allow = the lineage class only).
    if allow is not None and rg_score:
        _ina=lambda rg: rg in allow
        _best_in=max((s for rg,s in rg_score.items() if _ina(rg)), default=0.0)
        _esc=[rg for rg,s in rg_score.items() if (not _ina(rg)) and rg_hasdf.get(rg) and s>_best_in]
        if _esc: allow=None        # release the constraint -> the child may escape to a different region (BLA)
    region_groups=[rg for rg in region_groups_all if (allow is None or rg in allow)]
    rg_score={rg:s for rg,s in rg_score.items() if rg in region_groups}
    REGION_RATIO=0.6
    dom_s=max(rg_score.values(), default=0.0)
    dominant=set(rg for rg,s in rg_score.items() if dom_s>0 and s>=REGION_RATIO*dom_s)
    region_gated=bool(dominant)
    dominant_region=[lab.get(rg,rg) for rg in sorted(dominant, key=lambda r:-rg_score[r])]
    # ★ ANCHORS within the dominant region-group(s) (or all subtypes under the lineage, in fallback): specific
    #   terms whose OWN exclusive markers are ENRICHED and negatives clean (leaf markers — CA1=Fibcd1/Spink8,
    #   L5=Fezf2 — drive the call). If none anchors, surface the dominant region-group node(s) so the leaf lands
    #   at e.g. 'neocortical glutamatergic neuron', NOT the bare lineage class.
    anchors=[]
    for tid in terms:
        # dual-transmitter cross-root term (MHb): admitted as a candidate and allowed to bypass the region gate
        # when it carries the detected lineage's transmitter in R, even under a different root
        cross_dual=bool(lin_tx_sym and IDM["root"].get(tid)!=cls_root and tier.get(tid)!="class"
                        and ({str(g).upper() for g in (terms.get(tid,{}).get("R") or [])} & lin_tx_sym))
        if IDM["root"].get(tid)!=cls_root and not cross_dual: continue
        if tier.get(tid)=="class": continue
        if region_of.get(tid,tid) in _absent and not cross_dual: continue
        if allow is not None and tid not in allow: continue        # parent-constrained: only descendants of restrict_root
        if region_gated and not cross_dual and region_of.get(tid,tid) not in dominant: continue
        # ★ The region-group container node (neocortical / thalamic / ... = the element of dominant itself) is
        #   excluded from the main competition: it has few region-exclusive markers of its own (Slc30a3/Mef2c),
        #   so at ratio=1.0 it unfairly beats its child layers (L4 with excl_n 8 has ratio 0.25) and collapses
        #   every cortical layer to the generic node. It surfaces, as intended, only through the fallback below
        #   when no child anchors at all (see: 'If none anchors, surface the dominant region-group node').
        if region_gated and not cross_dual and tid in dominant: continue
        e=ev(tid)
        # off-panel-defining guard: when the KG defining markers are all off-panel (df_offpanel) and the
        #   independent exclusive evidence is weak (excl_on<2), this is a panel-blind force-fit -> excluded from
        #   the anchors -> the region-node fallback below surfaces the parent region/class (Kisspeptin: Kiss1 is
        #   off-panel and Prlr is a single exclusive -> descend to hypothalamic GABAergic). excl_on>=2 passes.
        if e["df_offpanel"] and e["excl_on"]<2: continue
        # ★ REGIME-C gate (df_absent + supporting-only): when the core (R u DF) defining marker is ON-PANEL yet
        #   none is ON (df_absent) and every ON exclusive marker is SUPPORTING (excl_on_core == 0), a confident
        #   specific anchor is impossible -> descend to the region/class floor. This blocks the bug where the
        #   engine merged the KG's distinction between defining/required (own identity) and supporting
        #   (co-expression) and minted a measured-absent term from supporting markers alone (CA2 Pfkp/Sv2b on
        #   oligo/cortex fragments, OB-mitral Robo3 in cortex, FC Notch2/Zfp536 on CA1).
        #   ★ Regime A is preserved: if the defining markers are all OFF-PANEL then df_absent=False (df_onpanel is
        #     empty so the guard above does not fire) -> anchoring on supporting markers proceeds normally.
        #   ★ Regime B is preserved: a genuine layer's quiet defining marker is rescued by auc_pos (0.65) ->
        #     on_pos -> df is non-empty -> df_absent=False -> no effect.
        #   When df_absent but at least one core-specific (R u DF) marker is ON (excl_on_core>=1) it passes
        #   (identity evidence exists).
        #   Generalized from df_absent (DF only) to core_absent (R u DF): a term with no DF and only REQUIRED
        #   markers, whose required markers were measured but are off, could still pass on supporting markers
        #   alone (immature granule with Prox1/Dcx OFF force-fit through Stmn1). Regime A (required markers all
        #   off-panel -> _core_set empty -> core_absent=False) and Regime B (a quiet required/df rescued by
        #   auc_pos 0.65 -> excl_on_core>=1 -> core_absent=False) both remain intact.
        if e.get("core_absent",False): continue
        # ★ EMPTY-REGION asymmetric anchor gate: when region_gated, the candidates are already confined to
        #   dominant_region and cross-region blips are structurally excluded, so a single exclusive marker is
        #   enough (unchanged behaviour). But when dominant_region is empty (region_gated=False) the candidates
        #   open up to the whole lineage and a distant region's subtype gets force-fit from one leaky exclusive
        #   marker (thalamic = Cbln4, OB mitral = Robo3; xenium cl11 cortex -> thalamic, cl8).
        #   With no region fixed, a specific anchor requires exclusive >=2 OR a defining marker ON — a single
        #   non-defining marker cannot carry a cross-region call. Genuine cortical layers survive through the df
        #   branch (Rspo1 -> L4, Ldb2 -> L5 ET, Cux2 -> L2/3 are ON), so cl11 -> L4 IT, while a thalamic call
        #   resting on Cbln4 alone with no defining marker is blocked and corrected to cortex. The region_gated
        #   path is unchanged.
        # ★ df_absent single-marker guard: even when region_gated, an anchor that is df_absent (it has on-panel
        #   defining markers but all are OFF) and has exactly one (non-defining) exclusive marker is not accepted.
        #   Otherwise, when region scoring wrongly assigns an unrelated subcluster to a region (a subcortical
        #   Slc17a6+/Gad1+ fragment becoming allocortical-dominant through Ndst4 alone), a child of that region
        #   becomes a clear_top anchor with df_absent + a single exclusive marker and produces a region-floor
        #   mislabel (allocortical). Three admissible paths: (a) df_on>=1 (a quiet defining marker ON = a real
        #   layer, L4 Rspo1 via auc_pos), (b) excl_on>=2 (multiple exclusives, L4 Rorb+Whrn), (c) excl_on>=1 and
        #   not df_absent (a type with no on-panel defining marker at all = off-panel-defining, preserved as
        #   panel-blind). Only df_absent & df_on==0 & excl_on==1 is blocked -> if that empties anchors, the leaf
        #   descends honestly to class_or_unknown.
        _anchor_ok = ((e["df_on"]>=1 or e["excl_on"]>=2 or (e["excl_on"]>=1 and not e.get("df_absent",False)))
                      if region_gated else (e["excl_on"]>=2 or e["df_on"]>=1))
        # ★ Relaxed negative veto: a strong negative (auc>=0.85 = rank-enriched, genuine counter-evidence) always
        #   vetoes. A weak negative (on_enr only = ambient/contamination level) is tolerated, at most one, and
        #   only when the positive evidence is strong (excl_on>=2) — a single weak negative introduced by KDE /
        #   neighbour bleed in spatial data used to reject a strong anchor outright (MHb Th at 32% rejected
        #   excl_on=3). Consistent with §4.1.
        _neg_ok = e.get("neg_strong",e["neg_on"])==0 and (e["neg_on"]==0 or e["excl_on"]>=2)
        if _anchor_ok and _neg_ok:
            e["path"]=path_to_root(tid); anchors.append(e)
    if not anchors and region_gated:
        for rg in sorted(dominant, key=lambda r:-rg_score[r]):
            e=ev(rg)
            if e["neg_on"]>0: continue          # a region node is not surfaced either when it has a negative leak (e.g. a DG region on an Slc17a6+ cluster)
            # ★ Fix A: to surface a region node as a floor, >=1 of that region's OWN exclusive/defining markers must
            #   be ON. This blocks a generic cluster carrying only pan markers (Slc17a7) from being force-fit to a
            #   region floor (DG etc.) — it removes the bogus cortical 'dentate gyrus glutamatergic neuron'
            #   (Prox1/C1ql2 = 0) and markerless region floors. When no region-specific marker (DG = Prox1/C1ql2,
            #   neocortical = Satb2/Slc30a3, thalamic = Tcf7l2) is ON, the node is not surfaced -> anchors are empty
            #   -> class_or_unknown -> honest descent to the lineage class ('glutamatergic neuron').
            #   (Unrelated to the leaf-anchor path and to region_gated auc_pos layer-marker preservation: this is only
            #   the 'region-node fallback when there is no anchor at all'.)
            if e["excl_on"]<1 and e["df_on"]<1: continue
            e["path"]=path_to_root(rg); anchors.append(e)
    # ★ EMBEDDED-TYPE RESCUE: vascular / immune cells (SMC, endothelium, pericyte ...) are embedded in the
    #   tissue, so KDE bleeds surrounding glial/neuronal RNA into their voxels -> the cluster's absolute
    #   profile is dominated by the surrounding cells and, being off-lineage, it drops out of the anchor
    #   candidates (cl69: Acta2 is 34x enriched but Mbp dominates absolutely -> oligo; cl103: Cldn5 18x ->
    #   astro). When a highly specific core marker (R u DF, share <= SPEC_REGION_MAX) is strong by BOTH rank
    #   (auc>=0.90) and fold (lf >= log2 10 = 10x), that term is surfaced as an anchor (ignoring lineage,
    #   negatives and absolute dominance). The triple gate keeps broad/ambient signal out — microglia at 3-5x
    #   have fold<10 and the floor is kept (correctly). The agent sees the rescued flag and rescue_markers and
    #   calls vascular instead of the absolutely-dominant glia (SKILL §4).
    if auc is not None and lf is not None:
        _RESC_AUC=0.90; _RESC_LF=float(np.log2(5.0)); _anch={a["term_id"] for a in anchors}
        # ★ Tuning: the fold threshold is 5, not 10 (microglia cl75's specific marker Tmem119 at 6x is genuine but
        #   was missed by a 10x bar; auc>=0.90 plus the non-parenchymal restriction already excludes ambient
        #   signal). DEFINING markers bypass the share filter (ependymal Foxj1 is a DF with share=4, so a share<=2
        #   rule rejected it despite 46x — a DF is curated-specific by construction).
        # ★ NON-PARENCHYMAL only: being embedded in tissue and losing absolute dominance to KDE smear happens only
        #   to the vascular / immune / meningeal / choroid-plexus / ependymal families. Neurons (glut, GABA,
        #   transmitter types) and macroglia (astro, oligo, OPC) are parenchymal and are not rescue targets (their
        #   mislabels are sibling/region problems handled elsewhere). Without this gate every type's defining
        #   marker reaches fold>=10 and the rescue floods.
        _NONPAR=('endothel','pericyte','smooth muscle','leptomening','mural','perivascular','fibroblast',
                 'macrophage','microg','choroid','ependym','arachnoid','vascular','erythro','neutrophil',
                 'lymphocyte','monocyte','mast','tanycyte')
        def _nonpar(_t):
            _rl=lab.get(IDM["root"].get(_t,_t),"").lower(); _tl=lab.get(_t,"").lower()
            return any(k in _rl or k in _tl for k in _NONPAR)
        for tid in terms:
            if tid in _anch or not _nonpar(tid): continue
            _dfset=set(defn.get(tid,()))   # DEFINING markers bypass the share filter (curated-specific; ependymal Foxj1 has share=4)
            strong=[g for g in core_map.get(tid,())
                    if (g in _dfset or share.get(g,99)<=SPEC_REGION_MAX) and auc[g]>=_RESC_AUC and lf[g]>=_RESC_LF]
            if strong:
                e=ev(tid); e["rescued"]=True; e["rescue_markers"]=[gname(g) for g in strong][:6]
                e["path"]=path_to_root(tid); anchors.append(e)
    # df_absent (a defining marker was measured and is OFF) is **only a weak tiebreak** — excl_on / ratio /
    #   n_on come first. Otherwise a generic region node with no DF (and therefore never df_absent) always
    #   beats a genuine layer whose defining marker is merely quiet (L4: Rorb/Whrn give excl_on 2 while Rspo1
    #   is quiet), collapsing every cortical layer to the generic node. A force-fit (mitral) is already pushed
    #   down by ratio, so no sort demotion is needed. df_absent is passed to the agent as a flag only (in the
    #   listing and in _validate) and breaks ties last.
    # rescued (embedded vascular/immune) is sorted to the top so it is not buried by absolute dominance — the
    # agent must see it (the final call is the agent's, SKILL §4).
    anchors.sort(key=lambda x:(x.get("rescued",False), x["excl_on"], round(x["excl_on"]/max(x["excl_n"],1),2), x["n_on"], not x.get("df_absent",False)), reverse=True)
    # verdict hint (the agent decides; this is a summary): clear top = unique max excl_on; tie = ambiguous.
    if not anchors: verdict="class_or_unknown"
    else:
        topv=anchors[0]["excl_on"]; ties=[a for a in anchors if a["excl_on"]==topv]
        verdict=("clear_top" if len(ties)==1 else "ambiguous_tie")
    anchor_labels=[a["label"] for a in anchors]
    pathlabels={l for a in anchors[:6] for l in a.get("path",[])}|set(anchor_labels[:6])
    # ★ ConfusionPair gate: a cp is matched by its terms (if the slim-index serves them) OR by its
    #   pair_label "<A> vs <B>" tokens against anchor labels (the served terms are often empty). If the TOP anchor
    #   and its cp-PARTNER BOTH anchor — both consistent, neither's specific markers exclude the other (e.g. FC &
    #   CA2 share Pcp4/Amigo2/Rgs14) — the identity is NOT marker-resolvable ⇒ confusion_unresolved. The agent then
    #   calls UNKNOWN, never force-fitting one (per user: a fasciola-cinerea cluster must be Unknown, not wrong CA2).
    def _cl(x): return re.sub(r'\s*\(.*?\)','',(x or '')).strip().lower()
    def _assoc(side, label):
        s=_cl(side); L=_cl(label)
        return len(s)>=5 and (s in L or L in s)
    cps=[]; confusion_unresolved=None
    top_label=anchor_labels[0] if anchor_labels else None
    a2tid={a["label"]:a["term_id"] for a in anchors}
    a2anchor={a["label"]:a for a in anchors}
    def _distinct_on(tid_a, tid_b):
        # among tid_a's own (unshared) exclusive markers, the ones that are ON -> switched-on evidence that
        # distinguishes tid_a from tid_b
        ea={g for g in disc.get(tid_a,[]) if share.get(g,99)<=SPEC_REGION_MAX}
        eb={g for g in disc.get(tid_b,[]) if share.get(g,99)<=SPEC_REGION_MAX}
        return [gname(g) for g in (ea-eb) if on_anchor(g)]
    for c in (octx.get("cp") or []):
        if len(c)<2: continue
        sides=[s for s in re.split(r'\s+vs\.?\s+', c[0] or '', flags=re.I) if s.strip()]
        side_anchors=[[al for al in anchor_labels if _assoc(s,al)] for s in sides]
        relevant=any(l in pathlabels for l in (c[1] or [])) or any(side_anchors)
        if relevant: cps.append({"pair":c[0],"terms":c[1],"discriminators":(c[2] if len(c)>2 else [])})
        if confusion_unresolved is None and top_label and len(sides)>=2:
            tsides={i for i,s in enumerate(sides) if _assoc(s,top_label)}
            if tsides:
                partners=[al for i,hits in enumerate(side_anchors) if i not in tsides for al in hits]
                if partners:
                    par=partners[0]; tt=a2tid.get(top_label); pt=a2tid.get(par)
                    # ★ Three-way resolution: decided by whether each candidate's *own* (unshared) markers are ON.
                    #   Both sides' own markers ON -> a mixture of two types -> split (needs_subclustering). Only one side ON
                    #   -> the discriminator supports that side -> that type. Both OFF (only shared markers) -> genuinely
                    #   ambiguous -> Unknown.
                    td=_distinct_on(tt,pt) if (tt and pt) else []
                    pd=_distinct_on(pt,tt) if (tt and pt) else []
                    # Both sides' own markers ON -> the two types coexist (cl64 CA2 Cacng5/Map3k15/Acan + FC Scgn) -> split.
                    #   Only one -> resolve to that side. Both OFF (shared only) -> unknown. On a split, needs_subclustering
                    #   prevents lumping FC into CA2: even if FC is not found, the fragments separate into CA2 / hippocampal /
                    #   Unknown and the FC part does not remain labelled CA2.
                    # ★ Resolve guard: if the defining marker that would confirm the resolve target is **off-panel
                    #   (df_offpanel) or on-panel but entirely OFF (df_absent)**, and that type has only **one** distinct ON
                    #   marker, then committing would be a force-fit from a single weak non-defining marker -> block the
                    #   resolve and fall to the common-ancestor class floor below. (FC's defining marker Necab2 is off-panel,
                    #   yet FC was being asserted from Zfp536/Igfbp4 alone. A genuine FC has distinct >= 2 and passes.)
                    def _df_unconf(lbl):
                        an=a2anchor.get(lbl) or {}; return bool(an.get("df_offpanel") or an.get("df_absent"))
                    if td and pd:      action,resolves="split",None
                    elif td and not (_df_unconf(top_label) and len(td)<2):  action,resolves="resolve",top_label
                    elif pd and not (_df_unconf(par) and len(pd)<2):        action,resolves="resolve",par
                    else:
                        # ★ Only shared markers are ON and both sides' own markers are OFF (or off-panel) -> neither can be
                        #   asserted. Rather than Unknown, fall honestly to the **deepest common ancestor CLASS** of the two
                        #   members. A confusion pair arises in a context where the split was made *expecting* these types, so
                        #   when only shared markers fire the common region/class is the right answer (FC vs CA2 -> hippocampal
                        #   pyramidal neuron). A type like FC, whose discriminating defining marker is off-panel, lands here and
                        #   gets the parent class instead of being asserted as CA2.
                        _pa=path_to_root(tt) if tt else []; _pb=path_to_root(pt) if pt else []
                        _common=None
                        for _x,_y in zip(_pa,_pb):
                            if _x==_y: _common=_x
                            else: break
                        action,resolves="unknown",_common
                    confusion_unresolved={"pair":c[0],"terms":[top_label]+partners[:3],
                                          "discriminators":(c[2] if len(c)>2 else []),
                                          "action":action,"mixture":bool(td and pd),"resolves_to":resolves,
                                          "top_distinct_on":td[:6],"partner_distinct_on":pd[:6]}
    if confusion_unresolved: verdict="confusion_unresolved"
    # ★ MIXTURE detection (SHOULD_SPLIT_MIXTURE): even when the leaf terminalizes at a region/class
    #   (no single subtype anchors strongly), if ≥2 of the dominant region's child subtypes each have their OWN
    #   exclusive marker WEAKLY co-present (auc≥WEAK_AUC=0.6 — below the strict on_anchor 0.85, OR fold/df-ON via
    #   on_anchor), the leaf is an UNDER-CLUSTERED MIXTURE that the strict anchor gate missed because the markers
    #   are diluted (2µm: cl2=Drd1+Drd2, cl15=Cux2+Rorb, cl7.1=L5 Fezf2+L6 Foxp2). A cluster with NO subtype's
    #   own exclusive marker present (cl4 region-less generic glut) is NOT flagged ⇒ no noise. ≥2 distinct → split.
    WEAK_AUC=0.6
    cand=[]
    for tid in terms:
        if IDM["root"].get(tid)!=cls_root or tier.get(tid)=="class": continue
        if region_of.get(tid,tid) in _absent: continue
        if allow is not None and tid not in allow: continue
        if region_gated and region_of.get(tid,tid) not in dominant: continue
        exc=[g for g in disc.get(tid,()) if share.get(g,99)<=SPEC_REGION_MAX]
        if not exc: continue
        wk=[g for g in exc if on_anchor(g) or (auc is not None and auc[g]>=WEAK_AUC)]
        if not wk or any(on_anchor(g) for g in neg.get(tid,())): continue
        cand.append((tid, region_of.get(tid,tid), wk))
    # ★ EMPTY-REGION mixture cleanup — KG-region grouping (expression based, no spatial input).
    #   NOTE: 'region' here means a **region-group in the KG hierarchy** (neocortical/thalamic = region_of, a
    #   graph structure) and the decision uses **marker expression only** (on_anchor/auc). Cell xy coordinates
    #   are never used, so this behaves identically on scRNA-seq (the same principle as dominant_region and
    #   the graph descent). It is unrelated to 'spatial coherence' (a 15-NN neighbour fraction).
    #   When region_gated, the region gate above has already removed cross-region candidates and catches
    #   diluted layers loosely (a single marker). But when the region is undetermined (region_gated=False) the
    #   mixture opens up to the whole lineage and cross-region noise from a single leaky marker creeps in (two
    #   thalamic types sharing one Cbln4, OB = Robo3, midbrain = Ntrk3).
    #   Discriminator: a genuine under-clustered mixture has **>=2 distinct exclusive markers ON within the
    #   same KG region-group** (cortex = Rspo1 + Ldb2 ...); noise is *a single marker scattered across
    #   different regions* or *one marker shared between siblings* (Cbln4). Count the union of distinct ON
    #   exclusive markers per region and keep only the subtypes of regions with >=2 — diluted genuine layers
    #   survive (cl12 L5 IT/ET/L5b, whose markers are distinct) while Cbln4-sharing thalamic types and isolated
    #   single markers are removed. (A criterion based on marker count OR df killed diluted layers and is not
    #   used.)
    if not region_gated and cand:
        _reg_mk={}
        for _tid,_reg,_wk in cand: _reg_mk.setdefault(_reg,set()).update(_wk)
        _ok={r for r,gs in _reg_mk.items() if len(gs)>=2}
        cand=[c for c in cand if c[1] in _ok]
    mix=[{"label":lab.get(tid,tid),"markers_present":[gname(g) for g in wk][:4]} for tid,_reg,wk in cand]
    mixture_subtypes=mix[:8] if len(mix)>=2 else []
    return {"lineage":lab.get(cls,cls),"verdict":verdict,"anchors":anchors[:8],"cp":cps[:4],
            "confusion_unresolved":confusion_unresolved,"mixture_subtypes":mixture_subtypes,
            "lineage_ambiguous":lineage_ambiguous,"codominant_lineages":codom if lineage_ambiguous else [],
            "lineage_repick":lineage_repick,"overlay_lineages":overlay_lineages,
            "lineage_unresolved":lineage_unresolved,"unresolved_to":unresolved_to,"dominant_region":dominant_region}

def _bimodal_split(Xleaf, IDM, octx, lineage_label, genes_arr,
                   corr_thr=-0.30, min_pole_frac=0.15, min_present_frac=0.10, min_cells=80):
    """★ BIMODAL-SPLIT backstop (entry-trigger gap). The cluster-level mixture
    detector (`graph_evidence.mixture_subtypes`) MISSES a *balanced* 50/50 mix of two
    SIBLING-region subtypes: each pole's markers are diluted below the fold/auc bar AND the 2nd
    pole sits OUTSIDE `dominant_region`, so neither the mixture loop nor the finalize gate flags
    it → the leaf terminalizes at class_floor and never enters the (already-implemented) γ-escalation
    loop (e.g. arterial+venous EC lumped as `endothelial cell`; alveolar+adventitial fibroblast
    lumped as `fibroblast`). This is invisible to MEAN-based detection but obvious PER-CELL.
    Here we test, within the leaf, whether ≥2 REGION-GROUPS under THIS lineage have
    ANTI-CORRELATED marker programs each high in a DISJOINT cell subset — the true bimodal
    signature. Region-INDEPENDENT (does not touch the brain-tuned region-gating), so purely
    ADDITIVE: existing triggers are unchanged. Continuum/ambient co-expression (corr≥0) and
    low-quality dropout clusters (no high subpopulation, <2 distinct markers present) are NOT
    flagged → no SMC↔pericyte continuum / lymphocyte-dropout false positives.
    Returns [{label, markers_present[]}, …] (≥2) when bimodal, else []."""
    terms=octx.get("terms",{}); disc=IDM["disc"]; share=IDM.get("region_share") or IDM["share"]
    tier=IDM["tier"]; region_of=IDM.get("region_of",{}); lab=IDM["lab"]; root=IDM["root"]
    cls=IDM["lin_class"].get(lineage_label)
    if cls is None: return []
    cls_root=root.get(cls,cls)
    if hasattr(Xleaf,"toarray"): Xleaf=Xleaf.toarray()
    Xleaf=np.asarray(Xleaf)
    if Xleaf.shape[0]<min_cells: return []
    det=(Xleaf>0).mean(axis=0)                                    # per-gene detection within the leaf
    reg_mk={}                                                     # region-group → its low-share exclusive markers PRESENT here
    for tid in terms:
        if root.get(tid)!=cls_root or tier.get(tid)=="class": continue
        if (terms.get(tid,{}).get("D") or "normal")!="normal": continue          # disease/dev state terms aren't discrete split poles
        exc=[g for g in disc.get(tid,()) if share.get(g,99)<=2 and 0<=g<det.shape[0] and det[g]>=min_present_frac]
        if exc: reg_mk.setdefault(region_of.get(tid,tid), set()).update(exc)
    regs=[(r,sorted(gs)) for r,gs in reg_mk.items() if len(gs)>=2]   # ≥2 distinct exclusive markers (ambient-noise guard)
    if len(regs)<2: return []
    scores={}
    for r,gs in regs:                                            # per-cell z-scored mean of each region-group's markers
        sub=Xleaf[:,gs]; mu=sub.mean(0); sd=sub.std(0)+1e-9
        scores[r]=((sub-mu)/sd).mean(1)
    rl=[r for r,_ in regs]; best=None
    for i in range(len(rl)):
        for j in range(i+1,len(rl)):
            a,b=scores[rl[i]],scores[rl[j]]
            cc=float(np.corrcoef(a,b)[0,1])
            if not np.isfinite(cc): continue
            za=(a-a.mean())/(a.std()+1e-9); zb=(b-b.mean())/(b.std()+1e-9)
            fa=float(((za>0.5)&(zb<0)).mean()); fb=float(((zb>0.5)&(za<0)).mean())   # disjoint high subpops
            if cc<=corr_thr and fa>=min_pole_frac and fb>=min_pole_frac and (best is None or cc<best[0]):
                best=(cc,rl[i],rl[j])
    if best is None: return []
    _reg_d=dict(regs)
    return [{"label":lab.get(r,r),"markers_present":[str(genes_arr[g]) for g in _reg_d.get(r,[])[:4]]}
            for r in (best[1],best[2])]

# ★ The KG tier is the **only** source of truth for the state decision.
#   Inferring state from label keywords (developmental / neuroblast / radial glia / immature / reactive
#   ...) excluded **38 normal cell types** that the KG explicitly marks tier='class'/'subclass' from the
#   top candidates purely because of their names (radial glia, neuroblast, immature B cell / neutrophil /
#   platelet / cardiomyocyte, ISP thymocyte, immature enterocyte, immature Sertoli cell ...). Maturation
#   stages are real cells that make up much of tissues such as bone marrow and thymus, so this suppression
#   was a systematic failure to detect them.
#   This simplification presumes the KG-side cleanup that precedes it:
#     1) 2 cells blocked by disease_context -> the disease was moved to an OBSERVED_IN_DISEASE edge and
#        the property removed
#     2) 200 developmental stages -> separated from disease_context into `developmental_stage`, exposed in
#        the slim-index as `DS`. Development is not disease, and in fetal data radial glia must be the
#        correct answer. Whether a sample is adult or fetal is judged by the **expert agent** from DS.
#     3) 158 tier='state' misclassifications resolved
#   As a result there is no longer any term that "looks like a state by label while the KG calls it an
#   identity in an abnormal context", so the keyword heuristic is unnecessary.
#   Disease is not consulted here — disease is a hint, not a filter (§7), and normal cells from a diseased
#   sample (cDC1 / Treg / naive CD4 T cell in breast cancer, 233 of them) must come out with normal labels.
def _is_state_term(label, tier):
    """STATE decision = whether the tier in the KG is 'state'. Labels and disease context are not consulted.
    A state is a base-agnostic condition (proliferating / IFN-stimulated / hypoxic) and is appended to the
    label as ', <state>' (STATE-AXIS). Anything that carries a cell name is a Term, not a state."""
    return tier == "state"
def _disease_terms(idx, disease):
    """Extract the ontology terms matching the disease the caller passed via --disease <name>.
    Matching: the term's D (disease_context) is not normal and overlaps the disease parameter in either
    direction (case-insensitive).
    Returns {tid:{L,D,R,S,DF}} — _profile_leaves surfaces these per leaf as disease_candidates (with marker
    support). Returns {} when disease is unset or normal (the normal path is unchanged)."""
    dz=(disease or "").strip().lower()
    if not dz or dz in ("normal","none"): return {}
    out={}
    for tid,t in ((idx or {}).get("terms",{}) or {}).items():
        D=(t.get("D") or "normal")
        if D and D.lower()!="normal" and (dz in D.lower() or D.lower() in dz):
            out[tid]={"L":t.get("L",tid),"D":D,
                      "R":list(t.get("R") or []),"S":list(t.get("S") or []),"DF":list(t.get("DF") or []),
                      "A":list(t.get("A") or []),"MP":list(t.get("MP") or [])}
    return out
_MIM_SPEC_SHARE_MAX=3   # maximum gene-degree (share) for a marker to count as part of a mimicked program; excludes pan markers (CD3E/PTPRC...)
def _mimicry_terms(idx):
    """lineage-mimicry / transition terms = terms that carry BOTH MIMICS_PHENOTYPE (MP) and an
    identity-anchor (A). Independent of tier (a mimetic TEC is tier=class, EMT is tier=state) — the defining
    property is holding MP + A.
    From these terms _profile_leaves builds per-leaf transition_candidates (origin anchors retained +
    the mimicked program present), so that 'do not force-fit to the mimicked lineage; call origin + state'
    is surfaced as strong evidence (STATE_TRANSITION_DESIGN §engine rules, anchor tiebreaker).
    ★ Qspec = only the SPECIFIC markers of the mimicked program (share<=MAX) — this stops pan-lineage
      markers (CD3D/CD52/IRF8...) from falsely triggering a transition in every cluster of that lineage
      (the same principle as VALID-EVIDENCE).
    Returns {tid:{L,D,A,MP,R,S,DF,P,Qspec}}."""
    terms=((idx or {}).get("terms",{}) or {})
    def _gm(t): return [str(x) for x in (list(t.get("R") or [])+list(t.get("S") or [])+list(t.get("DF") or []))]
    share={}                                            # gene -> #terms listing it (R+S+DF)
    for t in terms.values():
        for g in set(x.upper() for x in _gm(t)): share[g]=share.get(g,0)+1
    def _anc_markers(tid, seen=None):
        seen=seen or set(); acc=set()
        for p in (terms.get(tid,{}).get("P") or []):
            if p in seen or p not in terms: continue
            seen.add(p); acc|= {g.upper() for g in _gm(terms[p])} | _anc_markers(p, seen)
        return acc
    out={}
    for tid,t in terms.items():
        MP=[m for m in (t.get("MP") or []) if isinstance(m,dict) and m.get("lineage")]
        A=list(t.get("A") or [])
        if not (MP and A): continue
        _A={str(x).upper() for x in A}; _anc=_anc_markers(tid)
        def _dedup(seq):
            s=set(); o=[]
            for g in seq:
                u=str(g).upper()
                if u not in s: s.add(u); o.append(g)
            return o
        A=_dedup(A)
        # the mimicked program = a term's own markers that are (not an anchor) AND (low-share specific) AND (not an origin/ancestor marker), deduplicated
        Qspec=_dedup([g for g in _gm(t) if g.upper() not in _A and share.get(g.upper(),99)<=_MIM_SPEC_SHARE_MAX and g.upper() not in _anc])
        out[tid]={"L":t.get("L",tid),"D":(t.get("D") or "normal"),"A":A,"MP":MP,
                  "R":list(t.get("R") or []),"S":list(t.get("S") or []),"DF":list(t.get("DF") or []),
                  "P":list(t.get("P") or []),"Qspec":Qspec}
    # ★ StateAxis-based transitions: a StateAxis (sa) carrying mimics + anchor_markers is also absorbed as a
    #   transition candidate. The StateAxis restructuring moved EMT / EndMT / dedifferentiation from
    #   Term-level A/MP to sa, so they are merged here to keep the false-split guard at leaf judgment time
    #   (an epithelial base plus a mesenchymal program is the EMT state, not an epithelial/mesenchymal
    #   mixture -> do not split). StateAxis.markers is already a curated program signature, so it is used as
    #   Qspec directly (minus the anchors) with no share filter.
    def _dd(seq):
        s=set(); o=[]
        for g in seq:
            u=str(g).upper()
            if u and u not in s: s.add(u); o.append(g)
        return o
    for st in ((idx or {}).get("sa") or []):
        mim=st.get("mimics"); name=st.get("name")
        if not (mim and name): continue
        A=[str(x) for x in (st.get("anchor_markers") or [])]
        prog=[str(x) for x in (st.get("markers") or [])]
        if not (A and prog): continue
        _A={x.upper() for x in A}
        Qspec=_dd([g for g in prog if g.upper() not in _A])
        if not Qspec: continue
        out["SA:"+str(name)]={"L":str(name),"D":"normal","A":_dd(A),
                  "MP":[{"lineage":mim,"program":str(name)}],
                  "R":[],"S":[],"DF":[],"P":[],"Qspec":Qspec,
                  "of":[str(x) for x in (st.get("of") or [])]}
    return out
def _state_terms(idx):
    """Extract every state term in the module (tier=state, or a dev/reactive keyword).
    ★ disease is a hint, not a filter: these terms are always surfaced per leaf as
    disease_candidates even when --disease is unset, so the agent can assign that state when its markers
    dominate (SKILL §7). When disease IS given, they are intersected with _disease_terms and flagged
    hint=True. Returns {tid:{L,D,R,S,DF}}."""
    out={}
    for tid,t in ((idx or {}).get("terms",{}) or {}).items():
        if _is_state_term(t.get("L",""), t.get("T","")):
            out[tid]={"L":t.get("L",tid),"D":(t.get("D") or "normal"),
                      "R":list(t.get("R") or []),"S":list(t.get("S") or []),"DF":list(t.get("DF") or []),
                      "A":list(t.get("A") or []),"MP":list(t.get("MP") or [])}
    return out
def _identity_scores(fon, lf, M, pin=None, frac=0.30, lf_high=2.0, lf_on=1.5, pin_min=0.25, codom_ratio=0.5,
                     auc=None, auc_hi=0.85):
    """Absolute identity score (global reference, fragmentation-invariant).
    fon=(G,) fraction of leaf cells exceeding the global high-threshold (mu+sd);
    lf=(G,) global log2 fold (mean_in/global_mean); pin=(G,) detection rate (fraction of cells >0).
    Per term: n_on = number of its non-pan discriminative markers that are ON. STRONG = n_on>=2 (or one
    very-specific marker lf>=lf_high).

    ★ ON(g) = fon[g]>=frac  OR  (lf[g]>=lf_on AND pin[g]>=pin_min).
       The second (fold) branch rescues 'quiet-but-enriched' markers — exactly the cortical-layer / region
       markers of sparse spatial bins (broadly detected, fold-enriched, yet failing the mu+sd-in-30% gate).
       A naive fold-only loosening was rejected before because it dragged in spillover; here three guards
       make it safe: (a) detection floor pin_min, (b) coherence (a term needs n_on>=2), and (c) the
       co-dominance lineage test below, which suppresses ambient/neuropil bleed.
    ★ rule1 strength>depth: a deep subclass outranks a class only when itself STRONG.
    ★ rule2 class fallback: if no strong subclass, give the dominant lineage's class via lineage markers.
    ★ mixed (chimera): True ONLY when >=2 different lineage-roots are CO-DOMINANT — the runner-up root's
       strength >= codom_ratio * the strongest root's strength. When one lineage clearly dominates and the
       others are weak ambient/neuropil bleed (KDE/bin diffusion), it is NOT a chimera (single identity)."""
    disc=M["disc"]; neg=M["neg"]; tier=M["tier"]; lab=M["lab"]; root=M["root"]; share=M.get("share",{}); defn=M.get("defn",{})
    dzm=M.get("dz",{})
    if pin is None: pin=fon          # pin not provided → fold branch effectively inert (conservative)
    SPEC_SHARE_MAX=3; SPEC_COV_MIN=0.5
    def W(g): return 1.0/max(share.get(g,0),1)                  # marker specificity weight (1/#terms listing it)
    # ★ ON (RC2): detection-based (fon / fold+pin) OR — for a term's OWN low-share SPECIFIC marker
    #   (share≤SPEC_SHARE_MAX) — strong rank-enrichment (auc≥auc_hi). The auc branch rescues quiet-but-enriched
    #   subtype markers in SPARSE spatial data (2µm: Rorb auc 0.98 / Fezf2 0.90 / Dcx 0.72 yet detected <25% so
    #   fold+pin fails) so L4/L5/immature-granule resolve. GUARD: auc path applies ONLY to low-share specific
    #   markers — broad/shared genes (Slc17a7/Pcp4, share>3) still require real detection ⇒ no spillover force-fit.
    def _on(g): return fon[g]>=frac or (lf[g]>=lf_on and pin[g]>=pin_min) or \
                       (auc is not None and auc[g]>=auc_hi and share.get(g,99)<=SPEC_SHARE_MAX)
    # ★ DEFINING-MARKER CANDIDACY. CASE = a type whose discriminative identity rests on a
    #   region-BROAD defining marker (the *type itself* recurs across regions — interneuron neurochemical
    #   classes Sst/Pvalb/Vip, glia, immune, vascular). The pan/region_share filter strips such markers from
    #   `disc` (kept only in `defn`), so the type has too few non-pan `disc` markers to be built as a candidate
    #   → never scored → the cluster collapses to its lineage CLASS even though the DEFINING marker is strongly,
    #   specifically ON (e.g. an Sst+ interneuron cluster, up=Sst/Npy/Cort, fell to bare 'cortical GABAergic
    #   interneuron'; the agent then had no resolvable subtype to split on). RULE = a DEFINING marker encodes
    #   curated sufficiency ("a single defining marker is sufficient") that the region_share heuristic must not
    #   veto: a fold-ON defining marker (a) makes its term a candidate and (b) satisfies coherence on its own
    #   (df_on). The downstream DF-override + VALID-EVIDENCE gate (spec_cov≥0.5, with DF now counted in the spec
    #   set) + dec-competition guard A still bound any leaky-DF abuse — a leaky DF (Wfs1, curated CA1-only but
    #   present in cortex/amygdala) on a non-CA1 leaf has its OTHER specifics OFF ⇒ spec_cov<0.5 ⇒ still nulled,
    #   and its dec is dwarfed by the real competitor ⇒ guard A rejects. (Region-defined types are unaffected:
    #   their DF, e.g. Fezf2/Rorb, is low-share and already in `disc`.)
    # ★ DOMINANT broad-DEFINING marker. A region-BROAD defining marker (share>SPEC_SHARE_MAX —
    #   exactly the marker region_share strips from `disc`: Sst/Pvalb/Vip…, because the *type* recurs across
    #   regions) counts as that term's evidence ONLY when it DOMINATES the cluster, measured by strong vs-rest
    #   enrichment (auc≥auc_hi) i.e. it is a top up-marker — not a weak/ambient fold. This rescues an Sst+
    #   interneuron cluster (Sst = up-top, region-broad DF that was suppressed) while a region-SPECIFIC DF
    #   (low-share Th/Fezf2…) is unaffected (already in `disc`, normal path) and a non-dominant/leaky broad DF
    #   cannot force-fit a specific term (must be a top up-marker). auc unavailable ⇒ inert (no change).
    def _df_dom(tid):
        return [g for g in defn.get(tid,()) if _on(g) and share.get(g,99)>SPEC_SHARE_MAX
                and auc is not None and auc[g]>=auc_hi]
    cand=[]
    for tid,dc in disc.items():
        on=[g for g in dc if _on(g)]
        df_dom=_df_dom(tid)                                              # region-broad DEFINING marker that DOMINATES (vs-rest auc≥auc_hi)
        on_eff=list(dict.fromkeys(on+df_dom))                           # disc-ON ∪ dominant-broad-DEFINING
        if not on_eff: continue
        if not (len(on_eff)>=2 or lf[on_eff[0]]>=lf_high or df_dom): continue   # a dominant broad defining marker alone qualifies
        non=sum(1 for g in neg.get(tid,()) if _on(g) and W(g)>=0.33)     # SPECIFIC negative leak only (ambient broad N ignored)
        spec=round(float(sum(max(0.0,lf[g]) for g in on_eff)),2)
        wspec=round(float(sum(W(g)*max(0.0,lf[g]) for g in on_eff)),3)
        dec=round(float(max(W(g)*max(0.0,lf[g]) for g in on_eff)),3)     # decisive marker = most SPECIFIC×enriched single marker
        cand.append({"term_id":tid,"label":lab[tid],"tier":tier[tid],"dz":dzm.get(tid,"normal"),"n_on":len(on_eff),"df_on":bool(df_dom),
                     "cov":round(len(on_eff)/max(len(dc),1),2),"spec":spec,"wspec":wspec,"dec":dec,"neg":non,"_root":root[tid]})
    # coherent = n_on>=2 OR a dominant broad-DEFINING marker (defining=sufficient); non-state; no specific-neg leak
    coh=[c for c in cand if (c["n_on"]>=2 or c.get("df_on")) and c["neg"]==0 and not _is_state_term(c["label"],c["tier"])]
    # ★ SPECIFIC-EVIDENCE per candidate — computed HERE because both the valid-evidence `top` gate AND the
    #   chimera/co-dominance test below use it (see §5 of SKILL). (SPEC_SHARE_MAX/COV_MIN defined above for _on.)
    def _spec_stats(c):
        # discriminative markers with share <= SPEC_SHARE_MAX plus this term's DOMINANT broad-DEFINING marker
        #   (_df_dom; independent of region_share), so that a type defined by a broad DF (Sst+ interneuron) can
        #   pass spec_pass / spec_ok. Low-share DFs are already in disc. A leaky or non-dominant broad DF has
        #   auc < auc_hi and is excluded, so spec_cov cannot be inflated into a false positive.
        spec=set(g for g in disc.get(c["term_id"],()) if share.get(g,99)<=SPEC_SHARE_MAX) | set(_df_dom(c["term_id"]))
        return len(spec), sum(1 for g in spec if _on(g))
    for c in cand:
        sn,so=_spec_stats(c); c["spec_n"]=sn; c["spec_on"]=so
        c["spec_cov"]=round(so/sn,2) if sn else None
        c["spec_ok"]=bool(sn==0 or so>=1)
        c["spec_pass"]=bool(sn==0 or so/sn>=SPEC_COV_MIN)
    # robust dominant lineage / chimera = root with max SUM of spec across its coherent terms.
    # ★ CHIMERA GATE: count ONLY VALID-EVIDENCE terms (on-panel specific markers actually present —
    #   `spec_pass` AND `spec_n>0`). A force-fit term (shared/leaky markers, `spec_pass=false`) or a
    #   panel-blind term (`spec_n==0`, no on-panel specific evidence) must NOT inflate a second lineage into a
    #   FALSE chimera: a single striatal/region cluster is no longer flagged `mixed` just because it shares
    #   Hpca/Wfs1/Crym with hippocampal glut terms. Fall back to all-coherent ONLY when NO term qualifies
    #   (every specific marker off-panel ⇒ panel-blind: preserve prior behaviour).
    rs={}
    for c in coh:
        if c["spec_pass"] and c["spec_n"]>0: rs[c["_root"]]=rs.get(c["_root"],0.0)+c["spec"]
    if not rs:
        for c in coh: rs[c["_root"]]=rs.get(c["_root"],0.0)+c["spec"]
    dom_s=max(rs.values()) if rs else 0.0
    dom_root=max(rs, key=lambda r:rs[r]) if rs else None
    codom=[r for r,v in rs.items() if dom_s>0 and v>=codom_ratio*dom_s]
    # ★ DF-override: if a term IN the dominant lineage has a DEFINING marker that is fold-ON, it wins
    #   ("a single defining marker is sufficient"). Restricted to dom_root → no cross-lineage hijack.
    defn=M.get("defn",{})
    def _df_fold(tid): return max([lf[g] for g in defn.get(tid,()) if _on(g) and lf[g]>=lf_on], default=None)
    _key=lambda c:(c["dec"],c["wspec"],c["n_on"])
    # SPECIFIC-EVIDENCE per candidate (`spec_n`/`spec_on`/`spec_cov`/`spec_ok`/`spec_pass`) was computed above,
    # before the chimera test which also uses it. Semantics (also in SKILL §5):
    #   spec_n   = the term's on-panel high-specificity (share<=SPEC_SHARE_MAX) discriminative markers
    #   spec_on  = how many are ON ; spec_cov = spec_on/spec_n
    #   spec_ok  = (spec_on>=1 OR spec_n==0)              — DF-guard-B corroboration
    #   spec_pass= (spec_cov>=SPEC_COV_MIN OR spec_n==0)  — VALID-EVIDENCE gate (>= half its specifics present)
    # A leaky marker carried as plain R/DF (Wfs1, share=6) is OUT of the spec set, so a NON-hippocampal leaf
    # sharing Wfs1/Crym scores spec_cov~0-0.33 for CA1 (Fibcd1/Spink8 OFF) vs ~1.0 for a real CA1. spec_n==0
    # (all specific markers off-panel) ⇒ gate INERT (panel-blind: judge by supporting).
    # GUARD: DF-override only among COHERENT terms (n_on>=2, neg==0, non-state). A lone defining marker
    # firing on ambient/neuropil signal in an over-split fragment (e.g. Rgs14→CA2) must NOT confirm a type
    # without ≥1 corroborating marker — this prevents spurious DF over-calls (the 8µm CA2 balloon).
    # GUARD A+B: a LEAKY defining marker (curated as one term's DF/specific in the ontology but
    # biologically also expressed elsewhere — e.g. Wfs1 listed only on CA1 yet present in cortex/amygdala)
    # must not hijack a higher-evidence competitor. A DF-override candidate is kept ONLY if CORROBORATED:
    #   (A) dec-competition — its decisive marker is >= DF_DEC_FRAC × the best coherent same-lineage
    #       competitor's dec (a DF whose dec is dwarfed — CA1 dec 0.31 vs hilar/L2-3 2.6/2.0 on a cortical
    #       leaf — is not actually 'defining' here).
    #   (B) specific-support — the DF term itself must have >=1 of its OWN specific markers ON (spec_ok).
    DF_DEC_FRAC=0.34
    _best_dom_dec=max((c["dec"] for c in coh if c["_root"]==dom_root), default=0.0)
    def _df_corroborated(c):
        if _best_dom_dec>0 and c["dec"]<DF_DEC_FRAC*_best_dom_dec: return False          # (A)
        return c["spec_ok"]                                                              # (B)
    df_hits=[c for c in coh if c["_root"]==dom_root and _df_fold(c["term_id"]) is not None
             and _df_corroborated(c)]
    if df_hits:
        top=max(df_hits, key=lambda c:(_df_fold(c["term_id"]), c["dec"], c["wspec"]))
    else:
        # else DECISIVE-marker best WITHIN the dominant lineage (unique specific marker beats many shared ones)
        inlin=[c for c in coh if c["_root"]==dom_root]
        if inlin:  top=max(inlin, key=_key)
        elif coh:  top=max(coh, key=_key)
        else:
            nonst=[c for c in cand if not _is_state_term(c["label"],c["tier"])]
            top=max(nonst, key=_key) if nonst else (cand[0] if cand else None)
    # ★ VALID-EVIDENCE GATE: a SPECIFIC (non-class) `top` must have valid positive evidence — at least
    #   SPEC_COV_MIN of its OWN specific markers actually ON (spec_pass). A term whose specific markers are
    #   mostly OFF is a force-fit through shared/leaky markers, NOT a real identity (a non-hippocampal leaf
    #   landing on CA1 because it shares Wfs1; a non-CA3 leaf on CA3 via one shared marker) → drop `top` to
    #   None so the leaf falls to its lineage CLASS (`class_fallback`, Step 3) or Unknown (Step 4) — "specific
    #   only with valid evidence; otherwise don't commit". INERT for a class-tier top and for panel-blind
    #   terms (spec_n==0). The rejected candidate stays visible in `ranked` (spec_pass=false) so the agent can
    #   rescue it if `hp`/supporting clearly show the type IS present.
    if top is not None and top.get("tier")!="class" and not top["spec_pass"]:
        top=None
    # ranked: valid-evidence terms (spec_pass) surface first, then dominant-lineage, then decisiveness
    ranked=(sorted(coh, key=lambda c:(c["spec_pass"], c["_root"]==dom_root, c["dec"], c["wspec"]), reverse=True)[:5]
            or sorted(cand, key=lambda c:(c["spec_pass"], c["dec"], c["wspec"], c["n_on"]), reverse=True)[:5])
    if top is not None and top not in ranked: ranked=[top]+ranked[:4]
    # ★ NO STATE FALLBACK. disease/injury/metaplastic/malignant identities are ordinary celltype nodes
    #   (tier=class/subclass) with their OWN discriminative markers — they compete at the SAME level as normal
    #   types in `top`/`ranked` above, chosen when their markers WIN, never as a "before-floor" fallback.
    #   (`_is_state_term` is tier=='state' and there are no tier='state' Terms; the old state-visibility /
    #   disease_candidates path is retired. Base-agnostic PROGRAMS live in `sa` (state_axis) and are appended
    #   only in the interactive phase.) is_state is kept False for backward-compatible readers.
    for c in cand: c["is_state"]=False
    lin_idx=M["lin_idx"]; lin_class=M["lin_class"]; linbest=None
    for lx,idxs in lin_idx.items():
        c=sum(1 for g in idxs if _on(g))
        if c>=1 and (linbest is None or c>linbest[1]): linbest=(lx,c)   # >=1 lineage marker → class fallback
    classfb=({"term_id":lin_class[linbest[0]],"label":linbest[0]} if linbest and linbest[0] in lin_class else None)
    for c in cand: c.pop("_root",None)
    return {"top":top,"ranked":ranked,"class_fallback":classfb,
            "mixed":bool(len(codom)>=2),"n_strong_lineages":len(rs),
            "dominant_lineage":(lab.get(dom_root,dom_root) if dom_root else None),
            "codominant_lineages":[lab.get(r,r) for r in codom]}

# ===================================================================== profile core
def _rankcols(M): return np.asarray(rankdata(M, method="average", axis=0), dtype=np.float64)

def _exclusive_mixture(Xleaf, IDM, octx, lineage_label, genes_arr, split_frac=0.15, min_cells=80, hi_thr=None):
    """★ MUTUAL-EXCLUSIVITY mixture (CASE+RULE, no hardcoding).
    CASE: the leaf stopped at one lineage CLASS (class_floor/ambiguous), yet two sibling subtypes of that
      lineage are MUTUALLY EXCLUSIVE — X's positive (R/DF) marker is Y's NEGATIVE marker and vice versa.
      A single cell cannot carry both (CD4 vs CD8: CD4SP R=CD4, neg=CD8 / CD8SP R=CD8, neg=CD4), so when the
      cluster mixes X+Y- cells with X-Y+ cells, the cluster mean shows both as 'weakly present' (they dilute
      each other) -> a MEAN-based top/mixture test misses it and the leaf falls to class_floor (the signal of
      a mixture is not that one side is strong, but that both are low and similar).
    RULE: derive the mutually exclusive pair from the KG structure automatically
      (X.pos n Y.neg != {} AND Y.pos n X.neg != {}, same-lineage siblings).
      Classify each cell of the leaf PER-CELL as X-side (X marker+, Y marker-) or Y-side (Y+, X-):
        - both exclusive groups >= split_frac -> verdict 'split' (a genuine X+Y mixture -> needs_subclustering)
        - only one >= split_frac              -> 'dominant:<label>' (call the dominant subtype; the absence of
                                                 the other side is the evidence — even with CD4 dropout, the
                                                 absence of CD8 confirms CD4SP)
        - neither is sufficient (continuum/dropout) -> not triggered.
    Complements [[_bimodal_split]] (which uses only low-share positive markers): here reciprocal pos<->neg
    also catches high-share lineage markers (CD4/CD8). Purely ADDITIVE (existing triggers unchanged).
    Returns [{pair,markers,frac_exclusive,verdict}] or None."""
    terms=octx.get("terms",{}); lab=IDM["lab"]; root=IDM["root"]; tier=IDM["tier"]
    cls=IDM["lin_class"].get(lineage_label)
    if cls is None: return None
    cls_root=root.get(cls,cls)
    sibs=[t for t in terms if root.get(t)==cls_root and tier.get(t)!="class"
          and (terms[t].get("D") or "normal")=="normal"]
    U=lambda xs:{str(g).upper() for g in (xs or [])}
    # exclusive discriminators = DEFINING markers only (DF, curated as 'one is sufficient'). required (R)
    #   includes maturation-stage shared markers such as RAG1 and would create false pairs that are not
    #   lineage-exclusive (CD8 vs RAG1-immature) -> restricting to DF keeps only genuine lineage discriminators.
    pos={t:U(terms[t].get("DF")) for t in sibs}
    neg={t:U(terms[t].get("N")) for t in sibs}
    pairs=[]
    for i in range(len(sibs)):
        for j in range(i+1,len(sibs)):
            X,Y=sibs[i],sibs[j]
            mx=pos[X]&neg[Y]; my=pos[Y]&neg[X]
            if mx and my: pairs.append((X,Y,sorted(mx),sorted(my)))
    if not pairs: return None
    if hasattr(Xleaf,"toarray"): Xleaf=Xleaf.toarray()
    Xleaf=np.asarray(Xleaf)
    if Xleaf.shape[0]<min_cells: return None
    gidx={str(g).upper():i for i,g in enumerate(list(genes_arr))}
    out=[]
    for X,Y,mx,my in pairs:
        xi=[gidx[g] for g in mx if g in gidx]; yi=[gidx[g] for g in my if g in gidx]
        if not xi or not yi: continue
        # ★ ENRICHMENT-based ON: plain detection (>0) also counts noise (trace bleed) as ON on a sparse KDE
        #   vector field and manufactures fake mixtures (cl17: Wfs1/Prox1 have absolute means of 0.001 yet are
        #   'detected' in 19/20% of cells -> a fake CA1+DG split). When hi_thr (the global gmu+gsd, the same ON
        #   definition used by fon) is supplied, only values above it count as ON — removing the noise.
        if hi_thr is not None:
            xon=(Xleaf[:,xi]>=np.asarray(hi_thr)[xi]).any(axis=1); yon=(Xleaf[:,yi]>=np.asarray(hi_thr)[yi]).any(axis=1)
        else:
            xon=(Xleaf[:,xi]>0).any(axis=1); yon=(Xleaf[:,yi]>0).any(axis=1)
        fX=float((xon&~yon).mean()); fY=float((~xon&yon).mean()); fBoth=float((xon&yon).mean())
        # ★ Genuine mutual-exclusivity guard: double-positive cells (expressing both) must be rarer than the
        #   exclusive groups. When the markers frequently co-occur (CCR7 naive vs CD4: naive CD4 is CCR7+ CD4+),
        #   the pair is not truly exclusive even if pos n neg holds structurally -> excluded.
        if fBoth >= 0.5*min(fX, fY): continue
        # SPLIT only: both exclusive groups sufficiently large = a genuine X+Y mixture. If only one side is
        #   (a developmental continuum or a single identity), do not fire — that is not a mixture but 'which
        #   single one', which normal scoring handles (over-issuing dominant would be noise).
        if fX>=split_frac and fY>=split_frac:
            out.append({"pair":[lab.get(X,X),lab.get(Y,Y)],"markers":[mx,my],
                        "frac_exclusive":[round(fX,2),round(fY,2)],"verdict":"split",
                        "_m":min(fX,fY)})
    out.sort(key=lambda d:-d.pop("_m"))
    return out[:3] or None

def _profile_leaves(X, genes, leaf_arr, leaves, ont_gene_set, ont_protected,
                    octx, vf_norm=None, vf_low_thresh=None,
                    auc_up=0.6, auc_dn=0.4, top_hp=50, parent_up_auc=None,
                    gmu=None, gsd=None, restrict_root=None, dz_terms=None, st_terms=None, mt_terms=None, profile_cap=0, auc_pos=0.65, absent_rg=None,
                    adaptive_floor=False, af_F=3.0, af_Nmin=10):
    """Per-leaf profile. up/dn unbounded (AUC threshold only), hp=50.
    identity_evidence = density(vf_norm) + specificity_peak + lineage_markers_present +
    housekeeping_share + cross_lineage."""
    # ★ Stratified subsample (speed): computing the AUC rank over all cells (rankdata axis=0) takes ~11 min
    #   at 650k and is the pipeline's biggest bottleneck. Markers are a population signal, so the AUC is
    #   stable once a leaf has enough cells -> above profile_cap, subsample per leaf in a stratified way
    #   (small leaves are kept in full, only large leaves shrink proportionally). The global gmu/gsd are
    #   computed by init on the FULL data and passed in, so the absolute-identity reference is unchanged.
    #   The seed is fixed for determinism. profile_cap=0 means full (accuracy first).
    leaf_arr=np.asarray(leaf_arr); _n_full=X.shape[0]
    # ★ The TRUE leaf size (before subsampling) is preserved -> entry["n"] reports the real cell count, which
    #   keeps the agent's absolute-count reasoning and rare-type sanity checks valid. Statistics (AUC, pin,
    #   mean ...) are computed on the subsample, but the reported n / n_out are TRUE. (The finalize gate is
    #   unaffected: it uses the state's TRUE cluster_paths separately.)
    _lv,_lc=np.unique(leaf_arr, return_counts=True); _true_n={k:int(v) for k,v in zip(_lv.tolist(),_lc.tolist())}
    if profile_cap and _n_full>profile_cap:
        _MINKEEP=2500; _rng=np.random.default_rng(0); _frac=profile_cap/_n_full; _parts=[]
        for _l in np.unique(leaf_arr):
            _il=np.nonzero(leaf_arr==_l)[0]
            if _il.size<=_MINKEEP: _parts.append(_il)
            else:
                _k=max(_MINKEEP, int(round(_il.size*_frac)))
                _parts.append(_rng.choice(_il, min(_k,_il.size), replace=False))
        _sel=np.sort(np.concatenate(_parts))
        X=np.ascontiguousarray(X[_sel]); leaf_arr=leaf_arr[_sel]
        if vf_norm is not None: vf_norm=np.asarray(vf_norm)[_sel]
        print(f"[profile] stratified subsample {len(_sel)}/{_n_full} cells (cap={profile_cap}, MINKEEP={_MINKEEP}) — speeds up AUC/rank", flush=True)
    n_cells, n_genes = X.shape
    genes_arr=np.asarray(genes); ranks=_rankcols(X)
    gidx_U={str(g).upper():i for i,g in enumerate(genes_arr.tolist())}
    leaf_to_idx={l:k for k,l in enumerate(leaves)}
    cli=np.fromiter((leaf_to_idx.get(p,-1) for p in leaf_arr), dtype=np.int64, count=n_cells)
    valid=cli>=0; rows=cli[valid]; cols=np.nonzero(valid)[0]; K=len(leaves)
    M=csr_matrix((np.ones(rows.size),(rows,cols)), shape=(K,n_cells))
    n_in_per=np.asarray(M.sum(axis=1)).ravel()
    binX=(X>0).astype(np.float64)
    sumR=M@ranks; sumX=M@X; sumBin=M@binX
    totX=X.sum(axis=0); totBin=binX.sum(axis=0)
    auc_mat=np.zeros((K,n_genes)); mean_in_mat=np.zeros((K,n_genes)); mean_out_mat=np.zeros((K,n_genes))
    for k in range(K):
        ni=n_in_per[k]; no=n_cells-ni
        if ni==0 or no==0: continue
        auc_mat[k]=(sumR[k]-ni*(ni+1)/2.0)/(ni*no); mean_in_mat[k]=sumX[k]/ni; mean_out_mat[k]=(totX-sumX[k])/no
    # 3-layer HK
    hk1=set(g for g in genes if _is_hk(g))
    max_auc=auc_mat.max(axis=0) if K>0 else np.zeros(n_genes)
    hk2=_data_driven_hk(mean_in_mat, max_auc, genes_arr)
    hk=(hk1|hk2)-(ont_protected or set())
    hk_mask=np.array([g in hk for g in genes_arr])
    # ontology gene index (case-insensitive panel matching — SKILL contract "gene matching is
    # case-insensitive throughout"; mirrors the .upper() normalization at :180/:1774/:1836)
    gidx_U={str(g).upper():i for i,g in enumerate(genes_arr.tolist())}
    panelU=set(gidx_U)
    ont_on=sorted(g for g in (ont_gene_set or set()) if str(g).upper() in panelU)
    ont_off=sorted(g for g in (ont_gene_set or set()) if str(g).upper() not in panelU)
    ont_on_idx=np.array([gidx_U[str(g).upper()] for g in ont_on], dtype=np.int64) if ont_on else np.array([],dtype=np.int64)

    # absolute identity (global reference) precomputed: the global high threshold THR = gmu+gsd, and per leaf
    # frac_on = (M@HI)/n_in.
    # gmu/gsd are computed by init over the full data and persisted -> identical in every round
    # (invariant to fragmentation).
    IDM=None; _gpin_arr=_ghi_arr=None
    if gmu is not None and gsd is not None:
        IDM=_build_ident_maps(octx, genes_arr)
        gmu_arr=np.asarray(gmu,dtype=np.float64); gsd_arr=np.asarray(gsd,dtype=np.float64)
        sumHI=M@((X>=(gmu_arr+gsd_arr)).astype(np.float64))
        if adaptive_floor:   # ★ global baseline for the data-derived floor (reference-free): global detection rate / strong-expression rate (computed once)
            _gpin_arr=np.asarray(totBin,dtype=np.float64).ravel()/max(n_cells,1)
            _ghi_arr=np.asarray((X>=(gmu_arr+gsd_arr)).sum(axis=0),dtype=np.float64).ravel()/max(n_cells,1)
            print(f"[adaptive-floor] ON (F={af_F}, Nmin={af_Nmin}) — region-gated positive floor = min(fixed, ambient×F); "
                  f"median gpin={np.median(_gpin_arr):.4f} ghi={np.median(_ghi_arr):.4f}", flush=True)

    # union of all state-program markers on this panel → per-leaf mean stored for the WITHIN-BASE state gate (finalize)
    _sa_gi={str(_g).upper():gidx_U[str(_g).upper()] for _stx in (octx.get("sa") or []) for _g in (_stx.get("markers") or []) if str(_g).upper() in gidx_U}
    profiles={}
    for k,path in enumerate(leaves):
        ni=int(n_in_per[k]); no=n_cells-ni
        if ni==0 or no==0: continue
        auc=auc_mat[k]; mi=mean_in_mat[k]; mo=mean_out_mat[k]
        pin=sumBin[k]/ni
        order=np.argsort(mi)[::-1]
        rank_of=np.empty(n_genes,dtype=np.int64); rank_of[order]=np.arange(1,n_genes+1)
        up=[[genes_arr[i],round(float(auc[i]),3),round(float(mi[i]),3),round(float(mo[i]),3)]
            for i in np.argsort(auc)[::-1] if auc[i]>=auc_up]
        dn=[[genes_arr[i],round(float(auc[i]),3),round(float(mi[i]),3),round(float(mo[i]),3)]
            for i in np.argsort(auc) if auc[i]<=auc_dn]
        hp=[]
        for i in order:
            if hk_mask[i]: continue
            hp.append([genes_arr[i],round(float(mi[i]),3),int(rank_of[i]),round(float(pin[i]),3)])
            if len(hp)>=top_hp: break
        # ontology_gene_table
        ogt={}
        for j,i in enumerate(ont_on_idx):
            ogt[ont_on[j]]={"rank":int(rank_of[i]),"mean":round(float(mi[i]),3),"auc":round(float(auc[i]),3),"panel":True}
        for g in ont_off: ogt[g]={"rank":None,"mean":None,"auc":None,"panel":False}
        # flags (lineage / specific / sibling come from onto_ctx — ontology-derived, organ-agnostic)
        LINS=octx["lineages"]; SPEC=octx["specific"]; PAN=octx["pan"]
        up_genes={r[0] for r in up}; up_up={_U(g) for g in up_genes}
        multi=[ln for ln,gs in LINS.items() if len(up_up&gs)>=2]
        # subclass_mixing: >=2 sibling subclasses share a parent and each has >=2 required markers strong in up
        layer_mix=[p for p,ch in octx["siblings"].items()
                   if sum(1 for c in ch if len(octx["term_req"].get(c,set())&up_up)>=2)>=2]
        has_spec=bool(up_up&SPEC)
        any_class=any(len(up_up&gs)>=2 for gs in LINS.values())
        pan_class=(len(multi)<=1 and any_class and not has_spec)
        n_hk15=int(sum(1 for i in order[:15] if hk_mask[i])); bgscore=round(n_hk15/15.0,3)
        nonhk_order=[i for i in order if not hk_mask[i]]
        sp_peak=None
        if nonhk_order:
            best=max(nonhk_order, key=lambda i: auc[i])
            if auc[best]>=auc_up:
                sp_peak=[genes_arr[best],round(float(auc[best]),3),round(float(mi[best]),3),round(float(mo[best]),3)]
        lin_present=[g for g in ont_on if (ogt[g]["mean"] is not None) and pin[gidx_U[str(g).upper()]]>=0.4
                     and g.upper() not in PAN]
        # housekeeping_share + cross_lineage
        cross={"multi_lineage":multi,"n_in":ni,"specific_id_rescue":has_spec}
        # density (vf_norm)
        if vf_norm is not None:
            lv=vf_norm[valid][rows==k] if False else vf_norm[np.asarray(leaf_arr)==path]
            if lv.size:
                med=float(np.median(lv))
                pct=float((vf_norm<med).mean()*100)
                low=float((lv<vf_low_thresh).mean()) if vf_low_thresh is not None else None
                density={"median":round(med,4),"pct":round(pct,1),"low_frac":round(low,3) if low is not None else None}
            else: density=None
        else:
            density=None
        identity={"density":density,"specificity_peak":sp_peak,
                  "lineage_markers_present":{"n":len(lin_present),"genes":lin_present[:12]},
                  "housekeeping_share":bgscore,"cross_lineage":cross}
        # sibling_contrast
        sib=[]
        if K>=2:
            pct_leaf=sumBin/np.maximum(n_in_per[:,None],1.0)
            other=np.arange(K)!=k; mx=pct_leaf[other].max(axis=0); delta=pct_leaf[k]-mx
            for i in np.argsort(delta)[::-1]:
                if delta[i]<=0.1: break
                sib.append([genes_arr[i],round(float(pct_leaf[k][i]),2),round(float(mx[i]),2),round(float(delta[i]),3)])
                if len(sib)>=15: break
        # lineage_scan (#2): canonical lineage markers shown regardless of up/hp cutoff,
        # so moderate-AUC lineage signal is never invisible. case-insensitive panel match.
        gU={str(g).upper():i for i,g in enumerate(genes_arr)}
        lscan={}
        for ln,gs in LINS.items():
            hits=[[genes_arr[gU[g]], round(float(auc[gU[g]]),3), round(float(pin[gU[g]]),3)]
                  for g in gs if g in gU and pin[gU[g]]>=0.1]
            if hits: lscan[ln]=sorted(hits, key=lambda r:-r[1])
        # candidate_subtypes (a scorecard of side candidates + a one-level child lookahead) — lets the agent
        # review every subtype such as CA1/CA3 without relying on memory, and decide whether to split the chosen
        # term further.
        # region_share (region-aware discriminativeness, the same as in _build_ident_maps) is converted to symbol
        #   keys and passed in, so candidate_subtypes judges discriminativeness the same way graph_evidence's
        #   anchors do. None when there is no IDM (-> pan_disc fallback).
        _rs_sym=({_U(IDM["genes"][g]):c for g,c in (IDM.get("region_share") or {}).items() if 0<=g<len(IDM.get("genes") or [])}
                 if IDM is not None else None)
        cands, sig_targets=_candidate_subtypes(octx.get("terms",{}), ogt, up, octx, parent_up_auc=parent_up_auc, region_share=_rs_sym)
        # subtype_groups: when *>=2 sibling subtypes sharing the same parent coexist* among signal_targets, that
        # is a direct signal that this leaf can be split one level further into the subtypes under that parent
        # (used for maximum-depth recursion). E.g. if CA1 deep (Bcl6) and CA1 superficial (Calb1) are both up ->
        # {parent: CA1, children: [deep, superficial]}.
        terms_=octx.get("terms",{}); _grp={}
        _lab2tid={s["label"]:s["term_id"] for s in sig_targets}
        for s in sig_targets:
            for par in ((terms_.get(s["term_id"]) or {}).get("P") or []):
                if par in terms_: _grp.setdefault(par,[]).append(s["term_id"])
        subtype_groups=[{"parent_id":par,"parent_label":terms_[par].get("L",""),
                         "child_term_ids":ch,"children":[terms_[c].get("L","") for c in ch]}
                        for par,ch in _grp.items() if len(ch)>=2]
        _tn=_true_n.get(path, ni)
        entry={"n":_tn,"n_out":_n_full-_tn,"n_sub":ni,"up":up,"dn":dn,"hp":hp,"ontology_gene_table":ogt,
               "identity_evidence":identity, "lineage_scan":lscan, "candidate_subtypes":cands,
               "signal_targets_suggested":sig_targets, "subtype_groups":subtype_groups,
               "flags":{"multi_lineage":multi,"layer_mixing":layer_mix,"pan_class_only":bool(pan_class),"background_score":bgscore},
               "sibling_contrast":sib}
        if no<MIN_N_OUT: entry["warn"]="n_out_too_small"
        # absolute identity_scores (global reference, invariant to fragmentation) — the core evidence behind the Leiden + absolute judgment
        if IDM is not None:
            fon=sumHI[k]/max(ni,1); lf=np.log2((mi+1e-9)/(gmu_arr+1e-9))
            entry["identity_scores"]=_identity_scores(fon, lf, IDM, pin=pin, auc=auc)
            # ★ per-leaf data-derived floors (opt-in): min(fixed, ambient×F) with Nmin/ni cell-count guard.
            #   None(off) ⇒ on_pos predicate unchanged. Only relaxes (never above fixed 0.25/0.30).
            _pf=_ff=None
            if adaptive_floor and _gpin_arr is not None:
                _c=af_Nmin/max(ni,1)
                _pf=np.maximum(np.minimum(0.25,_gpin_arr*af_F),_c)
                _ff=np.maximum(np.minimum(0.30,_ghi_arr*af_F),_c)
            # ★ graph-descent EVIDENCE — the redesigned primary identity substrate (the agent reasons over
            #   the KG it built). identity_scores is kept only as a supplementary absolute-fold summary.
            entry["graph_evidence"]=_graph_evidence(fon, lf, pin, IDM, octx, mean=mi, auc=auc, restrict_root=restrict_root, auc_pos=auc_pos, absent_rg=absent_rg, pin_floor=_pf, fon_floor=_ff)
            # Processes are not labelled on the clustering-based path, and process detection is not part of this
            #  release.
            # DF-confirmed terms in this leaf (a DEFINING marker is fold-ON: log2fold>=1.5 AND detected>=25%).
            # Used by the finalize split-review gate (§8.1) + DF_MULTI_SUBTYPE to flag splits with a real 2nd identity.
            # ★ LINEAGE-GATED: only terms belonging to the leaf's dominant lineage (graph_evidence.lineage) go into
            #   df_present. Why: a shared DEFINING marker (CD4 is the DF of CD4 T cell but is also expressed by
            #   myeloid mono/macro/DC) picked up regardless of lineage would let a myeloid cluster put "CD4 T cell"
            #   into df_present and contaminate DF_MULTI_SUBTYPE / the gate. The lineage gate blocks the myeloid
            #   contamination while a genuine CD4(DF) + CD8B(DF) coexistence within the T lineage (a real CD4/CD8
            #   mixture) is still detected. (It also removes cross-lineage false split signals such as ambient oligo
            #   DFs on a glut leaf.) When lineage is None (no_lineage) the gate is not applied (conservative
            #   fallback). DF-override anchoring is unaffected — it is already gated by dom_root.
            _defn=IDM.get("defn",{}); _lab=IDM.get("lab",{}); _root=IDM.get("root",{})
            _lin_root=IDM.get("lin_class",{}).get((entry.get("graph_evidence") or {}).get("lineage"))
            _dfp=[tid for tid,gs in _defn.items() if any(lf[g]>=1.5 and pin[g]>=0.25 for g in gs)]
            if _lin_root is not None:
                _lr=_root.get(_lin_root,_lin_root); _dfp=[tid for tid in _dfp if _root.get(tid)==_lr]
            entry["df_present"]=sorted({_lab.get(tid,tid) for tid in _dfp})
            # ★ BIMODAL-SPLIT backstop: per-cell anti-correlated sibling programs that the
            #   cluster-level mixture detector missed (balanced 50/50 mix of sibling-region subtypes —
            #   arterial/venous EC, alveolar/adventitial fibroblast). Only when NO subtype confidently
            #   anchored (verdict≠clear_top) AND mixture_subtypes empty (the case it missed). Region-
            #   independent, so additive (existing triggers unchanged); finalize gate consumes it.
            _ge_b=entry.get("graph_evidence") or {}
            if (_ge_b.get("lineage") and not _ge_b.get("mixture_subtypes")
                    and _ge_b.get("verdict") in ("ambiguous_tie","class_or_unknown","confusion_unresolved")):
                _bim=_bimodal_split(X[np.asarray(leaf_arr)==path], IDM, octx, _ge_b.get("lineage"), genes_arr)
                if _bim: entry["bimodal_subtypes"]=_bim
            # ★ MUTUAL-EXCLUSIVITY mixture — detects a per-cell mixture of mutually exclusive marker pairs (CD4/CD8
            #   etc.). Evaluated whenever the lineage is known (regardless of verdict), which rescues a same-lineage
            #   subtype mixture or dominance that fell to class_floor. ADDITIVE (derived automatically from KG
            #   reciprocal pos<->neg, no hardcoding).
            _exm_lin=(entry.get("identity_scores") or {}).get("dominant_lineage") or _ge_b.get("lineage")
            if _exm_lin:                                    # prefer dominant_lineage (the graph lineage is perturbed by the transmitter gate)
                _Xl=X[np.asarray(leaf_arr)==path]
                _hi=(gmu_arr+gsd_arr) if (gmu is not None and gsd is not None) else None
                _exm=_exclusive_mixture(_Xl, IDM, octx, _exm_lin, genes_arr, hi_thr=_hi)
                if _exm: entry["exclusive_mixture"]=_exm
            # ★ CONTEXT-AXIS (§4.3) — DETERMINISTIC zonation/layer subtyping. When the lineage matches the applies_to
            #   of a ca axis declared by the module, score the per-pole context_markers -> if mixed (>=2 poles present)
            #   the parent becomes a split target; on a sub-leaf, propose dominant_pole -> suggested_subtype. The
            #   lineage is fixed by graph_evidence and never changed here.
            _ge_ca=entry.get("graph_evidence") or {}
            _ca_lin=_ge_ca.get("lineage")
            # labels used for region/anchor-aware matching: dominant_region + anchor labels + the anchor's ancestor path labels
            _ca_extra=set(_ge_ca.get("dominant_region") or [])
            for _an in (_ge_ca.get("anchors") or []):
                if _an.get("label"): _ca_extra.add(_an["label"])
                for _pid in (_an.get("path") or []):
                    _pl=IDM["lab"].get(_pid) if IDM else None
                    if _pl: _ca_extra.add(_pl)
            _ca=_context_axis_eval(_ca_lin, octx.get("ca") or [], fon, lf, pin, gidx_U, extra_labels=_ca_extra)
            if _ca:
                entry["context_axis"]=_ca
                # flag: >=2 poles present (mixed) -> subcluster the parent per §4.3; when a dominant_pole exists, that subtype is the target
                entry["flags"]["context_axis_mixed"]=any(a.get("mixed") for a in _ca)
                entry["flags"]["context_axis_pole"]=next((a.get("dominant_pole") for a in _ca if a.get("dominant_pole")), None)
            # ★ STATE-AXIS (base-agnostic PROGRAMS: proliferating / IFN / hypoxic …) — each judged
            #   independently (≥80% of its on-panel program markers ENRICHED). EVIDENCE ONLY: standard
            #   annotation never appends these; the INTERACTIVE phase (§8) reads state_axis to subdivide a
            #   base by state on user request. Not an identity, not a split trigger.
            _sax=_state_axis_eval(octx.get("sa") or [], fon, lf, pin, gidx_U)
            if _sax:
                entry["state_axis"]=_sax
            if _sa_gi:                                     # per-leaf mean of every state marker → within-base gate (interactive finalize)
                entry["state_marker_means"]={_g: round(float(mi[_i]),4) for _g,_i in _sa_gi.items()}
            # ★ NO disease_candidates / R7 fallback. disease/injury/metaplastic/malignant cells are ordinary
            #   celltype nodes (tier=class/subclass) with their OWN discriminative markers, scored at the SAME
            #   level as normal types in identity_scores (top/ranked) — chosen when their markers win, never as
            #   a before-floor fallback. (The old st_terms/dz_terms path required tier='state' Terms or a
            #   disease property, of which the KG now has none; it is retired.)
            # ★ transition_candidates — the anchor tiebreaker (STATE_TRANSITION_DESIGN §engine rules,
            #   evidence-strong). For a lineage-mimicry term (carrying MP + A, regardless of tier): if the origin
            #   identity-anchor (A) markers are still present AND the mimicked program (the non-anchor markers of
            #   R/S/DF) is present at the same time, then this leaf is not the mimicked lineage but 'the origin lineage
            #   in a transition state'. Surfaced as a top-priority signal to prevent a force-fit or chimera call
            #   toward the mimicked lineage.
            if mt_terms:
                def _on_g(_g):
                    _gi=gidx_U.get(str(_g).upper())
                    if _gi is None: return False
                    return fon[_gi]>=0.30 or (lf[_gi]>=1.5 and pin[_gi]>=0.25)
                # ★ origin-lineage gate: a transition fires only when the transition term's origin lineage equals the
                #   leaf's dominant lineage (it is 'a transition state of that origin' only if the origin matches, not the
                #   mimicked target). A mismatch would fire across lineages (an mTEC term firing on an endothelial or T
                #   cluster). When the leaf's lineage is undetermined (None) the gate is not applied (conservative).
                _geL=(entry.get("graph_evidence") or {}).get("lineage") if IDM is not None else None
                _leaf_root=IDM["root"].get(IDM.get("lin_class",{}).get(_geL)) if (IDM is not None and _geL) else None
                trc=[]
                for _tid,_info in mt_terms.items():
                    if _leaf_root is not None and IDM is not None:
                        _troot=IDM["root"].get(_tid)
                        if _troot is not None and _troot!=_leaf_root: continue   # origin lineage ≠ leaf lineage
                    anchor_on=[g for g in _info.get("A",[]) if _on_g(g)]
                    if not anchor_on: continue
                    spec_on=[g for g in _info.get("Qspec",[]) if _on_g(g)]
                    if len(spec_on)<2: continue                      # ★ fires only when >=2 specific mimicked-program markers are ON (blocks false firing from pan markers)
                    trc.append({"term_id":_tid,"label":_info["L"],"D":_info["D"],
                                "anchor_retained":anchor_on,"acquired_on":spec_on[:8],
                                "mimics":[{"lineage":m.get("lineage"),"program":m.get("program","")} for m in _info.get("MP",[])],
                                "n_anchor":len(anchor_on),"n_acquired":len(spec_on)})
                if trc:
                    trc.sort(key=lambda d:(-d["n_anchor"], -d["n_acquired"]))
                    entry["transition_candidates"]=trc[:6]
                    _mlin=sorted({m["lineage"] for d in trc for m in d["mimics"] if m.get("lineage")})
                    entry["transition_note"]=("identity anchors retained -> transition_candidates is the origin lineage in a transition state. "
                        "Do not call it purely as the mimicked lineage ("+", ".join(_mlin)+"); the anchor markers are evidence of the origin identity.")
        profiles[path]=entry
    return profiles

def _write_overview(out, rnd, profiles):
    """#1 cross-cluster overview: compare every leaf of a round side by side (restores the expert's
    'table comparison')."""
    rows={}
    for path,e in profiles.items():
        ls=e.get("lineage_scan",{})
        dom=sorted(ls.items(), key=lambda kv:-max((r[1] for r in kv[1]), default=0))
        sp=e.get("identity_evidence",{}).get("specificity_peak")
        rows[path]={"n":e["n"],
                    "dominant_lineages":[k for k,_ in dom[:2]],
                    "top_up":[r[0] for r in (e.get("up") or [])[:6]],
                    "specificity_peak":(sp[0] if sp else None)}
    json.dump(rows, open(os.path.join(out,f"cluster_overview_round{rnd}.json"),"w"), separators=(",",":"))

# ===================================================================== state
def _sp(out): return os.path.join(out,"_annotate_state.json")
def _load_state(out): return json.load(open(_sp(out)))
def _save_state(out,st):
    with open(_sp(out),"w") as f: json.dump(st,f)

def _vf_setup(adata):
    """Returns (array, bottom-25% threshold) when obs['vf_norm'] exists, else (None, None)."""
    if "vf_norm" in adata.obs:
        vf=np.asarray(adata.obs["vf_norm"].values, dtype=np.float64)
        return vf, float(np.percentile(vf,25))
    return None, None

def _check_input_matrix(adata, force=False):
    """init entry gate: the standard annotator reads only adata.X (there is no --layer). Two preconditions
    are verified — X = log-normalized (not integer raw counts) and var_names = gene symbols (not Ensembl
    IDs). Violating either still lets the pipeline run to completion silently, but the labels become
    garbage (the marker-axis statistics assume lognorm, and marker matching is symbol based). Hence
    fail-fast here. Can be overridden with --force (acknowledging non-standard input)."""
    import re as _re
    X=adata.X
    n=adata.n_obs
    # ---- sampling (read only the first N rows so a large h5ad is never densified in full) --------------
    rows=min(n,2000)
    Xs=X[:rows]
    Xs=np.asarray(Xs.todense()) if hasattr(Xs,"todense") else np.asarray(Xs)
    Xs=Xs.astype(np.float64, copy=False)
    problems=[]
    # ---- (1) var_names = gene symbol? --------------------------------------------
    vn=[str(g) for g in adata.var_names[:200]]
    ens=sum(1 for g in vn if _re.match(r'^ENS[A-Z]*[GT]?\d{6,}(\.\d+)?$', g) or _re.match(r'^ENSMUS[GT]\d', g))
    if vn and ens/len(vn) > 0.5:
        ex=next((g for g in vn if g.upper().startswith("ENS")), vn[0])
        problems.append(f"var_names look like Ensembl IDs rather than gene symbols (sampled {ens}/{len(vn)}, e.g. '{ex}'). "
                        f"Markers are matched by symbol (EPCAM, Snap25), so with Ensembl IDs no marker will match at all. "
                        f"Convert adata.var_names to gene symbols.")
    # ---- (2) is X log-normalized? (detect integer raw counts) ---------------------------
    finite=Xs[np.isfinite(Xs)]
    if finite.size:
        if np.any(finite < 0):
            problems.append("X contains negative values — this looks like a normalization other than log-normalization"
                            "(z-scored/scaled, or Pearson residuals, etc.). "
                            "The annotator expects non-negative log-normalized X — use the stage before scaling/residuals.")
        else:
            nz=finite[finite>0]
            xmax=float(finite.max())
            # Raw counts are detected by ANY of three independent signatures. Requiring ALL-integer was too
            # narrow: bin-to-cell toolchains (bin2cell / ENACT `weighted_by_area`) split a shared bin's counts
            # by overlap area, so a large share of the non-zeros is fractional and the matrix is still counts.
            sigs=[]
            #   (1) integrality SHARE — log1p output is essentially never integral (raw=100%, area-weighted
            #       counts ~60-70%, log-normalized ~0%), so a low threshold separates them with room to spare.
            int_share = float(np.mean(np.isclose(nz, np.round(nz)))) if nz.size else 0.0
            if int_share>=0.10 and xmax>1:
                sigs.append(f"{int_share*100:.0f}% of the non-zero values are whole numbers (max={xmax:.4g})")
            #   (2) out of range for log1p — log1p of even a 1e6-normalized value is 13.8, so max>20 cannot be
            #       log-normalized (raw counts, CPM/CP10K without log1p, or another linear scaling).
            if xmax>20:
                sigs.append(f"max={xmax:.4g} is far above what log1p can produce (log1p(1e6)=13.8)")
            #   (3) row sums reproduce a sequencing-depth column in obs -> the row still IS that cell's counts.
            depth_col=next((c for c in ("total_counts","num_transcripts","nCount_RNA","n_counts","nCount_Spatial")
                            if c in adata.obs.columns), None)
            if depth_col is not None:
                try:
                    rs=np.asarray(Xs.sum(axis=1)).ravel()
                    dv=np.asarray(adata.obs[depth_col].values[:rows], dtype=np.float64)
                    ok=(dv>0)&np.isfinite(dv)&np.isfinite(rs)
                    if ok.sum()>=20 and np.mean(np.isclose(rs[ok], dv[ok], rtol=1e-3))>=0.9:
                        sigs.append(f"row sums equal obs['{depth_col}'] — the rows are still that cell's transcript counts")
                except Exception:
                    pass
            if sigs:
                problems.append("X looks like raw counts, not log-normalized: " + "; ".join(sigs) + ". "
                                "The annotator expects log-normalized X (counts distort the marker-axis statistics). "
                                "Apply sc.pp.normalize_total(adata, target_sum=1e4) followed by sc.pp.log1p(adata). "
                                "Note that counts do not have to be integers — bin-to-cell pipelines "
                                "(bin2cell / ENACT weighted_by_area) split shared bins by overlap area, which makes "
                                "many values fractional while the matrix is still counts.")
    if problems:
        msg="\n".join(f"   • {p}" for p in problems)
        if force:
            print(f"[init][WARN:INPUT_MATRIX] input precondition violated (proceeding because of --force):\n{msg}", flush=True)
        else:
            sys.exit(f"[init][ERROR:INPUT_MATRIX] the input h5ad violates the annotator's preconditions:\n{msg}\n"
                     f"   -> fix it and rerun, or pass --force to proceed knowing the input is non-standard. "
                     f"-> the caller should relay the problem above to the user verbatim.")

# ===================================================================== init
def cmd_init(a):
    os.makedirs(a.out, exist_ok=True)
    print(f"[init] loading {a.h5ad} (cluster_col={a.cluster_col})", flush=True)
    adata=ad.read_h5ad(a.h5ad)
    if a.cluster_col not in adata.obs:
        sys.exit(f"[init] cluster column '{a.cluster_col}' not found in obs. Available: {list(adata.obs.columns)}")
    _check_input_matrix(adata, force=getattr(a,"force_input",False))
    cps=adata.obs[a.cluster_col].astype(str).values.copy()
    # every label in the cluster/annotation column the user supplied is a leaf (= a target for annotation).
    #   Clustering-based annotation takes the user's clustering (from a notebook or elsewhere) as given.
    leaves=sorted(set(cps.tolist()),
                  key=lambda s:[int(x) for x in s.split(".")] if s.replace('.','').isdigit() else [10**9,s])
    # ★ subcluster representation (use_rep): which obsm embedding the subcluster kNN is built on, used
    #   only when a marker axis cannot be built (the PCA fallback in cmd_subcluster).
    #   Unset -> X_pca, and normalize + log1p + PCA recomputed per subset when obsm has no X_pca.
    #   An integrated embedding (X_scVI, X_pca_harmony, ...) is NEVER picked implicitly: it changes which
    #   splits are reachable, so it has to be named. Pass the obsm key to use one.
    #   The recompute fallback log-normalizes rather than using raw counts, which are dominated by stress
    #   and library size and fail to split identities.
    rep=a.use_rep
    if rep in (None,"auto",""):
        rep="X_pca" if "X_pca" in adata.obsm else None
    elif rep in ("recompute","none","raw","RECOMPUTE"):
        rep=None
    elif rep not in adata.obsm:
        sys.exit(f"[init] --use-rep '{rep}' is not in obsm. Available obsm: {list(adata.obsm)} "
                 "(or use --use-rep recompute to redo normalize + log1p + PCA).")
    print(f"[init] subcluster representation: {rep or 'recompute (normalize_total+log1p+PCA)'}", flush=True)
    # ★ The subcluster metric / n_neighbors **must match the initial clustering** — if they differ, results
    #   silently drift. When the clustering that produced the input h5ad (a user notebook or an external tool)
    #   records metric/n_neighbors in uns['cellman_cluster_params'], they are inherited here; otherwise the
    #   defaults (euclidean/15) apply. Passing --subcluster-metric / -n-neighbors explicitly overrides uns
    #   (for an h5ad clustered elsewhere with different settings).
    _cp={}
    try:
        _raw=adata.uns.get("cellman_cluster_params")
        _cp=json.loads(_raw) if isinstance(_raw,str) else (dict(_raw) if _raw else {})
    except Exception: _cp={}
    sc_metric=getattr(a,"sc_metric",None) or _cp.get("metric") or "euclidean"
    sc_n_neighbors=int(getattr(a,"sc_n_neighbors",None) or _cp.get("n_neighbors") or 15)
    if _cp and not getattr(a,"sc_metric",None):
        print(f"[init] inheriting the initial clustering settings from uns['cellman_cluster_params']: "
              f"metric={sc_metric} n_neighbors={sc_n_neighbors} (backend={_cp.get('backend')})", flush=True)
    if rep is None:
        print(f"[init] subcluster feature: log-normalize + metric={sc_metric} + n_neighbors={sc_n_neighbors}", flush=True)
    st={"h5ad_path":os.path.abspath(a.h5ad),"cluster_col":a.cluster_col,
        "cluster_paths":cps.tolist(),"leaves":leaves,"round":1,"parent_of":{p:None for p in leaves},
        "output":a.output,"disease":a.disease,"use_rep":rep,
        "sc_metric":sc_metric,"sc_n_neighbors":sc_n_neighbors,
        # ★ adaptive-floor is persisted as well: passing --adaptive-floor once at init makes every subcluster
        #   round inherit it (the same mechanism as subcluster-metric) — no need to repeat it each round.
        "adaptive_floor":bool(getattr(a,"adaptive_floor",False)),"af_F":float(getattr(a,"af_F",3.0)),"af_Nmin":int(getattr(a,"af_Nmin",10))}
    _save_state(a.out, st)
    # ontology URL resolution: an explicit --ontology-url > (species/organ + the default or given server); a local file comes from --ontology.
    url=a.ontology_url
    if not url and not a.ontology:
        if not (a.species and a.organ):
            sys.exit("[init] --species/--organ are required (or --ontology-url / a local --ontology). "
                     "The client connects to the default Cell-Ontology server automatically.")
        # --organ accepts SEVERAL modules, comma-separated ("brain,spinal_cord"): the user was shown the
        # candidates and asked for all of them. The wings are fetched independently and merged into one index.
        organs=[o.strip() for o in str(a.organ).split(",") if o.strip()]
        url=[f"{_resolve_server(a.server_url)}/module/{a.species}/{o}/slim-index" for o in organs]
        if len(url)==1: url=url[0]
        print(f"[init] ontology server: {_resolve_server(a.server_url)} "
              f"(module{'s' if len(organs)>1 else ''} {a.species}/{'+'.join(organs)})", flush=True)
    idx=_fetch_slim_index(a.out, url, a.ontology)
    if not idx or not _ontology_gene_set(idx):
        sys.exit("[init] FATAL: the ontology (slim-index) is empty — check the server / module. Running ontology-blind is not allowed.")
    ogs=_ontology_gene_set(idx); prot=_ontology_protected(idx); octx=_onto_ctx(idx)
    print(f"[init] ontology markers {len(ogs)} (protected {len(prot)}); lineages {len(octx['lineages'])} "
          f"(derived={'yes' if _derive_lineages(idx) else 'fallback'})", flush=True)
    X=np.asarray(adata.X.todense() if hasattr(adata.X,"todense") else adata.X)
    vf,low=_vf_setup(adata)
    print(f"[init] vf_norm: {'present' if vf is not None else 'absent (density axis disabled)'}", flush=True)
    # ★ Global per-gene statistics (over the full data) are computed once and persisted. Every round's
    #   absolute-identity score reuses this global reference, so identity does not collapse under
    #   fragmentation (no dependence on a local contrast).
    gmu=np.asarray(X.mean(axis=0)).ravel(); gsd=np.asarray(X.std(axis=0)).ravel()
    np.savez(os.path.join(a.out,"_global_stats.npz"), mean=gmu, std=gsd, genes=np.array(list(adata.var_names),dtype=object))
    print(f"[init] global per-gene stats saved (_global_stats.npz, {len(gmu)} genes) — the absolute-identity reference", flush=True)
    dz_terms=_disease_terms(idx, a.disease); st_terms=_state_terms(idx); mt_terms=_mimicry_terms(idx)   # dz/st_terms retired (no tier='state' Terms / disease property) — kept for signature compat
    if mt_terms: print(f"[init] mimicry/transition terms {len(mt_terms)} (MP+anchor) -> per-leaf transition_candidates (anchor tiebreaker, §state-transition)", flush=True)
    print(f"[init] state-axis programs are EVIDENCE only (interactive §8); disease/injury/malignant are class/subclass celltypes judged at the same level as normal — no before-floor state fallback", flush=True)
    # ★ F-B: the anatomical region-presence gate (computed once over the full data and persisted). Absent
    #   regions (cerebellum, pons, ...) are removed from the anchor / dominant / floor candidates, blocking
    #   force-fits. Only region-groups matching brain anatomy are affected (no effect on non-brain modules).
    absent_rg, _rpres = _absent_region_groups(X, list(adata.var_names), octx, gmu, gsd)
    _rlab=_build_ident_maps(octx, np.asarray(list(adata.var_names)))["lab"]
    json.dump({"absent_region_groups":sorted(absent_rg),
               "absent_labels":sorted(_rlab.get(r,r) for r in absent_rg),
               "region_present":_rpres, "thr":REGION_PRESENT_THR},
              open(os.path.join(a.out,"_region_presence.json"),"w"), ensure_ascii=False, indent=1)
    if absent_rg:
        print(f"[init] F-B region-presence: {len(absent_rg)} absent region-group(s) gated → "
              f"{sorted(_rlab.get(r,r) for r in absent_rg)[:8]}", flush=True)
    profiles=_profile_leaves(X, list(adata.var_names), np.array(cps), leaves, ogs, prot, octx, vf, low, gmu=gmu, gsd=gsd, dz_terms=dz_terms, st_terms=st_terms, mt_terms=mt_terms, profile_cap=getattr(a,"profile_cap",0), auc_pos=getattr(a,"auc_pos",0.65), absent_rg=absent_rg, adaptive_floor=getattr(a,"adaptive_floor",False), af_F=getattr(a,"af_F",3.0), af_Nmin=getattr(a,"af_Nmin",10))
    json.dump(profiles, open(os.path.join(a.out,"cluster_gene_profiles_round1.json"),"w"), separators=(",",":"))
    _write_overview(a.out, 1, profiles)
    json.dump(_ontology_hits(list(adata.var_names), idx), open(os.path.join(a.out,"ontology_hits.json"),"w"), separators=(",",":"))
    print(f"[init] {len(leaves)} top-level leaves profiled. No h5ad write.", flush=True)

# ===================================================================== subcluster
def _pick_backend(gpu):
    if gpu is not None:
        try:
            import rapids_singlecell as rsc, cupy
            if cupy.cuda.runtime.getDeviceCount()>0:
                return "rapids", rsc
        except Exception: pass
    import scanpy as sc
    return "scanpy", sc

def _hcmd_subcluster(a):
    st=_load_state(a.out)
    # ★ Unknown re-pool mode: the cells of leaves decided Unknown in the current round are gathered across
    #   parents into a single synthetic parent U{round} and re-clustered with Leiden. Scattered Unknown
    #   fragments of the same type re-aggregate and are re-judged in the next round. Only what is still
    #   genuinely Unknown at the last round is labelled 'Unknown' at finalize. (Unknown is not terminal until
    #   the final round.)
    repool=bool(getattr(a,"repool_unknown",False))
    unk_leaves=[]
    if repool:
        if not a.decisions: print("[error] --repool-unknown requires --decisions"); return
        _dec=json.load(open(a.decisions))
        unk_leaves=[lf for lf in st["leaves"] if str((_dec.get(lf) or {}).get("decision","")).strip().lower()=="unknown"]
        if not unk_leaves: print("[repool] no Unknown leaves to re-pool — nothing to do."); return
        paths=[f"U{st['round']+1}"]                       # synthetic pooled parent
    else:
        paths=json.loads(a.paths)
    # --resolution: a scalar (the same for every path) or a JSON dict {path: res} (per cluster — the finalize start_resolution).
    try: _rj=json.loads(a.resolution)
    except Exception: _rj=None
    if isinstance(_rj, dict): res_map={str(k):float(v) for k,v in _rj.items()}; res_def=None
    else: res_map={}; res_def=float(a.resolution)
    def _res_p(path): return res_map.get(path, res_def if res_def is not None else 1.0)
    backend,mod=_pick_backend(a.gpu)
    print(f"[round {st['round']+1}] subcluster {[(p,_res_p(p)) for p in paths]} backend={backend}", flush=True)
    adata=ad.read_h5ad(st["h5ad_path"])
    cps=np.array(st["cluster_paths"], dtype=object)
    genes=list(adata.var_names)
    rep=st.get("use_rep")                              # ★ subcluster representation (decided at init). None -> recompute normalize + log1p + PCA.
    new_all=[]; parent_prof={}
    for path in paths:
        if repool:
            mask=np.isin(cps, unk_leaves); n=int(mask.sum())
            print(f"[repool] pooling {n} cells from {len(unk_leaves)} Unknown leaves -> {path}", flush=True)
        else:
            if path not in st["leaves"]: print(f"[skip] {path} not leaf"); continue
            mask=(cps==path); n=int(mask.sum())
        # there is no policy min-cells (a small leaf is never treated as background or blocked from splitting —
        # rare cells are protected).
        # Only sizes for which PCA / the neighbour graph are mathematically impossible are skipped (= the split
        # operation is undefined, not background).
        if n<3:
            st.setdefault("unresolved_paths",[]).append(path)
            print(f"[skip] {path} n={n}<3: too small to split (the operation is undefined). Not background — the current leaf is kept.", flush=True)
            continue
        res=_res_p(path)                                  # ★ per-cluster resolution
        sub=adata[mask].copy()
        npc=min(50, sub.n_vars-1, max(2, sub.n_obs-1))
        nn=min(int(st.get("sc_n_neighbors",15)),sub.n_obs-1)  # ★ n_neighbors persisted at init (15 when unset)
        # ★ use_rep (resolved at init) builds neighbors on that obsm key restricted to the subset; otherwise
        #   normalize_total + log1p + PCA precede neighbors, since raw-count PCA is dominated by library size.
        use_rep=rep if (rep and rep in sub.obsm) else None
        # ★ metric consistent with the initial clustering (persisted at init). Applies only on the recompute path.
        sc_metric=st.get("sc_metric","euclidean")
        if backend=="rapids":
            mod.get.anndata_to_GPU(sub)
            if use_rep is None:
                mod.pp.normalize_total(sub,target_sum=1e4); mod.pp.log1p(sub)
                mod.pp.pca(sub,n_comps=npc)
                mod.pp.neighbors(sub,n_neighbors=nn,use_rep="X_pca",metric=sc_metric)
            else:
                mod.pp.neighbors(sub,n_neighbors=nn,use_rep=use_rep,metric="euclidean")
            mod.tl.leiden(sub,resolution=res,random_state=0); mod.get.anndata_to_CPU(sub)
        else:
            import scanpy as sc
            if use_rep is None:
                sc.pp.normalize_total(sub,target_sum=1e4); sc.pp.log1p(sub)
                sc.pp.pca(sub,n_comps=npc)
                sc.pp.neighbors(sub,n_neighbors=nn,use_rep="X_pca",metric=sc_metric)
            else:
                sc.pp.neighbors(sub,n_neighbors=nn,use_rep=use_rep,metric="euclidean")
            sc.tl.leiden(sub,resolution=res,random_state=0,flavor="igraph",n_iterations=2,directed=False)
        lbls=sub.obs["leiden"].astype(str).values
        if len(set(lbls))<2: st.setdefault("unresolved_paths",[]).append(path); continue
        new=np.array([f"{path}.{l}" for l in lbls]); cps[mask]=new
        nh=sorted(set(new.tolist())); new_all+=nh
        for nl in nh: st["parent_of"][nl]=path
        parent_prof[path]=(mask.copy(),new)
    if repool:
        # if the repool actually separated anything (new_all), replace the original Unknown leaves with the U sub-leaves; if it did not (a single blob), leave them as they were.
        leaves=([p for p in st["leaves"] if p not in unk_leaves]+new_all) if new_all else st["leaves"]
        if not new_all: print("[repool] pooled Unknown did not split (single blob) — Unknown leaves unchanged.", flush=True)
    else:
        leaves=[p for p in st["leaves"] if p not in parent_prof]+new_all
    st["leaves"]=leaves; st["cluster_paths"]=cps.tolist(); st["round"]+=1; _save_state(a.out,st)
    idx=_load_ontology_index(a.out); ogs=_ontology_gene_set(idx); prot=_ontology_protected(idx); octx=_onto_ctx(idx)
    dz_terms=_disease_terms(idx, st.get("disease")); st_terms=_state_terms(idx); mt_terms=_mimicry_terms(idx)   # states are always surfaced (disease is a hint), for sub-leaves too; mt = the transition anchor tiebreaker
    X=np.asarray(adata.X.todense() if hasattr(adata.X,"todense") else adata.X)
    vf,low=_vf_setup(adata)
    # ★ F-B: load the absent region-groups persisted at init (a subset cannot compute them itself, so they are reused)
    _arp=os.path.join(a.out,"_region_presence.json")
    absent_rg=set(json.load(open(_arp)).get("absent_region_groups",[])) if os.path.exists(_arp) else set()
    # ★ load the global statistics persisted at init -> this round uses the same absolute-identity reference (from the full data).
    gmu=gsd=None
    _gsp=os.path.join(a.out,"_global_stats.npz")
    if os.path.exists(_gsp):
        _z=np.load(_gsp, allow_pickle=True); gmu=_z["mean"]; gsd=_z["std"]
    # preload the previous round's parent profiles (used for candidate_subtypes parent pooling and for injecting parent_ctx)
    prior={}
    for r in range(1,st["round"]):
        pf=os.path.join(a.out,f"cluster_gene_profiles_round{r}.json")
        if os.path.exists(pf): prior.update(json.load(open(pf)))
    # ★ PARENT-CONSTRAINED subclustering: each parent's children may only be the parent's KG
    #   subtree. restrict_root = the parent's resolved DOMINANT REGION-GROUP (single) so genuine SIBLINGS stay
    #   findable (CA1 inside a CA3 cluster — both under hippocampal pyramidal); if the parent had 0 or multiple
    #   dominant regions, fall back to its LINEAGE class (still blocks cross-lineage cortex→GABA/oligo). A child
    #   thus can NEVER force-fit a distant region (the 8µm cortex→pontine bug).
    _lab2tid={}
    for _tid,_t in (octx.get("terms",{}) or {}).items(): _lab2tid.setdefault(_t.get("L"),_tid)
    def _restrict_for(parent):
        ge=(prior.get(parent,{}) or {}).get("graph_evidence") or {}
        dr=ge.get("dominant_region") or []
        # ★ Constrain to a single region only when the parent HAS a single region. If the parent mixes **two or
        #   more upper regions** (dominant_region >= 2), do not pin a single region — constrain to the **lineage
        #   class only** so a child can resolve into either region (as in stomics cl6, where a cortical arc and a
        #   BLA blob share one parent and the child must be able to escape to amygdalar). Pinning dr[0] trapped a
        #   minority region (BLA) inside the majority region (neocortical).
        #   (Siblings within one region stay findable: CA1 inside CA3, both hippocampal pyramidal.)
        if len(dr)==1 and dr[0] in _lab2tid: return _lab2tid[dr[0]]
        lin=ge.get("lineage")
        return _lab2tid.get(lin)
    profiles={}
    for parent,(mask,sub_paths) in parent_prof.items():
        sl=sorted(set(sub_paths.tolist()))
        p_up={r[0]:r[1] for r in (prior.get(parent,{}).get("up") or [])}
        rr=_restrict_for(parent)
        pr=_profile_leaves(X[mask], genes, sub_paths, sl, ogs, prot, octx,
                           vf[mask] if vf is not None else None, low, parent_up_auc=p_up, gmu=gmu, gsd=gsd, restrict_root=rr, dz_terms=dz_terms, st_terms=st_terms, mt_terms=mt_terms, profile_cap=getattr(a,"profile_cap",0), auc_pos=getattr(a,"auc_pos",0.65), absent_rg=absent_rg, adaptive_floor=(bool(st.get("adaptive_floor",False)) or getattr(a,"adaptive_floor",False)), af_F=float(st.get("af_F",getattr(a,"af_F",3.0))), af_Nmin=int(st.get("af_Nmin",getattr(a,"af_Nmin",10))))
        profiles.update(pr)
    for leaf in list(profiles):
        par=st["parent_of"].get(leaf)
        if par and par in prior:
            profiles[leaf]["parent_ctx"]={"path":par,"up":prior[par].get("up",[])[:10],"hp":prior[par].get("hp",[])[:10]}
    json.dump(profiles, open(os.path.join(a.out,f"cluster_gene_profiles_round{st['round']}.json"),"w"), separators=(",",":"))
    _write_overview(a.out, st["round"], profiles)
    print(f"[round {st['round']}] {len(profiles)} new leaves. No h5ad write.", flush=True)

# ===================================================================== validate (reporting only)
# Principle: the helper never changes a decision. It only detects and reports violations; re-deciding is
# the agent's job.
def _validate(decisions, st, out):
    import glob
    prof={}
    for pf in sorted(glob.glob(os.path.join(out,"cluster_gene_profiles_round*.json"))): prof.update(json.load(open(pf)))
    oh=json.load(open(os.path.join(out,"ontology_hits.json"))) if os.path.exists(os.path.join(out,"ontology_hits.json")) else {}
    _idx_full=_load_ontology_index(out) or {}
    idx=_idx_full.get("terms",{})
    octx=_onto_ctx(_idx_full); LINS=octx["lineages"]; SPEC=octx["specific"]; PAN=octx["pan"]
    def upstrong(p): return {_U(r[0]) for r in (p.get("up") or []) if r[1]>=0.6}
    def lins(gs): return sorted(l for l,s in LINS.items() if gs&s)
    def pool(p):
        g=set()
        for b in ("up","hp"):
            for r in (p.get(b) or []): g.add(_U(r[0]))
        pc=p.get("parent_ctx") or {}
        for b in ("up","hp"):
            for r in (pc.get(b) or []): g.add(_U(r[0]))
        return g
    _NONP=('endothel','pericyte','smooth muscle','leptomening','mural','perivascular','fibroblast',
           'macrophage','microg','choroid','ependym','arachnoid','vascular','erythro','neutrophil',
           'lymphocyte','monocyte','mast','tanycyte')
    def _isnonpar(s): return any(k in (s or '').lower() for k in _NONP)
    V=[]
    def add(lf,check,sev,msg,sugg):
        V.append({"cluster_path":lf,"check":check,"severity":sev,"message":msg,
                  "current_decision":(decisions.get(lf) or {}).get("decision"),
                  "current_label":(decisions.get(lf) or {}).get("label"),"suggestion":sugg})
    for lf in st["leaves"]:
        d=decisions.get(lf); p=prof.get(lf)
        if d is None:
            add(lf,"MISSING_DECISION","error","No decision was written for this leaf.",
                "Write one of the four decisions for this cluster."); continue
        dec=d.get("decision")
        if dec=="needs_subclustering":
            add(lf,"NEEDS_SUB_REMAINING","error",
                "Still marked needs_subclustering at finalize time (it was never split further).",
                "Turn it into a terminal decision per the Step 6 unsplittable rule: a class-final celltype when a common class exists, otherwise Unknown. Do not downgrade it to Background.")
            continue
        if p is None: continue
        us=upstrong(p)
        # ★ Two advisory checks — not enforced: they present evidence and prompt the agent to re-judge (the helper leaves decisions untouched).
        _ge=p.get("graph_evidence") or {}
        # A) RESCUE_IGNORED: the engine rescued an embedded vascular/immune anchor but the label is parenchymal,
        #    so the call may have been swept along by absolutely-dominant glia (KDE smear) -> re-judge per §4.1b.
        _resc=[an for an in (_ge.get("anchors") or []) if an.get("rescued")]
        if dec=="celltype" and _resc and not _isnonpar(d.get("label","")):
            _r=_resc[0]
            add(lf,"RESCUE_IGNORED","warn",
                f"The engine rescued the embedded vascular/immune anchor '{_r.get('label')}' (its specific markers {_r.get('rescue_markers')} are strongly enriched), yet the label is the parenchymal '{d.get('label')}'. The glia/neuron signal may be KDE smear.",
                "§4.1b: when rescued specific vascular/immune markers are strong, re-judge as that type instead of the absolutely-dominant glia (keep the label if you have real evidence).")
        # B) NO_ANCHOR_SPECIFIC: the engine found 0 positive anchors (class_or_unknown) yet the label is a
        #    specific term, which may be a region/subtype force-fit with no marker evidence (allocortex,
        #    subiculum, ...) -> reconsider class / Unknown.
        if dec=="celltype":
            _verd=_ge.get("verdict"); _linc=str(_ge.get("lineage") or "").strip().lower()
            _ll=str(d.get("label","")).strip().lower()
            _is_proc_term=((idx.get((d.get("term_id") or "").strip(),{}) or {}).get("T") in _PROC_TIERS)
            if (_verd=="class_or_unknown" and _ll and _ll!=_linc and _ll not in ("unknown","background")
                    and not _is_proc_term):
                # ★ Flagged only when none of the label's own markers (R u DF) appears in the cluster's strong up (if a marker such as astrocyte Aqp4 is ON the call is legitimate -> exempt).
                _tid=(d.get("term_id") or "").strip(); _term=idx.get(_tid) or {}
                _mk={_U(g) for g in ((_term.get("R") or [])+(_term.get("DF") or []))}
                if _mk and not (_mk & us):
                    add(lf,"NO_ANCHOR_SPECIFIC","warn",
                        f"The engine found 0 positive anchors (class_or_unknown) and none of the own markers of the specific term '{d.get('label')}' is strongly ON in this cluster — a region/subtype force-fit is suspected.",
                        f"Keep it if its own markers are ON; otherwise re-judge as the lineage class ('{_ge.get('lineage')}') or Unknown (§3 positive evidence).")
        # ambient/neuropil(KDE) bleed makes raw `up` list many lineages → only warn on a REAL co-dominant
        # mixture (identity_scores.mixed = ≥2 codominant lineage-roots). ambient-only bleed is NOT flagged.
        ismixed=bool((p.get("identity_scores") or {}).get("mixed"))
        if ismixed and len(lins(us))>=2 and not (us&SPEC) and dec=="celltype":
            add(lf,"UP_CROSS_LINEAGE","warn",
                f"≥2 co-dominant lineages (identity_scores.mixed, up={lins(us)}) but a single celltype.",
                "genuine chimera? → needs_subclustering, or re-check dominance (ambient bleed alone is not flagged).")
        if dec=="celltype":
            tid=(d.get("term_id") or "").strip()
            _lab_l=str(d.get("label","")).strip().lower()
            # ★ CONTEXT-AXIS enforcement (§4.3): the engine computed a context_axis (zonation/layer) but the leaf was
            #   terminated at the bare lineage class — a pole subtype was missed. With a dominant_pole, relabel to that
            #   subtype; without one but mixed (>=2 poles), require a subcluster. (The helper leaves decisions
            #   untouched — check only reports; because the engine already computed suggested_subtype, the agent's
            #   re-decision is mechanical.) Fires only when the leaf was terminated at a lineage class label (a real
            #   subtype label is exempt).
            _cax=p.get("context_axis") or []
            _calin=str((p.get("graph_evidence") or {}).get("lineage") or "").strip().lower()
            _is_top=("." not in lf)
            _nleaf=int(p.get("n",0) or 0)
            for _ax in _cax:
                # 1) ★ Both poles are *substantially* present (mixed) in a large top-level parent -> whether it was
                #    labelled by one pole or left at the class, split first to separate both zones (so the minority pole
                #    is not buried). A sub-leaf (already split) is exempt — the lean label is correct there.
                if _ax.get("mixed") and _is_top and _nleaf>=1500:
                    add(lf,"CONTEXT_AXIS_SPLIT","warn",
                        f"Both poles of context axis '{_ax['axis']}' are substantially present (poles_present={_ax['poles_present']}) in a large top-level cluster (n={_nleaf}), yet it was terminated undivided as '{d.get('label')}' — labelling by one pole buries the minority pole. §4.3: subcluster to separate the two zones.",
                        "Split it (needs_subclustering, reason_code=context_axis) and label each sub-leaf by its dominant_pole subtype, recovering both poles.")
                # 2) terminated at a bare lineage class while a dominant_pole exists -> relabel to that pole subtype
                elif _lab_l==_calin and _ax.get("dominant_pole") and _ax.get("suggested_subtype"):
                    add(lf,"CONTEXT_AXIS_SUBTYPE","warn",
                        f"Context axis '{_ax['axis']}' has dominant_pole={_ax['dominant_pole']} (context lead), yet the leaf was terminated at the bare class '{d.get('label')}' — §4.3 requires labelling it '{_ax['suggested_subtype']}'.",
                        f"Change the label to '{_ax['suggested_subtype']}' (reason_code=context_axis, confidence={_ax.get('pole_confidence') or 'low'}). The call comes from the pole context, not from the (off-panel) intrinsic markers.")
            # ★ General over-split rule: a "magnitude-only subtype" is a subtype label whose *own* markers
            #   distinguishing it from the parent class are not ON -> the split rests only on the *magnitude* of
            #   shared markers -> collapse to the parent class. Zero hardcoding (it branches only on the KG hierarchy
            #   and the data) and applies to every module. Genuine subtypes (L4 = Rorb, CA3 = Grik4, Opalin+ oligo)
            #   keep their labels because their own markers are ON.
            #   #1 = ordinary subtypes (term-marker differential); #2 = context-axis poles (the opposite pole exists
            #   across leaves). They are separated because a liver zonation pole (separated by ambient context) cannot
            #   be judged by a per-leaf differential — only the coexistence of the opposite pole distinguishes signal
            #   from noise.
            _ca_pole_subs=set()
            for _ax in (p.get("context_axis") or []):
                for _po in (_ax.get("poles") or []):
                    if _po.get("subtype"): _ca_pole_subs.add(str(_po["subtype"]).strip().lower())
            # ── #1 DIFFERENTIAL_UNSUPPORTED (non-context-axis subtype) ──
            if tid in idx and _lab_l not in _ca_pole_subs:
                _S=idx[tid]; _Pids=[pp for pp in (_S.get("P") or []) if pp in idx]
                if _Pids:
                    _Smk={_U(g) for g in (_S.get("R") or [])+(_S.get("S") or [])+(_S.get("DF") or [])}
                    _Pmk=set()
                    for _pp in _Pids: _Pmk|={_U(g) for g in (idx[_pp].get("R") or [])+(idx[_pp].get("S") or [])+(idx[_pp].get("DF") or [])}
                    _diff=_Smk-_Pmk
                    _ogtv={_U(g):v for g,v in (p.get("ontology_gene_table") or {}).items()}
                    _diff_op={g for g in _diff if not ((_ogtv.get(g) or {}).get("panel") is False)}
                    if _diff_op and not (_diff_op & us):                 # the term has own markers but 0 of them are enriched (up auc>=0.6)
                        _Cl=idx[_Pids[0]].get("L") or _Pids[0]
                        add(lf,"DIFFERENTIAL_UNSUPPORTED","warn",
                            f"None of the discriminative markers {sorted(_diff_op)} that distinguish the subtype '{_S.get('L')}' from its parent '{_Cl}' is enriched (up) — this split rests only on a level difference in shared markers (magnitude-only subtype, over-split suspected).",
                            f"Change the label to the parent class '{_Cl}'. A real subtype must have its own markers ON, and siblings of the same identity merge into the parent.")
            # ── #2 CONTEXT_AXIS_LONELY_POLE (context-axis pole) ──
            if _lab_l in _ca_pole_subs:
                _all_lab={str((dd or {}).get("label","")).strip().lower() for dd in decisions.values()}
                for _ax in (p.get("context_axis") or []):
                    _psubs={str(po.get("subtype")).strip().lower() for po in (_ax.get("poles") or []) if po.get("subtype")}
                    if _lab_l not in _psubs: continue
                    _others=_psubs-{_lab_l}
                    if _others and not (_others & _all_lab):
                        _Pp=idx.get(tid,{}).get("P") or []
                        _Cl=idx.get(_Pp[0],{}).get("L") if _Pp else None
                        add(lf,"CONTEXT_AXIS_LONELY_POLE","warn",
                            f"The leaf was labelled with the '{d.get('label')}' pole of context axis '{_ax['axis']}', but the opposite pole ({sorted(_others)}) appears in no cluster at all — a single-pole noise lean the axis did not actually separate (genuine zonation shows both poles).",
                            f"Change the label to the parent class '{_Cl or 'parent'}' (split by pole later, once the opposite pole really appears). This is judged across leaves, so liver — where both poles coexist — is unaffected.")
            # ★ PROCESS (tier='process' — neuropil, fiber tract, ...) is not a celltype but a tissue process.
            #   It should never appear in a clustering-based decision JSON, but if a process term_id does show up,
            #   the celltype checks (REQUIRED_ABSENT_ONPANEL etc.) are skipped (continue). Since this release has no
            #   process-detection path, a warning is emitted telling the agent not to label it.
            _is_proc=((idx.get(tid,{}) or {}).get("T") in _PROC_TIERS)
            if _is_proc:
                add(lf,"PROCESS_IN_CLUSTERING","warn",
                    f"'{d.get('label')}' is tier='process' (a tissue process) — clustering-based annotation does not label these.",
                    "Detection of processes (neuropil, fiber tract, ...) is not supported in this version — use a celltype label instead.")
                continue
            elif not tid or (idx and tid not in idx):
                add(lf,"BAD_TERM_ID","error",
                    f"The decision is celltype but term_id is empty or absent from the ontology (term_id='{tid}').",
                    "Fix it to a valid term_id or change the decision. The helper never silently turns this into Unknown. (For a tissue compartment you may use label='neuropil'.)")
            # ★ Own required-badge check (REQUIRED_ABSENT_ONPANEL): if this term has any non-pan required marker that
            # is on-panel, at least one must be present. UBC has Eomes (on-panel); when it is absent the call rests on
            # a misassigned supporting marker (Grm1) -> flag. CA1 has all required markers off-panel -> excluded from
            # the check (it is identified by the supporting badge Fibcd1, §5 panel-blind).
            if tid in idx:
                t_=idx[tid]; ogt_l={_U(g):v for g,v in (p.get("ontology_gene_table") or {}).items()}
                up_l={_U(r[0]):r[1] for r in (p.get("up") or [])}
                # ★ GLIAL_MISLABEL backstop: the top 8 of up (vs-rest) is dominated by glial markers (>=3) with 0
                #   neuronal / pan-neuronal markers, yet the label is a neuronal term -> a fiber-tract / vascular /
                #   microglial / meningeal contamination fragment was force-fit to a neuronal subtype because of the
                #   parent constraint (a ~19k-cell bug). Exempt when the chosen term is itself glial. Because up is
                #   vs-rest it does not trigger on ambient (non-enriched) myelin (cl9 L6b: up = Cplx3/Nxph3, so it does
                #   not fire). For a real neuron, up consists of identity markers so glial < 3.
                _GLIA_M={'PLP1','MBP','MOBP','MAG','MOG','CLDN11','SOX10','MAL','CNP','OPALIN','ASPA','ERMN','ENPP6','BCAS1',
                         'AQP4','GFAP','SLC1A2','SLC1A3','GJA1','AGT','MLC1','NTSR2','FGFR3','GJB6',
                         'C1QA','C1QB','C1QC','CSF1R','CX3CR1','P2RY12','HEXB','TMEM119','CTSS','SELPLG','LAPTM5',
                         'CLDN5','PECAM1','FLT1','PDGFRB','RGS5','KCNJ8','COL1A1','COL1A2','COL3A1','DCN','PTGDS','VTN','BNC2','SLC6A13','IGFBP7','LUM','PCOLCE',
                         'TTR','FOLR1','ENPP2','KCNE2','CLIC6','OTX2','AQP1','ECRG4','MFRP','CALML4',
                         'FOXJ1','PIFO','DNAH11','TMEM212','CCDC153','HDC','RARRES2'}
                _NEUR_M={'SLC17A7','SLC17A6','SLC17A8','GAD1','GAD2','SLC32A1','SLC6A5','TH','DBH','DDC','SLC6A3','CHAT','SLC18A3','SLC5A7','TPH2',
                         'SNAP25','SYT1','STMN2','RBFOX3','MEG3','NRGN','TUBB3','GAP43','SYP','STMN3','MAP2','NEFL','NEFM','SYT4','RESP18','SCG2',
                         'CAMK2A','CAMK2N1','CAMK2B','PCP4','CALM1','CALM2','CALM3','NCALD'}
                _ut=[_U(r[0]) for r in (p.get('up') or [])[:8]]
                _ng=[g for g in _ut if g in _GLIA_M]; _nm=[g for g in _ut if g in _NEUR_M]
                _chosen_glial=len({_U(g) for g in (t_.get('R') or [])+(t_.get('S') or [])} & _GLIA_M)>=2
                # exempt: skip when the chosen term's own markers appear in up (a real neuron with glial bleed).
                _own_in_up=len({_U(g) for g in (t_.get('R') or [])+(t_.get('S') or [])+(t_.get('DF') or [])} & set(_ut))>=1
                if len(_ng)>=3 and len(_nm)==0 and not _own_in_up and not _chosen_glial:
                    # ★ Fix B: classify the glial markers in up by sub-family and suggest *which* glia it is. This prevents
                    #   the agent from over-specifying to microglia when vascular/oligo signal dominates (microglia 75 is
                    #   vascular-mixed with Stab1/Sparc, yet near-noise Cx3cr1 led to a microglia call). Message only
                    #   (decisions unchanged).
                    _SUBFAM={
                        "oligodendrocyte":{'PLP1','MBP','MOBP','MAG','MOG','CLDN11','SOX10','MAL','CNP','OPALIN','ASPA','ERMN','ENPP6','BCAS1','OLIG1','OLIG2'},
                        "astrocyte":{'AQP4','GFAP','SLC1A2','SLC1A3','GJA1','AGT','MLC1','NTSR2','FGFR3','GJB6'},
                        "microglia/macrophage":{'C1QA','C1QB','C1QC','CSF1R','CX3CR1','P2RY12','HEXB','TMEM119','CTSS','SELPLG','LAPTM5','TYROBP'},
                        "vascular/leptomeningeal (endothelial/mural/VLMC)":{'CLDN5','PECAM1','FLT1','PDGFRB','RGS5','KCNJ8','COL1A1','COL1A2','COL3A1','DCN','PTGDS','VTN','BNC2','SLC6A13','IGFBP7','LUM','PCOLCE','STAB1'},
                        "ependymal/choroid plexus":{'TTR','FOLR1','ENPP2','KCNE2','CLIC6','OTX2','AQP1','ECRG4','MFRP','CALML4','FOXJ1','PIFO','DNAH11','TMEM212','CCDC153'}}
                    _fam_hit={k:[g for g in _ng if g in v] for k,v in _SUBFAM.items()}
                    _fam_hit={k:v for k,v in _fam_hit.items() if v}
                    _fam_sorted=sorted(_fam_hit.items(), key=lambda kv:-len(kv[1]))
                    _dom=_fam_sorted[0][0] if _fam_sorted else "glia"
                    _fam_str="; ".join(f"{k}={v}" for k,v in _fam_sorted)
                    _hint=(f"The dominant glial sub-family indicated by up is **{_dom}** ({_fam_str}). Relabel to a celltype of that sub-family. "
                           "If another sub-family's markers are near-noise (absolute expression ~0), do not settle on it — e.g. when vascular/oligo markers dominate up but "
                           "only Cx3cr1/Tmem119 are rank-ON (absolute ~0), do not over-specify to microglia; use the dominant sub-family or its lineage class.")
                    add(lf,"GLIAL_MISLABEL","error",   # error = blocks finalize (robust: glial up is abundant and unambiguous, and the own-marker / pan-neuronal exemptions make false firing ~0)
                        f"The top 8 of up (vs-rest) is dominated by glial markers ({_ng}) with zero neuronal / pan-neuronal / own-identity markers, yet the label is '{t_.get('L')}' (a neuron) — this may be a fiber-tract / vascular / glial contamination fragment.",
                        _hint+" (Even if graph_evidence showed only neurons because of the parent constraint, glial up means glia. If it really is a neuron, give the neuronal identity / transmitter evidence from up.)")
                # ★ INTERNEURON_MISLABEL backstop: the top 6 of up (vs-rest) has an ENRICHED GABA transmitter
                #   (Gad1/2/Slc32a1) plus >=1 interneuron subclass marker (Pvalb/Sst/Vip/Erbb4/Pthlh...), yet the labelled
                #   term is glut (its own markers contain no GABA transmitter) -> an interneuron fragment embedded in
                #   cortical glut whose lineage was misread as glut from ambient Slc17a7 and force-fit to L2/3 IT or
                #   similar. A real glut neuron cannot have Gad enriched vs-rest, so false firing is ~0. A GABAergic term
                #   (carrying Gad in its own R) has _is_glut_term=False and is exempt.
                _IN_M={'PVALB','SST','VIP','NPY','ERBB4','PTHLH','RELN','CCK','SNCG','CALB2','CXCL14','HTR3A','ADARB2','LHX6','CNR1','KCNC1','KCNC2','KCNC3'}
                _GABA_TX={'GAD1','GAD2','SLC32A1'}
                _term_mk={_U(g) for g in (t_.get('R') or [])+(t_.get('S') or [])+(t_.get('DF') or [])}
                _gaba_up=[g for g in _ut[:6] if g in _GABA_TX]; _in_up=[g for g in _ut[:6] if g in _IN_M]
                # warn, not error: in L4/L5 real Pvalb interneurons are dense enough that >=2 can appear as ambient, so
                # forcing a relabel is risky -> the agent decides (relabel only clear Pvalb-FS / CGE fragments). Requires
                # an enriched GABA transmitter plus >=2 interneuron subclass markers.
                _is_glut_term=not (_term_mk & (_GABA_TX|{'SLC6A5'}))
                if _gaba_up and len(_in_up)>=2 and _is_glut_term and not _chosen_glial:
                    add(lf,"INTERNEURON_MISLABEL","warn",
                        f"The top 6 of up (vs-rest) has an enriched GABA transmitter ({_gaba_up}) plus interneuron markers ({_in_up}), yet the label is '{t_.get('L')}' (glut) — this may be a cortical GABAergic interneuron fragment whose lineage was misread from ambient Slc17a7.",
                        "If up is interneuron-dominated, relabel as a cortical GABAergic interneuron (Pvalb+/Sst+/Vip+ etc.). Keep the label when the glut identity is clear (as in L4/L5) and Gad is ambient — but state the positive glut evidence.")
                onp=[]
                for g in (t_.get("R") or []):
                    gu=_U(g)
                    if gu in PAN: continue
                    e=ogt_l.get(gu)
                    if e is None or e.get("panel") is False: continue
                    onp.append((gu, up_l.get(gu,0)>=0.6 or (e.get("auc") or 0)>=0.55))
                if onp and not any(pr for _,pr in onp):
                    add(lf,"REQUIRED_ABSENT_ONPANEL","warn",
                        f"The own on-panel required badge of '{t_.get('L')}' is not expressed ({', '.join(g for g,pr in onp if not pr)}) — the call may rest on a misassigned supporting marker.",
                        "Switch to a term whose required badge does fire, or drop to the parent class / Unknown (§5: an on-panel required marker that is absent = fail).")
            # ★ DEFINING_ABSENT (the DF-absent guard): when the chosen term has df_absent=True in graph_evidence, its
            #   defining marker — the strongest badge, where one is sufficient — was measured and is entirely OFF, so
            #   the call rests on supporting markers alone (e.g. cl37 mitral with Cdhr1 ON and Tbx21 OFF). Since
            #   graph_evidence already demoted it, choosing it anyway warrants a strong warning. If a competing
            #   non-force-fit anchor exists, use it (the verdict points at it); otherwise fall back to the parent
            #   class / Unknown.
            ge_=p.get("graph_evidence") or {}
            _amatch=next((a for a in (ge_.get("anchors") or []) if a.get("term_id")==tid or a.get("label")==t_.get("L")), None) if tid in idx else None
            #   However, excl_on>=2 (strong positive evidence including required markers) is not a force-fit even with a quiet defining marker (L4 = Rorb/Whrn) -> no warning.
            if _amatch and _amatch.get("df_absent") and _amatch.get("excl_on",0)<2:
                add(lf,"DEFINING_ABSENT","warn",
                    f"The on-panel defining markers of '{t_.get('L')}' ({', '.join(_amatch.get('df_onpanel') or [])}) are OFF and the exclusive evidence is weak too (excl_on<2) — a force-fit from supporting markers is suspected.",
                    "If another term has stronger exclusive markers, use it; otherwise verify and keep, or fall back to the parent class. (This warning is not raised when excl_on>=2.)")
            # ★ OFF_EVIDENCE: the chosen specific term is absent from graph_evidence.anchors while the engine gave a
            #   clear_top (a confident candidate in a different region) — that means a term the region-coherent engine
            #   excluded was picked from raw markers (cl49 cortex -> immature granule: the engine's clear_top is
            #   L2/3 IT and immature granule, a DG-region term, is not in anchors). Exempt when the regions match.
            if tid in idx and _amatch is None and ge_.get("verdict")=="clear_top" and (ge_.get("anchors") or []):
                def _rg(tt):
                    cur=tt; seen={tt}
                    while True:
                        ps=[q for q in (idx.get(cur,{}).get("P") or []) if q in idx]
                        if not ps: return cur
                        par=ps[0]
                        if idx.get(par,{}).get("T")=="class" or par in seen: return cur
                        seen.add(par); cur=par
                _dom=set(ge_.get("dominant_region") or []); _crg=idx.get(_rg(tid),{}).get("L")
                if _dom and _crg not in _dom:
                    _topa=(ge_.get("anchors") or [{}])[0].get("label")
                    add(lf,"OFF_EVIDENCE","warn",
                        f"'{t_.get('L')}' is not in graph_evidence.anchors (region '{_crg}'). The engine's clear_top is '{_topa}' ({'/'.join(sorted(_dom))}) — you may have picked a term from raw markers that the region-coherent engine had excluded.",
                        f"Use the anchor the engine surfaced ('{_topa}'), or demonstrate from hp / supporting that this term's OWN markers clearly dominate.")
            # ★ SPEC_UNSUPPORTED (the valid-evidence gate): when the chosen term has spec_pass==false in
            # identity_scores (fewer than half of its own discriminative markers with share<=3 are ON — an identity
            # force-fit through broad/leaky markers, e.g. calling a non-hippocampal leaf CA1 because Wfs1 is shared,
            # or calling a non-CA3 leaf CA3 from one marker), terminating at a specific celltype is not allowed.
            # There is no valid positive evidence -> use a class floor when the lineage is clear, Unknown when it is
            # not.
            # (spec_pass is True when a term's own markers are all off-panel (spec_n=0), so a legitimate panel-blind
            # termination still passes.)
            isc=p.get("identity_scores") or {}
            _spass={}
            for c in ([isc.get("top")]+(isc.get("ranked") or [])):
                if c and c.get("term_id"): _spass.setdefault(c["term_id"], c.get("spec_pass"))
            if bool(d.get("subclass_resolved",True)) and _spass.get(tid) is False:
                add(lf,"SPEC_UNSUPPORTED","warn",
                    f"Fewer than half of the discriminative own markers (share<=3) of '{idx.get(tid,{}).get('L',tid)}' are ON (spec_pass=false) — an identity force-fitted through broad/leaky markers (no valid positive evidence).",
                    "Do not settle on the specific term -> when the lineage is clear, use a class floor (subclass_resolved=false); when it is not, use Unknown. Keep it only if hp/supporting genuinely show that type's markers dominating (§5 spec_pass).")
            # ★ SHOULD_SPLIT_DEEPER (all tiers): when >=2 one-level children of a term terminated as a celltype are
            # resolvable in this cluster through non-pan markers (= a further split would separate them), recommend
            # splitting.
            # (The same rule applies whether the term is a class or CA1. It uses the profile's candidate_subtypes
            # scorecard directly.)
            if tid in idx:
                cs={c["term_id"]:c for c in (p.get("candidate_subtypes") or [])}
                kids=(cs.get(tid) or {}).get("children_resolvable") or []
                strong_kids=[k for k in kids if k.get("n_nonpan_present",0)>=1]
                if len(strong_kids)>=2:
                    add(lf,"SHOULD_SPLIT_DEEPER","warn",
                        f"The leaf was terminated as '{idx[tid].get('L')}', yet the discriminative markers of {len(strong_kids)} one-level children"
                        f"({', '.join(k['label'] for k in strong_kids[:4])}) coexist in this cluster.",
                        "Splitting further would separate them -> needs_subclustering. (If the children's markers are off-panel, or the split does not separate them, terminate as is.)")
            # ★ DF_MULTI_SUBTYPE: if in-round split detection relied on children_resolvable's strict-ON
            # (n_nonpan_present), quiet markers on sparse data such as 2 um would never clear ON and it would not
            # fire. df_present uses fold-ON of DEFINING markers (lf>=1.5 and detection>=0.25) and therefore catches
            # quiet-but-enriched markers — exposing, during the round, the very signal the finalize split-review
            # uses, so cl11 (a CA1+CA2+CA3 mixture) is steered to needs_subclustering before finalize. Top-level
            # (depth-0) only, celltype terminations only, >=2 distinct subtypes.
            # ★ CONFUSION_UNRESOLVED: chosen a specific term whose ConfusionPair PARTNER also anchors
            # (markers can't separate them — FC vs CA2 share Pcp4/Amigo2/Rgs14). Don't force-fit one -> Unknown.
            cu=(p.get("graph_evidence") or {}).get("confusion_unresolved")
            if cu and bool(d.get("subclass_resolved",True)):
                chosen=idx.get(tid,{}).get("L","")
                if chosen and (chosen==(cu.get("terms") or [None])[0] or chosen in (cu.get("terms") or [])):
                    act=cu.get("action") or "unknown"; pair=cu.get("pair"); partners=', '.join((cu.get("terms") or [])[1:4])
                    if act=="split":
                        add(lf,"CONFUSION_UNRESOLVED","warn",
                            f"The *own* markers of '{chosen}' and of its partner(s) ({partners}) are BOTH ON (top: {cu.get('top_distinct_on')} / partner: {cu.get('partner_distinct_on')}) "
                            f"— a mixture of the two types ({pair}).",
                            "-> needs_subclustering (split it). After splitting, decide each fragment as CA2 / FC / Unknown / the parent class.")
                    elif act=="resolve":
                        add(lf,"CONFUSION_UNRESOLVED","warn",
                            f"This is a confusion pair ({pair}) but only one side's own markers are ON -> the evidence supports '{cu.get('resolves_to')}'.",
                            f"-> decide '{cu.get('resolves_to')}' (the other side's own markers are OFF). If the current choice '{chosen}' is not that one, replace it.")
                    else:
                        _cls=cu.get("resolves_to")
                        add(lf,"CONFUSION_UNRESOLVED","warn",
                            f"'{chosen}' and its partner(s) ({partners}) are anchored only by shared markers, and the own markers of both are OFF or off-panel — they cannot be separated by markers ({pair}).",
                            (f"-> do not commit to either; decide the common parent class '{_cls}' (e.g. when the FC discriminator is off-panel, use '{_cls}' instead of committing to CA2). "
                             f"If a discriminator (position etc.) is clear, keep that type." if _cls else
                             "-> decide Unknown (do not commit to either one)."))
            if dec=="celltype" and lf.count(".")==0:
                dfp=sorted({x for x in (p.get("df_present") or []) if x})
                if len(dfp)>=2:
                    add(lf,"DF_MULTI_SUBTYPE","warn",
                        f"The DEFINING markers of >=2 subtypes coexist fold-ON in this cluster ({', '.join(dfp[:4])}) — "
                        f"it was terminated as the single celltype '{idx.get(tid,{}).get('L',tid)}', but it may be a mixture of identities.",
                        "-> needs_subclustering (split within the round). On sparse data strict-ON is weak, but the fold signal is real. "
                        "If a split does not separate them, or one type clearly dominates, terminate as is.")
            # ★ SHOULD_SPLIT_MIXTURE: graph_evidence.mixture_subtypes = the own markers of >=2 child subtypes of the
            # dominant region coexisting WEAKLY (a diluted mixture). It is a weaker signal than df fold-ON
            # (DF_MULTI_SUBTYPE) and therefore catches under-clustered mixtures on sparse data such as 2 um
            # (cl2 Drd1+Drd2 / cl15 Cux2+Rorb / cl7.1 L5+L6). A leaf with no own markers at all (cl4) has an empty
            # mixture_subtypes and does not fire. depth<=1 only (prevents over-splitting and loops; complemented by
            # max_split_depth and split_no_gain).
            if dec=="celltype" and lf.count(".")<=1:
                ms=(p.get("graph_evidence") or {}).get("mixture_subtypes") or []
                if len(ms)>=2:
                    labs=", ".join("%s(%s)"%(m.get("label",""), "/".join(m.get("markers_present") or [])) for m in ms[:4])
                    add(lf,"SHOULD_SPLIT_MIXTURE","warn",
                        f"Terminated as '{idx.get(tid,{}).get('L',tid)}', yet the own markers of >=2 sibling subtypes from the same region coexist (even if diluted): {labs}.",
                        "-> needs_subclustering; split at a high enough resolution (each anchor is weak because it is a mixture; over-splitting is fine because same-label pieces merge). "
                        "If the split yields only one type, terminate with split_no_gain.")
            # ★ SIBLING_RESOLVABLE: looks at *siblings*, not only children. If the chosen term coexists with >=1
            # other normal sibling subtype within a subtype_groups group (carrying the discriminative markers of the
            # same parent), this leaf may also contain that sibling type as a minority -> recommend splitting. This
            # is the signal that lets the agent catch, with no post-processing, a case terminated as CA3 that
            # actually contained 23% CA1 (absolute identity cannot see a diluted minority).
            # depth-0 (top level) only: at depth>=1 the leaf is already a split product (handled by Step 5
            # split_no_gain), so sibling cross-listing there is mostly noise.
            if tid in idx and lf.count(".")==0:
                sibs=[]
                for grp in (p.get("subtype_groups") or []):
                    ct=grp.get("child_term_ids") or []
                    if tid not in ct: continue
                    for c in ct:
                        if c==tid or c not in idx: continue
                        lab=idx[c].get("L","")
                        if not lab or lab in sibs or "DEPRECATED" in lab: continue
                        if (idx[c].get("D") or "normal")!="normal": continue
                        sibs.append(lab)
                if sibs:
                    add(lf,"SIBLING_RESOLVABLE","warn",
                        f"Terminated as '{idx[tid].get('L')}', yet the discriminative markers of sibling subtypes under the same parent "
                        f"({', '.join(sibs[:4])}) coexist in this cluster.",
                        "A minority sibling type may be mixed in -> needs_subclustering. (If the split does not separate them, terminate as is.)")
            hit=oh.get(tid)
            if hit and bool(d.get("subclass_resolved",True)):
                up_auc={_U(r[0]):r[1] for r in (p.get("up") or [])}
                hp_pct={_U(r[0]):r[3] for r in (p.get("hp") or []) if len(r)>3}
                pl=pool(p)
                rson=set(map(_U,hit.get("R_on",[])))|set(map(_U,hit.get("S_on",[])))
                resc=any(g in SPEC and up_auc.get(g,0)>=0.6 for g in set(map(_U,hit.get("R_on",[]))))
                if len(rson&pl)<2 and not resc:
                    add(lf,"RS_LT2_BUT_RESOLVED","warn",
                        "subclass_resolved=true but fewer than 2 on-panel R/S markers.",
                        "Lower it to class-final (subclass_resolved=false) or consider a different term.")
                N_on=set(map(_U,hit.get("N_on",[])))-PAN
                if any(up_auc.get(g,0)>=0.6 or hp_pct.get(g,0)>=0.9 for g in N_on):
                    add(lf,"NEGATIVE_LEAK","warn",
                        "A negative marker of the chosen term is strongly expressed (a leak).",
                        "A close sibling is possible — use needs_subclustering or consider a different term.")
    return V

def cmd_check(a):
    st=_load_state(a.out); decisions=json.load(open(a.decisions))
    V=_validate(decisions, st, a.out)
    vp=os.path.join(a.out,"_annotate_violations.json")
    json.dump(V, open(vp,"w"), ensure_ascii=False, indent=1)
    ne=sum(1 for v in V if v["severity"]=="error"); nw=len(V)-ne
    for v in V: print(f"  [{v['severity']}] {v['cluster_path']} {v['check']}: {v['message']}", flush=True)
    print(json.dumps({"clean":len(V)==0,"violations":len(V),"errors":ne,"warnings":nw,"file":vp}, ensure_ascii=False))

def _res_for(k):
    """Leiden start resolution per expected #types (k). Scaled by heterogeneity (0.25 per type), floored at
    0.1 and capped at 2.0. Start low to avoid over-split noise; when a residual mixture remains
    (SHOULD_SPLIT_MIXTURE) the round loop raises it by gamma x1.5. gamma <-> #clusters is data dependent,
    so this is a starting point for convergence, not a *prediction*. (k=2 -> 0.5, 4 -> 1.0, 6 -> 1.5; the
    floor of 0.1 is the lower bound used when over-fragmenting.)"""
    return round(min(2.0, max(0.1, 0.25*k)), 2)
def _finalize_review(decisions, st, out, max_depth=1, large_min_n=1500):
    """For the finalize gate: precisely select the large leaves that are 'worth splitting once more'.
    Two kinds:
    (1) bare class (subclass unresolved): >=2 one-level *children* that a split would separate
        (children_resolvable, non-pan markers).
    (2) resolved subclass: >=1 *sibling* subtype under the same parent coexists (subtype_groups) -> a
        minority sibling mixed in (23% CA1 inside CA3). Absolute identity cannot see a diluted minority,
        so this gate is the backstop that needs no post-processing.
    Common conditions: decision is celltype; large (n >= large_min_n); dense (not neuropil; low_frac<0.6);
    reason_code != panel_limited (if the child/sibling markers are off-panel, splitting is pointless);
    split depth <= max_depth (leaves already split several times are excluded -> no infinite loop).
    Unlike detection (SHOULD_SPLIT_DEEPER / SIBLING_RESOLVABLE), what is caught here is an *action*
    candidate the caller can put to the user."""
    import glob
    from collections import Counter
    prof={}
    for pf in sorted(glob.glob(os.path.join(out,"cluster_gene_profiles_round*.json"))): prof.update(json.load(open(pf)))
    _full_idx=_load_ontology_index(out) or {}
    idx=_full_idx.get("terms",{})
    # ★ ContextAxis escape hatch: when a mixture that will not separate is a declared spatial-axis
    #   (zonation/layer) continuum, label it by the axis pole and stop instead of re-splitting forever. The
    #   set of pole subtype labels of the ca axis is prepared here (lower-cased).
    _CA=_full_idx.get("ca") or []
    def _axis_pole_labels(ax):
        out_s=set()
        for p in (ax.get("poles") or []):
            for v in (p.get("subtypes") or {}).values():
                if v: out_s.add(str(v).strip().lower())
        return out_s
    def _axis_match(lineage, match_labels):
        """If the lineage / region / anchor label matches the applies_to of some ca axis, return that axis
    (region- and anchor-aware, the same principle as _context_axis_eval)."""
        cl={str(lineage).strip().lower()} | {str(x).strip().lower() for x in match_labels if x}
        for ax in _CA:
            if ax.get("lineage") and str(ax["lineage"]).strip().lower()!=str(lineage).strip().lower(): continue
            if any(str(a).strip().lower() in cl for a in (ax.get("applies_to") or [])): return ax
        return None
    lab2D={}
    for t in idx.values(): lab2D.setdefault(t.get("L"),t.get("D"))
    ncnt=Counter(st.get("cluster_paths") or [])
    cands=[]; axis_cont=[]; converged=[]
    for lf in st["leaves"]:
        d=decisions.get(lf)
        if not d or d.get("decision")!="celltype": continue
        if (d.get("reason_code") or "")=="panel_limited": continue
        if lf.count(".")>max_depth: continue                            # split-depth cap (prevents infinite loops)
        n=int(ncnt.get(lf,0))
        if n<large_min_n: continue
        p=prof.get(lf) or {}
        dfp=set(p.get("df_present") or [])        # terms whose DEFINING marker is fold-ON in this leaf
        dens=((p.get("identity_evidence") or {}).get("density") or {})
        lfrac=dens.get("low_frac")
        if lfrac is not None and lfrac>=0.6: continue                   # low density (neuropil) -> splitting is pointless
        tid=(d.get("term_id") or "").strip()
        resolved=bool(d.get("subclass_resolved",True))
        _ge=p.get("graph_evidence") or {}
        _mix=_ge.get("mixture_subtypes") or []
        _cu=_ge.get("confusion_unresolved") or {}
        # ★ MIXTURE backstop: the df_present (defining marker fold-ON) >= 2 path misses a mixture involving a type
        #   that has no DEFINING marker (cl64 FC has none, cl66 immature granule has none -> strong<2 and they
        #   drop out). A confusion action=split, or mixture_subtypes (the own markers of >=2 children of the
        #   dominant region coexisting), detected directly by graph_evidence is surfaced as a split candidate at
        #   finalize even when the agent terminated the leaf — a backstop that does not depend on the agent's
        #   decision. Two guards prevent false firing:
        #   (a) lineage agreement — the root lineage of the agent's label == graph_evidence.lineage (for glia and
        #       vascular cells the agent labels against the ambient glut graph_evidence, so they disagree and are
        #       excluded; this blocks bogus candidates such as 'split an oligodendrocyte into L5 IT')
        #   (b) the mixture subtypes lie inside dominant_region — excludes cross-region ambient bleed (L2/3 inside
        #       CA1) and keeps only genuine co-populations of the same region (immature granule inside DG).
        def _root_lab(tt):
            x=tt; seen=set()
            while x and x not in seen:
                seen.add(x); ps=[q for q in (idx.get(x,{}).get("P") or []) if q in idx]
                if not ps: return idx.get(x,{}).get("L")
                x=ps[0]
            return idx.get(x,{}).get("L")
        def _reg_of(tt):
            cur=tt; seen={tt}
            while True:
                ps=[q for q in (idx.get(cur,{}).get("P") or []) if q in idx]
                if not ps: return cur
                par=ps[0]
                if idx.get(par,{}).get("T")=="class" or par in seen: return cur
                seen.add(par); cur=par
        _lin_ok=bool(_ge.get("lineage") and tid in idx and _root_lab(tid)==_ge.get("lineage"))
        _dom=set(_ge.get("dominant_region") or [])
        _lab2tid={t.get("L"):k for k,t in idx.items()}
        _mix_regs=[(m, idx.get(_reg_of(_lab2tid[m["label"]]),{}).get("L")) for m in _mix if _lab2tid.get(m.get("label"))]
        if _dom:
            _mix_inreg=[m for m,r in _mix_regs if r in _dom]
        else:
            # ★ When dominant_region is empty: a laminar cortical mixture often shares several region-exclusive
            #   markers weakly across layers, so region scoring cannot fix a dominant_region (L2/3, L4 and L5 are all
            #   neocortical yet _dom=[]). The old condition 'the mixture lies inside dominant_region' then rejected
            #   everything -> the cortex never became a split candidate and its layers collapsed to the generic node.
            #   Instead, when the mixture subtypes gather into **the same region-group with >=2 members** (e.g. all
            #   neocortical), treat it as a genuine mixture of that region and surface it. Cross-region ambient (a
            #   lone L2/3 inside CA1) cannot reach two members of one region and is still excluded, so the
            #   false-firing guard is preserved.
            _rc=Counter(r for m,r in _mix_regs if r)
            _common={r for r,c in _rc.items() if c>=2}
            _mix_inreg=[m for m,r in _mix_regs if r in _common]
        mixture_trigger=_lin_ok and ((_cu.get("action")=="split") or (len(_mix_inreg)>=2))
        # ★ BIMODAL backstop trigger: per-cell anti-correlated sibling programs (≥2) the
        #   mean-based mixture detector missed (balanced sibling-region mix). Region-independent, so it
        #   needs no _mix_inreg region check; _lin_ok still guards cross-lineage ambient.
        _bim=p.get("bimodal_subtypes") or []
        bimodal_trigger=_lin_ok and len(_bim)>=2
        if not resolved:
            cs={c["term_id"]:c for c in (p.get("candidate_subtypes") or [])}
            kids=(cs.get(tid) or {}).get("children_resolvable") or []
            strong=[]
            for k in kids:
                lab=k.get("label","")
                if k.get("n_nonpan_present",0)<1 or lab in strong or "DEPRECATED" in lab: continue
                if (lab2D.get(lab) or "normal")!="normal": continue
                strong.append(lab)
            # ★ REGION-FLOOR descend backstop: when the leaf is REGION-ESTABLISHED (dominant_region
            #   set), its children_resolvable are that region's OWN subtypes — cross-region ambient false children
            #   are already excluded by region-gating — so a bare-region floor with ≥2 resolvable region-children
            #   was simply NOT descended, and must be HELD even without aggregate DF-confirm (dilution hides a
            #   subtype's defining marker at the mixed mean; the descend is what un-dilutes it). Guards the exact
            #   failure this run hit: a whole cortex floored to `neocortical glutamatergic neuron` with L2/3/L4/L5
            #   resolvable but never descended. For a NON-region-established floor keep the strict DF-confirm
            #   filter (ambient/shared-marker guard, unchanged).
            _region_gated=bool(_ge.get("dominant_region"))
            if not (_region_gated and len(strong)>=2):
                strong=[lab for lab in strong if lab in dfp]            # non-region floor: only DF-CONFIRMED children
            if len(strong)<2 and not mixture_trigger and not bimodal_trigger: continue   #   >=2 children (region floor: subtypes unique to that region) OR a graph_evidence mixture OR per-cell bimodality
            split_kind="children"
        else:
            # ★ resolved subclass: looks at *siblings* rather than children. If the chosen subtype coexists with >=1
            # other normal sibling subtype in subtype_groups (same parent), a sibling type is present in this leaf as
            # a minority (CA1 inside CA3). Absolute identity cannot see a diluted minority, so this gate is the
            # backstop that needs no post-processing.
            # ★ depth-0 (top level) only: sibling cross-listing is noisy, and at depth>=1 the leaf is already a split
            # product (handled by Step 5 split_no_gain), so re-splitting would loop. The lesson of CA1/CA3 is
            # fundamentally about top-level under-clustering.
            if lf.count("."): continue
            strong=[]
            for grp in (p.get("subtype_groups") or []):
                ct=grp.get("child_term_ids") or []
                if tid not in ct: continue
                for c in ct:
                    if c==tid or c not in idx: continue
                    lab=idx[c].get("L","")
                    if not lab or lab in strong or "DEPRECATED" in lab: continue
                    if (lab2D.get(lab) or "normal")!="normal": continue
                    strong.append(lab)
            strong=[lab for lab in strong if lab in dfp]                # ★ only DF-CONFIRMED siblings — kills ambient false siblings (BLA/PAG/thalamic in a cortical leaf)
            if len(strong)<1 and not mixture_trigger and not bimodal_trigger: continue
            split_kind="siblings"
        # k = an estimate of how many distinct types would separate: DF-confirmed siblings (strong) u graph_evidence.mixture_subtypes (diluted coexistence) u bimodal.
        if not strong and _mix_inreg: strong=[m.get("label","") for m in _mix_inreg if m.get("label")]
        if not strong and _bim: strong=[m.get("label","") for m in _bim if m.get("label")]
        nexp=max(len(strong), len(_mix), len(_bim), 2)
        depth=lf.count(".")
        mix_labs={str(x).strip().lower() for x in strong} | {str((m.get("label") or "")).strip().lower() for m in _mix}
        mix_labs.discard("")
        # ★ (1) ContextAxis escape hatch: the leaf has already been split at least once (depth>=1) and still shows
        #   the same mixture, and that mixture consists of the poles of a declared spatial axis = a continuum that
        #   further splitting will not separate -> drop it from the re-split candidates and terminate as
        #   axis_continuum (the caller labels it by the context_axis pole).
        _anchor_labs=[a.get("label") for a in (_ge.get("anchors") or []) if a.get("label")]
        _ax=_axis_match(_ge.get("lineage"), list(_dom)+_anchor_labs+[d.get("label","")])
        if depth>=1 and _ax is not None:
            _poles=_axis_pole_labels(_ax)
            if mix_labs and mix_labs<=_poles:                        # the mixture consists entirely of that axis's pole subtypes -> a continuum
                axis_cont.append({"cluster_path":lf,"n_cells":n,"label":d.get("label",""),
                                  "axis":_ax.get("name"),"poles":sorted(_poles),
                                  "action":"label by context_axis pole (stop re-splitting — spatial continuum won't separate)"})
                continue
        # ★ (2) Delta-separation stop: an already-split leaf (depth>=1) whose mixture is identical to its parent's (separation gain ~ 0) has converged -> re-splitting is futile -> stop (class_floor).
        if depth>=1:
            _par=lf.rsplit(".",1)[0]
            _pmix={str((m.get("label") or "")).strip().lower() for m in ((prof.get(_par,{}).get("graph_evidence") or {}).get("mixture_subtypes") or [])}
            _pmix.discard("")
            if _pmix and mix_labs and mix_labs==_pmix:               # the same mixture as the parent -> the split cannot separate it -> stop
                converged.append({"cluster_path":lf,"reason":"mixture unchanged vs parent (Δ separation ≈ 0)"})
                continue
        # ★ (3) gamma escalation: each retry (depth) raises the Leiden resolution by x1.5 (capped at 2.0) — start low, raise when it does not separate, and converge.
        start_res=round(min(2.0, _res_for(nexp) * (1.5 ** depth)), 2)
        cands.append({"cluster_path":lf,"n_cells":n,"label":d.get("label",""),"term_id":tid,
                      "subclass_resolved":resolved,"split_kind":split_kind,
                      "density_pct":dens.get("pct"),"low_frac":lfrac,"split_depth":depth,
                      "reason_code":d.get("reason_code",""),"resolvable_children":strong[:8],
                      "n_expected":nexp,"start_resolution":start_res})
    cands.sort(key=lambda c:-c["n_cells"])
    # ★ FLOORED-OVER-ANCHOR backstop: the agent floored a leaf to a CLASS/region-group ancestor
    #   even though the engine's OWN verdict was `clear_top` with a deeper subtype anchor carrying ≥2 of its
    #   OWN exclusive markers ON (SKILL R2/R3 = REAL) and NO strong negative leak. This is a RELABEL, not a
    #   split — no re-clustering, so ZERO over-fragmentation risk (unlike a marker-axis descend, which on dense
    #   whole-transcriptome Stereo-seq fragments into cross-region ambient — the exact failure that makes the
    #   agent over-floor in the first place). Grounded entirely in the engine's already-region-coherence-gated
    #   signals (verdict/excl_on/neg_strong/anchor.path), so it inherits their ambient protection — it flags
    #   L5/L6/L4/L2-3 IT, subiculum, claustrum, DG-granule, thalamic-relay calls the agent buried at
    #   `glutamatergic neuron`, never a cross-region force-fit. Advisory HOLD: reassign to the anchor OR justify.
    floored_over=[]
    FLOOR_MIN_N=50
    for lf in st["leaves"]:
        d=decisions.get(lf)
        if not d or d.get("decision")!="celltype": continue
        n=int(ncnt.get(lf,0))
        if n<FLOOR_MIN_N: continue
        _clab=str(d.get("label","")).strip()
        if not _clab: continue
        _p=prof.get(lf) or {}
        _ge2=_p.get("graph_evidence") or {}
        if _ge2.get("verdict")!="clear_top": continue                # engine's own clean-top conclusion (region-coherence gated)
        _anc=_ge2.get("anchors") or []
        if not _anc: continue
        _a=_anc[0]                                                   # top anchor
        # REAL by R3 = ≥2 of its OWN exclusive low-share markers ON (excl_on≥2) OR its DEFINING marker ON
        #   (df_on≥1 — DF-override: a single defining marker is sufficient, so a clear_top df-anchored subtype
        #   like Fibrous astrocyte [Gfap, shared→excl_on=0 but df_on=1] is REAL and must not be buried under
        #   the bare class). Either path qualifies; a strong negative leak still disqualifies.
        if int(_a.get("excl_on",0) or 0)<2 and int(_a.get("df_on",0) or 0)<1: continue
        if int(_a.get("neg_strong",0) or 0)>0: continue             # a STRONG negative leak = a legitimate reason to floor → skip
        _apath=[str(x).strip() for x in (_a.get("path") or [])]
        _alab=str(_a.get("label","")).strip()
        if not _alab or _clab==_alab: continue                      # agent already took the anchor → fine
        if _clab not in _apath[:-1]: continue                       # chosen must be an ANCESTOR of the anchor on ITS path (= stopped short)
        if (lab2D.get(_alab) or "normal")!="normal": continue       # normal-only (disease_context set → not a floor target)
        if _is_state_term(_alab, _a.get("tier","")): continue       # ★ disease/reactive STATE term (DOL, DAA, A1/A2 reactive…) — its broad
        #   defining markers (Serpina3n/B2m/C3) are ambiently ON so df_on fires falsely; states are never a
        #   base-identity floor target (handled by R7, not here). Keeps the df_on path clean (Fibrous/Protoplasmic
        #   = real subtypes pass; DOL/reactive excluded).
        floored_over.append({"cluster_path":lf,"n_cells":n,"floored_label":_clab,
            "anchor_label":_alab,"anchor_term_id":_a.get("term_id"),"tier":_a.get("tier"),
            "excl_on":int(_a.get("excl_on",0)),"excl_markers":_a.get("excl_markers_on") or [],
            "df_on":int(_a.get("df_on",0) or 0)})
    floored_over.sort(key=lambda c:-c["n_cells"])
    # ★ DF-PRESENT SUBTYPE backstop: a leaf floored to a class/group whose `df_present` names a
    #   DEEPER same-lineage subtype whose SPECIFIC (low-share) defining marker is fold-ON (≥3× enriched, detected
    #   ≥25% — a genuine signal, not ambient). The floored_over_anchor scan above misses this because the top
    #   ANCHOR is not that subtype: in a tiny ambient-dominated fragment (rare cortical interneurons on dense
    #   Stereo-seq) a cross-lineage vascular/immune type (pericyte/endothelial KDE smear) out-ranks the interneuron
    #   for the top anchor (verdict≠clear_top), so the subtype's identity is DETECTED (df_present) but never
    #   SELECTED → floored to `GABAergic neuron`/`CGE-derived`. This backstop surfaces it from df_present directly.
    #   Two guards keep it from touching anything else: (a) the subtype must be a DESCENDANT of the floored label
    #   (df_present is lineage-gated AND the ancestor check ⇒ same lineage, never cross-lineage ambient), (b) its
    #   defining marker must be LOW-SHARE (share≤2, a specific discriminator like Vip/Lamp5/Pvalb) — a broad/shared
    #   defining marker never triggers. RELABEL, not split (no re-cluster).
    _dfr_share={}
    for _t in idx.values():
        for _g in set((_t.get("DF") or [])+(_t.get("R") or [])): _dfr_share[str(_g).upper()]=_dfr_share.get(str(_g).upper(),0)+1
    def _anc_ids(tid):
        seen=set(); stack=[tid]                       # ALL parents (DAG — a term can have >1 IS_A parent, e.g. Vip+ under BOTH
        while stack:                                  # cortical-GABAergic-interneuron AND CGE-derived after the flatten)
            x=stack.pop()
            if x in seen or x not in idx: continue
            seen.add(x)
            for p in (idx.get(x,{}).get("P") or []): stack.append(p)
        return seen
    _lab2tid_full={t.get("L"):k for k,t in idx.items()}
    def _root_id(tid):
        x=tid; seen=set()
        while x and x not in seen:
            seen.add(x); ps=[q for q in (idx.get(x,{}).get("P") or []) if q in idx]
            if not ps: return x
            x=ps[0]
        return x
    _DFS_MIN_N=5    # interneurons are RARE & tiny (7-35 cell fragments) — a high floor defeats rare-cell recovery; the
                    #   cross-lineage-hijack + ancestor + defining-fold-ON guards (not size) keep it honest. Agent verifies.
    df_present_sub=[]
    for lf in st["leaves"]:
        d=decisions.get(lf)
        if not d or d.get("decision")!="celltype": continue
        n=int(ncnt.get(lf,0))
        if n<_DFS_MIN_N: continue
        _ctid=(d.get("term_id") or "").strip()
        if not _ctid or _ctid not in idx: continue
        _p=prof.get(lf) or {}
        _ge3=_p.get("graph_evidence") or {}
        _anchs=_ge3.get("anchors") or []
        # ★ CROSS-LINEAGE HIJACK guard — only fire when the top anchor is CROSS-LINEAGE (a different root than the
        #   leaf's lineage), i.e. an ambient vascular/immune type out-ranked the real same-lineage identity. This
        #   is the interneuron-in-a-tiny-fragment case (pericyte/endothelial anchor over Vip+/Lamp5+). When the top
        #   anchor is SAME-lineage (oligo→Myelin-forming, cortex→a layer) the subtype was NOT hijacked → handled by
        #   floored_over_anchor / the split gate → excluded here (no redundant flood).
        _lin_root=_lab2tid_full.get(_ge3.get("lineage"))
        _lin_root=_root_id(_lin_root) if _lin_root else None
        _top_root=_root_id(_anchs[0].get("term_id")) if _anchs else None
        if not (_lin_root and _top_root and _top_root!=_lin_root): continue  # require a cross-lineage top anchor (ambient hijack)
        for _slab in (_p.get("df_present") or []):
            _stid=_lab2tid_full.get(_slab)
            if not _stid or _stid==_ctid or _stid not in idx: continue
            if (lab2D.get(_slab) or "normal")!="normal": continue           # normal only
            if _is_state_term(_slab, idx.get(_stid,{}).get("T","")): continue
            if _root_id(_stid)!=_lin_root: continue                         # subtype must be the leaf's OWN lineage (not the ambient one)
            if _ctid not in _anc_ids(_stid): continue                       # floored label an ANCESTOR of the subtype (stopped short)
            # NO share filter here: an interneuron subclass marker (Pvalb share=8, Vip=4) recurs across regional
            #   GABAergic variants, but the ancestor check already pins it to THIS branch (cortical); the
            #   cross-lineage-hijack guard above is the real protection. Require only that the subtype HAS a DF.
            _sdf=list(idx.get(_stid,{}).get("DF") or [])
            if not _sdf: continue
            df_present_sub.append({"cluster_path":lf,"n_cells":n,"floored_label":str(d.get("label","")).strip(),
                "subtype_label":_slab,"subtype_term_id":_stid,"specific_df":_sdf[:4],
                "hijack_anchor":(_anchs[0].get("label") if _anchs else None)})
            break
    df_present_sub.sort(key=lambda c:-c["n_cells"])
    # ★ MUST-DESCEND — the un-bypassable "always descend before floor" enforcement. The other
    #   backstops key on a pre-descend SIGNAL (df_present / children_resolvable / a clear_top anchor), but a
    #   subtype is DILUTED at the whole-cluster mean — a real Fibrous astrocyte (Gfap auc 0.53 across the mixed
    #   astrocyte pool, 0.99 only AFTER the split concentrates it) shows NOTHING at round 1, so no signal-based
    #   gate can flag it. The only correct rule is the SKILL's: **descend EVERY floor to CHECK.** So: a leaf the
    #   agent floored to a class / region-group (subclass_resolved=False) that (a) was NEVER descended (top-level,
    #   no "."), (b) is large & dense, (c) is NOT panel_limited/compartment, and (d) has ≥2 KG children to descend
    #   into — MUST be descended at least once before it can be finalized. `cmd_finalize` blocks `--accept-large`
    #   on these (only an explicit `--force` + rationale passes) so a run cannot init→judge→floor→finalize without
    #   a single descend. After a descent that genuinely fails to separate, the sub-leaves are depth≥1 → this no
    #   longer fires → an honest floor is finalizable. Threshold-free, dilution-proof (it does NOT read markers).
    _kid_parents=set()
    for _t in idx.values():
        for _pp in (_t.get("P") or []): _kid_parents.add(_pp)
    must_descend=[]
    for lf in st["leaves"]:
        if lf.count("."): continue                                          # only top-level = never descended
        d=decisions.get(lf)
        if not d or d.get("decision")!="celltype": continue
        if bool(d.get("subclass_resolved",True)): continue                  # only class/region FLOORS (agent stopped short)
        if (d.get("reason_code") or "") in ("panel_limited",): continue
        n=int(ncnt.get(lf,0))
        if n<large_min_n: continue
        _p=prof.get(lf) or {}
        _lf=((_p.get("identity_evidence") or {}).get("density") or {}).get("low_frac")
        if _lf is not None and _lf>=0.6: continue                           # neuropil (low density) → nothing to split
        _tid=(d.get("term_id") or "").strip()
        if not _tid or _tid not in _kid_parents: continue                   # its class/region node has KG children to descend into
        must_descend.append({"cluster_path":lf,"n_cells":n,"label":str(d.get("label","")).strip()})
    must_descend.sort(key=lambda c:-c["n_cells"])
    # ★ PRINCIPLED resolution: Leiden resolution γ has NO closed-form map to #clusters or which
    #   subtypes split — it is data-dependent (Traag 2019; γ in CPM = a DENSITY THRESHOLD, in modularity scales the
    #   null model w/ a resolution-limit √(2m) that UNDER-resolves sub-types inside a large cluster — the 2µm blob
    #   bug). So a formula like 0.12·n was unfounded, and the 0.5 cap throttled it. REDESIGN: γ is not PREDICTED,
    #   it is CONVERGED. `start_resolution` per candidate scales with expected #types (`_res_for`: 0.25·k, floor 0.1,
    #   ceil 2.0 — k=2→0.5, 4→1.0, 6→1.5). Start LOW (avoid over-fragmentation noise — high γ on sparse 2µm bins
    #   makes many ambient fragments), then the ROUND LOOP converges UP using the pipeline's unique oracle, the
    #   IDENTITY check: (a) a sub-leaf still flagged SHOULD_SPLIT_MIXTURE/DF_MULTI_SUBTYPE (≥2 subtype markers
    #   co-present) ⇒ still under-resolved → raise γ ×1.5 and re-split; (b) sub-leaves with no distinct identity
    #   (split_no_gain) ⇒ over-fragmented → already merge by identity, stop. Per-cluster (each cand carries
    #   start_resolution); the global value = max so a single --resolution covers the most heterogeneous path.
    _rec=round(max((c["start_resolution"] for c in cands), default=0.5), 2) if cands else None
    _resmap={c["cluster_path"]: c["start_resolution"] for c in cands}     # ★ per-cluster resolution
    return {"split_candidates":cands,"paths":[c["cluster_path"] for c in cands],
            "recommended_resolution":_rec,"resolution_map":_resmap,
            "floored_over_anchor":floored_over,                     # ★ RELABEL (not a split): leaves floored to an ancestor despite a clear_top + excl_on>=2 subtype anchor
            "df_present_subtype":df_present_sub,                    # ★ RELABEL: leaves floored because a df_present (specific defining fold-ON) same-lineage subtype was buried by ambient signal
            "must_descend":must_descend,                            # ★ UN-BYPASSABLE: a never-descended large class/region floor -> a descend is mandatory (--accept-large has no effect; only --force)
            "axis_continuum":axis_cont,                              # ★ (1) continuum: label by the context_axis pole and stop rather than re-splitting
            "converged_stop":converged,                             # ★ (2) Delta-separation stop: the same mixture as the parent -> re-splitting is futile
            "max_split_depth":max_depth,"large_min_n":large_min_n,
            "note":"Resolution is *converged*, not *predicted* (gamma <-> #clusters is data dependent; there is no closed form — Traag/CPM). "
                   "start_resolution scales with the expected #types (0.25 x k, floor 0.1, cap 2.0; k=2 -> 0.5, 4 -> 1.0, 6 -> 1.5; the old 0.12 x n with a 0.5 cap is retired). "
                   "Start low (to avoid over-split noise); when a residual mixture remains (SHOULD_SPLIT_MIXTURE / DF_MULTI_SUBTYPE), re-split at gamma x1.5 to converge further; "
                   "children that cannot be told apart (split_no_gain) merge by label and terminate. Per cluster this is cand.start_resolution. "
                   "★ Termination / escape (no infinite loop): start_resolution is raised automatically by split_depth as gamma x 1.5^depth (retries = depth, bounded by depth <= max_split_depth and gamma <= 2.0); "
                   "a leaf at depth>=1 whose mixture is identical to its parent is converged_stop (separation delta ~ 0 -> stop); a mixture that will not separate but is a declared ContextAxis continuum becomes "
                   "axis_continuum (label by the §4.3 pole and stop instead of re-splitting). Both are dropped from split_candidates so they raise no HOLD (the current label is final)."}

def cmd_finalize(a):
    import pandas as pd, glob, tarfile
    st=_load_state(a.out); decisions=json.load(open(a.decisions))
    # Validation: the helper does not change decisions. Only errors block (returning the case to the agent);
    # warnings are advisory (recorded, non-blocking).
    V=_validate(decisions, st, a.out)
    vp=os.path.join(a.out,"_annotate_violations.json")
    json.dump(V, open(vp,"w"), ensure_ascii=False, indent=1)
    errs=[v for v in V if v["severity"]=="error"]
    if errs and not a.force:
        print(f"[finalize][ABORT] {len(errs)} validation error(s). The h5ad was not written. "
              f"The caller should hand {vp} to the agent, have only those clusters re-decided, and run finalize again. "
              f"(The helper never changes a decision on its own. Pass --force to write it as is.)", flush=True)
        for v in errs: print(f"  [error] {v['cluster_path']} {v['check']}: {v['suggestion']}", flush=True)
        sys.exit(2)
    # ★ Split review for large bare-class leaves (the finalize gate): if a large, dense leaf remains whose
    #   subtypes would separate under a further split, stop instead of writing. The caller asks the user
    #   "split once more?"; on YES the paths go through subcluster -> re-judgment -> finalize, on NO
    #   finalize --accept-large records them as they are.
    review=_finalize_review(decisions, st, a.out, max_depth=a.max_split_depth, large_min_n=a.large_min_n)
    rp=os.path.join(a.out,"_annotate_finalize_review.json")
    json.dump(review, open(rp,"w"), ensure_ascii=False, indent=1)
    cands=review["split_candidates"]
    # ★ Loop-safe terminations (no re-split; the current labels are final): a ContextAxis continuum, or Delta-separation convergence
    for c in (review.get("axis_continuum") or []):
        print(f"[finalize] axis-exit: {c['cluster_path']} '{c['label']}' = '{c['axis']}' continuum -> stop re-splitting, terminate at the context_axis pole.", flush=True)
    for c in (review.get("converged_stop") or []):
        print(f"[finalize] converged: {c['cluster_path']} {c['reason']} -> stop re-splitting.", flush=True)
    # ★ MUST-DESCEND HARD BLOCK: a never-descended large class/region floor cannot be bypassed with
    #   `--accept-large` — finalize is possible only after a descend has actually been attempted (if it
    #   genuinely does not separate, the resulting sub-leaves are at depth>=1 and fall outside this block).
    #   This makes the failure mode "init -> judge -> floor -> finalize (round 1, zero descends)" impossible.
    #   To force it anyway: --force plus a rationale.
    md=review.get("must_descend") or []
    if md and not a.force:
        print(f"[finalize][BLOCK] {len(md)} large class/region floor(s) that were never descended ({sum(c['n_cells'] for c in md)} cells in total). "
              f"The h5ad was not written — **--accept-large cannot bypass this** (a descend is mandatory).\n"
              f"  A floor is earned only when a descend fails to separate (SKILL R8). These are not featureless; their subtypes are merely diluted in the aggregate,\n"
              f"  and a descend surfaces them (e.g. astrocyte -> Fibrous: whole-cluster Gfap auc 0.53, but 0.99 after the split).\n"
              f"   - For each leaf: `descend --node '<that leaf's class/region label>'` -> re-judge -> finalize. If it genuinely does not separate, the floor is then earned automatically.\n"
              f"   - To proceed with a legitimate reason: --force (and record the reason in rationale). Details: {rp}", flush=True)
        for c in md[:40]:
            print(f"  [descend!] {c['cluster_path']} n={c['n_cells']} '{c['label']}' — must be descended into this node's KG children", flush=True)
        sys.exit(3)
    # ★ FLOORED-OVER-ANCHOR HOLD: a RELABEL, not a split. The engine pointed at a specific subtype with
    #   clear_top + excl_on>=2 (its own exclusive markers ON, no strong negative leak) and the agent floored
    #   the leaf to its ancestor (class / region-group). Only the label needs to change to the anchor — no
    #   re-clustering — so there is zero over-split risk. On dense Stereo-seq this prevents the regression
    #   where the agent, wary of over-splitting during descend, covers L5 / L6 / subiculum with a generic
    #   label.
    fova=review.get("floored_over_anchor") or []
    if fova and not a.accept_large and not a.force:
        print(f"[finalize][HOLD] {len(fova)} leaf/leaves floored to an ancestor despite a clear_top anchor with excl_on>=2 ({sum(c['n_cells'] for c in fova)} cells in total). The h5ad was not written.\n"
              f"  This is a RELABEL problem, not a split problem — the engine already pointed at a specific subtype (>=2 of its own exclusive markers ON) as clear_top, and it was overwritten by the parent class/region.\n"
              f"   - Either replace each leaf's decision with the anchor subtype (label + term_id), or record a legitimate reason for flooring (a negative leak, a confusion pair, ...) in rationale and run finalize ... --accept-large.\n"
              f"  (No re-clustering is needed — edit only label/term_id in the decisions JSON and rerun finalize.) Details: {rp}", flush=True)
        for c in fova[:40]:
            print(f"  [relabel?] {c['cluster_path']} n={c['n_cells']} '{c['floored_label']}' → '{c['anchor_label']}' "
                  f"(excl_on={c['excl_on']}: {', '.join(c['excl_markers'][:4])})", flush=True)
        sys.exit(3)
    # ★ DF-PRESENT SUBTYPE HOLD: same-lineage subtype whose SPECIFIC defining marker is fold-ON
    #   (df_present) but was buried under a cross-lineage ambient anchor → RELABEL, not split.
    dfs=review.get("df_present_subtype") or []
    if dfs and not a.accept_large and not a.force:
        print(f"[finalize][HOLD] {len(dfs)} leaf/leaves floored to a parent class/group even though df_present holds a same-lineage subtype whose specific defining marker is fold-ON ({sum(c['n_cells'] for c in dfs)} cells in total). The h5ad was not written.\n"
              f"  A RELABEL problem — the subtype's own identity was detected (its specific defining marker is >=3x expressed), but in a tiny fragment cross-lineage ambient signal (vascular/immune) hijacked the top anchor and the label was never applied.\n"
              f"   - Either relabel to that subtype (after checking the defining marker in the data), or, if the contamination really dominates, record it in rationale and run finalize ... --accept-large.\n"
              f"  (No re-clustering is needed — edit only label/term_id.) Details: {rp}", flush=True)
        for c in dfs[:40]:
            print(f"  [relabel?] {c['cluster_path']} n={c['n_cells']} '{c['floored_label']}' → '{c['subtype_label']}' "
                  f"(specific defining fold-ON: {', '.join(c['specific_df'][:4])})", flush=True)
        sys.exit(3)
    if cands and not a.accept_large and not a.force:
        print(f"[finalize][HOLD] {len(cands)} large bare-class leaf/leaves can still be split. The h5ad was not written.\n"
              f"  The caller should ask the user: 'Shall we split these once more? (y/n)'.\n"
              f"   - YES -> subcluster --paths '{json.dumps(review['paths'])}' --resolution '{json.dumps(review.get('resolution_map') or {})}' (per cluster; a single value also works) -> re-judge -> finalize\n"
              f"   - NO  -> finalize ... --accept-large (keep the current labels as final)\n"
              f"  Details: {rp}", flush=True)
        for c in cands:
            print(f"  [split?] {c['cluster_path']} n={c['n_cells']} '{c['label']}'(density_pct={c['density_pct']}) "
                  f"-> children: {', '.join(c['resolvable_children'][:4])}", flush=True)
        sys.exit(3)
    nw=len(V)-len(errs)
    if nw: print(f"[finalize] {nw} advisory warning(s) recorded only; proceeding (not blocking). Details: {vp}", flush=True)
    if cands: print(f"[finalize] {len(cands)} large leaf/leaves finalized with their current labels, without splitting (--accept-large/--force).", flush=True)
    # passed (or --force): the agent's decisions are trusted and recorded as they are (no changes)
    # ★ Column names: standard annotation = `cellman_annotation`; the result of user interaction
    #   (post-standard, the SKILL §3.5 interactive phase) = `cellman_free_annotation` (chosen by the caller
    #   via --column). A free finalize reads the already-written annotated h5ad and adds only the free
    #   column, preserving the standard one.
    col=getattr(a,"column",None) or "cellman_annotation"
    out_h5ad=a.output or st.get("output") or os.path.join(a.out, re.sub(r"\.h5ad$","",os.path.basename(st["h5ad_path"]))+"_annotated.h5ad")
    src=out_h5ad if os.path.exists(out_h5ad) else st["h5ad_path"]
    adata=ad.read_h5ad(src)
    cps=np.array(st["cluster_paths"], dtype=object)
    ann=np.full(adata.n_obs,"",dtype=object)
    # ★ STATE-AXIS label synthesis: append base-agnostic active cell states as ", <state>" (multiple allowed)
    #   → "CD8 T cell, proliferating, interferon-stimulated". Source per leaf = the agent's explicit
    #   dec["states"] override if set, else the engine's `state_axis` (ENRICHED program markers) from the
    #   latest profile. Only for `celltype` decisions; states already named in the base label are dropped.
    #   Modules without a state axis (sa empty) get nothing → fully additive.
    _state_of={}; _state_means={}
    for _r in range(1, int(st.get("round",1))+1):
        _pf=os.path.join(a.out, f"cluster_gene_profiles_round{_r}.json")
        if os.path.exists(_pf):
            try:
                for _p,_e in json.load(open(_pf)).items():
                    _sx=[s.get("name") for s in (_e.get("state_axis") or []) if s.get("name")]
                    if _sx: _state_of[_p]=_sx
                    _mm=_e.get("state_marker_means")
                    if _mm: _state_means[_p]=_mm
            except Exception: pass
    # ★ STATE LINEAGE GATE: a lineage-restricted state (M2=myeloid, EMT/mesenchymal=epithelial,
    #   estrogen-response=luminal — declared via st.applies_to in the slim-index `sa`) is kept ONLY when the
    #   FINAL base celltype OR one of its IS_A ancestors is in applies_to. Global states (empty applies_to)
    #   always pass. Applied here (authoritative, on the FINAL base label — works on finalize-only re-run),
    #   so a marker-bleed state never lands on the wrong lineage ("CD8 T cell, M2 polarization" ✗).
    try: _full=json.load(open(os.path.join(a.out,"_ontology_index.json")))
    except Exception: _full={}
    _sa_appl={(x.get("name") or "").strip().lower(): {str(y).strip().lower() for y in (x.get("applies_to") or [])}
              for x in (_full.get("sa") or [])}
    _gterms=(_full.get("terms") or {}); _glab2tid={}
    for _tid,_t in _gterms.items(): _glab2tid.setdefault((_t.get("L") or "").strip().lower(), _tid)
    def _base_lineages(base_label):
        out=set(); bl=str(base_label).strip().lower(); out.add(bl)
        stack=[_glab2tid.get(bl)]; seen=set()
        while stack:
            x=stack.pop()
            if not x or x in seen or x not in _gterms: continue
            seen.add(x); t=_gterms[x]; L=(t.get("L") or "").strip().lower()
            if L: out.add(L)
            for p in (t.get("P") or []): stack.append(p)
        return out
    def _state_ok(state_name, base_lins):
        appl=_sa_appl.get(str(state_name).strip().lower())
        return (not appl) or bool(base_lins & appl)      # empty applies_to = GLOBAL state → always ok
    # ★ WITHIN-BASE STATE GATE: a program state is kept only when ≥2 of its markers are enriched vs
    #   SIBLING leaves of the SAME final base celltype — not merely vs the whole dataset. auc/lf (vs-rest) cannot
    #   tell a base-lineage-identity gene (macrophage C1q, Apoe) — uniform across the celltype yet high vs the
    #   dataset — from a genuine program (MKI67, high only in the cycling subset); only within-base variation can.
    #   So a pan-lineage 'state' (complement on every macrophage, lipid=Apoe) is dropped while proliferating/IFN
    #   (which vary across siblings) survive. No siblings / no stored means → cannot compare → kept (conservative).
    _sa_mk={(x.get("name") or "").strip().lower(): [str(g).upper() for g in (x.get("markers") or [])]
            for x in (_full.get("sa") or [])}
    _base_of_p={}
    for _p,_d in decisions.items():
        if _p not in st["leaves"]: continue
        _ll=_d.get("label","")
        _base_of_p[_p]=";".join(map(str,_ll)) if isinstance(_ll,(list,tuple)) else _ll
    _base_paths={}
    for _p,_b in _base_of_p.items(): _base_paths.setdefault(_b,[]).append(_p)
    def _within_base_states(path, states):
        if not states: return states
        _b=_base_of_p.get(path); _sibs=[x for x in _base_paths.get(_b,[]) if x!=path and x in _state_means]
        _me=_state_means.get(path)
        if not _sibs or not _me: return states            # cannot compare (single leaf of this base / no means) → keep
        _kept=[]
        for _s in states:
            _mk=_sa_mk.get(str(_s).strip().lower()) or []
            if not _mk: _kept.append(_s); continue        # markers unknown (index missing/name mismatch) → cannot evaluate → keep
            _nwl=0
            for _g in _mk:
                if _g not in _me: continue
                _sv=[_state_means[x][_g] for x in _sibs if _g in _state_means[x]]
                if not _sv: continue
                _ref=float(np.median(_sv))
                if _me[_g]>0 and _me[_g]>=max(1.5*_ref, _ref+1e-6): _nwl+=1
            if _nwl>=2: _kept.append(_s)                  # ≥2 markers elevated vs same-base siblings → real program
        return _kept
    def _synth(path, base, dec):
        base_s=str(base); _bl=base_s.strip().lower()
        _is_proc=((_gterms.get((dec.get("term_id") or "").strip(),{}) or {}).get("T") in _PROC_TIERS)
        if ((dec.get("decision") or "")!="celltype" or base_s in ("","Background","Unknown")
                or _is_proc):
            return base_s
        # ★ STANDARD annotation does NOT decide/append state — the base celltype is the whole label.
        #   State-axis programs (proliferating / IFN / hypoxic …) are base-agnostic OVERLAYS, not identity;
        #   they are appended ONLY in the INTERACTIVE phase (§8), when the user asks to subdivide by state,
        #   by the agent explicitly setting dec["states"]. The engine's per-leaf state_axis (coverage 0.80)
        #   is EVIDENCE for that step, never auto-applied here. (disease/injury/metaplastic identities are
        #   class/subclass celltypes judged at the same level as normal — NOT states, no fallback.)
        states=dec.get("states") or []
        states=_within_base_states(path, states)          # ★ (interactive) drop pan-lineage states not enriched vs same-base siblings
        bl=base_s.lower(); blins=_base_lineages(base_s); seen=set(); keep=[]
        for s in (states or []):
            s=str(s).strip()
            if not s or s.lower() in bl or s.lower() in seen: continue
            if not _state_ok(s, blins): continue          # ★ lineage gate (base+ancestors ∩ applies_to)
            seen.add(s.lower()); keep.append(s)
        keep.sort(key=lambda x:x.lower())                 # ★ canonical order (alphabetical) → no order-dup labels
        return base_s + "".join(", "+s for s in keep)
    for path,dec in decisions.items():
        if path not in st["leaves"]: continue
        lab=dec.get("label","")
        base=";".join(map(str,lab)) if isinstance(lab,(list,tuple)) else lab
        ann[cps==path]=_synth(path, base, dec)
    adata.obs[col]=ann                            # ★ only the key column is added (the helper never demotes or alters anything)
    adata.write_h5ad(out_h5ad)
    print(f"[finalize] wrote {out_h5ad} (input copy + obs['{col}'])", flush=True)
    rows=[]
    for path in st["leaves"]:
        dec=decisions[path]
        rows.append({"cluster_path":path,"n_cells":int((cps==path).sum()),
            "decision":dec.get("decision",""),"label":dec.get("label",""),"term_id":dec.get("term_id",""),
            "subclass_resolved":dec.get("subclass_resolved",""),"confidence":dec.get("confidence",""),
            "reason_code":dec.get("reason_code",""),"key_markers":";".join(dec.get("key_markers",[]) if isinstance(dec.get("key_markers"),list) else [str(dec.get("key_markers",""))]),
            "rationale":dec.get("rationale",""),"alternatives_considered":dec.get("alternatives_considered",""),
            "open_question_flag":bool(dec.get("open_question_flag",False))})
    sfx="_free" if "free" in col else ""
    csv_path=os.path.join(a.out,f"expert_judgment_annotation{sfx}.csv")
    pd.DataFrame(rows).to_csv(csv_path,index=False)
    print(f"[finalize] wrote {csv_path}", flush=True)
    # ★ annotation_rationale.md (SKILL §3.5) — a natural-language account of 'how each cluster was decided':
    #   the decision path, the evidence (markers ON, rejected candidates) and supporting PMIDs (from the
    #   chosen term's KG grounding = slim `GM`). English.
    try:
        _idx=json.load(open(os.path.join(a.out,"_ontology_index.json"))).get("terms",{})
    except Exception:
        _idx={}
    rl=["# Annotation rationale","",
        "How each cluster was decided — decision path, evidence, and supporting literature PMIDs "
        "(from the chosen term's KG grounding).",""]
    for path in st["leaves"]:
        dec=decisions[path]
        lab=dec.get("label",""); lab=";".join(map(str,lab)) if isinstance(lab,(list,tuple)) else (lab or "")
        n=int((cps==path).sum()); deck=dec.get("decision",""); conf=dec.get("confidence","")
        km=dec.get("key_markers",[]); km=km if isinstance(km,list) else [str(km)]
        gm=(_idx.get(dec.get("term_id",""),{}) or {}).get("GM",{}) or {}
        sup=[]
        for g in km:
            ps=gm.get(g) or gm.get(str(g).upper())
            if ps: sup.append(f"{g} (PMID {', '.join(ps[:4])})")
        rat=(dec.get("rationale") or "").strip(); alt=str(dec.get("alternatives_considered") or "").strip()
        rl.append(f"## cl{path} → {lab or deck}  ({deck}, confidence={conf}, n={n})")
        if rat: rl.append(rat)
        if alt: rl.append(f"Alternatives considered & rejected: {alt}")
        if sup: rl.append("Supported by: "+"; ".join(sup)+".")
        elif km: rl.append("Key markers: "+", ".join(map(str,km))+". (chosen term has no KG-grounded PMIDs on these markers)")
        rl.append("")
    rat_path=os.path.join(a.out,f"annotation{sfx}_rationale.md")
    open(rat_path,"w").write("\n".join(rl))
    print(f"[finalize] wrote {rat_path} (per-cluster natural-language rationale + PMIDs)", flush=True)
    # every working artifact other than the h5ad and the final CSV (state, decisions, ontology, profiles,
    # global stats, ...) goes into a single archive.
    # -> annotation_out: {input}_annotated.h5ad + expert_judgment_annotation.csv (the merge input) +
    #    annotation_artifacts.tar.gz.
    pats=["_global_stats.npz","_annotate_state.json","_annotate_decisions.json","_ontology_index.json","ontology_hits.json",
          "_annotate_recommendation.json","_annotate_violations.json","_annotate_finalize_review.json",
          "cluster_gene_profiles_round*.json","cluster_overview_round*.json","expert_judgment_round*.csv"]
    tmp=[]
    for pat in pats: tmp+=glob.glob(os.path.join(a.out,pat))
    if tmp:
        arc=os.path.join(a.out,"annotation_artifacts.tar.gz")
        with tarfile.open(arc,"w:gz") as tar:
            for f in tmp: tar.add(f,arcname=os.path.basename(f))
        for f in tmp:
            try: os.remove(f)
            except OSError: pass
        print(f"[finalize] archived {len(tmp)} artifacts -> {arc} (annotation_out keeps only the h5ad + the archive)", flush=True)

# ===================================================================== main
def _helper_cli():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest="cmd"); sp.required=True
    pi=sp.add_parser("init"); pi.add_argument("--h5ad",required=True); pi.add_argument("--cluster-col",dest="cluster_col",required=True)
    pi.add_argument("--out",required=True)
    pi.add_argument("--species",default=None)
    pi.add_argument("--organ",default=None,
                    help="canonical KG module organ (e.g. brain). Several comma-separated modules "
                         "(e.g. 'brain,spinal_cord') are merged into one index — use ONLY when the user "
                         "explicitly asked for several candidate modules at once.")
    pi.add_argument("--server-url",dest="server_url",default=None)
    pi.add_argument("--ontology-url",dest="ontology_url",default=None)
    pi.add_argument("--ontology",default=None); pi.add_argument("--output",default=None)
    pi.add_argument("--force",dest="force_input",action="store_true",
                    help="Downgrade the input-matrix checks (X=lognorm / var_names=symbol) to warnings and proceed. Use only when you know the input is non-standard.")
    pi.add_argument("--disease",default=None,help="(retired hint) Disease context of the input data. Disease/injury/malignant cells are now ordinary class/subclass celltypes judged by their own markers at the same level as normal — not a special fallback; flag kept for backward compatibility.")
    pi.add_argument("--use-rep",dest="use_rep",default="auto",
                    help="obsm key the subcluster kNN is built on, used only when a marker axis cannot be built. "
                         "Unset = X_pca, falling back to a per-subset normalize_total + log1p + PCA when obsm has no X_pca. "
                         "Name an integrated embedding (e.g. X_scVI, X_pca_harmony) to use one — none is ever chosen implicitly. "
                         "'recompute' = ignore obsm and always redo normalize_total + log1p + PCA.")
    # ★ subcluster metric / n_neighbors: inputs that keep the subcluster consistent with the initial
    #   clustering (cluster_vectors).
    #   A subcluster recompute always log-normalizes (+ PCA).
    #   When use_rep resolves to an obsm key, metric is ignored and that embedding is used with euclidean.
    pi.add_argument("--subcluster-metric",dest="sc_metric",default=None,
                    help="kNN metric for subcluster (correlation|euclidean). Defaults to correlation (assuming log-normalized input).")
    pi.add_argument("--subcluster-n-neighbors",dest="sc_n_neighbors",type=int,default=None,
                    help="Number of kNN neighbors for subcluster (n_neighbors). Defaults to 30. Smaller values split fine structure more aggressively; larger values smooth the graph.")
    pi.add_argument("--auc-pos",dest="auc_pos",type=float,default=0.65,
                    help="AUC bar for calling a positive layer/subtype marker ON inside a region-gated leaf (0.85 = the strict default, 0.65 = recovers quiet layers)")
    pi.add_argument("--profile-cap",dest="profile_cap",type=int,default=150000,
                    help="Maximum number of cells used to compute the AUC/rank profile (stratified subsample; 0 = unlimited). Ranking all 650k cells takes ~11 min and "
                         "is the bottleneck -> a 150k cap takes ~2 min (accuracy is preserved because markers are a population signal, rare clusters are kept in full, and the global "
                         "gmu/gsd still come from the full data). Use 0 (full) when accuracy matters most.")
    pi.add_argument("--adaptive-floor",dest="adaptive_floor",action="store_true",
                    help="(opt-in) Data-derived detection floor for sparse platforms: lower the fon/pin thresholds of region-gated positive anchors to "
                         "min(fixed, gpin/ghi x F) (reference-free). Recovers low-copy defining markers on a 0.5 um vector field (Rspo1 -> L4). "
                         "Not needed for scRNA / 10 um data (leaving it off is exactly the previous behaviour). Lineage, negative and region scoring are unchanged.")
    pi.add_argument("--af-F",dest="af_F",type=float,default=3.0,help="adaptive-floor multiplier F relative to the global level (default 3)")
    pi.add_argument("--af-Nmin",dest="af_Nmin",type=int,default=10,help="adaptive-floor guard on the minimum number of detected cells (default 10)")
    pi.set_defaults(f=cmd_init)
    ps=sp.add_parser("subcluster"); ps.add_argument("--out",required=True); ps.add_argument("--paths",default=None)
    ps.add_argument("--resolution",default="1.0")   # manual fallback only (the old 0.2 under-splits). In practice the finalize start_resolution (_res_for = 0.25 x k) overrides it
    ps.add_argument("--gpu",default=None); ps.add_argument("--ontology",default=None)
    # ★ Unknown carry-forward: with --repool-unknown, instead of --paths, the cells of every leaf decided
    #   Unknown in the current round are gathered across parents into one pool and re-clustered with Leiden.
    #   Scattered Unknown fragments of the same type re-aggregate and become identifiable.
    ps.add_argument("--repool-unknown",dest="repool_unknown",action="store_true")
    ps.add_argument("--decisions",default=None,help="With --repool-unknown, the current round's _annotate_decisions.json (used to identify Unknown leaves)")
    ps.add_argument("--profile-cap",dest="profile_cap",type=int,default=150000)
    ps.add_argument("--auc-pos",dest="auc_pos",type=float,default=0.65)
    ps.add_argument("--adaptive-floor",dest="adaptive_floor",action="store_true")
    ps.add_argument("--af-F",dest="af_F",type=float,default=3.0); ps.add_argument("--af-Nmin",dest="af_Nmin",type=int,default=10)
    ps.set_defaults(f=_hcmd_subcluster)
    # signal_split has been removed (the Leiden + absolute paradigm): it prevents force-assignment contamination. Splitting is done only by subcluster (Leiden).
    pc=sp.add_parser("check"); pc.add_argument("--out",required=True); pc.add_argument("--decisions",required=True); pc.set_defaults(f=cmd_check)
    pf=sp.add_parser("finalize"); pf.add_argument("--out",required=True); pf.add_argument("--decisions",required=True)
    pf.add_argument("--output",default=None); pf.add_argument("--force",action="store_true")
    pf.add_argument("--accept-large",dest="accept_large",action="store_true")
    pf.add_argument("--max-split-depth",dest="max_split_depth",type=int,default=1)    # split-depth cap (prevents infinite loops)
    pf.add_argument("--large-min-n",dest="large_min_n",type=int,default=1500)
    pf.add_argument("--column",default="cellman_annotation",
                    help="obs column name. Standard annotation = cellman_annotation (default); the result of user interaction = cellman_free_annotation.")
    pf.set_defaults(f=cmd_finalize)
    a=p.parse_args(); a.f(a)


# ===================================================================================
# annotate-mode additions — marker-axis split (subcluster) + KG descent (descend)
# ===================================================================================
# ---------------------------------------------------------------- candidate marker axis (KG-aware)
def _term_full_markers(terms, tid):
    """FULL signature of term tid = its own DF u R plus that of every ancestor (the IS_A P-chain), which
    reconstructs the normalized delta.
    supporting/negative are excluded (the discriminating axis is built from the identity markers DF u R;
    supporting markers are broadly co-expressed and carry no separating signal)."""
    seen = set(); stack = [tid]; genes = set()
    while stack:
        x = stack.pop()
        if x in seen or x not in terms: continue
        seen.add(x); t = terms[x]
        for g in (t.get("DF") or []) + (t.get("R") or []):
            if g: genes.add(g)
        for p in (t.get("P") or []): stack.append(p)
    return genes

def _children_of(terms, node_label):
    """Labels of the **direct children** of the term node_label in the slim-index (terms whose P contains that
    node's tid).
    The candidates for descending a class/region floor are exactly that node's KG children (region groups /
    subtypes)."""
    nl = (node_label or "").strip().lower()
    tid = next((k for k, t in terms.items() if (t.get("L") or "").strip().lower() == nl), None)
    if tid is None: return []
    return sorted({t.get("L") for t in terms.values() if tid in (t.get("P") or []) and t.get("L")})

def _candidate_axis(terms, cand_tids, var_up):
    """FULL markers per candidate (_term_full_markers) -> the discriminating axis.
    Removed: (a) off-panel genes and (b) only those shared by **every** candidate (= a common gate, not a
    discriminator). Neither a measured-breadth (global_on) filter nor a share filter is applied — the KG is
    normalized and curated, so DF u R is itself the specific signal, and a breadth cut would also discard
    local discriminators. Returns (axis_genes[list], per_cand{tid:set})."""
    per = {tid: {g for g in _term_full_markers(terms, tid) if g.upper() in var_up} for tid in cand_tids}
    ncand = len(per)
    from collections import Counter
    cnt = Counter(g for gs in per.values() for g in gs)
    axis = [g for g, c in cnt.items() if not (ncand >= 2 and c >= ncand)]
    return sorted(axis), per

def _marker_axis_leiden(sub, axis_genes, gUp, resolution, nn, backend, mod):
    """Subset sub (AnnData) to axis_genes only -> centering-free 90th-percentile scaling (0-preserving) ->
    kNN -> Leiden.
    ★ NO-SIGNAL GUARD: cells that switch on none of the axis genes (an all-zero row) are
    excluded from clustering and returned as **'u' (unassigned)**, so the caller floors them to the parent
    (bare class). Why: (1) when all-zero cells are numerous the kNN graph collapses and degenerates into one
    cluster per cell (oligo, 26504); (2) force-assigning a cell that switches on no region marker into some
    region sub-leaf gives it that region's label and causes a **spatial leak** (thalamic cells scattered
    through cortex). Only cells that actually switch on a region marker are split by region; the rest go
    honestly to the bare class.
    Returns labels[str] ('u' = unassigned), or None when separation fails or there is too little signal."""
    cols = [gUp[g.upper()] for g in axis_genes if g.upper() in gUp]
    if len(cols) < 2 or sub.n_obs < 3: return None
    Xa = np.asarray(sub.X.todense() if hasattr(sub.X, "todense") else sub.X)[:, cols].astype(float)
    has_sig = (Xa > 0).any(axis=1)
    n_sig = int(has_sig.sum())
    if n_sig < 3: return None                                         # too few cells carry any axis signal -> cannot separate (keep the parent)
    for j in range(Xa.shape[1]):                                      # centering-free: nonzero 90pct
        nz = Xa[:, j][Xa[:, j] > 0]
        s = np.percentile(nz, 90) if nz.size >= 10 else (nz.max() if nz.size else 1.0)
        Xa[:, j] = Xa[:, j] / (s + 1e-9)
    Xs = Xa[has_sig]
    import scanpy as sc
    axd = sc.AnnData(Xs); axd.obsm["Xm"] = Xs
    k = min(nn, max(2, n_sig - 1))
    if backend == "rapids":
        mod.get.anndata_to_GPU(axd); mod.pp.neighbors(axd, use_rep="Xm", n_neighbors=k)
        mod.tl.leiden(axd, resolution=resolution, key_added="ma", random_state=0); mod.get.anndata_to_CPU(axd)
    else:
        sc.pp.neighbors(axd, use_rep="Xm", n_neighbors=k)
        sc.tl.leiden(axd, resolution=resolution, key_added="ma", flavor="igraph", n_iterations=2, directed=False)
    labels = np.full(sub.n_obs, "u", dtype=object)                    # no-signal → 'u'(→ parent/bare class)
    labels[has_sig] = axd.obs["ma"].astype(str).values
    return labels

# ---------------------------------------------------------------- subcluster (marker-axis + PCA fallback)
def cmd_subcluster(a):
    st = _load_state(a.out)
    paths = json.loads(a.paths)
    # --candidates: {path:[labels]} (per path) or [labels] (shared by all paths). A path with none -> PCA fallback (same as annotate-mode).
    craw = json.loads(a.candidates) if a.candidates else {}
    cand_of = (lambda p: craw.get(p, [])) if isinstance(craw, dict) else (lambda p: list(craw))
    try: _rj = json.loads(a.resolution)
    except Exception: _rj = None
    res_of = (lambda p: float(_rj.get(p, 1.0))) if isinstance(_rj, dict) else (lambda p: float(a.resolution))
    backend, mod = _pick_backend(a.gpu)
    print(f"[round {st['round']+1}] v2 marker-axis subcluster {paths} backend={backend}", flush=True)
    adata = ad.read_h5ad(st["h5ad_path"])
    cps = np.array(st["cluster_paths"], dtype=object)
    genes = list(adata.var_names); gUp = {g.upper(): i for i, g in enumerate(genes)}; var_up = set(gUp)
    Xall = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    idx = _load_ontology_index(a.out); terms = (idx or {}).get("terms", {})
    lab2tid = {}
    for tid, t in terms.items(): lab2tid.setdefault((t.get("L") or "").strip().lower(), tid)
    rep = st.get("use_rep"); sc_metric = st.get("sc_metric", "euclidean")
    new_all = []; parent_prof = {}
    for path in paths:
        if path not in st["leaves"]: print(f"[skip] {path} not leaf"); continue
        mask = (cps == path); n = int(mask.sum())
        if n < 3:
            st.setdefault("unresolved_paths", []).append(path)
            print(f"[skip] {path} n={n}<3: too small to split.", flush=True); continue
        sub = adata[mask].copy(); res = res_of(path)
        cands = [c for c in cand_of(path)]
        tids = [lab2tid.get(c.strip().lower()) for c in cands]; tids = [t for t in tids if t]
        lbls = None; mode = "pca"
        if len(tids) >= 2:
            axis, per = _candidate_axis(terms, tids, var_up)
            print(f"[{path}] candidates={cands} → axis {len(axis)} genes (ancestor-walked, share-free): {axis[:12]}{'…' if len(axis)>12 else ''}", flush=True)
            nn = max(2, min(int(getattr(a, "sc_n_neighbors", 15) or 15), n - 1))   # marker-axis kNN: default 15, override --subcluster-n-neighbors
            lbls = _marker_axis_leiden(sub, axis, gUp, res, nn, backend, mod)
            mode = "marker-axis"
            if lbls is None: print(f"[{path}] marker-axis unusable (axis<2 genes) → PCA fallback", flush=True)
        if lbls is None:                                             # ── PCA fallback = annotate-mode subcluster ──
            npc = min(50, sub.n_vars - 1, max(2, sub.n_obs - 1)); nn = min(int(st.get("sc_n_neighbors", 15)), sub.n_obs - 1)
            use_rep = rep if (rep and rep in sub.obsm) else None
            import scanpy as sc
            if backend == "rapids":
                mod.get.anndata_to_GPU(sub)
                if use_rep is None:
                    mod.pp.normalize_total(sub, target_sum=1e4); mod.pp.log1p(sub); mod.pp.pca(sub, n_comps=npc)
                    mod.pp.neighbors(sub, n_neighbors=nn, use_rep="X_pca", metric=sc_metric)
                else: mod.pp.neighbors(sub, n_neighbors=nn, use_rep=use_rep, metric="euclidean")
                mod.tl.leiden(sub, resolution=res, random_state=0); mod.get.anndata_to_CPU(sub)
            else:
                if use_rep is None:
                    sc.pp.normalize_total(sub, target_sum=1e4); sc.pp.log1p(sub); sc.pp.pca(sub, n_comps=npc)
                    sc.pp.neighbors(sub, n_neighbors=nn, use_rep="X_pca", metric=sc_metric)
                else: sc.pp.neighbors(sub, n_neighbors=nn, use_rep=use_rep, metric="euclidean")
                sc.tl.leiden(sub, resolution=res, random_state=0, flavor="igraph", n_iterations=2, directed=False)
            lbls = sub.obs["leiden"].astype(str).values
        if len(set(lbls)) < 2:
            st.setdefault("unresolved_paths", []).append(path)
            print(f"[{path}] {mode}: did not split (1 cluster).", flush=True); continue
        new = np.array([f"{path}.{l}" for l in lbls]); cps[mask] = new
        nh = sorted(set(new.tolist())); new_all += nh
        for nl in nh: st["parent_of"][nl] = path
        parent_prof[path] = (mask.copy(), new)
        print(f"[{path}] {mode}: → {len(nh)} sub-leaves", flush=True)
    st["leaves"] = [p for p in st["leaves"] if p not in parent_prof] + new_all
    st["cluster_paths"] = cps.tolist(); st["round"] += 1; _save_state(a.out, st)
    # -- re-profile: reuses helper.py (the same substrate the annotate-mode judgment runs on) --
    ogs = _ontology_gene_set(idx); prot = _ontology_protected(idx); octx = _onto_ctx(idx)
    dz_terms = _disease_terms(idx, st.get("disease")); st_terms = _state_terms(idx); mt_terms = _mimicry_terms(idx)
    vf, low = _vf_setup(adata)
    _arp = os.path.join(a.out, "_region_presence.json")
    absent_rg = set(json.load(open(_arp)).get("absent_region_groups", [])) if os.path.exists(_arp) else set()
    gmu = gsd = None; _gsp = os.path.join(a.out, "_global_stats.npz")
    if os.path.exists(_gsp): _z = np.load(_gsp, allow_pickle=True); gmu = _z["mean"]; gsd = _z["std"]
    prior = {}
    for r in range(1, st["round"]):
        pf = os.path.join(a.out, f"cluster_gene_profiles_round{r}.json")
        if os.path.exists(pf): prior.update(json.load(open(pf)))
    _lab2tid = {}
    for _tid, _t in (octx.get("terms", {}) or {}).items(): _lab2tid.setdefault(_t.get("L"), _tid)
    def _restrict_for(parent):
        ge = (prior.get(parent, {}) or {}).get("graph_evidence") or {}
        dr = ge.get("dominant_region") or []
        if len(dr) == 1 and dr[0] in _lab2tid: return _lab2tid[dr[0]]
        return _lab2tid.get(ge.get("lineage"))
    profiles = {}
    for parent, (mask, sub_paths) in parent_prof.items():
        sl = sorted(set(sub_paths.tolist()))
        p_up = {r[0]: r[1] for r in (prior.get(parent, {}).get("up") or [])}
        pr = _profile_leaves(Xall[mask], genes, sub_paths, sl, ogs, prot, octx,
                               vf[mask] if vf is not None else None, low, parent_up_auc=p_up, gmu=gmu, gsd=gsd,
                               restrict_root=_restrict_for(parent), dz_terms=dz_terms, st_terms=st_terms, mt_terms=mt_terms,
                               profile_cap=getattr(a, "profile_cap", 0), auc_pos=getattr(a, "auc_pos", 0.65), absent_rg=absent_rg,
                               adaptive_floor=bool(st.get("adaptive_floor", False)), af_F=float(st.get("af_F", 3.0)), af_Nmin=int(st.get("af_Nmin", 10)))
        profiles.update(pr)
    for leaf in list(profiles):
        par = st["parent_of"].get(leaf)
        if par and par in prior:
            profiles[leaf]["parent_ctx"] = {"path": par, "up": prior[par].get("up", [])[:10], "hp": prior[par].get("hp", [])[:10]}
    json.dump(profiles, open(os.path.join(a.out, f"cluster_gene_profiles_round{st['round']}.json"), "w"), separators=(",", ":"))
    _write_overview(a.out, st["round"], profiles)
    print(f"[round {st['round']}] {len(profiles)} new leaves. No h5ad write. (init/check/finalize: use helper.py)", flush=True)

def cmd_descend(a):
    """Unconditionally descend a class/region-floor leaf onto the **direct KG children** of a node
    (region -> subtype) along the marker axis. This is a kg-mode style hierarchical descent judged by
    annotate-mode. It does **not wait** for flat-multiplicity in graph_evidence — splitting a region-mixed
    floor along the region axis first concentrates each sub-leaf by region, and only then can re-judgment
    pick up dominant_region/anchors and resolve region/subtype (layer markers come out of dilution).
    The child candidates are used as the marker axis and the work is delegated to the subcluster core.
    Iteration (node -> children -> that child node -> ...) is driven by the agent per the SKILL.

    Usage: helper.py descend --out O --paths '["cl3"]' --node "glutamatergic neuron"
      -> subcluster on the marker axis of that node's children (neocortical / thalamic / hippocampal glut ...).
    STOP condition (agent): if every sub-leaf re-judges to the same class as the parent, that is a genuine
    floor -> stop."""
    idx = _load_ontology_index(a.out); terms = (idx or {}).get("terms", {})
    kids = _children_of(terms, a.node)
    if len(kids) < 2:
        print(f"[descend] node '{a.node}' has <2 KG children in slim-index — nothing to descend."); return
    paths = json.loads(a.paths)
    a.candidates = json.dumps({p: kids for p in paths})
    print(f"[descend] node '{a.node}' → {len(kids)} KG children as marker-axis candidates: {kids[:6]}{'…' if len(kids)>6 else ''}", flush=True)
    cmd_subcluster(a)


# ================================================================ annotate-mode CLI (single entry)
def main():
    """SINGLE entry point for annotate-mode. subcluster/descend = v2 marker-axis (this file);
    init/check/finalize = the inlined helper engine (_helper_cli), run IN-PROCESS (no subprocess)."""
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: helper.py {init|descend|subcluster|check|finalize} ...\n"
              "  init | check | finalize = the inlined annotate engine (reads the user-specified cluster column)\n"
              "  descend    = marker-axis descent into a KG node's children (region->subtype, unconditional)\n"
              "  subcluster = marker-axis split by EXPLICIT named candidates (PCA fallback when 0/1 candidate)")
        return
    cmd, rest = argv[0], argv[1:]
    # ★ scoring-based annotation (score-profile/apply/finalize) is not in this release — its fixed-quota
    #   candidates, absence of agent-judgeable material and unjudged auto-subtyping are unresolved, so
    #   annotate-mode provides clustering-based annotation only (development continues out of tree).
    if cmd in ("score-profile", "score-apply", "score-finalize"):
        print(f"[helper] '{cmd}' is not supported in this version — annotate-mode provides clustering-based annotation "
              "only (init/descend/subcluster/check/finalize).\n"
              "         Detection of processes (neuropil, fiber tract, ...) is not supported yet.")
        raise SystemExit(2)
    if cmd in ("subcluster", "descend"):
        ap = argparse.ArgumentParser(prog=f"helper.py {cmd}")
        ap.add_argument("--out", required=True)
        ap.add_argument("--paths", required=True)
        if cmd == "subcluster":
            ap.add_argument("--candidates", default=None)
        else:
            ap.add_argument("--node", required=True)
        ap.add_argument("--resolution", default="1.0")
        ap.add_argument("--subcluster-n-neighbors", dest="sc_n_neighbors", type=int, default=15)
        ap.add_argument("--gpu", type=int, default=None)
        ap.add_argument("--profile-cap", dest="profile_cap", type=int, default=150000)
        ap.add_argument("--auc-pos", dest="auc_pos", type=float, default=0.65)
        a = ap.parse_args(rest)
        (cmd_descend if cmd == "descend" else cmd_subcluster)(a)
    elif cmd in ("init", "check", "finalize"):
        _helper_cli()   # inlined helper engine's argparse dispatch, in-process (reads sys.argv)
    else:
        print(f"[helper] unknown command '{cmd}' — use init|subcluster|descend|check|finalize")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
