#!/usr/bin/env python3
"""UNNAMED annotate helper — in-memory recursive subclustering + ontology-aware profiling.

cellularspace `_annotate_helper.py`를 계승하되 UNNAMED 변경 반영:
  - cluster 컬럼 인자화 (--cluster-col; leiden 하드코딩 제거)
  - ontology = **서버 slim-index fetch** (init에서 1회) + `--ontology` 로컬 폴백
  - subcluster 백엔드 **자동 분기**: GPU+rapids_singlecell → rapids, 아니면 scanpy
  - profile `bg` → **identity_evidence**(vf_norm 밀도 + 정체 신호) ; up/dn 무제한, hp=50
  - 출력 = **{input}_annotated.h5ad** (입력 전체 복사 + obs['celltype_annotation']만 추가)

서브커맨드:
  init       --h5ad --cluster-col --out [--ontology-url | --ontology]
  subcluster --out --paths '["4","11"]' --resolution 0.2 [--gpu 0] [--ontology ...]
  finalize   --out --decisions <json> [--output <path>]
"""
import argparse, json, os, sys, re, datetime
import numpy as np
import anndata as ad
from scipy.stats import rankdata
from scipy.sparse import csr_matrix

# ===================================================================== ontology
# 우리가 배포하는 단일 Cell-Ontology 서버. 사용자는 보통 아무것도 안 줘도 여기로 자동 연결된다.
# 우선순위: --server-url 플래그 > 환경변수 CELLONTO_SERVER_URL > 이 기본값.
# (배포 시 이 상수를 공개 서버 URL로 교체.)
DEFAULT_ONTOLOGY_SERVER = "http://localhost:8088"
def _resolve_server(server_url=None):
    return (server_url or os.environ.get("CELLONTO_SERVER_URL") or DEFAULT_ONTOLOGY_SERVER).rstrip("/")

def _fetch_slim_index(out, ontology_url=None, ontology_file=None):
    """slim-index를 확보해 {out}/_ontology_index.json 으로 저장.
    우선순위: 서버 fetch(ontology_url) > 로컬 파일(ontology_file) > 이미 있는 파일."""
    dst = os.path.join(out, "_ontology_index.json")
    if ontology_url:
        import urllib.request, urllib.error
        try:
            with urllib.request.urlopen(ontology_url, timeout=60) as r:
                idx = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                sys.exit(f"[init][ERROR:WING_NOT_FOUND] 서버에 해당 wing이 없습니다 ({ontology_url}, HTTP 404). "
                         f"(species, organ)을 확인하거나 그 wing의 빌드를 요청하세요. → caller는 사용자에게 "
                         f"'요청한 조직(species/organ)의 온톨로지가 서버에 없어 annotation을 진행할 수 없습니다'라고 알릴 것.")
            sys.exit(f"[init][ERROR:SERVER_HTTP_{e.code}] 온톨로지 서버가 오류 {e.code}를 반환했습니다 ({ontology_url}).")
        except urllib.error.URLError as e:
            sys.exit(f"[init][ERROR:SERVER_UNREACHABLE] 온톨로지 서버에 연결할 수 없습니다 ({ontology_url}; {e.reason}). "
                     f"서버 기동 여부와 CELLONTO_SERVER_URL을 확인하세요. → caller는 사용자에게 "
                     f"'온톨로지 서버에 연결할 수 없습니다(서버 다운 또는 주소 오류)'라고 알릴 것.")
        if not idx or not idx.get("terms"):
            sys.exit(f"[init][ERROR:WING_EMPTY] wing은 응답했으나 term이 비어 있습니다 ({ontology_url}). "
                     f"그 wing의 빌드가 필요합니다. → caller는 사용자에게 '해당 조직의 온톨로지가 비어 있어 진행 불가'라고 알릴 것.")
        with open(dst, "w") as f: json.dump(idx, f, separators=(",", ":"))
        print(f"[ontology] fetched slim-index from server -> {dst} ({len(idx.get('terms',{}))} terms)", flush=True)
        return idx
    if ontology_file and os.path.exists(ontology_file):
        # celltypes.json(원형) 또는 slim-index 둘 다 허용
        raw = json.load(open(ontology_file))
        if "terms" in raw and isinstance(raw["terms"], dict):
            idx = raw                                   # 이미 slim-index
        else:                                            # celltypes.json → slim 변환
            idx = {"terms": {}, "cp": [], "oq": []}
            for t in raw.get("terms", []):
                m = t.get("markers", {})
                idx["terms"][t["id"]] = {"L": t.get("label",""), "T": t.get("tier",""),
                    "P": t.get("is_a", []), "R": list(m.get("required", [])),
                    "S": list(m.get("supporting", []))[:6], "N": list(m.get("negative", []))[:4],
                    "E": list((t.get("experience") or {}).get("consensus_up", []))[:8],
                    "D": t.get("disease_context","normal")}
        with open(dst, "w") as f: json.dump(idx, f, separators=(",", ":"))
        print(f"[ontology] loaded slim-index from {ontology_file} -> {dst}", flush=True)
        return idx
    if os.path.exists(dst):
        return json.load(open(dst))
    return None

def _load_ontology_index(out):
    p = os.path.join(out, "_ontology_index.json")
    return json.load(open(p)) if os.path.exists(p) else None

def _ontology_gene_set(idx):
    s = set()
    for t in (idx or {}).get("terms", {}).values():
        for k in ("R","S","N","E","ES","DF"): s.update(t.get(k, []) or [])
    return s

def _ontology_protected(idx):
    s = set()
    for t in (idx or {}).get("terms", {}).values():
        for k in ("R","S","E","ES","DF"): s.update(t.get(k, []) or [])  # negative는 보호 안 함
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
# 과거의 뇌-하드코딩(_LINEAGE/_LAYER/_SPECIFIC/_PAN)을 폐기하고, lineage/specific/pan/sibling을
# 그 wing의 온톨로지(slim-index)에서 유도한다. 유도 불가(계층/마커 빈약) 시에만 아래 cross-organ
# fallback을 쓴다. → 어떤 organ에도 일반화.
_BG_MAX_N=500; MIN_N_OUT=20
def _U(s): return str(s).upper()

# cross-organ fallback(온톨로지 유도가 비었을 때만): 흔한 broad lineage 마커.
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
    """lineage = 계층의 root(부모 없음) 노드. lineage 마커 = 그 root 노드 **자신의** required 만
    (canonical pan-LINEAGE transmitter: glut→Slc17a7/6/8, GABA→Gad1/2·Slc32a1, DA→Th/Ddc/Slc6a3 …).
    ★ root∪후손을 쓸어담지 않는다(2026-06-13): 쓸어담으면 후손에 붙은 pan-NEURONAL(Camk2a)·subtype
    마커까지 lineage 집합에 섞여 'which lineage'가 아닌 'neuron'을 가리키게 됨 — cluster 38이 Slc17a7=0
    인데도 Camk2a로 glut 오판된 원인. root-own-R 로 좁히면 transmitter만 남아 lineage 판정이 깨끗해진다.
    계층이 없으면 {} 반환(→ fallback). {lineage_label: {MARKER_UPPER}}."""
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
    """≤2개 term에만 required인 마커(=고특이 식별자) ∪ DF(defining). 빈약하면 {}."""
    terms=(idx or {}).get("terms",{}); cnt={}; spec=set()
    for t in terms.values():
        for m in (t.get("R") or []): cnt[m.upper()]=cnt.get(m.upper(),0)+1
        spec.update(m.upper() for m in (t.get("DF") or []))
    spec.update(m for m,c in cnt.items() if c<=2)
    return spec

def _derive_pan(idx, min_terms=4):
    """많은 term이 공유하는 R/S 마커(=비변별 pan/class 마커). neg-leak 예외·lineage_present 제외에 사용."""
    terms=(idx or {}).get("terms",{}); cnt={}
    for t in terms.values():
        for k in ("R","S"):
            for m in (t.get(k) or []): cnt[m.upper()]=cnt.get(m.upper(),0)+1
    return {m for m,c in cnt.items() if c>=min_terms}

def _sibling_groups(idx):
    """부모를 공유하는 ≥2 자식(=sibling subclass 그룹). {parent_id:[child_id...]} + child별 required."""
    terms=(idx or {}).get("terms",{}); bp={}
    for tid,t in terms.items():
        for p in (t.get("P") or []):
            if p in terms: bp.setdefault(p,[]).append(tid)
    return {p:ch for p,ch in bp.items() if len(ch)>=2}

def _onto_ctx(idx):
    """profiling/validate가 쓰는 도메인 컨텍스트(온톨로지 유도, fallback 포함)."""
    terms=(idx or {}).get("terms",{})
    return {"lineages": _derive_lineages(idx) or {k:set(v) for k,v in _LINEAGE_FALLBACK.items()},
            "specific": _derive_specific(idx) or set(_SPECIFIC_FALLBACK),
            "pan": _derive_pan(idx) or set(_PAN_FALLBACK),
            "siblings": _sibling_groups(idx),
            "terms": terms,
            "cp": (idx or {}).get("cp") or [],   # confusion pairs from slim-index → surfaced by _graph_evidence
            "term_req": {tid:{m.upper() for m in (t.get("R") or [])} for tid,t in terms.items()}}

def _candidate_subtypes(terms, ogt, up_rows, octx, parent_up_auc=None):
    """이 클러스터에서 *정의 마커가 실제로 보이는* 온톨로지 term을 증거 scorecard로 랭킹(옆 후보).
    + 각 후보의 1단계 자식이 이 클러스터에서 해소 가능한지(children_resolvable)도 같이 — 더 split할지 판단용(아래 lookahead).
    품질 필터: pan(비변별) 마커만 떠선 후보로 안 침. → CA1/CA3 같은 진짜 subtype만 소수로 남음.
    전부 온톨로지 런타임 유도 → organ 무관·하드코딩 0. 비강제(Step5 게이트가 최종).
    ★ 부모 풀링(parent_up_auc): 균질 subtype이 과분할로 사라지는 것 방지. 자식 up(=형제 대비)에선
    layer/relay 마커가 ~0.5로 평준화되지만, 부모(=뇌 전체 대비)에서 enriched였다면 그 정체성은 모든
    자식에 상속된다. 그래서 term의 *자신* required 평가엔 부모 enrichment를 함께 본다(inherit=True).
    단 children_resolvable(=더 쪼개면 분리?)엔 부모 풀링을 쓰지 않는다 — 균질 마커는 분할로 안 갈라지므로."""
    pan=octx.get("pan") or set()
    up_auc={_U(r[0]):r[1] for r in (up_rows or [])}
    p_auc={_U(g):a for g,a in (parent_up_auc or {}).items()}
    ogt_u={_U(g):v for g,v in (ogt or {}).items()}
    # 부모→자식 인덱스(1단계)
    children={}
    for tid,t in (terms or {}).items():
        for p in (t.get("P") or []):
            if p in terms: children.setdefault(p,[]).append(tid)
    def ev(g, inherit):
        """(present_bool, auc, on_panel_bool, is_pan_bool). inherit=True면 부모 enrichment도 인정."""
        gu=_U(g); isp=gu in pan
        if gu in up_auc and up_auc[gu]>=0.6: return (True, round(up_auc[gu],3), True, isp)
        if inherit and p_auc.get(gu,0)>=0.6:                        # 부모에서 enriched → 상속된 정체성
            return (True, round(p_auc[gu],3), True, isp)
        e=ogt_u.get(gu)
        if e is None: return (False, None, True, isp)               # 패널엔 있고 발현 약함
        if e.get("panel") is False: return (None, None, False, isp) # off-panel: 미지(기각 근거 아님)
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
        """이 term의 모든 마커(R∪S∪DF∪ES) 중 '비-pan · on-panel · 강한(AUC≥0.55)'인 것들(유전자 dedup).
        required가 off-panel(CA1의 Wfs1)이어도 supporting(Fibcd1)/ES로 잡힘 → signal_split target 자동
        도출의 핵심(누락 방지). 진짜로 존재하는 type은 *여러* 변별마커가 잡히고, 마커 1개만 교차상장된
        spurious(예: 피질에 Cerebellar Golgi via Neurod6)는 1개라 아래 랭킹에서 밀린다."""
        tt=terms.get(tid) or {}; d={}
        for g in (tt.get("R") or [])+(tt.get("S") or [])+(tt.get("DF") or [])+(tt.get("ES") or []):
            pres,a,onpanel,isp=ev(g, True)
            if onpanel and not isp and a is not None and a>=0.55:
                gu=_U(g)
                if gu not in d or a>d[gu]: d[gu]=a
        return sorted(d.items(), key=lambda x:-x[1])
    def required_ok(tid):
        """'고유(비-pan) 필수명찰' 검사. 이 term의 비-pan required 중 on-panel인 게 있으면 적어도
        하나는 present(AUC≥0.55)여야 True. 전부 off-panel이거나 비-pan required가 없으면 True(보조명찰로
        식별 — CA1: Wfs1/Mpped1 off-panel). UBC처럼 on-panel 필수(Eomes)가 안 뜨면 False → target 제외."""
        onp=[]
        for g in ((terms.get(tid) or {}).get("R") or []):
            pres,a,onpanel,isp=ev(g, True)
            if onpanel and not isp:                              # 흔한(pan) 필수명찰은 무시
                onp.append(a is not None and a>=0.55)
        return (not onp) or any(onp)
    out=[]; sugg=[]   # 자동 target 제안용(R-필터와 무관)
    for tid,t in (terms or {}).items():
        if not (t.get("R")): continue
        dms=disc_markers(tid)                                      # 고유 변별마커들(모든 마커류에서)
        if dms and required_ok(tid):                              # 변별마커 有 + 고유 필수명찰 검사 통과
            sugg.append({"term_id":tid,"label":t.get("L",""),"n_disc":len(dms),
                         "top_auc":dms[0][1],"disc_markers":[g for g,a in dms[:5]]})
        rec,npres,nnonpan,noff=term_evidence(tid, inherit=True)     # 자신 정체성: 부모 상속 인정
        # 품질 필터: 비-pan required ≥1, 또는 required ≥2 present (pan-only는 제외)
        if not (nnonpan>=1 or npres>=2): continue
        # child lookahead(1단계): 자식 중 비-pan required가 present인 것 개수/라벨
        # inherit=False — 더 쪼개서 분리 가능한지는 '이 leaf 내부' 신호로만 판단(부모 상속 제외).
        kids=[]
        for kt in children.get(tid,[]):
            krec,knp,knn,kno=term_evidence(kt, inherit=False)
            if knn>=1 or knp>=2:
                kids.append({"label":(terms.get(kt) or {}).get("L",""),"n_req_present":knp,"n_nonpan_present":knn})
        out.append({"term_id":tid,"label":t.get("L",""),"tier":t.get("T",""),
                    "n_req_present":npres,"n_nonpan_present":nnonpan,"n_req":len(t.get("R")),"n_req_offpanel":noff,
                    "required":rec,"children_resolvable":kids})
    out.sort(key=lambda d:(-d["n_nonpan_present"], -d["n_req_present"], -max((r[1] or 0) for r in d["required"])))
    # signal_targets_suggested: 고유 변별마커가 강한 type을 자동 target으로 제안.
    # 랭킹 = (변별마커 개수, 최고 AUC) 내림차순 → 여러 마커가 잡히는 진짜 type(CA1 Fibcd1+Spink8,
    # CA3 Grik4+Npy2r+Nectin3, DG Prox1+Dsp+Neurod1)이 상위, 1개만 교차상장된 spurious는 하위.
    # CA1/CA3/DG는 required가 off-panel이어도 supporting으로 surface → agent가 누락 불가.
    signal_targets=sorted(sugg, key=lambda d:(-d["n_disc"], -d["top_auc"]))[:15]
    return out, signal_targets

# ===================================================================== absolute identity (Leiden+absolute paradigm)
_TIERD={"class":0,"subclass":2,"supertype":2,"cluster":3,"state":1,"":0}
def _build_ident_maps(octx, genes, pan_n=12):
    """온톨로지→identity 스코어용 인덱스(1회). term별 비-pan on-panel 변별마커/음성마커 인덱스, tier,
    lineage 마커 인덱스 + lineage→class term, lineage-root 맵. 전부 유전자 전역 인덱스 기준(organ 무관).
    ★pan 임계는 octx['pan'](≥4 term, signal_split용)보다 높은 pan_n(기본 12)을 별도 계산해 쓴다.
    ≥4는 Cux2(4)/Rorb(5)/Fezf2(5) 같은 *층 변별 마커*까지 housekeeping처럼 제거해 피질층이 안 잡힌다.
    ≥12면 진짜 pan(Slc17a7=39, Gad1=139, Calb1=46, Foxp2=18)만 거르고 층 마커는 변별마커로 살아남는다."""
    terms=octx.get("terms",{}) or {}; LIN=octx.get("lineages") or {}
    _cnt={}
    for _t in terms.values():
        for _k in ("R","S"):
            for _m in (_t.get(_k) or []): _cnt[_m.upper()]=_cnt.get(_m.upper(),0)+1
    PAN={_m for _m,_c in _cnt.items() if _c>=pan_n}
    gUp={_U(g):i for i,g in enumerate(genes)}
    term_disc={}; term_neg={}; term_tier={}; term_lab={}; term_def={}
    for tid,t in terms.items():
        term_tier[tid]=t.get("T",""); term_lab[tid]=t.get("L",tid); dc=[]; seen=set()
        for g in (t.get("R") or [])+(t.get("S") or [])+(t.get("DF") or [])+(t.get("ES") or []):
            gu=_U(g)
            if gu in PAN: continue
            j=gUp.get(gu)
            if j is not None and j not in seen: seen.add(j); dc.append(j)
        if dc: term_disc[tid]=dc
        term_neg[tid]=[gUp[_U(g)] for g in (t.get("N") or []) if _U(g) in gUp]
        # DEFINING markers (a single one is sufficient) — kept separately for the DF-override.
        dfi=[gUp[_U(g)] for g in (t.get("DF") or []) if _U(g) in gUp]
        if dfi: term_def[tid]=dfi
    lin_idx={}; lin_class={}
    for lx,gs in LIN.items():
        idxs=[gUp[_U(g)] for g in gs if _U(g) in gUp]
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
    # ★ HIERARCHY-AWARE specificity (region_share, 2026-06-13): a marker's true specificity is how many
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
    return {"disc":term_disc,"neg":term_neg,"tier":term_tier,"lab":term_lab,"share":share,
            "region_share":region_share,"defn":term_def,"region_of":_rmap,
            "lin_idx":lin_idx,"lin_class":lin_class,"root":{tid:root(tid) for tid in terms},
            "genes":list(genes), "P":{tid:[p for p in (t.get("P") or []) if p in terms] for tid,t in terms.items()}}

def _graph_evidence(fon, lf, pin, IDM, octx, frac=0.30, lf_on=1.5, pin_min=0.25, SPEC_REGION_MAX=2,
                    mean=None, auc=None, auc_hi=0.85, anchor_floor_ratio=0.5, floor_pct=90, restrict_root=None):
    """★ GRAPH-DESCENT EVIDENCE (not a verdict) — the redesigned identity substrate (2026-06-12).
    The agent is the expert; the KG it built is its memory. helper provides EVIDENCE, the agent reasons.
    Anchors = specific terms whose OWN **exclusive** markers are ON and negatives clean, ranked by exclusive
    support, each with its ancestor `path` so the agent can STOP at a region when the subtype is ambiguous.
    **Exclusivity is HIERARCHY-AWARE (`region_share`, 2026-06-13):** a marker counts as exclusive for a term
    when it spans ≤SPEC_REGION_MAX distinct region-groups — so a region master TF (Tcf7l2→thalamus, shared
    across thalamic subtypes but absent from cortex/hippocampus) IS exclusive at the region level and lets the
    region node anchor; a pan/broad marker (Slc17a7, Calb2) spans many regions and is excluded. scRNA-
    compatible (no spatial): a leaf descends to CA1 only if CA1's exclusive Fibcd1/Spink8 are ON, to the
    thalamic region only if Tcf7l2/Gbx2 are ON — not because it shares a leaky marker."""
    disc=IDM["disc"]; neg=IDM["neg"]; defn=IDM["defn"]
    share=IDM.get("region_share") or IDM["share"]          # hierarchy-aware degree (fallback to term-degree)
    lab=IDM["lab"]; tier=IDM["tier"]; genes=IDM.get("genes"); P=IDM.get("P",{})
    lin_idx=IDM["lin_idx"]; lin_class=IDM["lin_class"]; terms=octx.get("terms",{})
    region_of=IDM.get("region_of",{})
    # ★ TWO predicates (2026-06-13 region-coherent redesign, replaces the absolute noise-floor gate):
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
    def on_present(g): return pin[g]>=pin_min and (mean is None or mean[g]>0)
    def gname(g): return str(genes[g]) if (genes is not None and 0<=g<len(genes)) else str(g)
    children={}
    for tid,ps in P.items():
        for p in ps: children.setdefault(p,[]).append(tid)
    def ev(tid):   # anchor evidence — uses LOOSE on_anchor (auc branch) since region-gating already excludes cross-region blips
        dc=disc.get(tid,[])
        excl=[g for g in dc if share.get(g,99)<=SPEC_REGION_MAX]  # exclusive: spans ≤N region-groups (hierarchy-aware)
        excl_on=[g for g in excl if on_anchor(g)]
        ng=[g for g in neg.get(tid,()) if on_anchor(g)]           # negative-marker leak
        df=[g for g in defn.get(tid,()) if on_anchor(g)]
        n_on=sum(1 for g in dc if on_anchor(g))
        supported=bool((len(excl_on)>=1 or (len(excl)==0 and n_on>=2)) and len(ng)==0)
        # ★ DF-absent 가드 (2026-06-15): defn[tid]은 _build_ident_maps가 패널(adata.var_names)로 만들어 ON-PANEL
        #   defining 마커만 담는다(off-panel DF는 애초 제외 → CA1 Wfs1류는 면제). defining은 "하나면 충분"한 최강
        #   등급이므로, 그 역 — 측정됐는데(on-panel) 하나도 ON이 아니면(df==[]) 그 type일 가능성이 매우 낮다 →
        #   force-fit. 이 term은 supporting만으로 anchor했어도 강등(아래 sort 1차키)·verdict에서 제외. auc≥0.85
        #   분기로 quiet-but-enriched DF(cl50 Grik4 auc0.92)는 ON 처리되어 면제. DF 없는 term(대부분)은 영향 0.
        df_onpanel=defn.get(tid,())
        df_absent=bool(len(df_onpanel)>=1 and len(df)==0)
        return {"term_id":tid,"label":lab.get(tid,tid),"tier":tier.get(tid,""),
                "excl_on":len(excl_on),"excl_n":len(excl),"excl_markers_on":[gname(g) for g in excl_on][:8],
                "neg_on":len(ng),"neg_markers_on":[gname(g) for g in ng][:6],
                "df_on":len(df),"df_absent":df_absent,"df_onpanel":[gname(g) for g in df_onpanel][:6],
                "n_on":n_on,"n_disc":len(dc),"supported":supported}
    def lvl(g): return mean[g] if mean is not None else fon[g]   # expression level (fon fallback when mean absent)
    def path_to_root(tid):
        chain=[tid]; seen={tid}
        while True:
            nxt=next((x for x in P.get(chain[-1],[]) if x not in seen), None)
            if nxt is None: break
            chain.append(nxt); seen.add(nxt)
        return [lab.get(x,x) for x in chain[::-1]]
    # ★ PARENT-CONSTRAINED descent (2026-06-14): restrict_root = the parent's resolved term (its dominant
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
    allow=None; lineage_ambiguous=False; codom=[]
    if restrict_root and restrict_root in terms:
        allow={t for t in terms if _under(t, restrict_root)}
        cls_root=IDM["root"].get(restrict_root,restrict_root); cls=cls_root; lin_tx_sym=set()
    else:
        # LINEAGE (transmitter HARD gate): prefer the lineage whose transmitter is ENRICHED (rejects ambient
        #   myelin Mbp ⇒ not oligodendrocyte), else the strongest PRESENT transmitter by MEAN (cortical Slc17a7
        #   is everywhere ⇒ low fold but high mean ⇒ glut). NEURONAL PRIORITY: a detected neuronal transmitter
        #   excludes glial lineages (white-matter ambient Mbp can't make a neuron oligodendrocyte). MAX mean
        #   (not enrichment): a RARE ambient transmitter (Chat, high fold at trace mean) must not beat the real
        #   Gad (Gad 0.01 ≫ ambient Chat 0.001 ⇒ GABA).
        NEURONAL_TX={"SLC17A7","SLC17A6","SLC17A8","GAD1","GAD2","SLC32A1","SLC6A5","TH","DBH","DDC","SLC6A3",
                     "SLC6A4","SLC6A2","TPH2","CHAT","SLC18A3","SLC5A7","HDC"}
        def _is_neur(ln): return any(gname(g).upper() in NEURONAL_TX for g in lin_idx.get(ln,[]))
        _neur_present=any(_is_neur(ln) and any(on_present(g) for g in idxs) for ln,idxs in lin_idx.items())
        lin_pool={ln:idxs for ln,idxs in lin_idx.items() if (not _neur_present) or _is_neur(ln)}
        strength={ln:max((lvl(g) for g in idxs if on_present(g)), default=0.0) for ln,idxs in lin_pool.items()}
        strength={ln:s for ln,s in strength.items() if s>0}
        if not strength or max(strength, key=lambda k:strength[k]) not in lin_class:
            return {"lineage":None,"anchors":[],"verdict":"no_lineage","cp":[],
                    "lineage_ambiguous":False,"codominant_lineages":[],"dominant_region":[]}
        best=max(strength.items(), key=lambda kv:kv[1]); LIN_RATIO=0.5
        codom=sorted([ln for ln,s in strength.items() if s>=LIN_RATIO*best[1]], key=lambda ln:-strength[ln])
        lineage_ambiguous=len(codom)>=2
        cls=lin_class[best[0]]; cls_root=IDM["root"].get(cls,cls)
        # ★ dual-transmitter cross-root 도달(2026-06-15): MHb(cholinergic-glutamatergic)는 root=glut인데 cl28처럼
        #   glut transmitter OFF·cholinergic ON이면 cholinergic lineage로 검출돼 도달 불가했음. 검출 lineage의 전달물질
        #   심볼을 모아, 그걸 자기 R에 가진 cross-root term(MHb=Chat/Slc18a3 보유)을 아래 anchor 루프가 region-gate
        #   우회로 후보에 포함시킨다(공동방출 뉴런 복원).
        lin_tx_sym={gname(g).upper() for g in lin_idx.get(best[0],[])}
    # ★ REGION-COHERENT DESCENT WITHIN the lineage: among the lineage class's region-groups (depth-1 children —
    #   neocortical / thalamic / hippocampal pyramidal …), pick the DOMINANT by expression-weighted (relative,
    #   not an absolute floor) sum over its subtree's ENRICHED EXCLUSIVE markers; ≥2 concordant markers required
    #   so a lone broad gene (Stmn1) can't raise a region. ONLY that region's subtypes are anchor candidates ⇒
    #   cross-region blips (CA2 in a thalamic glut cluster) are structurally excluded WITHOUT a magnitude floor,
    #   so low-abundance layer markers (Fezf2→L5) survive. Gene-based, spatial-free.
    rg_terms={}
    for t in terms:
        if IDM["root"].get(t)==cls_root and (allow is None or t in allow): rg_terms.setdefault(region_of.get(t,t),[]).append(t)
    region_groups=[rg for rg in children.get(cls,[]) if tier.get(rg)!="class" and (allow is None or rg in allow)]
    rg_score={}
    for rg in region_groups:
        gg=set(disc.get(rg,()))
        for t in rg_terms.get(rg,()): gg|=set(disc.get(t,()))
        gg=[g for g in gg if share.get(g,99)<=SPEC_REGION_MAX]
        # RC3 (2026-06-14): region-exclusive markers count via on_anchor (auc≥0.85 branch), so a quiet-but-enriched
        #   region marker (Fezf2→neocortical, auc 0.90 yet detected<25%) resolves the region in sparse 2µm data —
        #   else a deep-cortical (L5) leaf loses neocortical to hippocampal (shared Slc17a7/Pcp4) and is mislabeled.
        #   gg is already region-exclusive (share≤SPEC_REGION_MAX) and ≥2 concordant required ⇒ no lone-blip region.
        onm=[g for g in gg if on_anchor(g)]
        if len(onm)>=2: rg_score[rg]=sum(lvl(g) for g in onm)
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
        # dual-transmitter cross-root term(MHb): root이 달라도 검출 lineage의 전달물질을 R에 가지면 후보 + region-gate 우회
        cross_dual=bool(lin_tx_sym and IDM["root"].get(tid)!=cls_root and tier.get(tid)!="class"
                        and ({str(g).upper() for g in (terms.get(tid,{}).get("R") or [])} & lin_tx_sym))
        if IDM["root"].get(tid)!=cls_root and not cross_dual: continue
        if tier.get(tid)=="class": continue
        if allow is not None and tid not in allow: continue        # parent-constrained: only descendants of restrict_root
        if region_gated and not cross_dual and region_of.get(tid,tid) not in dominant: continue
        # ★ region-group 컨테이너 노드(neocortical/thalamic/… = dominant의 원소 자신)는 main 경쟁에서 제외 —
        #   자기 region-exclusive 마커(Slc30a3/Mef2c)가 적어 ratio=1.0로 자식 층(L4 excl_n8 ratio0.25)을 부당하게
        #   이겨 모든 피질 층을 generic으로 붕괴시킴(2026-06-15). 의도대로 자식 leaf가 하나도 anchor 안 될 때만
        #   아래 fallback으로 surface한다(주석: 'If none anchors, surface the dominant region-group node').
        if region_gated and not cross_dual and tid in dominant: continue
        e=ev(tid)
        if e["excl_on"]>=1 and e["neg_on"]==0:
            e["path"]=path_to_root(tid); anchors.append(e)
    if not anchors and region_gated:
        for rg in sorted(dominant, key=lambda r:-rg_score[r]):
            e=ev(rg)
            if e["neg_on"]>0: continue          # region node도 negative leak이면 surface 안 함(Slc17a6+ 클러스터에 DG region 등)
            e["path"]=path_to_root(rg); anchors.append(e)
    # df_absent(defining 측정+OFF)는 **약한 tiebreak일 뿐** — excl_on/ratio/n_on이 우선. 이렇게 안 하면 DF 없는
    #   generic region node(절대 df_absent 아님)가 정의마커만 quiet한 진짜 층(L4=Rorb/Whrn excl_on2, Rspo1 quiet)을
    #   항상 이겨 피질 층이 전부 generic으로 붕괴함(2026-06-15 회귀). force-fit(mitral)은 ratio로 이미 밀리므로 정렬
    #   강등 불필요. df_absent는 flag(목록·_validate)로만 agent에 전달, 정렬은 동률일 때만 마지막으로 가른다.
    anchors.sort(key=lambda x:(x["excl_on"], round(x["excl_on"]/max(x["excl_n"],1),2), x["n_on"], not x.get("df_absent",False)), reverse=True)
    # verdict hint (the agent decides; this is a summary): clear top = unique max excl_on; tie = ambiguous.
    if not anchors: verdict="class_or_unknown"
    else:
        topv=anchors[0]["excl_on"]; ties=[a for a in anchors if a["excl_on"]==topv]
        verdict=("clear_top" if len(ties)==1 else "ambiguous_tie")
    anchor_labels=[a["label"] for a in anchors]
    pathlabels={l for a in anchors[:6] for l in a.get("path",[])}|set(anchor_labels[:6])
    # ★ ConfusionPair gate (2026-06-14): a cp is matched by its terms (if the slim-index serves them) OR by its
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
    def _distinct_on(tid_a, tid_b):
        # tid_a의 고유(공유 아닌) exclusive 마커 중 ON인 것 → tid_a를 tid_b와 구별하는 켜진 증거
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
                    # ★ 3분기(2026-06-14): 각 후보의 *고유*(공유 아닌) 마커가 ON인지로 해소.
                    #   양쪽 고유 ON → 두 type 혼합 → split(needs_subclustering). 한쪽만 ON → discriminator가
                    #   그쪽 지지 → 그 type. 둘 다 OFF(공유 마커만) → 진짜 모호 → Unknown.
                    td=_distinct_on(tt,pt) if (tt and pt) else []
                    pd=_distinct_on(pt,tt) if (tt and pt) else []
                    # 양쪽 고유 마커가 ON → 두 type 공존(cl64 CA2 Cacng5/Map3k15/Acan + FC Scgn) → split. 한쪽만 → 그쪽
                    #   resolve. 둘 다 OFF(공유만) → unknown. split이면 needs_subclustering으로 FC를 CA2로 뭉치지 않게(사용자
                    #   2026-06-15): FC 못 찾아도 fragment는 CA2 / hippocampal / Unknown으로 갈리고 FC부분이 CA2로 안 남음.
                    if td and pd:      action,resolves="split",None
                    elif td:           action,resolves="resolve",top_label
                    elif pd:           action,resolves="resolve",par
                    else:              action,resolves="unknown",None
                    confusion_unresolved={"pair":c[0],"terms":[top_label]+partners[:3],
                                          "discriminators":(c[2] if len(c)>2 else []),
                                          "action":action,"mixture":bool(td and pd),"resolves_to":resolves,
                                          "top_distinct_on":td[:6],"partner_distinct_on":pd[:6]}
    if confusion_unresolved: verdict="confusion_unresolved"
    # ★ MIXTURE detection (SHOULD_SPLIT_MIXTURE, 2026-06-14): even when the leaf terminalizes at a region/class
    #   (no single subtype anchors strongly), if ≥2 of the dominant region's child subtypes each have their OWN
    #   exclusive marker WEAKLY co-present (auc≥WEAK_AUC=0.6 — below the strict on_anchor 0.85, OR fold/df-ON via
    #   on_anchor), the leaf is an UNDER-CLUSTERED MIXTURE that the strict anchor gate missed because the markers
    #   are diluted (2µm: cl2=Drd1+Drd2, cl15=Cux2+Rorb, cl7.1=L5 Fezf2+L6 Foxp2). A cluster with NO subtype's
    #   own exclusive marker present (cl4 region-less generic glut) is NOT flagged ⇒ no noise. ≥2 distinct → split.
    WEAK_AUC=0.6
    mix=[]
    for tid in terms:
        if IDM["root"].get(tid)!=cls_root or tier.get(tid)=="class": continue
        if allow is not None and tid not in allow: continue
        if region_gated and region_of.get(tid,tid) not in dominant: continue
        exc=[g for g in disc.get(tid,()) if share.get(g,99)<=SPEC_REGION_MAX]
        if not exc: continue
        wk=[gname(g) for g in exc if on_anchor(g) or (auc is not None and auc[g]>=WEAK_AUC)]
        if wk and not any(on_anchor(g) for g in neg.get(tid,())):
            mix.append({"label":lab.get(tid,tid),"markers_present":wk[:4]})
    mixture_subtypes=mix[:8] if len(mix)>=2 else []
    return {"lineage":lab.get(cls,cls),"verdict":verdict,"anchors":anchors[:8],"cp":cps[:4],
            "confusion_unresolved":confusion_unresolved,"mixture_subtypes":mixture_subtypes,
            "lineage_ambiguous":lineage_ambiguous,"codominant_lineages":codom if lineage_ambiguous else [],
            "dominant_region":dominant_region}

_DEV_STATE_KW=("developmental","neuroblast","radial glia","immature","reactive","aging astro",
               "disease-associated","interferon-responsive","proliferative-region","committed oligodendrocyte")
# ★ real celltypes that contain a state keyword but ARE discrete published populations (adult neurogenesis):
#   exempt so they can be a `top`/anchor. "immature granule neuron" (DG adult neurogenesis) is a standard
#   celltype name (Neurod1/Dcx/Sox11+), not a transient state to be demoted.
_STATE_KW_EXEMPT=("immature granule",)
def _is_state_term(label, tier):
    """developmental/reactive STATE terms share broad markers (Neurod/Dcx/Sox2, Gfap-reactive…) and are
    NOT discrete spatial cell populations → exclude from the chimera co-dominance test and from `top`.
    Exempts _STATE_KW_EXEMPT (bona-fide adult-neurogenesis celltypes)."""
    L=(label or "").lower()
    if any(k in L for k in _STATE_KW_EXEMPT): return False
    return tier=="state" or any(k in L for k in _DEV_STATE_KW)
def _disease_terms(idx, disease):
    """caller가 --disease <name>로 지정한 질병에 해당하는 ontology term들을 추출.
    매칭: term의 D(disease_context)가 normal이 아니고 disease 파라미터와 양방향 부분일치(대소문자 무시).
    반환 {tid:{L,D,R,S,DF}} — _profile_leaves가 leaf별 disease_candidates(마커 지지도)로 surface한다.
    disease 미지정/normal이면 {} (정상 경로 그대로)."""
    dz=(disease or "").strip().lower()
    if not dz or dz in ("normal","none"): return {}
    out={}
    for tid,t in ((idx or {}).get("terms",{}) or {}).items():
        D=(t.get("D") or "normal")
        if D and D.lower()!="normal" and (dz in D.lower() or D.lower() in dz):
            out[tid]={"L":t.get("L",tid),"D":D,
                      "R":list(t.get("R") or []),"S":list(t.get("S") or []),"DF":list(t.get("DF") or [])}
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
    disc=M["disc"]; neg=M["neg"]; tier=M["tier"]; lab=M["lab"]; root=M["root"]; share=M.get("share",{})
    if pin is None: pin=fon          # pin not provided → fold branch effectively inert (conservative)
    SPEC_SHARE_MAX=3; SPEC_COV_MIN=0.5
    def W(g): return 1.0/max(share.get(g,0),1)                  # marker specificity weight (1/#terms listing it)
    # ★ ON (RC2, 2026-06-14): detection-based (fon / fold+pin) OR — for a term's OWN low-share SPECIFIC marker
    #   (share≤SPEC_SHARE_MAX) — strong rank-enrichment (auc≥auc_hi). The auc branch rescues quiet-but-enriched
    #   subtype markers in SPARSE spatial data (2µm: Rorb auc 0.98 / Fezf2 0.90 / Dcx 0.72 yet detected <25% so
    #   fold+pin fails) so L4/L5/immature-granule resolve. GUARD: auc path applies ONLY to low-share specific
    #   markers — broad/shared genes (Slc17a7/Pcp4, share>3) still require real detection ⇒ no spillover force-fit.
    def _on(g): return fon[g]>=frac or (lf[g]>=lf_on and pin[g]>=pin_min) or \
                       (auc is not None and auc[g]>=auc_hi and share.get(g,99)<=SPEC_SHARE_MAX)
    cand=[]
    for tid,dc in disc.items():
        on=[g for g in dc if _on(g)]
        if not on: continue
        if not (len(on)>=2 or lf[on[0]]>=lf_high): continue
        non=sum(1 for g in neg.get(tid,()) if _on(g) and W(g)>=0.33)     # SPECIFIC negative leak only (ambient broad N ignored)
        spec=round(float(sum(max(0.0,lf[g]) for g in on)),2)
        wspec=round(float(sum(W(g)*max(0.0,lf[g]) for g in on)),3)
        dec=round(float(max(W(g)*max(0.0,lf[g]) for g in on)),3)         # decisive marker = most SPECIFIC×enriched single marker
        cand.append({"term_id":tid,"label":lab[tid],"tier":tier[tid],"n_on":len(on),
                     "cov":round(len(on)/len(dc),2),"spec":spec,"wspec":wspec,"dec":dec,"neg":non,"_root":root[tid]})
    # coherent, non-state, no specific-negative leak
    coh=[c for c in cand if c["n_on"]>=2 and c["neg"]==0 and not _is_state_term(c["label"],c["tier"])]
    # ★ SPECIFIC-EVIDENCE per candidate — computed HERE because both the valid-evidence `top` gate AND the
    #   chimera/co-dominance test below use it (see §5 of SKILL). (SPEC_SHARE_MAX/COV_MIN defined above for _on.)
    def _spec_stats(c):
        spec=[g for g in disc.get(c["term_id"],()) if share.get(g,99)<=SPEC_SHARE_MAX]
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
    # (all specific markers off-panel) ⇒ gate INERT (panel-blind: judge by experience/supporting).
    # GUARD: DF-override only among COHERENT terms (n_on>=2, neg==0, non-state). A lone defining marker
    # firing on ambient/neuropil signal in an over-split fragment (e.g. Rgs14→CA2) must NOT confirm a type
    # without ≥1 corroborating marker — this prevents spurious DF over-calls (the 8µm CA2 balloon).
    # GUARD A+B (2026-06-11): a LEAKY defining marker (curated as one term's DF/specific in the ontology but
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
    #   rescue it if `hp`/experience clearly show the type IS present.
    if top is not None and top.get("tier")!="class" and not top["spec_pass"]:
        top=None
    # ranked: valid-evidence terms (spec_pass) surface first, then dominant-lineage, then decisiveness
    ranked=(sorted(coh, key=lambda c:(c["spec_pass"], c["_root"]==dom_root, c["dec"], c["wspec"]), reverse=True)[:5]
            or sorted(cand, key=lambda c:(c["spec_pass"], c["dec"], c["wspec"], c["n_on"]), reverse=True)[:5])
    if top is not None and top not in ranked: ranked=[top]+ranked[:4]
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

def _profile_leaves(X, genes, leaf_arr, leaves, ont_gene_set, ont_protected,
                    octx, vf_norm=None, vf_low_thresh=None,
                    auc_up=0.6, auc_dn=0.4, top_hp=50, parent_up_auc=None,
                    gmu=None, gsd=None, restrict_root=None, dz_terms=None):
    """leaf별 profile. up/dn 무제한(AUC 임계만), hp=50.
    identity_evidence = density(vf_norm) + specificity_peak + lineage_markers_present + housekeeping_share + cross_lineage."""
    n_cells, n_genes = X.shape
    genes_arr=np.asarray(genes); ranks=_rankcols(X)
    gidx_U={str(g).upper():i for i,g in enumerate(genes_arr.tolist())}   # disease_candidates 마커 매칭(대소문자 무시)
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
    # ontology gene index
    panel=set(genes_arr.tolist())
    ont_on=sorted(g for g in (ont_gene_set or set()) if g in panel)
    ont_off=sorted(g for g in (ont_gene_set or set()) if g not in panel)
    ont_on_idx=np.array([np.where(genes_arr==g)[0][0] for g in ont_on], dtype=np.int64) if ont_on else np.array([],dtype=np.int64)

    # absolute identity (전역 기준) 사전계산: 전역 high임계 THR=gmu+gsd, leaf별 frac_on = (M@HI)/n_in.
    # gmu/gsd는 init에서 전체 데이터로 계산해 persist → 모든 round에서 동일(조각화 불변).
    IDM=None
    if gmu is not None and gsd is not None:
        IDM=_build_ident_maps(octx, genes_arr)
        gmu_arr=np.asarray(gmu,dtype=np.float64); gsd_arr=np.asarray(gsd,dtype=np.float64)
        sumHI=M@((X>=(gmu_arr+gsd_arr)).astype(np.float64))   # (K, G) leaf별 high세포 수

    profiles={}
    for k,path in enumerate(leaves):
        ni=int(n_in_per[k]); no=n_cells-ni
        if ni==0 or no==0: continue
        auc=auc_mat[k]; mi=mean_in_mat[k]; mo=mean_out_mat[k]
        pin=sumBin[k]/ni
        order=np.argsort(mi)[::-1]
        rank_of=np.empty(n_genes,dtype=np.int64); rank_of[order]=np.arange(1,n_genes+1)
        # up/dn (무제한 — AUC 임계만)
        up=[[genes_arr[i],round(float(auc[i]),3),round(float(mi[i]),3),round(float(mo[i]),3)]
            for i in np.argsort(auc)[::-1] if auc[i]>=auc_up]
        dn=[[genes_arr[i],round(float(auc[i]),3),round(float(mi[i]),3),round(float(mo[i]),3)]
            for i in np.argsort(auc) if auc[i]<=auc_dn]
        # hp (top 50, HK 제외)
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
        # flags (lineage/specific/sibling는 onto_ctx에서 — 온톨로지 유도, organ 무관)
        LINS=octx["lineages"]; SPEC=octx["specific"]; PAN=octx["pan"]
        up_genes={r[0] for r in up}; up_up={_U(g) for g in up_genes}
        multi=[ln for ln,gs in LINS.items() if len(up_up&gs)>=2]
        # subclass_mixing: 부모를 공유하는 sibling subclass가 ≥2개, 각각 required ≥2개가 up에 강함
        layer_mix=[p for p,ch in octx["siblings"].items()
                   if sum(1 for c in ch if len(octx["term_req"].get(c,set())&up_up)>=2)>=2]
        has_spec=bool(up_up&SPEC)
        any_class=any(len(up_up&gs)>=2 for gs in LINS.values())
        pan_class=(len(multi)<=1 and any_class and not has_spec)
        n_hk15=int(sum(1 for i in order[:15] if hk_mask[i])); bgscore=round(n_hk15/15.0,3)
        # identity_evidence (bg 대체)
        nonhk_order=[i for i in order if not hk_mask[i]]
        # specificity_peak: non-HK 중 AUC 최고 (>=auc_up)
        sp_peak=None
        if nonhk_order:
            best=max(nonhk_order, key=lambda i: auc[i])
            if auc[best]>=auc_up:
                sp_peak=[genes_arr[best],round(float(auc[best]),3),round(float(mi[best]),3),round(float(mo[best]),3)]
        # lineage_markers_present: ontology R∪S on-panel 중 pct_in>=0.4
        lin_present=[g for g in ont_on if (ogt[g]["mean"] is not None) and pin[np.where(genes_arr==g)[0][0]]>=0.4
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
        # candidate_subtypes (옆 후보 scorecard + 1단계 child lookahead) — agent가 기억 의존 없이
        # CA1/CA3 등 subtype을 빠짐없이 검토 + 고른 term을 더 쪼갤지 판단.
        cands, sig_targets=_candidate_subtypes(octx.get("terms",{}), ogt, up, octx, parent_up_auc=parent_up_auc)
        # subtype_groups: signal_targets 중 *같은 부모를 가진 sibling subtype이 ≥2개 공존* → 이 leaf를 그
        # 부모 밑 subtype들로 한 단계 더 쪼갤 수 있다는 직접 신호(최대 깊이 재귀용). 예: CA1 deep(Bcl6)+
        # CA1 superficial(Calb1)이 둘 다 떠 있으면 → {parent:CA1, children:[deep,superficial]}.
        terms_=octx.get("terms",{}); _grp={}
        _lab2tid={s["label"]:s["term_id"] for s in sig_targets}
        for s in sig_targets:
            for par in ((terms_.get(s["term_id"]) or {}).get("P") or []):
                if par in terms_: _grp.setdefault(par,[]).append(s["term_id"])
        subtype_groups=[{"parent_id":par,"parent_label":terms_[par].get("L",""),
                         "child_term_ids":ch,"children":[terms_[c].get("L","") for c in ch]}
                        for par,ch in _grp.items() if len(ch)>=2]
        entry={"n":ni,"n_out":no,"up":up,"dn":dn,"hp":hp,"ontology_gene_table":ogt,
               "identity_evidence":identity, "lineage_scan":lscan, "candidate_subtypes":cands,
               "signal_targets_suggested":sig_targets, "subtype_groups":subtype_groups,
               "flags":{"multi_lineage":multi,"layer_mixing":layer_mix,"pan_class_only":bool(pan_class),"background_score":bgscore},
               "sibling_contrast":sib}
        if no<MIN_N_OUT: entry["warn"]="n_out_too_small"
        # absolute identity_scores (전역 기준, 조각화 불변) — Leiden+absolute 판단의 핵심 근거
        if IDM is not None:
            fon=sumHI[k]/max(ni,1); lf=np.log2((mi+1e-9)/(gmu_arr+1e-9))
            entry["identity_scores"]=_identity_scores(fon, lf, IDM, pin=pin, auc=auc)
            # ★ graph-descent EVIDENCE — the redesigned primary identity substrate (the agent reasons over
            #   the KG it built). identity_scores is kept only as a supplementary absolute-fold summary.
            entry["graph_evidence"]=_graph_evidence(fon, lf, pin, IDM, octx, mean=mi, auc=auc, restrict_root=restrict_root)
            # DF-confirmed terms in this leaf (a DEFINING marker is fold-ON: log2fold>=1.5 AND detected>=25%).
            # Used by the finalize split-review gate (§8.1) to only flag splits with a real 2nd identity.
            _defn=IDM.get("defn",{}); _lab=IDM.get("lab",{})
            entry["df_present"]=sorted({_lab.get(tid,tid) for tid,gs in _defn.items()
                                        if any(lf[g]>=1.5 and pin[g]>=0.25 for g in gs)})
            # ★ disease_candidates — caller가 --disease 지정 시, 그 질병의 state term(tumor 등)을 leaf별로
            #   marker 지지도와 함께 surface. _is_state_term이 정상 경로의 top/coh에서 state를 빼므로 여기서
            #   별도로 노출(정상 scoring 코어 불변). 같은 ON 정의(fon>=0.30 or fold-on) 사용. 판정은 agent(§7).
            if dz_terms:
                dzc=[]
                for _tid,_info in dz_terms.items():
                    ons=[]; tot=0
                    for _g in (_info.get("R",[])+_info.get("S",[])+_info.get("DF",[])):
                        _gi=gidx_U.get(str(_g).upper())
                        if _gi is None: continue
                        tot+=1
                        if fon[_gi]>=0.30 or (lf[_gi]>=1.5 and pin[_gi]>=0.25): ons.append(_g)
                    if ons:
                        dzc.append({"term_id":_tid,"label":_info["L"],"D":_info["D"],
                                    "on_markers":ons,"n_on":len(ons),"n_markers":tot})
                dzc.sort(key=lambda d:(-d["n_on"], -(d["n_on"]/max(d["n_markers"],1))))
                if dzc: entry["disease_candidates"]=dzc[:8]
        profiles[path]=entry
    return profiles

def _write_overview(out, rnd, profiles):
    """#1 cross-cluster overview: 한 round의 모든 leaf를 나란히 비교(전문가의 '표 비교' 복원)."""
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
    """obs['vf_norm'] 있으면 (배열, 하위25% 임계) 반환, 없으면 (None,None)."""
    if "vf_norm" in adata.obs:
        vf=np.asarray(adata.obs["vf_norm"].values, dtype=np.float64)
        return vf, float(np.percentile(vf,25))
    return None, None

# ===================================================================== init
def cmd_init(a):
    os.makedirs(a.out, exist_ok=True)
    print(f"[init] loading {a.h5ad} (cluster_col={a.cluster_col})", flush=True)
    adata=ad.read_h5ad(a.h5ad)
    if a.cluster_col not in adata.obs:
        sys.exit(f"[init] obs에 cluster 컬럼 '{a.cluster_col}' 없음. 사용가능: {list(adata.obs.columns)}")
    cps=adata.obs[a.cluster_col].astype(str).values.copy()
    leaves=sorted(set(cps.tolist()), key=lambda s:[int(x) for x in s.split(".")] if s.replace('.','').isdigit() else [10**9,s])
    st={"h5ad_path":os.path.abspath(a.h5ad),"cluster_col":a.cluster_col,
        "cluster_paths":cps.tolist(),"leaves":leaves,"round":1,"parent_of":{p:None for p in leaves},
        "output":a.output,"disease":a.disease}            # disease 컨텍스트 persist(모든 round 공유)
    _save_state(a.out, st)
    # ontology URL 자동 구성: --ontology-url 직접지정 > (species/organ + 기본/지정 서버) ; 로컬은 --ontology.
    url=a.ontology_url
    if not url and not a.ontology:
        if not (a.species and a.organ):
            sys.exit("[init] --species/--organ 가 필요합니다(또는 --ontology-url / 로컬 --ontology). "
                     "서버는 자동으로 기본 Cell-Ontology 서버에 연결됩니다.")
        url=f"{_resolve_server(a.server_url)}/wing/{a.species}/{a.organ}/slim-index"
        print(f"[init] ontology server: {_resolve_server(a.server_url)} (wing {a.species}/{a.organ})", flush=True)
    idx=_fetch_slim_index(a.out, url, a.ontology)
    if not idx or not _ontology_gene_set(idx):
        sys.exit("[init] FATAL: ontology(slim-index) 비어있음 — 서버/ wing 확인. ontology-blind 금지.")
    ogs=_ontology_gene_set(idx); prot=_ontology_protected(idx); octx=_onto_ctx(idx)
    print(f"[init] ontology markers {len(ogs)} (protected {len(prot)}); lineages {len(octx['lineages'])} "
          f"(derived={'yes' if _derive_lineages(idx) else 'fallback'})", flush=True)
    X=np.asarray(adata.X.todense() if hasattr(adata.X,"todense") else adata.X)
    vf,low=_vf_setup(adata)
    print(f"[init] vf_norm: {'있음' if vf is not None else '없음(밀도축 비활성)'}", flush=True)
    # ★ 전역 per-gene 통계(전체 데이터) 1회 계산 → persist. 모든 round의 absolute identity 스코어가
    #   이 전역 기준을 재사용 → 조각나도 정체성 안 붕괴(local 대조 의존 제거).
    gmu=np.asarray(X.mean(axis=0)).ravel(); gsd=np.asarray(X.std(axis=0)).ravel()
    np.savez(os.path.join(a.out,"_global_stats.npz"), mean=gmu, std=gsd, genes=np.array(list(adata.var_names),dtype=object))
    print(f"[init] global per-gene stats saved (_global_stats.npz, {len(gmu)} genes) — absolute identity 기준", flush=True)
    dz_terms=_disease_terms(idx, a.disease)
    if a.disease:
        print(f"[init] disease='{a.disease}' → matching state terms {len(dz_terms)} (leaf별 disease_candidates로 surface; 판정은 agent §7)", flush=True)
    profiles=_profile_leaves(X, list(adata.var_names), np.array(cps), leaves, ogs, prot, octx, vf, low, gmu=gmu, gsd=gsd, dz_terms=dz_terms)
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

def cmd_subcluster(a):
    st=_load_state(a.out); paths=json.loads(a.paths)
    # --resolution: 스칼라(모든 path 동일) 또는 JSON dict {path:res}(클러스터별 맞춤 — finalize start_resolution).
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
    new_all=[]; parent_prof={}
    for path in paths:
        if path not in st["leaves"]: print(f"[skip] {path} not leaf"); continue
        mask=(cps==path); n=int(mask.sum())
        # policy성 min-cells 없음(작다고 background 취급/분할 차단 안 함 — 희귀 세포 보호).
        # 단 PCA/이웃그래프가 수학적으로 불가능한 극소만 건너뜀(=분할 연산 불가, background 아님).
        if n<3:
            st.setdefault("unresolved_paths",[]).append(path)
            print(f"[skip] {path} n={n}<3: 분할 연산 불가(극소). background 아님 — 현 leaf 유지.", flush=True)
            continue
        res=_res_p(path)                                  # ★ 클러스터별 맞춤 resolution
        sub=adata[mask].copy()
        npc=min(50, sub.n_vars-1, max(2, sub.n_obs-1))
        if backend=="rapids":
            mod.get.anndata_to_GPU(sub); mod.pp.pca(sub,n_comps=npc)
            mod.pp.neighbors(sub,n_neighbors=min(30,sub.n_obs-1),metric="correlation")
            mod.tl.leiden(sub,resolution=res,random_state=0); mod.get.anndata_to_CPU(sub)
        else:
            import scanpy as sc
            sc.pp.pca(sub,n_comps=npc); sc.pp.neighbors(sub,n_neighbors=min(30,sub.n_obs-1),metric="correlation")
            sc.tl.leiden(sub,resolution=res,random_state=0,flavor="igraph",n_iterations=2,directed=False)
        lbls=sub.obs["leiden"].astype(str).values
        if len(set(lbls))<2: st.setdefault("unresolved_paths",[]).append(path); continue
        new=np.array([f"{path}.{l}" for l in lbls]); cps[mask]=new
        nh=sorted(set(new.tolist())); new_all+=nh
        for nl in nh: st["parent_of"][nl]=path
        parent_prof[path]=(mask.copy(),new)
    leaves=[p for p in st["leaves"] if p not in parent_prof]+new_all
    st["leaves"]=leaves; st["cluster_paths"]=cps.tolist(); st["round"]+=1; _save_state(a.out,st)
    idx=_load_ontology_index(a.out); ogs=_ontology_gene_set(idx); prot=_ontology_protected(idx); octx=_onto_ctx(idx)
    dz_terms=_disease_terms(idx, st.get("disease"))     # init에서 persist한 disease로 sub-leaf도 동일 surface
    X=np.asarray(adata.X.todense() if hasattr(adata.X,"todense") else adata.X)
    vf,low=_vf_setup(adata)
    # ★ init에서 persist한 전역 통계 로드 → absolute identity 기준(전체 데이터)을 이 round에도 그대로 사용.
    gmu=gsd=None
    _gsp=os.path.join(a.out,"_global_stats.npz")
    if os.path.exists(_gsp):
        _z=np.load(_gsp, allow_pickle=True); gmu=_z["mean"]; gsd=_z["std"]
    # 이전 라운드 부모 프로파일 미리 로드(candidate_subtypes 부모 풀링 + parent_ctx 주입에 사용)
    prior={}
    for r in range(1,st["round"]):
        pf=os.path.join(a.out,f"cluster_gene_profiles_round{r}.json")
        if os.path.exists(pf): prior.update(json.load(open(pf)))
    # ★ PARENT-CONSTRAINED subclustering (2026-06-14): each parent's children may only be the parent's KG
    #   subtree. restrict_root = the parent's resolved DOMINANT REGION-GROUP (single) so genuine SIBLINGS stay
    #   findable (CA1 inside a CA3 cluster — both under hippocampal pyramidal); if the parent had 0 or multiple
    #   dominant regions, fall back to its LINEAGE class (still blocks cross-lineage cortex→GABA/oligo). A child
    #   thus can NEVER force-fit a distant region (the 8µm cortex→pontine bug).
    _lab2tid={}
    for _tid,_t in (octx.get("terms",{}) or {}).items(): _lab2tid.setdefault(_t.get("L"),_tid)
    def _restrict_for(parent):
        ge=(prior.get(parent,{}) or {}).get("graph_evidence") or {}
        dr=ge.get("dominant_region") or []
        if dr and dr[0] in _lab2tid: return _lab2tid[dr[0]]      # TOP (highest-scoring) region → its subtree
        #   (siblings within it stay findable: CA1 inside a CA3 cluster, both under hippocampal pyramidal; but a
        #   cortical cluster's false 2nd region 'pontine' is dropped — children can't escape to it).
        lin=ge.get("lineage")
        return _lab2tid.get(lin)                                  # no region resolved → lineage class (blocks cross-lineage)
    profiles={}
    for parent,(mask,sub_paths) in parent_prof.items():
        sl=sorted(set(sub_paths.tolist()))
        p_up={r[0]:r[1] for r in (prior.get(parent,{}).get("up") or [])}   # 부모 up(=뇌 전체 대비 AUC)
        rr=_restrict_for(parent)
        pr=_profile_leaves(X[mask], genes, sub_paths, sl, ogs, prot, octx,
                           vf[mask] if vf is not None else None, low, parent_up_auc=p_up, gmu=gmu, gsd=gsd, restrict_root=rr, dz_terms=dz_terms)
        profiles.update(pr)
    # parent_ctx 주입
    for leaf in list(profiles):
        par=st["parent_of"].get(leaf)
        if par and par in prior:
            profiles[leaf]["parent_ctx"]={"path":par,"up":prior[par].get("up",[])[:10],"hp":prior[par].get("hp",[])[:10]}
    json.dump(profiles, open(os.path.join(a.out,f"cluster_gene_profiles_round{st['round']}.json"),"w"), separators=(",",":"))
    _write_overview(a.out, st["round"], profiles)
    print(f"[round {st['round']}] {len(profiles)} new leaves. No h5ad write.", flush=True)

# ===================================================================== validate (보고 전용)
# 원칙: helper는 결정을 절대 바꾸지 않는다. 위반을 '검출해 보고'만 하고, 재결정은 agent가 한다.
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
    V=[]
    def add(lf,check,sev,msg,sugg):
        V.append({"cluster_path":lf,"check":check,"severity":sev,"message":msg,
                  "current_decision":(decisions.get(lf) or {}).get("decision"),
                  "current_label":(decisions.get(lf) or {}).get("label"),"suggestion":sugg})
    for lf in st["leaves"]:
        d=decisions.get(lf); p=prof.get(lf)
        if d is None:
            add(lf,"MISSING_DECISION","error","이 leaf에 대한 결정이 없음.",
                "이 클러스터에 4지선다 결정을 작성하세요."); continue
        dec=d.get("decision")
        if dec=="needs_subclustering":
            add(lf,"NEEDS_SUB_REMAINING","error",
                "finalize 시점에 needs_subclustering로 남아 있음(더 분할되지 않음).",
                "Step6 unsplittable 규칙으로 종착 결정으로 바꾸세요: 공통 class 있으면 class-final celltype, 없으면 Unknown. Background로 강등 금지.")
            continue
        if p is None: continue
        us=upstrong(p)
        # ambient/neuropil(KDE) bleed makes raw `up` list many lineages → only warn on a REAL co-dominant
        # mixture (identity_scores.mixed = ≥2 codominant lineage-roots). ambient-only bleed is NOT flagged.
        ismixed=bool((p.get("identity_scores") or {}).get("mixed"))
        if ismixed and len(lins(us))>=2 and not (us&SPEC) and dec=="celltype":
            add(lf,"UP_CROSS_LINEAGE","warn",
                f"≥2 co-dominant lineages (identity_scores.mixed, up={lins(us)}) but a single celltype.",
                "genuine chimera? → needs_subclustering, or re-check dominance (ambient bleed alone is not flagged).")
        if dec=="celltype":
            tid=(d.get("term_id") or "").strip()
            if not tid or (idx and tid not in idx):
                add(lf,"BAD_TERM_ID","error",
                    f"celltype인데 term_id가 비었거나 온톨로지에 없음(term_id='{tid}').",
                    "유효한 term_id로 고치거나 결정을 바꾸세요. helper가 임의로 Unknown 처리하지 않습니다.")
            # ★ 고유 필수명찰 검사(REQUIRED_ABSENT_ONPANEL): 이 term의 비-pan required 중 on-panel인 게
            # 있으면 적어도 하나는 present여야. UBC=Eomes(on-panel)인데 absent면 보조마커(Grm1) 오배정 →
            # flag. CA1=required 전부 off-panel → 검사 제외(보조명찰 Fibcd1로 식별, §5 panel-blind).
            if tid in idx:
                t_=idx[tid]; ogt_l={_U(g):v for g,v in (p.get("ontology_gene_table") or {}).items()}
                up_l={_U(r[0]):r[1] for r in (p.get("up") or [])}
                onp=[]
                for g in (t_.get("R") or []):
                    gu=_U(g)
                    if gu in PAN: continue                         # 흔한 pan 필수명찰 무시
                    e=ogt_l.get(gu)
                    if e is None or e.get("panel") is False: continue   # off-panel/미측정 제외
                    onp.append((gu, up_l.get(gu,0)>=0.6 or (e.get("auc") or 0)>=0.55))
                if onp and not any(pr for _,pr in onp):
                    add(lf,"REQUIRED_ABSENT_ONPANEL","warn",
                        f"'{t_.get('L')}'의 고유 on-panel 필수명찰 미발현({', '.join(g for g,pr in onp if not pr)}) — 보조마커 오배정 가능.",
                        "필수명찰이 뜨는 다른 term으로 바꾸거나 상위 class/Unknown으로 내리세요(§5: on-panel required absent=fail).")
            # ★ DEFINING_ABSENT (DF-absent 가드, 2026-06-15): 선택 term이 graph_evidence에서 df_absent=True면 — 그 term의
            #   defining 마커(하나면 충분한 최강 명찰)가 측정됐는데 전부 OFF → 보조마커만으로 force-fit(예: cl37 mitral=Cdhr1
            #   ON·Tbx21 OFF). graph_evidence가 이미 강등했으니, 그래도 골랐다면 강한 경고. 경쟁하는 비-force-fit anchor가
            #   있으면 그쪽으로(verdict는 그걸 가리킴), 없으면 상위 class/Unknown.
            ge_=p.get("graph_evidence") or {}
            _amatch=next((a for a in (ge_.get("anchors") or []) if a.get("term_id")==tid or a.get("label")==t_.get("L")), None) if tid in idx else None
            #   단, excl_on≥2(required 포함 강한 양성 근거)면 quiet-defining이라도 force-fit 아님(L4=Rorb/Whrn) → 경고 안 함.
            if _amatch and _amatch.get("df_absent") and _amatch.get("excl_on",0)<2:
                add(lf,"DEFINING_ABSENT","warn",
                    f"'{t_.get('L')}'의 on-panel defining 마커({', '.join(_amatch.get('df_onpanel') or [])}) OFF이고 exclusive 근거도 약함(excl_on<2) — 보조마커 force-fit 의심.",
                    "다른 exclusive 마커가 더 강한 term이 있으면 그쪽을, 없으면 verify 후 유지하거나 상위 class로. (excl_on≥2면 이 경고는 안 뜸.)")
            # ★ OFF_EVIDENCE (2026-06-15): 선택 specific term이 graph_evidence.anchors에 전혀 없는데 엔진이 clear_top(다른
            #   region의 확신 후보)을 줬다면 — region-coherent 엔진이 배제한 term을 raw 마커로 고른 것(cl49 피질→immature
            #   granule: 엔진은 L2/3 IT clear_top, immature granule(=DG region)은 anchors에 없음). 같은 region이면 면제.
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
                        f"'{t_.get('L')}'는 graph_evidence.anchors에 없음(region '{_crg}'). 엔진 clear_top은 '{_topa}'({'/'.join(sorted(_dom))}) — region-coherent 엔진이 배제한 term을 raw 마커로 골랐을 수 있음.",
                        f"엔진이 surface한 anchor('{_topa}')를 쓰거나, 이 term의 OWN 고유마커가 명확히 우세함을 hp/experience로 입증하세요.")
            # ★ SPEC_UNSUPPORTED(valid-evidence gate): 선택 term이 identity_scores에서 spec_pass==false
            # (자기 고유 변별마커 share≤3 중 절반 미만만 ON — broad/leaky 마커로 force-fit된 정체성, 예:
            # 비-해마 leaf를 Wfs1 공유로 CA1, 비-CA3 leaf를 한 마커로 CA3)면 specific celltype 종결 금지.
            # 타당한 양성 근거 없음 → lineage 분명하면 class-floor, 불분명하면 Unknown.
            # (spec_pass는 고유마커가 모두 off-panel(spec_n=0)일 땐 True → panel-blind 정상 종결은 통과.)
            isc=p.get("identity_scores") or {}
            _spass={}
            for c in ([isc.get("top")]+(isc.get("ranked") or [])):
                if c and c.get("term_id"): _spass.setdefault(c["term_id"], c.get("spec_pass"))
            if bool(d.get("subclass_resolved",True)) and _spass.get(tid) is False:
                add(lf,"SPEC_UNSUPPORTED","warn",
                    f"'{idx.get(tid,{}).get('L',tid)}'의 고유 변별마커(share≤3) 중 절반 미만만 ON(spec_pass=false) — broad/leaky 마커로 force-fit된 정체성(타당한 양성 근거 부족).",
                    "specific으로 확정하지 마세요 → lineage 분명하면 class-floor(subclass_resolved=false), 불분명하면 Unknown. hp/experience에서 그 type 마커가 실제 우세하면 그때만 유지(§5 spec_pass).")
            # ★ SHOULD_SPLIT_DEEPER(모든 tier): celltype으로 종결한 term의 1단계 자식 ≥2개가
            # 이 클러스터에서 비-pan 마커로 해소 가능하면(=더 쪼개면 분리됨) → 더 split 권고.
            # (class든 CA1이든 동일 규칙. profile의 candidate_subtypes scorecard를 그대로 사용.)
            if tid in idx:
                cs={c["term_id"]:c for c in (p.get("candidate_subtypes") or [])}
                kids=(cs.get(tid) or {}).get("children_resolvable") or []
                strong_kids=[k for k in kids if k.get("n_nonpan_present",0)>=1]
                if len(strong_kids)>=2:
                    add(lf,"SHOULD_SPLIT_DEEPER","warn",
                        f"'{idx[tid].get('L')}'로 종결했는데 1단계 자식 {len(strong_kids)}개"
                        f"({', '.join(k['label'] for k in strong_kids[:4])})의 변별 마커가 이 클러스터에 공존함.",
                        "더 쪼개면 분리됩니다 → needs_subclustering. (자식 마커가 off-panel이거나 split해도 안 나뉘면 그대로 종결.)")
            # ★ DF_MULTI_SUBTYPE (RC4, 2026-06-14): in-round split 감지가 children_resolvable의 strict-ON
            # (n_nonpan_present)에 의존하면 2um 같은 희소 데이터에서 quiet 마커가 ON을 못 넘겨 미발화한다.
            # df_present는 DEFINING 마커의 fold-ON(lf≥1.5 & 검출≥0.25)이라 quiet-but-enriched 마커도 잡는다 —
            # finalize split-review가 쓰는 바로 그 신호를 round 중에도 노출해 cl11(CA1+CA2+CA3 혼합)을 finalize
            # 전에 needs_subclustering으로 유도한다. top-level(depth-0)만, celltype 종결만, ≥2 distinct subtype.
            # ★ CONFUSION_UNRESOLVED (2026-06-14): chosen a specific term whose ConfusionPair PARTNER also anchors
            # (markers can't separate them — FC vs CA2 share Pcp4/Amigo2/Rgs14). Don't force-fit one → Unknown.
            cu=(p.get("graph_evidence") or {}).get("confusion_unresolved")
            if cu and bool(d.get("subclass_resolved",True)):
                chosen=idx.get(tid,{}).get("L","")
                if chosen and (chosen==(cu.get("terms") or [None])[0] or chosen in (cu.get("terms") or [])):
                    act=cu.get("action") or "unknown"; pair=cu.get("pair"); partners=', '.join((cu.get("terms") or [])[1:4])
                    if act=="split":
                        add(lf,"CONFUSION_UNRESOLVED","warn",
                            f"'{chosen}'와 짝({partners})의 *고유* 마커가 BOTH ON(top:{cu.get('top_distinct_on')} / 짝:{cu.get('partner_distinct_on')}) "
                            f"— 두 type 혼합({pair}).",
                            "→ needs_subclustering (쪼개세요). 쪼갠 뒤 각 조각을 CA2/FC/Unknown/상위 class로 결정.")
                    elif act=="resolve":
                        add(lf,"CONFUSION_UNRESOLVED","warn",
                            f"confusion-pair({pair})이나 한쪽 고유 마커만 ON → '{cu.get('resolves_to')}'를 지지.",
                            f"→ '{cu.get('resolves_to')}'로 결정(반대쪽 고유 마커는 OFF). 현재 '{chosen}'가 그쪽이 아니면 교체.")
                    else:
                        add(lf,"CONFUSION_UNRESOLVED","warn",
                            f"'{chosen}'와 짝({partners})이 공유 마커로만 anchor, 양쪽 고유 마커 모두 OFF — 마커로 분리 불가({pair}).",
                            "→ Unknown으로 결정하세요(둘 중 하나로 단정 금지). discriminator(위치 등)가 분명하면 그 type 유지.")
            if dec=="celltype" and lf.count(".")==0:
                dfp=sorted({x for x in (p.get("df_present") or []) if x})
                if len(dfp)>=2:
                    add(lf,"DF_MULTI_SUBTYPE","warn",
                        f"이 클러스터에 ≥2개 subtype의 DEFINING 마커가 fold-ON으로 공존함({', '.join(dfp[:4])}) — "
                        f"단일 celltype('{idx.get(tid,{}).get('L',tid)}')으로 종결했으나 다중 정체성 혼합 가능.",
                        "→ needs_subclustering (round 중 분할). 희소 데이터라 strict-ON은 약해도 fold 신호는 실재. "
                        "split해도 안 나뉘거나 한 type이 분명히 우세하면 그대로 종결.")
            # ★ SHOULD_SPLIT_MIXTURE (2026-06-14): graph_evidence.mixture_subtypes = dominant region의 ≥2 자식
            # subtype 고유마커가 WEAK 공존(희석된 혼합). df fold-ON(DF_MULTI_SUBTYPE)보다 약한 신호라 2µm 같은
            # 희소 데이터의 under-clustered 혼합을 잡는다(cl2 Drd1+Drd2 / cl15 Cux2+Rorb / cl7.1 L5+L6). 고유마커
            # 전무(cl4)는 mixture_subtypes 비어 안 뜸. depth≤1만(과분할·루프 방지; max_split_depth + split_no_gain 보조).
            if dec=="celltype" and lf.count(".")<=1:
                ms=(p.get("graph_evidence") or {}).get("mixture_subtypes") or []
                if len(ms)>=2:
                    labs=", ".join("%s(%s)"%(m.get("label",""), "/".join(m.get("markers_present") or [])) for m in ms[:4])
                    add(lf,"SHOULD_SPLIT_MIXTURE","warn",
                        f"종결('{idx.get(tid,{}).get('L',tid)}')했으나 같은 region의 ≥2 형제 subtype 고유마커가 (희석돼도) 공존함: {labs}.",
                        "→ needs_subclustering, 충분히 높은 resolution으로 분할(혼합이라 각 anchor가 약함; over-split해도 같은 라벨끼리 병합됨). "
                        "split해도 한 type만 나오면 split_no_gain으로 종결.")
            # ★ SIBLING_RESOLVABLE: 자식뿐 아니라 *형제* subtype도 본다. 고른 term이 subtype_groups의
            # 한 그룹에서 ≥1개의 다른 normal 형제 subtype과 공존하면(같은 부모의 변별마커 보유) → 이 leaf는
            # 형제 type(소수 섞인)을 함께 담고 있을 수 있음 → 더 split 권고. CA3로 종결했지만 23% CA1이
            # 섞여 있던 사례를 agent가 후처리 없이 잡게 하는 신호(절대 identity는 소수 희석으로 못 봄).
            # depth-0(최상위)만: depth-1+는 이미 분할 산물(Step5 split_no_gain) → 형제 교차상장 노이즈 억제.
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
                        f"'{idx[tid].get('L')}'로 종결했는데 같은 부모의 형제 subtype "
                        f"({', '.join(sibs[:4])})의 변별 마커가 이 클러스터에 공존함.",
                        "소수 섞인 형제 type일 수 있습니다 → needs_subclustering. (split해도 안 나뉘면 그대로 종결.)")
            hit=oh.get(tid)
            if hit and bool(d.get("subclass_resolved",True)):
                up_auc={_U(r[0]):r[1] for r in (p.get("up") or [])}
                hp_pct={_U(r[0]):r[3] for r in (p.get("hp") or []) if len(r)>3}
                pl=pool(p)
                rson=set(map(_U,hit.get("R_on",[])))|set(map(_U,hit.get("S_on",[])))
                resc=any(g in SPEC and up_auc.get(g,0)>=0.6 for g in set(map(_U,hit.get("R_on",[]))))
                if len(rson&pl)<2 and not resc:
                    add(lf,"RS_LT2_BUT_RESOLVED","warn",
                        "subclass_resolved=true인데 on-panel R/S가 2개 미만.",
                        "class-final(subclass_resolved=false)로 낮추거나 다른 term을 검토하세요.")
                N_on=set(map(_U,hit.get("N_on",[])))-PAN   # 비변별 pan/class 마커는 leak에서 제외(organ 무관)
                if any(up_auc.get(g,0)>=0.6 or hp_pct.get(g,0)>=0.9 for g in N_on):
                    add(lf,"NEGATIVE_LEAK","warn",
                        "선택 term의 negative 마커가 강하게 발현(누출).",
                        "close-sibling 가능 — needs_subclustering 또는 다른 term을 검토하세요.")
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
    """expected #type(k)별 Leiden start resolution. heterogeneity로 스케일(0.25/type), 하한 0.1·상한 2.0.
    낮게 시작해 과분할 노이즈를 피하고, 잔여 혼합(SHOULD_SPLIT_MIXTURE)이면 round loop가 γ×1.5로 올린다.
    γ↔#클러스터는 데이터 의존이라 *예측*이 아닌 수렴의 시작값. (k=2→0.5, 4→1.0, 6→1.5; 하한 0.1은 over-fragment
    시 낮추는 하한)."""
    return round(min(2.0, max(0.1, 0.25*k)), 2)
def _finalize_review(decisions, st, out, max_depth=1, large_min_n=1500):
    """finalize 게이트용: '한 번 더 쪼갤 가치가 있는' 대형 leaf를 정밀하게 추린다. 두 종류:
    ① bare-class(subclass 미해소): 분할로 분리될 1단계 *자식* ≥2개(children_resolvable, 비-pan 마커).
    ② resolved subclass: 같은 부모의 *형제* subtype ≥1개가 공존(subtype_groups) → 소수 섞인 형제 type
       (CA3 안의 23% CA1). 절대 identity는 소수 희석으로 못 보므로 이 게이트가 후처리 없는 backstop.
    공통 조건: celltype 결정 · 큼(n>=large_min_n) · 고밀도(neuropil 아님; low_frac<0.6)
    · reason_code != panel_limited(자식/형제 마커 off-panel이면 분할해도 무의미)
    · 분할 깊이 <= max_depth(이미 여러 번 쪼갠 leaf는 제외 → 무한루프 방지).
    detection(SHOULD_SPLIT_DEEPER/SIBLING_RESOLVABLE)과 달리, 여기서 잡힌 것은 caller가 user에게 물어볼 '행동 후보'다."""
    import glob
    from collections import Counter
    prof={}
    for pf in sorted(glob.glob(os.path.join(out,"cluster_gene_profiles_round*.json"))): prof.update(json.load(open(pf)))
    idx=(_load_ontology_index(out) or {}).get("terms",{})
    lab2D={}                                                            # label→disease_context(첫 매칭)
    for t in idx.values(): lab2D.setdefault(t.get("L"),t.get("D"))
    ncnt=Counter(st.get("cluster_paths") or [])
    cands=[]
    for lf in st["leaves"]:
        d=decisions.get(lf)
        if not d or d.get("decision")!="celltype": continue
        if (d.get("reason_code") or "")=="panel_limited": continue      # 자식/형제 마커 off-panel → 분할 무의미
        if lf.count(".")>max_depth: continue                            # 분할 깊이 상한(무한루프 방지)
        n=int(ncnt.get(lf,0))
        if n<large_min_n: continue                                      # 대형만
        p=prof.get(lf) or {}
        dfp=set(p.get("df_present") or [])        # terms whose DEFINING marker is fold-ON in this leaf
        dens=((p.get("identity_evidence") or {}).get("density") or {})
        lfrac=dens.get("low_frac")
        if lfrac is not None and lfrac>=0.6: continue                   # 저밀도(neuropil) → 분할해도 무의미
        tid=(d.get("term_id") or "").strip()
        resolved=bool(d.get("subclass_resolved",True))
        _ge=p.get("graph_evidence") or {}
        _mix=_ge.get("mixture_subtypes") or []
        _cu=_ge.get("confusion_unresolved") or {}
        # ★ MIXTURE backstop (2026-06-15): df_present(정의마커 fold-ON) ≥2 경로는 DEFINING이 없는 type과의 혼합을
        #   놓친다(cl64 FC=DF없음, cl66 immature granule=DF없음 → strong<2 탈락). graph_evidence가 직접 잡은
        #   confusion action=split 또는 mixture_subtypes(dominant region의 ≥2 자식 고유마커 공존)는 agent가 종결했어도
        #   finalize에서 split 후보로 surface — agent 미결정에 의존하지 않는 백스톱. 단 2중 가드로 오발동 차단:
        #   (a) lineage 일치 — agent 라벨의 root lineage == graph_evidence.lineage (glia/혈관은 agent가 ambient glut
        #       graph_evidence를 무시하고 라벨하므로 불일치 → 제외; oligo가 'L5 IT로 split' 같은 거짓 후보 차단)
        #   (b) mixture subtype이 dominant_region 안 — cross-region ambient bleed(CA1 안의 L2/3) 제외, 같은 region의
        #       진짜 공동집단(DG 안의 immature granule)만.
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
        _mix_inreg=[m for m in _mix if (_lab2tid.get(m.get("label")) and idx.get(_reg_of(_lab2tid[m["label"]]),{}).get("L") in _dom)]
        mixture_trigger=_lin_ok and ((_cu.get("action")=="split") or (len(_mix_inreg)>=2))
        if not resolved:
            # bare-class(subtype 미해소): 분할로 분리될 1단계 *자식* ≥2개
            cs={c["term_id"]:c for c in (p.get("candidate_subtypes") or [])}
            kids=(cs.get(tid) or {}).get("children_resolvable") or []
            strong=[]
            for k in kids:
                lab=k.get("label","")
                if k.get("n_nonpan_present",0)<1 or lab in strong or "DEPRECATED" in lab: continue
                if (lab2D.get(lab) or "normal")!="normal": continue     # 질병/상태/발생 자식은 subtype 분할로 보지 않음
                strong.append(lab)                                      # 라벨 dedup + deprecated/비-normal 제외
            strong=[lab for lab in strong if lab in dfp]                # ★ only DF-CONFIRMED children (defining marker fold-ON) — kills ambient/shared-marker false children
            if len(strong)<2 and not mixture_trigger: continue          # DF-자식 ≥2개 OR graph_evidence가 혼합 직접 감지
            split_kind="children"
        else:
            # ★ resolved subclass: 자식이 아니라 *형제*를 본다. 고른 subtype이 subtype_groups에서 ≥1개의
            # 다른 normal 형제 subtype과 공존하면(같은 부모) → 이 leaf에 형제 type이 소수 섞여 있음
            # (CA3 안의 CA1). 절대 identity는 소수 희석으로 못 보므로, 이 게이트가 후처리 없는 backstop.
            # ★ depth-0(최상위)만: 형제 교차상장은 노이즈가 많고, depth-1+는 이미 분할 산물(Step5
            # split_no_gain이 담당)이라 재분할하면 루프. CA1/CA3 교훈의 본질은 top-level under-clustering.
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
            if len(strong)<1 and not mixture_trigger: continue          # 형제 ≥1개 OR graph_evidence 혼합 감지
            split_kind="siblings"
        # k = 분리될 distinct type 추정: DF-confirmed 형제(strong) ∪ graph_evidence.mixture_subtypes(희석 공존).
        if not strong and _mix_inreg: strong=[m.get("label","") for m in _mix_inreg if m.get("label")]  # 혼합 trigger인데 DF-자식 없으면 in-region mixture 라벨
        nexp=max(len(strong), len(_mix), 2)                          # split 후보면 최소 2-way
        cands.append({"cluster_path":lf,"n_cells":n,"label":d.get("label",""),"term_id":tid,
                      "subclass_resolved":resolved,"split_kind":split_kind,
                      "density_pct":dens.get("pct"),"low_frac":lfrac,
                      "reason_code":d.get("reason_code",""),"resolvable_children":strong[:8],
                      "n_expected":nexp,"start_resolution":_res_for(nexp)})
    cands.sort(key=lambda c:-c["n_cells"])
    # ★ PRINCIPLED resolution (2026-06-15): Leiden resolution γ has NO closed-form map to #clusters or which
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
    _resmap={c["cluster_path"]: c["start_resolution"] for c in cands}     # ★ 클러스터별 맞춤 res
    return {"split_candidates":cands,"paths":[c["cluster_path"] for c in cands],
            "recommended_resolution":_rec,"resolution_map":_resmap,
            "max_split_depth":max_depth,"large_min_n":large_min_n,
            "note":"resolution은 *예측*이 아니라 *수렴*이다(γ↔#클러스터는 데이터 의존, 닫힌 공식 없음 — Traag/CPM). "
                   "start_resolution=expected #type로 스케일(0.25·k, 하한 0.1·상한 2.0; k=2→0.5,4→1.0,6→1.5; 옛 0.12·n·상한0.5 폐기). "
                   "낮게 시작(과분할 노이즈 회피), 잔여 혼합(SHOULD_SPLIT_MIXTURE/DF_MULTI_SUBTYPE)이면 γ×1.5 재분할로 수렴↑; "
                   "구별 안 되는 자식(split_no_gain)은 같은 라벨끼리 병합·종결. per-cluster는 cand.start_resolution."}

def cmd_finalize(a):
    import pandas as pd, glob, tarfile
    st=_load_state(a.out); decisions=json.load(open(a.decisions))
    # 검증: helper는 결정을 바꾸지 않는다. error만 차단(agent에 되돌림)하고, warn은 자문(기록만, 비차단).
    V=_validate(decisions, st, a.out)
    vp=os.path.join(a.out,"_annotate_violations.json")
    json.dump(V, open(vp,"w"), ensure_ascii=False, indent=1)
    errs=[v for v in V if v["severity"]=="error"]
    if errs and not a.force:
        print(f"[finalize][ABORT] 검증 error {len(errs)}건. h5ad 미작성. "
              f"caller는 {vp} 를 agent에게 주어 해당 클러스터만 재결정하게 한 뒤 다시 finalize 하세요. "
              f"(helper는 결정을 임의로 바꾸지 않습니다. 정말 그대로 쓰려면 --force.)", flush=True)
        for v in errs: print(f"  [error] {v['cluster_path']} {v['check']}: {v['suggestion']}", flush=True)
        sys.exit(2)
    # ★ 대형 bare-class leaf 분할 리뷰(finalize 게이트): 더 쪼개면 subtype이 분리될 대형·고밀도
    #   leaf가 남아 있으면 바로 쓰지 않고 멈춘다. caller가 user에게 "한 번 더 쪼갤지" 물어,
    #   YES면 paths를 subcluster→재판정→finalize, NO면 finalize --accept-large로 그대로 기록.
    review=_finalize_review(decisions, st, a.out, max_depth=a.max_split_depth, large_min_n=a.large_min_n)
    rp=os.path.join(a.out,"_annotate_finalize_review.json")
    json.dump(review, open(rp,"w"), ensure_ascii=False, indent=1)
    cands=review["split_candidates"]
    if cands and not a.accept_large and not a.force:
        print(f"[finalize][HOLD] 더 분할 가능한 대형 bare-class leaf {len(cands)}개 감지. h5ad 미작성.\n"
              f"  caller는 user에게 '이들을 한 번 더 쪼갤까요?(y/n)'를 물으세요.\n"
              f"   • YES → subcluster --paths '{json.dumps(review['paths'])}' --resolution '{json.dumps(review.get('resolution_map') or {})}' (클러스터별 맞춤; 단일값도 가능) → 재판정 → finalize\n"
              f"   • NO  → finalize ... --accept-large (현재 라벨 그대로 확정)\n"
              f"  상세: {rp}", flush=True)
        for c in cands:
            print(f"  [split?] {c['cluster_path']} n={c['n_cells']} '{c['label']}'(density_pct={c['density_pct']}) "
                  f"→ 자식: {', '.join(c['resolvable_children'][:4])}", flush=True)
        sys.exit(3)
    nw=len(V)-len(errs)
    if nw: print(f"[finalize] 자문 warn {nw}건은 기록만 하고 진행(차단 아님). 상세: {vp}", flush=True)
    if cands: print(f"[finalize] 대형 leaf {len(cands)}개를 분할 없이 현재 라벨로 확정(--accept-large/--force).", flush=True)
    # 통과(또는 --force): agent 결정을 그대로 신뢰해 기록(변경 없음)
    adata=ad.read_h5ad(st["h5ad_path"])          # 입력 전체 복사
    cps=np.array(st["cluster_paths"], dtype=object)
    ann=np.full(adata.n_obs,"",dtype=object)
    for path,dec in decisions.items():
        if path not in st["leaves"]: continue
        lab=dec.get("label","")
        ann[cps==path]=";".join(map(str,lab)) if isinstance(lab,(list,tuple)) else lab
    adata.obs["celltype_annotation"]=ann          # ★ 핵심 컬럼만 추가(helper 강등/변경 없음)
    # 출력 경로: --output > state.output > {out}/{input}_annotated.h5ad (annotation_out 안)
    out_h5ad=a.output or st.get("output") or os.path.join(a.out, re.sub(r"\.h5ad$","",os.path.basename(st["h5ad_path"]))+"_annotated.h5ad")
    adata.write_h5ad(out_h5ad)
    print(f"[finalize] wrote {out_h5ad} (입력 복사 + celltype_annotation)", flush=True)
    # 상세 메타는 CSV
    rows=[]
    for path in st["leaves"]:
        dec=decisions[path]
        rows.append({"cluster_path":path,"n_cells":int((cps==path).sum()),
            "decision":dec.get("decision",""),"label":dec.get("label",""),"term_id":dec.get("term_id",""),
            "subclass_resolved":dec.get("subclass_resolved",""),"confidence":dec.get("confidence",""),
            "reason_code":dec.get("reason_code",""),"key_markers":";".join(dec.get("key_markers",[]) if isinstance(dec.get("key_markers"),list) else [str(dec.get("key_markers",""))]),
            "rationale":dec.get("rationale",""),"alternatives_considered":dec.get("alternatives_considered",""),
            "open_question_flag":bool(dec.get("open_question_flag",False))})
    csv_path=os.path.join(a.out,"expert_judgment_annotation.csv")
    pd.DataFrame(rows).to_csv(csv_path,index=False)
    print(f"[finalize] wrote {csv_path}", flush=True)
    # h5ad·최종 CSV 이외의 모든 작업 산출물(state/decisions/ontology/profiles/global stats 등)을 단일 압축으로.
    # → annotation_out: {input}_annotated.h5ad + expert_judgment_annotation.csv(merge 입력) + annotation_artifacts.tar.gz.
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
        print(f"[finalize] archived {len(tmp)} artifacts → {arc} (annotation_out: h5ad + 압축파일만 남김)", flush=True)

# ===================================================================== main
if __name__=="__main__":
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest="cmd"); sp.required=True
    pi=sp.add_parser("init"); pi.add_argument("--h5ad",required=True); pi.add_argument("--cluster-col",dest="cluster_col",required=True)
    pi.add_argument("--out",required=True)
    pi.add_argument("--species",default=None); pi.add_argument("--organ",default=None)
    pi.add_argument("--server-url",dest="server_url",default=None)   # 보통 불필요(기본 서버 자동연결)
    pi.add_argument("--ontology-url",dest="ontology_url",default=None)  # 전체 URL 직접지정(고급)
    pi.add_argument("--ontology",default=None); pi.add_argument("--output",default=None)
    pi.add_argument("--disease",default=None,help="입력 데이터의 질병 컨텍스트(예: 'brain cancer'). 그 질병의 state term을 leaf별 disease_candidates로 surface(§7).")
    pi.set_defaults(f=cmd_init)
    ps=sp.add_parser("subcluster"); ps.add_argument("--out",required=True); ps.add_argument("--paths",required=True)
    ps.add_argument("--resolution",default="1.0")   # 수동 fallback만(옛 0.2는 under-split). 실제론 finalize start_resolution(_res_for=0.25·k)이 덮어씀
    ps.add_argument("--gpu",default=None); ps.add_argument("--ontology",default=None); ps.set_defaults(f=cmd_subcluster)
    # signal_split 제거(Leiden+absolute paradigm): 강제배정 오염 방지. 분할은 subcluster(Leiden) 전용.
    pc=sp.add_parser("check"); pc.add_argument("--out",required=True); pc.add_argument("--decisions",required=True); pc.set_defaults(f=cmd_check)
    pf=sp.add_parser("finalize"); pf.add_argument("--out",required=True); pf.add_argument("--decisions",required=True)
    pf.add_argument("--output",default=None); pf.add_argument("--force",action="store_true")
    pf.add_argument("--accept-large",dest="accept_large",action="store_true")        # 대형 leaf 분할 리뷰 통과(user가 NO)
    pf.add_argument("--max-split-depth",dest="max_split_depth",type=int,default=1)    # 분할 깊이 상한(무한루프 방지)
    pf.add_argument("--large-min-n",dest="large_min_n",type=int,default=1500)         # '대형' 기준 셀 수
    pf.set_defaults(f=cmd_finalize)
    a=p.parse_args(); a.f(a)
