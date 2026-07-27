#!/usr/bin/env python3
# NOTE: helper.py is a STANDALONE build — the helper.py engine is INLINED below,
#       plus the annotate-mode marker-axis split (subcluster/descend). Do NOT edit
#       helper.py expecting changes here; regenerate via scratchpad/build_helper.py.
"""UNNAMED annotate helper — in-memory recursive subclustering + ontology-aware profiling.

cellularspace `_annotate_helper.py`를 계승하되 UNNAMED 변경 반영:
  - cluster 컬럼 인자화 (--cluster-col; leiden 하드코딩 제거)
  - ontology = **서버 slim-index fetch** (init에서 1회) + `--ontology` 로컬 폴백
  - subcluster 백엔드 **자동 분기**: GPU+rapids_singlecell → rapids, 아니면 scanpy
  - profile `bg` → **identity_evidence**(vf_norm 밀도 + 정체 신호) ; up/dn 무제한, hp=50
  - 출력 = **{input}_annotated.h5ad** (입력 전체 복사 + obs['cellman_annotation'](표준) / obs['cellman_free_annotation'](사용자 상호작용)만 추가)

서브커맨드:
  init       --h5ad --cluster-col --out [--ontology-url | --ontology] [--use-rep X_scVI|auto|recompute]
  subcluster --out --paths '["4","11"]' --resolution 0.2 [--gpu 0] [--ontology ...]
  finalize   --out --decisions <json> [--output <path>]
"""
import argparse, json, os, sys, re, datetime
import numpy as np
import anndata as ad
from scipy.stats import rankdata
from scipy.sparse import csr_matrix

# ===================================================================== ontology
# 우리가 배포하는 단일 Cell-Ontology 서버(공개, 읽기 전용). 사용자는 보통 아무것도 안 줘도 여기로 자동 연결된다.
# 우선순위: --server-url 플래그 > 환경변수 CELLONTO_SERVER_URL > 이 기본값.
# 내부(랩) 개발자는 CELLONTO_SERVER_URL로 사설 주소(예: http://192.168.101.31:8088)를 덮어쓴다.
DEFAULT_ONTOLOGY_SERVER = "https://cellman.pnucolab.com"
def _resolve_server(server_url=None):
    return (server_url or os.environ.get("CELLONTO_SERVER_URL") or DEFAULT_ONTOLOGY_SERVER).rstrip("/")

def _dedup_parents(idx):
    """KG 빌드 결함 방어 (2026-06-17): term의 P(부모 리스트)에 같은 부모가 중복 등장하는 경우가 있다
    (관측: mouse_whole_brain 28/373 term이 P=['X','X']). children-index/subtype_groups가 `for p in P`를
    dedup 없이 돌아 같은 자식을 두 번 세어 → 자식이 1종뿐인데 허위 '≥2 형제' subtype_group이 성립하고
    children_resolvable에도 중복이 생긴다(허위 분할 신호). 인덱스를 얻는 모든 지점에서 순서 보존 dedup으로
    모든 소비처(subtype_groups·_candidate_subtypes·_graph_evidence·_build_ident_maps)를 한 번에 정상화."""
    if idx and isinstance(idx.get("terms"), dict):
        for t in idx["terms"].values():
            P = t.get("P")
            if P and len(P) != len(dict.fromkeys(P)):
                t["P"] = list(dict.fromkeys(P))      # 순서 보존 dedup
    return idx

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
                sys.exit(f"[init][ERROR:MODULE_NOT_FOUND] 서버에 해당 module이 없습니다 ({ontology_url}, HTTP 404). "
                         f"(species, organ)을 확인하거나 그 module의 빌드를 요청하세요. → caller는 사용자에게 "
                         f"'요청한 조직(species/organ)의 온톨로지가 서버에 없어 annotation을 진행할 수 없습니다'라고 알릴 것.")
            sys.exit(f"[init][ERROR:SERVER_HTTP_{e.code}] 온톨로지 서버가 오류 {e.code}를 반환했습니다 ({ontology_url}).")
        except urllib.error.URLError as e:
            sys.exit(f"[init][ERROR:SERVER_UNREACHABLE] 온톨로지 서버에 연결할 수 없습니다 ({ontology_url}; {e.reason}). "
                     f"서버 기동 여부와 CELLONTO_SERVER_URL을 확인하세요. → caller는 사용자에게 "
                     f"'온톨로지 서버에 연결할 수 없습니다(서버 다운 또는 주소 오류)'라고 알릴 것.")
        if not idx or not idx.get("terms"):
            sys.exit(f"[init][ERROR:MODULE_EMPTY] module은 응답했으나 term이 비어 있습니다 ({ontology_url}). "
                     f"그 module의 빌드가 필요합니다. → caller는 사용자에게 '해당 조직의 온톨로지가 비어 있어 진행 불가'라고 알릴 것.")
        idx = _dedup_parents(idx)                       # P 중복 부모 정리(허위 형제 그룹 방지)
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
                    "D": t.get("disease_context","normal"),
                    "A": list(t.get("anchor", []) or []),          # identity-anchor (origin 잔존) 마커
                    "MP": list(t.get("mimics", []) or [])}         # mimics[{lineage,program}] (흉내 대상)
        idx = _dedup_parents(idx)                       # P 중복 부모 정리(허위 형제 그룹 방지)
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
        for k in ("R","S","DF"): s.update(t.get(k, []) or [])  # negative는 보호 안 함
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
# HVG feeding PCA는 (1) technical gene을 빼고 (2) KG에 있고 데이터에도 있는데 HVG가 놓친 마커를 도로 넣는다.
_PANEL_DENY_MOUSE=("mt-","Rps","Rpl","Hba","Hbb")
_PANEL_DENY_HUMAN=("MT-","RPS","RPL","MRPL","MRPS","HBA","HBB")
_PANEL_DENY_HUMAN_RE=re.compile(r"^(ENSG|LINC|MIR|SNOR)|(-AS\d*$)|(-DT$)|(-IT\d*$)|(-OT\d*$)"
                                r"|(^RP\d+-)|(^A[CLP]\d{6})|(^Z\d{5})|(\.\d+$)")
def _panel_denier(species):
    """technical/low-info gene predicate (g->bool). species 미지정이면 mouse 기본."""
    sp=(species or "").lower()
    if sp.startswith("human") or sp in ("hs","homo_sapiens"):
        return lambda g: g.startswith(_PANEL_DENY_HUMAN) or g=="MALAT1" or bool(_PANEL_DENY_HUMAN_RE.search(g))
    return lambda g: g.startswith(_PANEL_DENY_MOUSE) or bool(re.match(r"Gm\d",g)) or g.endswith("Rik")

def _panel_clean(g):
    """marker 심볼을 var_names에 맞게 정규화(-/_ suffix 제거) — _panel._clean과 동일."""
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
# 과거의 뇌-하드코딩(_LINEAGE/_LAYER/_SPECIFIC/_PAN)을 폐기하고, lineage/specific/pan/sibling을
# 그 module의 온톨로지(slim-index)에서 유도한다. 유도 불가(계층/마커 빈약) 시에만 아래 cross-organ
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

# ★ PROCESS tier (2026-07-15, 'compartment'→'process' 개명 2026-07-17): 세포체가 아니라 **세포에서 뻗어나온
#   부분**(process)으로 채워진 조직 영역. 정의 = "mRNA는 있는데 핵/세포체가 아닌 영역". neuropil(수상돌기·축삭·
#   교세포 돌기의 mesh)과 fiber tract(수초화 축삭 다발)이 그 신경계 사례고, 망막 plexiform layer·신장 podocyte
#   발돌기·폐 I형 폐포세포 확장부처럼 **비신경 조직으로도 확장**된다.
#   ★ 2026-07-21 리팩터: process를 **pre-Leiden 자동 파티션**으로 찾던 로직(Step P + cluster-level Step 0.5)은
#   제거됐다. process는 양성 마커를 그 전사체를 만든 소마와 **공유**하므로 마커축 클러스터링(clustering-based
#   annotation)으로는 소마에서 안 갈린다 — 오직 **음성축(MALAT1 등 핵-보유)** 으로만 갈린다. 그래서 process
#   탐지는 이제 사용자가 명시적으로 요청하는 **scoring-based annotation**(score-profile/apply/finalize) 전용이다.
#   _PROC_TIERS는 (1) scoring 기본 대상(target 미지정 시 tier='process' class) + (2) _onto_ctx에서 process term을
#   celltype 엔진(lineage/anchor)에서 배제(회귀 방지: neuropil이 lineage root로 잡혀 oligodendrocyte 오판)에 쓰인다.
#   'compartment'는 legacy tier로 계속 받는다.
_PROC_TIERS={"process","compartment"}

# ================================================================ SCORING-based annotation
# ★ 2026-07-21. process P 로직을 **일반화**한 per-vector scoring 엔진. clustering-based annotation(마커축
#   Leiden→클러스터당 판정)과 대비되는 두 번째 모드다. 대상 class의 positive 마커 합성점수 상위 N%를 후보로,
#   negative 마커 합성점수 상위 M%를 배제해 keep을 정한다(클러스터 아님, 벡터 단위). neuropil/fiber tract처럼
#   양성 마커를 소마와 공유해 클러스터로 안 갈리는 대상에 필수. 기본 대상=tier='process' class지만 --targets로
#   임의 KG class도 스코어링할 수 있다(pos=자신 R∪DF, neg=자신 N∪조상 N).
_PROC_CAND_PCT = 10.0     # 후보 = pos_score 상위 N%   (default; 스킬과 같은 값)
_PROC_NEG_PCT  = 10.0     # 제거 = neg_score 상위 M%   (default; 우연 제거율 = M% 이 되어 배수 해석이 자명)

def _score_targets_ctx(idx, target_labels=None):
    """scoring-based annotation의 대상 class를 KG 구조에서 유도. 하드코딩 이름 없음.
      target_labels=None  → module의 tier='process' class(process 루트의 **직속 자식**: neuropil / fiber tract / …)
                            를 기본 대상으로 (process 탐지의 기본 동작).
      target_labels=[...] → 임의 KG class 라벨(대소문자 무시)을 대상으로. "score 기반으로 <X> 찾아" 처럼
                            사용자가 명시한 임의 class를 스코어링.
    각 class:
      pos       = 그 class의 R∪DF
      neg       = N(자신) ∪ N(조상 전부)  ← 범용 핵 축(MALAT1)은 루트에만 있고 전원이 상속한다(process),
                  임의 class면 그 계보의 curated N을 상속.
      children  = 직속 자식(typed subtype)들의 델타(R∪DF).
    {} 반환 = 대상 없음(기본 모드에서 process term 없는 module, 또는 --targets가 어떤 term에도 매칭 안 됨)."""
    terms=(idx or {}).get("terms",{}) or {}
    if not terms: return {}
    def _U(gs): return {str(g).upper() for g in (gs or [])}
    def _rdf(t): return _U(t.get("R"))|_U(t.get("DF"))
    def _par(tid): return [p for p in dict.fromkeys(terms[tid].get("P") or []) if p in terms and p!=tid]  # slim이 P를 중복해 실음
    def _kid(tid): return [c for c in terms if tid in _par(c)]
    def _neg(tid, seen=None):
        if seen is None: seen=set()
        if tid in seen or tid not in terms: return set()
        seen.add(tid)
        out=_U(terms[tid].get("N"))
        for q in _par(tid): out |= _neg(q, seen)
        return out
    # 대상 term id 결정
    if target_labels:
        want={str(l).strip().lower() for l in target_labels}
        target_ids=[tid for tid,t in terms.items() if str(t.get("L","")).strip().lower() in want]
    else:
        P={tid for tid,t in terms.items() if (t.get("T") or "") in _PROC_TIERS}
        if not P: return {}
        def _parP(tid): return [p for p in _par(tid) if p in P]
        roots=[tid for tid in P if not _parP(tid)]
        target_ids=[c for r in roots for c in P if r in _parP(c)]      # process 루트의 직속 자식
    classes=[]
    for c in target_ids:
        classes.append({"id":c, "label":terms[c].get("L",c),
                        "pos":sorted(_rdf(terms[c])), "neg":sorted(_neg(c)),
                        "children":[{"id":k,"label":terms[k].get("L",k),"delta":sorted(_rdf(terms[k]))}
                                    for k in _kid(c)]})
    return {"classes":classes}


def _zc(X, cols):
    """표준화 composite. ★ AND/OR 조합(k-of-n)을 쓰지 않는 이유: 수율이 유전자 **개수**에 지수적으로
    끌려간다 — 같은 데이터에서 neuropil(7개 AND)은 0%, fiber tract(2개 AND)는 6.4%가 나와 마커
    품질 차이처럼 보였지만 전부 0.25^7 vs 0.25^2 이라는 산수였다. 유전자별 z → 평균 → **그 평균을
    다시 표준화**하면 2개든 20개든 축의 분포가 같아져 임계값과 수율이 비교 가능해진다."""
    import numpy as _np
    Z=_np.column_stack([(X[:,i]-X[:,i].mean())/(X[:,i].std()+1e-9) for i in cols])
    c=Z.mean(1)
    return (c-c.mean())/(c.std()+1e-9)


def _proc_scores(adata, ctx, cand_pct=_PROC_CAND_PCT, neg_pct=_PROC_NEG_PCT, restrict=None):
    """class별 pos/neg score와 판단 재료를 계산. 라벨을 붙이지 않는다 — 판정은 에이전트 몫.
    restrict = 전체 길이 bool 마스크(None=전 벡터). 지정 시 후보 percentile을 그 subset **안에서만** 잡는다
    ('이 neuron들 안에서 상위 N%')·keep/pct도 subset 기준. z-score는 전 벡터 대비(=whole-dataset enrichment).
    반환 마스크(_mask/_cand)는 전체 길이(restrict 밖은 False)라 하류 인덱스 매핑이 필요 없다."""
    import numpy as _np
    X=adata.X; X=_np.asarray(X.todense()) if hasattr(X,"todense") else _np.asarray(X)
    um={str(g).upper():i for i,g in enumerate(adata.var_names)}
    vf=adata.obs["vf_norm"].to_numpy() if "vf_norm" in adata.obs else None
    rm=_np.ones(adata.n_obs,bool) if restrict is None else _np.asarray(restrict,bool)
    nrm=max(int(rm.sum()),1)
    out=[]
    for cl in ctx["classes"]:
        pc=[um[g] for g in cl["pos"] if g in um]; pmiss=[g for g in cl["pos"] if g not in um]
        nc=[um[g] for g in cl["neg"] if g in um]; nmiss=[g for g in cl["neg"] if g not in um]
        if not pc:
            out.append({"label":cl["label"],"term_id":cl["id"],"usable":False,
                        "reason":"positive 마커가 패널에 하나도 없음",
                        "pos_missing":pmiss,"neg_missing":nmiss}); continue
        ps=_zc(X,pc)
        cand=(ps>_np.quantile(ps[rm],1-cand_pct/100.0)) & rm            # 후보는 restrict 안에서만
        rec={"label":cl["label"],"term_id":cl["id"],"usable":True,
             "pos_used":[adata.var_names[i] for i in pc],"pos_missing":pmiss,
             "neg_used":[adata.var_names[i] for i in nc],"neg_missing":nmiss,
             "n_candidate":int(cand.sum()),"pct_candidate":round(100*float(cand.sum())/nrm,2)}
        if nc:
            ns=_zc(X,nc)
            negon=(ns>_np.quantile(ns[rm],1-neg_pct/100.0)) & rm
            removed=float((cand&negon).sum())/max(int(cand.sum()),1)
            keep=cand&~negon
            rec.update({
                "corr_pos_neg":round(float(_np.corrcoef(ps[rm],ns[rm])[0,1]),3),
                # ★ 배수 해석: neg_score 상위 M%는 정의상 restrict의 M%를 지운다. neg가 후보와 무관하면
                #   후보에서도 M%가 지워진다 => 우연=M%. 배수>1 = 후보에 소마가 실제로 몰려 있고
                #   필터가 그걸 걷어냄. 배수≈1 = 필터가 무정보. 배수<1 = 후보가 이미 소마-결핍.
                "removal":{"observed_pct":round(100*removed,2),"chance_pct":neg_pct,
                           "ratio":round(removed/(neg_pct/100.0),2)},
                "n_final":int(keep.sum()),"pct_final":round(100*float(keep.sum())/nrm,2)})
        else:
            keep=cand
            rec.update({"removal":None,"n_final":int(keep.sum()),
                        "pct_final":round(100*float(keep.sum())/nrm,2)})
        if vf is not None:
            rec["density"]={"median_vf_final":round(float(_np.median(vf[keep])),4) if keep.sum() else None,
                            "median_vf_all":round(float(_np.median(vf[rm])),4)}
        subs=[]
        for k in cl["children"]:
            dc=[um[g] for g in k["delta"] if g in um]
            subs.append({"label":k["label"],"term_id":k["id"],
                         "delta_used":[adata.var_names[i] for i in dc],
                         "delta_missing":[g for g in k["delta"] if g not in um]})
        rec["subtypes"]=subs
        rec["_mask"]=keep
        rec["_cand"]=cand
        rec["_pos"]=ps
        out.append(rec)
    return out


def _onto_ctx(idx):
    """profiling/validate가 쓰는 도메인 컨텍스트(온톨로지 유도, fallback 포함)."""
    terms_all=(idx or {}).get("terms",{})
    # ★ PROCESS 분리 (2026-07-15, 게이트 일반화 2026-07-21): tier='process' term(neuropil/fiber tract 등)은
    #   celltype이 아니므로 lineage/specific/pan/sibling 유도 및 IDM(anchor/top)에서 제외한다 — state term을
    #   top에서 빼는 것과 같은 취지. 제외 안 하면 neuropil/fiber-tract가 root=lineage로 잡혀 진짜
    #   oligodendrocyte(MBP+)를 'fiber tract' lineage로 오판한다(process module은 KG에 남아 있으므로 필수).
    #   process 마커는 _ontology_gene_set(full idx)엔 남아 scoring-based annotation이 측정한다. process term이
    #   없는 module이면 idx 그대로 통과 → 완전 무영향.
    _has_proc=any((t.get("T") or "") in _PROC_TIERS for t in terms_all.values())
    idx_ct=({**(idx or {}), "terms":{tid:t for tid,t in terms_all.items() if (t.get("T") or "") not in _PROC_TIERS}}
            if _has_proc else idx)
    terms=(idx_ct or {}).get("terms",{})
    return {"lineages": _derive_lineages(idx_ct) or {k:set(v) for k,v in _LINEAGE_FALLBACK.items()},
            "specific": _derive_specific(idx_ct) or set(_SPECIFIC_FALLBACK),
            "pan": _derive_pan(idx_ct) or set(_PAN_FALLBACK),
            # ★ pan_disc (2026-06-17): candidate_subtypes/children_resolvable의 변별마커 판정용 pan 집합.
            #   octx["pan"](≥4 term)은 Cux2(4)/Rorb(5)/Fezf2(5) 같은 *층 변별 마커*까지 pan으로 죽여 피질 층이
            #   children_resolvable·n_nonpan_present에서 사라진다(SHOULD_SPLIT_DEEPER 과소발화). _build_ident_maps가
            #   graph_evidence용으로 쓰는 것과 **동일한 ≥12 임계**로 맞춰 통일 — 진짜 pan(Slc17a7=52 등)만 거르고
            #   층 마커는 변별마커로 살린다. octx["pan"](≥4)은 _validate/lineage_present 등 기존 용처가 있어 불변.
            "pan_disc": _derive_pan(idx_ct, 12) or set(_PAN_FALLBACK),
            "siblings": _sibling_groups(idx_ct),
            "terms": terms,
            "cp": (idx or {}).get("cp") or [],   # confusion pairs from slim-index → surfaced by _graph_evidence
            "ca": (idx or {}).get("ca") or [],   # context axes (zonation/layer) → surfaced by _context_axis_eval (§4.3)
            "sa": (idx or {}).get("sa") or [],   # state axes (base-agnostic cell states) → surfaced by _state_axis_eval
            "term_req": {tid:{m.upper() for m in (t.get("R") or [])} for tid,t in terms.items()}}

def _context_axis_eval(lineage_label, ca, fon, lf, pin, gidx_U, lean_ratio=1.15, sig_floor=0.03, extra_labels=None):
    """★ Context-axis subtyping (§4.3) — DETERMINISTIC engine support. lineage의 subtype이 세포-내재 마커가
    아니라 *공간 위치*(zonation/layer)로 갈리는 경우, module이 선언한 ca 축의 pole별 context_markers(=이웃 우세
    세포형 마커, 분할데이터선 ambient)를 점수화해 dominant pole→subtype을 제안한다. lineage는 graph_evidence가
    내재마커로 이미 확정(여기선 절대 안 바꿈) — pole은 subtype 축에만 적용. off-panel context 마커는 스킵.
    ★ WEAK-LEAN (2026-06-21): zonation은 연속체라 신호가 약하다 — 강한 우세를 요구하면 대부분 bare class로
    빠진다. 그래서 (a) 점수는 graded(하드 ON 임계 없이 fon 합 — 약한 ambient도 반영), (b) 한 극이 *약하더라도*
    분명히(>=lean_ratio×) 앞서면 그 극으로 판정(pole_confidence=low; >=1.5×면 medium). 둘 다 실재하나 동률
    (near-tie)이면 mixed(부모면 분할). bounded same-lineage 축이라 최악도 인접 zone(치명적 오판 아님)."""
    if not ca: return None
    # ★ region/anchor-aware 매칭 (2026-06-22): liver는 lineage=celltype("Hepatocyte")라 lineage_label로 충분하나,
    #   brain은 lineage=transmitter class("glutamatergic neuron")로 너무 broad → applies_to를 dominant_region/anchor
    #   라벨(extra_labels)에도 매칭해 축을 올바른 region/celltype에만 적용(thalamic core/matrix, CA1 deep/superficial).
    cand={str(lineage_label).strip().lower()} if lineage_label else set()
    for x in (extra_labels or []):
        if x: cand.add(str(x).strip().lower())
    if not cand: return None
    def lvl(g):
        i=gidx_U.get(str(g).upper())
        if i is None: return None                       # off-panel — 무시(증거로 안 씀)
        return float(max(fon[i],0.0))                   # graded presence (약한 신호도 반영)
    out=[]
    for ax in ca:
        # ★ lineage 게이트 (2026-06-22): 축이 lineage를 선언하면 cluster의 transmitter lineage가 일치할 때만 적용.
        #   CA1 radial 같은 glut-pyramidal 축이 GABAergic interneuron 클러스터(parent가 CA1라 anchor/path로 매칭)에
        #   잘못 발화해 "CA1 deep pyramidal"로 오라벨하던 문제 차단. lineage 미선언 축은 게이트 없음(liver 호환).
        ax_lin=ax.get("lineage")
        if ax_lin and lineage_label and str(ax_lin).strip().lower()!=str(lineage_label).strip().lower(): continue
        appl=[str(x).strip().lower() for x in (ax.get("applies_to") or [])]
        matched=[a for a in appl if a in cand]
        if not matched: continue
        mkey=matched[0]                                  # 매칭된 applies_to(소문자) → subtype lookup 키
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
        if top>0 and top>=lean_ratio*max(second,1e-9):  # 약하더라도 한 극이 분명히 우세 → 그 극으로 lean
            dom=present[0]; conf=("medium" if top>=1.5*max(second,1e-9) else "low")
        # ★ mixed = 둘째 극이 *실질적*(>=0.4×)으로 공존 → 큰 부모면 양극 분리 위해 subcluster(소수 극 매몰 방지).
        #   dominant_pole이 있어도 mixed일 수 있다(부모가 한쪽으로 기울지만 반대 극도 상당). lean은 분할 후
        #   sub-leaf 라벨용, mixed는 분할 트리거(§4.3). 둘째 극이 미미하면(pure-ish) mixed=False → 바로 lean-label.
        both_substantial=(len(present)>=2 and second>=0.4*max(top,1e-9))
        out.append({"axis":ax.get("name"),"poles":poles,
                    "poles_present":[p["pole"] for p in present],
                    "mixed":both_substantial,
                    "dominant_pole":(dom["pole"] if dom else None),
                    "suggested_subtype":(dom["subtype"] if dom else None),
                    "pole_confidence":conf,
                    "pmids":ax.get("pmids") or []})
    return out or None

def _state_axis_eval(sa, fon, lf, pin, gidx_U, frac=0.30, lf_on=1.5, pin_min=0.25, min_on=2):
    """★ STATE-AXIS eval — base-agnostic cell states (proliferating / interferon-stimulated / hypoxic …).
    Unlike `_context_axis_eval` (pole competition for a subtype), each State is judged INDEPENDENTLY and any
    number can be active: a State is ACTIVE when ≥`min_on` of its OWN program markers are **ENRICHED** —
    elevated vs the whole dataset (fon≥frac OR fold≥lf_on & detected≥pin_min), NOT mere detection. State
    markers are program markers (MKI67=proliferation …), curated base-agnostically in the KG, so the
    discriminative side is baked in; the engine only checks enrichment + coherence (≥2). Ranked by summed
    enrichment. States with an IDENTICAL marker signature (EMT ≡ mesenchymal) are de-duplicated (keep one).
    Returns [{name, markers_on[], n_on, n_onpanel, score, pmids}] (highest first); [] if none active.
    The label is later synthesized as `base, <state1>, <state2>` (finalize)."""
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
        if len(on)>=min_on:                                        # coherence: ≥2 program markers enriched
            seen_sig.add(sig)
            out.append({"name":st.get("name"),"markers_on":on,"n_on":len(on),"n_onpanel":n_onpanel,
                        "score":round(score,3),"pmids":st.get("pmids") or []})
    out.sort(key=lambda x:-x["score"])
    return out

def _candidate_subtypes(terms, ogt, up_rows, octx, parent_up_auc=None, region_share=None):
    """이 클러스터에서 *정의 마커가 실제로 보이는* 온톨로지 term을 증거 scorecard로 랭킹(옆 후보).
    + 각 후보의 1단계 자식이 이 클러스터에서 해소 가능한지(children_resolvable)도 같이 — 더 split할지 판단용(아래 lookahead).
    품질 필터: pan(비변별) 마커만 떠선 후보로 안 침. → CA1/CA3 같은 진짜 subtype만 소수로 남음.
    전부 온톨로지 런타임 유도 → organ 무관·하드코딩 0. 비강제(Step5 게이트가 최종).
    ★ 부모 풀링(parent_up_auc): 균질 subtype이 과분할로 사라지는 것 방지. 자식 up(=형제 대비)에선
    layer/relay 마커가 ~0.5로 평준화되지만, 부모(=뇌 전체 대비)에서 enriched였다면 그 정체성은 모든
    자식에 상속된다. 그래서 term의 *자신* required 평가엔 부모 enrichment를 함께 본다(inherit=True).
    단 children_resolvable(=더 쪼개면 분리?)엔 부모 풀링을 쓰지 않는다 — 균질 마커는 분할로 안 갈라지므로."""
    # ★ pan_disc(≥12) 사용 — graph_evidence(_build_ident_maps)와 동일 기준. octx["pan"](≥4)을 쓰면 Cux2/Rorb/Fezf2
    #   같은 층 변별 마커가 pan으로 죽어 n_nonpan_present가 0이 되고 SHOULD_SPLIT_DEEPER가 과소발화한다(통일, 2026-06-17).
    pan=octx.get("pan_disc") or octx.get("pan") or set()
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
        # ★ 변별/비변별(isp) 판정을 graph_evidence의 anchor와 통일(2026-06-17): region_share가 주어지면
        #   "비변별=영역 >2개에 퍼짐(region_share>SPEC_REGION_MAX=2)"으로 판정 — pan≥12로 broad 마커(PCP4 등)를
        #   못 거르던 과다발화를 막고, 층 마커(Rorb/Cux2/Fezf2: region_share=1)는 변별로 살린다. region_share가
        #   없으면(IDM 미생성) pan_disc(≥12) 집합으로 폴백.
        gu=_U(g); isp=(region_share.get(gu,99)>2) if region_share else (gu in pan)
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
        """이 term의 모든 마커(R∪S∪DF) 중 '비-pan · on-panel · 강한(AUC≥0.55)'인 것들(유전자 dedup).
        required가 off-panel(CA1의 Wfs1)이어도 supporting(Fibcd1)로 잡힘 → signal_split target 자동
        도출의 핵심(누락 방지). 진짜로 존재하는 type은 *여러* 변별마커가 잡히고, 마커 1개만 교차상장된
        spurious(예: 피질에 Cerebellar Golgi via Neurod6)는 1개라 아래 랭킹에서 밀린다."""
        tt=terms.get(tid) or {}; d={}
        for g in (tt.get("R") or [])+(tt.get("S") or [])+(tt.get("DF") or []):
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
    gUp={_U(g):i for i,g in enumerate(genes)}
    # ★ 접미사 마커명 정규화 (2026-06-16): KG 마커의 15%가 문맥 접미사(Drd1_strict, Foxp2_high, Tac1_subset,
    #   Slc17a6_neuronal, Pou4f1_PAG, Aromatase_Cyp19a1, Sf1_Nr5a1 …)를 달고 있어 패널 심볼과 매칭 실패 →
    #   의도한 마커가 죽어있었다. base 심볼로 재매칭: full→첫 토큰(gene_suffix)→마지막 토큰(alias_gene:
    #   Aromatase_Cyp19a1→Cyp19a1). ★중요: 정규화는 NEGATIVE·DF·lineage에만 적용하고 DISC/PAN/region_share는
    #   RAW 유지한다 — disc에 base를 복원하면(Foxp2_high×31 → Foxp2) region_share가 부풀어 Fezf2(L5 ET)·Drd2(D2-MSN)·
    #   Foxp2(L6 CT) 같은 검증된 exclusive/defining anchor가 비-exclusive로 강등돼 광범위 회귀가 난다. 실제 피해를
    #   준 건 죽은 NEGATIVE(eSPN의 Drd1_strict/Drd2_strict가 죽어 D1/D2로 veto 안 돼 모든 선조체 leaf 점령)이므로
    #   negative만 살리면 충분(eSPN→Drd1/Drd2 negative 복원 → D1/D2 leaf에서 veto). DF 정규화는 df_offpanel 정확도
    #   +DF-override만 돕고 region_share 불변(별도 맵). 엔진 on_enr/auc 임계가 발현 강도를 처리하므로 뉘앙스 손실 무해.
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
            for _m in (_t.get(_k) or []): _cnt[_m.upper()]=_cnt.get(_m.upper(),0)+1   # RAW(원본 동작 보존)
    PAN={_m for _m,_c in _cnt.items() if _c>=pan_n}
    term_disc={}; term_neg={}; term_neg_soft={}; term_tier={}; term_lab={}; term_def={}; term_core={}
    term_dz={}                                    # tid -> disease_context ('normal' 기본). _is_state_term 이 KG 신뢰 판정에 씀
    for tid,t in terms.items():
        term_tier[tid]=t.get("T",""); term_lab[tid]=t.get("L",tid); dc=[]; seen=set()
        term_dz[tid]=(t.get("D") or "normal")
        for g in (t.get("R") or [])+(t.get("S") or [])+(t.get("DF") or []):
            gu=_U(g)                                   # RAW 매칭 — disc/region_share는 정규화 안 함(anchor 보존)
            if gu in PAN: continue
            j=gUp.get(gu)
            if j is not None and j not in seen: seen.add(j); dc.append(j)
        if dc: term_disc[tid]=dc
        # ★ CORE identity markers = R∪DF (defining/required), SEPARATE from supporting(S). regime-C 게이트용
        #   (_graph_evidence): KG는 HAS_DEFINING/REQUIRED(고유 정체)와 HAS_SUPPORTING(공발현·비배타)을 구분하는데
        #   disc는 둘을 합쳐 supporting도 exclusive로 카운트한다 → core가 측정됐는데(on-panel) 꺼져 있어도(df_absent)
        #   supporting 1~2개로 specific anchor가 성립하던 버그(CA2 Pfkp/Sv2b, OB-mitral Robo3, FC Notch2). RAW 매칭
        #   (disc 동일 기준, PAN 제외)이라 share/정규화 불변.
        _core=set()
        for g in (t.get("R") or [])+(t.get("DF") or []):
            gu=_U(g)
            if gu in PAN: continue
            j=gUp.get(gu)
            if j is not None: _core.add(j)
        if _core: term_core[tid]=_core
        term_neg[tid]=[j for j in (_gidx(g) for g in (t.get("N") or [])) if j is not None]   # 정규화(죽은 negative 복원)
        # ★ "_low" graded-negative = SOFT (2026-06-19): "Pcp4_low" 류는 "이 유전자가 낮아야 함"이라는 **상대적 발현
        #   힌트**(절대 부재 아님)인데 _gidx가 base(Pcp4)로 정규화 → 광범위발현 유전자가 auc≥0.85면 강한 negative로
        #   genuine identity를 hard-veto(MHb: Chrna3/Gpr151/Avil excl_on=3인데 ambient Pcp4 auc0.861이 전면거부).
        #   "_low"는 graded라 hard-veto에서 제외(neg_strong 불산입, neg_on엔 남아 excl_on≥2일 때만 1개 허용=§4.1).
        #   "_high"(Foxp2_high 등)는 "높으면 안 됨"이라 auc≥0.85 veto가 올바르므로 그대로 hard 유지.
        _soft=set(_gidx(g) for g in (t.get("N") or []) if str(g).lower().endswith("_low"))
        term_neg_soft[tid]={j for j in _soft if j is not None}
        # DEFINING markers (a single one is sufficient) — kept separately for the DF-override.
        dfi=[j for j in (_gidx(g) for g in (t.get("DF") or [])) if j is not None]             # 정규화(df_offpanel/override)
        if dfi: term_def[tid]=dfi
    lin_idx={}; lin_class={}
    for lx,gs in LIN.items():
        idxs=[j for j in (_gidx(g) for g in gs) if j is not None]                             # 정규화(lineage 마커)
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
    return {"disc":term_disc,"neg":term_neg,"neg_soft":term_neg_soft,"tier":term_tier,"lab":term_lab,"share":share,
            "dz":term_dz,
            "region_share":region_share,"defn":term_def,"core":term_core,"region_of":_rmap,
            "lin_idx":lin_idx,"lin_class":lin_class,"root":{tid:root(tid) for tid in terms},
            "genes":list(genes), "P":{tid:[p for p in (t.get("P") or []) if p in terms] for tid,t in terms.items()}}

# ★ ANATOMICAL REGION-PRESENCE GATE (F-B, 2026-06-22) — brain module force-fit 차단.
#   배경: 저감도 spatial(Stereo-seq)에서 정체성 약한 클러스터가, KG의 promiscuous region-exclusive 마커가
#   noise로 ON 되며 **절편에 존재하지도 않는 region**(소뇌/뇌교)의 rare nucleus로 force-fit됨(예: 시상 뉴런
#   749개가 cerebellar unipolar brush cell로). KG region 마커로는 부재 판정이 불가(promiscuous → cerebellar
#   GABA가 thalamic보다 높게 나옴 = 정상 차단 회귀)이므로, **curated anatomical-region 마커**(아래)로 데이터셋
#   수준 존재 여부를 1회 판정해, 부재 region의 region-group을 dominant/anchor 후보에서 제외한다.
#   anatomical(coarse) 수준이라 weak-marker 하위(SNr=midbrain GABAergic; midbrain union이 Th/Slc6a3 포함)는
#   부당 차단되지 않음. 비-brain module은 region-group 라벨이 매칭 안 돼 게이트 비활성(무영향).
#   검증(stereoseq_brain): PRESENT hippocampus11%/cortex7.9/thalamus4.3/striatum3.1/midbrain2.5/amygdala2.2
#   vs ABSENT olfactory_bulb0.026/pons-medulla0.024/cerebellum0.007 (present_thr=0.002로 명백 부재만 차단).
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
# (keyword, anat) — region-group 라벨 substring 매칭, 위→아래 first-match (hypothalam이 thalamic보다 먼저!)
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
REGION_PRESENT_THR = 0.002   # dataset 존재율(>=2 anat 마커 co-expr cell 비율) 미만이면 그 region 부재

def _absent_region_groups(X, genes, octx, gmu, gsd, thr=REGION_PRESENT_THR):
    """전체 데이터(init)에서 1회 계산. 반환: (absent_rg set[region-group term-id], present dict[anat→frac]).
    subcluster는 부분집합이라 직접 계산 불가 → init 결과를 persist해 재사용."""
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
    disc=IDM["disc"]; neg=IDM["neg"]; neg_soft=IDM.get("neg_soft",{}); defn=IDM["defn"]; core_map=IDM.get("core",{})
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
    # ★ LINEAGE adaptive floor (2026-07-02): on_present는 lineage 판정 substrate. adaptive_floor ON이면 pin_floor를
    #   여기에도 적용(region-agnostic — lineage는 region보다 먼저 결정) → rare lineage의 변별 required 마커(NK GNLY
    #   pin~0.24 < 고정 0.25인데 15/6-cell dropout)가 codom에 뜬다. SAFE: codom 정체성은 여전히 disc_on(변별 마커)을
    #   요구하므로(line ~728 repick/overlay), pin_floor를 넘은 leaky pan 마커는 false identity를 못 만들고 hijack-기각된다.
    #   pin_floor 자체가 min(fixed, gpin×F)+Nmin guard라 흔한 ambient 마커엔 floor를 안 내림(rare-specific만). None(off)=무변경.
    def on_present(g): return (pin[g]>=pin_min or (pin_floor is not None and pin[g]>=pin_floor[g])) and (mean is None or mean[g]>0)
    def gname(g): return str(genes[g]) if (genes is not None and 0<=g<len(genes)) else str(g)
    children={}
    for tid,ps in P.items():
        for p in ps: children.setdefault(p,[]).append(tid)
    def ev(tid):   # anchor evidence — uses LOOSE on_anchor (auc branch) since region-gating already excludes cross-region blips
        dc=disc.get(tid,[])
        excl=[g for g in dc if share.get(g,99)<=SPEC_REGION_MAX]  # exclusive: spans ≤N region-groups (hierarchy-aware)
        # ★ POSITIVE-anchor bar within a GATED region (2026-06-17): low-copy layer/subtype markers (Rorb/Rspo1→L4,
        #   Cux2→L2/3, Fezf2→L5) are RANK-enriched yet below both the absolute on_enr floor AND auc_hi=0.85 in
        #   sparse platforms (Stereo-seq: a real L4 leaf shows Rorb auc~0.68, Rspo1~0.72, mean~0.01). With only the
        #   strict bar NO child subtype passes ⇒ anchors empty ⇒ the region node surfaces via fallback ⇒ the whole
        #   cortex collapses to generic 'neocortical glutamatergic neuron' even though the layer signal is present
        #   (user-reported regression vs the force-assign era). FIX: when region_gated (cross-region blips already
        #   structurally excluded, container node excluded from competition), POSITIVE markers (excl/df/n_on) use a
        #   looser auc_pos bar so a quiet-but-enriched layer marker anchors its layer. NEGATIVES stay STRICT
        #   (on_anchor, 0.85) — a weak negative must never veto a term; region scoring (onm) also stays strict.
        # ★ DATA-DERIVED detection floor (opt-in, 2026-07-01) — positive-anchor rescue for sparse platforms.
        #   FIXED pin_min/frac are absolute fractions unreachable under heavy dropout (0.5µm SSAM: L4 defining
        #   Rspo1 pin~0.19 < 0.25 ⇒ dropped though lf~1.94 genuinely specific). pin_floor/fon_floor are per-gene
        #   REFERENCE-FREE floors = min(fixed, gpin[g]×F) / min(fixed, ghi[g]×F) with an Nmin cell-count guard
        #   folded in (computed in _profile_leaves). Applied ONLY here (positive anchors) + ONLY region_gated —
        #   so negatives(on_anchor)/region-scoring stay STRICT. (2026-07-02: on_present(lineage) ALSO uses pin_floor —
        #   region-agnostic — for rare-lineage recovery; safe because codom identity still requires disc_on, see on_present.)
        #   None (default/off) ⇒ predicate identical to before (zero change to existing modules). Recovers Rspo1→L4;
        #   Rorb/Etv1 stay off (low fold+high global detection). auc bars unchanged (already rank-normalized).
        def on_pos(g): return on_anchor(g) or (region_gated and auc is not None and auc[g]>=auc_pos) \
            or (region_gated and pin_floor is not None and lf[g]>=lf_on and pin[g]>=pin_floor[g]) \
            or (region_gated and fon_floor is not None and fon[g]>=fon_floor[g])
        excl_on=[g for g in excl if on_pos(g)]
        _core_set=core_map.get(tid,())                       # regime-C: ON exclusive 중 core(R∪DF)인 것 수
        excl_on_core=sum(1 for g in excl_on if g in _core_set)
        ng=[g for g in neg.get(tid,()) if on_anchor(g) and gname(g).upper() not in _cross_tx]  # negative-marker leak (cross-lineage transmitter ambient 제외)
        _soft=neg_soft.get(tid,())  # "_low" graded 힌트는 hard-veto 제외(상대발현 힌트≠절대부재)
        ng_strong=[g for g in ng if auc is not None and auc[g]>=auc_hi and g not in _soft]  # 강한 negative(rank-enriched, ambient 아님) — 진짜 반대증거
        df=[g for g in defn.get(tid,()) if on_pos(g)]
        # ★ CORE-absent (req-veto 일반화, 2026-06-27; df-guard 2026-06-28): core(R∪DF, pan제외, on-panel)가 ≥1 있는데
        #   exclusive-core도 defining(df)도 하나도 ON이 아님 → 정체 근거 전무. df_absent(DEFINING 전용)이 DF 없고
        #   required만 있는 term(immature granule: R=Dcx…, DF=[])을 못 막아 supporting(Neurod1)으로 force-fit되던 것
        #   일반화. ★len(df)==0 필수: 비-exclusive defining이 ON인 df-anchored 진짜 type(L5 ET/L6 CT/L6b/MH-LH/
        #   Sncg+)을 오차단하던 회귀(2026-06-28 회귀검증서 발견)를 방지 — defining이 ON이면 정당 anchor.
        core_absent=bool(len(_core_set)>=1 and excl_on_core==0 and len(df)==0)
        n_on=sum(1 for g in dc if on_pos(g))
        supported=bool((len(excl_on)>=1 or (len(excl)==0 and n_on>=2)) and len(ng)==0)
        # ★ DF-absent 가드 (2026-06-15): defn[tid]은 _build_ident_maps가 패널(adata.var_names)로 만들어 ON-PANEL
        #   defining 마커만 담는다(off-panel DF는 애초 제외 → CA1 Wfs1류는 면제). defining은 "하나면 충분"한 최강
        #   등급이므로, 그 역 — 측정됐는데(on-panel) 하나도 ON이 아니면(df==[]) 그 type일 가능성이 매우 낮다 →
        #   force-fit. 이 term은 supporting만으로 anchor했어도 강등(아래 sort 1차키)·verdict에서 제외. auc≥0.85
        #   분기로 quiet-but-enriched DF(cl50 Grik4 auc0.92)는 ON 처리되어 면제. DF 없는 term(대부분)은 영향 0.
        df_onpanel=defn.get(tid,())
        df_absent=bool(len(df_onpanel)>=1 and len(df)==0)
        # ★ off-panel-defining 가드 (2026-06-16): df_absent(on-panel인데 OFF, quiet 가능 → 약한 tiebreak)와
        #   범주가 다르다. term이 KG상 defining 마커를 가지는데(terms[tid].DF 비어있지 않음) 그게 *전부 off-panel*
        #   (패널에 아예 없음 → defn[tid] 비어 df_onpanel==() )이면 그 type의 고유 정체는 이 패널로 **영원히
        #   확인 불가**(quiet이 아니라 측정 자체가 없음). 이런 term은 broad/false-exclusive 보조마커(Kisspeptin의
        #   Prlr/Tac2 — Kiss1 off-panel)만으로 force-fit되기 쉽다. df_absent와 달리 cortical 층은 절대 안 걸림
        #   (층 defining Cux2/Fezf2/Foxp2/Rspo1/Car3는 모두 on-panel). 독립 exclusive 근거가 강하면(excl_on≥2)
        #   defining 없이도 식별 가능하므로 면제; excl_on<2(단일 가능-false-exclusive)면 panel-blind force-fit.
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
    NEURONAL_TX={"SLC17A7","SLC17A6","SLC17A8","GAD1","GAD2","SLC32A1","SLC6A5","TH","DBH","DDC","SLC6A3",
                 "SLC6A4","SLC6A2","TPH2","CHAT","SLC18A3","SLC5A7","HDC"}
    def _is_neur(ln): return any(gname(g).upper() in NEURONAL_TX for g in lin_idx.get(ln,[]))
    _neur_present=any(_is_neur(ln) and any(on_present(g) for g in idxs) for ln,idxs in lin_idx.items())
    # ★ 비-뉴런 오염 감지 (2026-06-16, ENRICHMENT 기반): 비-뉴런 class lineage(oligo=Plp1/Mbp/Mag, astro=Aqp4/Slc1a2,
    #   micro=C1qa/Csf1r, VLMC=Col1a1/Dcn …)가 ≥2개 마커 ENRICHED(on_anchor=vs-rest auc≥0.85)이고 **뉴런 transmitter는
    #   하나도 enriched 아니면** → 진짜 비-뉴런 fragment. presence(ambient neuropil)가 아니라 ENRICHMENT로 판정하므로,
    #   뉴런 클러스터에 박힌 fiber-tract/혈관/미세아교 오염 조각이 ambient Snap25/Gad를 가져도 glia로 풀린다(parent-
    #   constraint 해제 + 뉴런우선 해제 → ~19k cell glia가 striatal/midbrain GABA로 오라벨되던 버그). ★진짜 뉴런은
    #   glial이 ambient(non-enriched, auc<0.85)라 _nn_enr<2 → 안 풀림(cl9 L6b: up=Cplx3/Nxph3 enriched, Plp1/Mbp는
    #   ambient라 enrich 안 됨 → 보호). transmitter가 enriched면(_neur_enr>0) 진짜 뉴런이므로 절대 해제 안 함.
    #   (보수적: 뉴런 transmitter가 하나도 present 안 할 때만 — ambient neuropil이 검출되는 fiber-tract/혈관 오염은
    #   여기서 못 풀고, 아래 _validate GLIAL_MISLABEL 백스톱이 up-기반으로 잡아 agent가 glia로 재라벨한다.)
    # ★ 뉴런우선 예외 = ENRICHMENT 대칭 (2026-07-08): 비-뉴런 lineage 가 **≥2 마커 ENRICHED**(on_anchor)이고
    #   뉴런 transmitter 는 **enriched 가 아님**(ambient 로 present 만)이면 진짜 비-뉴런. 이전엔 `not _neur_present`
    #   (뉴런 transmitter 가 하나도 검출 안 될 때만)이라 neuropil 에 박힌 microglia(ambient 뉴런신호 항상 present)를
    #   못 구제 → GABA/뉴런으로 오배정(cl24). present→enriched 로 바꿔, 진짜 뉴런(transmitter enriched)은 그대로 보호하고
    #   ≥2 특이마커 enriched 인 microglia(뉴런신호는 ambient=non-enriched)만 빠져나오게 한다. glia≥2 는 단일마커 노이즈 차단.
    _neur_enriched=any(_is_neur(ln) and any(on_anchor(g) for g in idxs) for ln,idxs in lin_idx.items())
    _nn_on=any((not _is_neur(ln)) and ln in lin_class and sum(1 for g in idxs if on_anchor(g))>=2 for ln,idxs in lin_idx.items())
    # ★ COMPARATIVE non-neuronal dominance (2026-07-15): on dense/KDE spatial data (Stereo-seq) glial identity
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
        # ★ NON-IDENTITY LINEAGE OVERLAY (2026-06-23, 일반화). CASE: 한 클러스터의 변별 정체성은 lineage B인데
        #   (B의 변별 required 마커 ON), 다른 lineage A의 프로그램도 함께 present하다. 단 A의 존재가 *오직 비특이
        #   (pan_disc=R+S를 ≥12 term이 공유) required 마커로만* 뒷받침되고 A의 변별(∉pan_disc) required 마커가 0개면
        #   → A는 경쟁 정체성이 아니라 B 위에 얹힌 프로그램 overlay(활성화/전이 상태, 예: EndMT/EMT) 또는 ambient.
        #   lineage strength는 "단일 required 마커 max"라 A의 비특이 마커 하나가 broad-shared로 높게 뜨면 lineage를
        #   가로챌(hijack) 수 있다. 계열·유전자명 비종속: "변별 ON-required=0" 구조로만 정의 — KG marker-sharing 통계
        #   (octx.pan_disc, _build 동일 기준)만 사용. evidence-not-verdict: overlay_lineages/lineage_repick으로 surface.
        _PAN=octx.get("pan_disc") or set()
        def _disc_on(ln): return [g for g in lin_idx.get(ln,[]) if on_present(g) and gname(g).upper() not in _PAN]
        # (1) REPICK action: overlay lineage(disc_on=∅)가 raw max strength를 차지 → 변별 ON-required≥2인 codom
        #   경쟁자로 best 재지정(≥2는 단일 ambient bleed 마커로 인한 오재지정 차단). 자기보호: 진짜 SMC는 MYH11/CNN1
        #   (∉pan_disc) ON → disc_on≥1 → 절대 강등 안 됨; ¬lineage_ambiguous면 발동 0.
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
        # (2) OVERLAY surface: codominant present lineage 중 정체성(best)이 아니고 변별 ON-required=0인 것 = 프로그램/
        #   ambient(정체성 아님). agent가 고발현 program 보고 정체성을 그쪽으로 갈아타는 것을 차단(cl8 EC→myofibroblast).
        overlay_lineages=[{"lineage":lab.get(lin_class.get(ln),ln),"present":[gname(g) for g in lin_idx.get(ln,[]) if on_present(g)]}
                          for ln in codom if ln!=best[0] and not _disc_on(ln)]
        # (3) INDISCRIMINABLE LINEAGES → 공통조상/Unknown (2026-06-23, cl12 type). CASE: best lineage의 on_present
        #   required 마커가 *전부 다른 codom lineage와 공유*돼(PTPRC ∈ myeloid.R & lymphocyte.R, 둘의 top 신호)
        #   best 고유(codom-private) 변별 마커가 0개 = codom을 구분할 증거가 없음 → 어느 한쪽을 confident하게 고를 수
        #   없다. RULE: 최심 공통조상 class로 라벨(없으면 Unknown). ★pan_disc 아님 — codom-private로만 판정하므로 뇌
        #   transmitter(Slc17a7 ∈ glut만, GABA엔 없음=private)는 안 걸려 진짜 glut/GABA doublet은 보호(split 경로
        #   유지). repick이 이미 정체성(B)을 찾았으면(lineage_repick≠None) skip — cl8은 여기 안 옴.
        if lineage_ambiguous and lineage_repick is None:
            _oth_best=set().union(*[{gname(g).upper() for g in lin_idx.get(o,[])} for o in codom if o!=best[0]])
            _priv_best=[g for g in lin_idx.get(best[0],[]) if on_present(g) and gname(g).upper() not in _oth_best]
            if not _priv_best:                                          # best 고유 변별 0 → 구분 불가
                lineage_unresolved=True
                def _anc(tid):
                    ch=[tid]; seen={tid}
                    while True:
                        nx=next((x for x in P.get(ch[-1],[]) if x not in seen),None)
                        if nx is None: break
                        ch.append(nx); seen.add(nx)
                    return ch
                _common=set.intersection(*[set(_anc(lin_class.get(ln,ln))) for ln in codom])
                for x in _anc(lin_class.get(best[0],best[0])):           # best 경로에서 가장 깊은 공통조상
                    if x in _common: unresolved_to=lab.get(x,x); break
        # ★ dual-transmitter cross-root 도달(2026-06-15): MHb(cholinergic-glutamatergic)는 root=glut인데 cl28처럼
        #   glut transmitter OFF·cholinergic ON이면 cholinergic lineage로 검출돼 도달 불가했음. 검출 lineage의 전달물질
        #   심볼을 모아, 그걸 자기 R에 가진 cross-root term(MHb=Chat/Slc18a3 보유)을 아래 anchor 루프가 region-gate
        #   우회로 후보에 포함시킨다(공동방출 뉴런 복원).
        lin_tx_sym={gname(g).upper() for g in lin_idx.get(best[0],[])}
    # ★ cross-lineage transmitter negative 무시 (2026-06-16): lineage gate가 이미 glut/GABA/등을 확정했으므로,
    #   leaf lineage가 아닌 다른 lineage의 transmitter(GABAergic leaf의 pan-glut Slc17a7/6/8, glut leaf의 Gad1/2)를
    #   negative로 카운트하면 8µm ambient bleed에 항상 leak해 진짜 type을 부당 veto한다 — D1/D2-MSN이 pan-glut
    #   negative 때문에 모든 선조체 leaf에서 탈락하고 eSPN(가짜 negative)만 생존하던 직접 원인. leaf 자신의 lineage
    #   transmitter는 그대로 유지(진짜 음성 근거). ev()의 ng 계산에서 _cross_tx를 제외한다.
    _leaf_lins=[ln for ln,tid in lin_class.items() if IDM["root"].get(tid,tid)==cls_root]
    _own_tx={gname(g).upper() for ln in _leaf_lins for g in lin_idx.get(ln,[]) if gname(g).upper() in NEURONAL_TX}
    _cross_tx={x for x in NEURONAL_TX if x not in _own_tx}
    # ★ REGION-COHERENT DESCENT WITHIN the lineage: among the lineage class's region-groups (depth-1 children —
    #   neocortical / thalamic / hippocampal pyramidal …), pick the DOMINANT by expression-weighted (relative,
    #   not an absolute floor) sum over its subtree's ENRICHED EXCLUSIVE markers; ≥2 concordant markers required
    #   so a lone broad gene (Stmn1) can't raise a region. ONLY that region's subtypes are anchor candidates ⇒
    #   cross-region blips (CA2 in a thalamic glut cluster) are structurally excluded WITHOUT a magnitude floor,
    #   so low-abundance layer markers (Fezf2→L5) survive. Gene-based, spatial-free.
    # region scoring을 lineage의 **모든** region에 대해 먼저(allow와 무관) — fragment-escape 판정에 필요.
    rg_terms={}
    for t in terms:
        if IDM["root"].get(t)==cls_root: rg_terms.setdefault(region_of.get(t,t),[]).append(t)
    _absent=absent_rg or set()   # F-B: 데이터셋에 부재한 region-group은 후보 제외(dominant/floor/anchor)
    region_groups_all=[rg for rg in children.get(cls,[]) if tier.get(rg)!="class" and rg not in _absent]
    rg_score={}; rg_hasdf={}
    for rg in region_groups_all:
        gg=set(disc.get(rg,())); rdf=set(defn.get(rg,()))
        for t in rg_terms.get(rg,()): gg|=set(disc.get(t,())); rdf|=set(defn.get(t,()))
        gg=[g for g in gg if share.get(g,99)<=SPEC_REGION_MAX]
        # RC3 (2026-06-14): region-exclusive markers count via on_anchor (auc≥0.85 branch), so a quiet-but-enriched
        #   region marker (Fezf2→neocortical, auc 0.90 yet detected<25%) resolves the region in sparse 2µm data —
        #   else a deep-cortical (L5) leaf loses neocortical to hippocampal (shared Slc17a7/Pcp4) and is mislabeled.
        #   gg is already region-exclusive (share≤SPEC_REGION_MAX) and ≥2 concordant required ⇒ no lone-blip region.
        onm=[g for g in gg if on_anchor(g)]
        if len(onm)>=2:
            rg_score[rg]=sum(lvl(g) for g in onm)
            rg_hasdf[rg]=any(on_anchor(g) for g in rdf if share.get(g,99)<=SPEC_REGION_MAX)  # defining 마커 ON?
    # ★ COMPARATIVE region establishment (2026-07-15): the strict pass above needs ≥2 region-exclusive markers
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
    # ★ FRAGMENT-ESCAPE (2026-06-18): subcluster가 부모 region(restrict_root→allow)에 제약됐는데, 제약 **밖** region이
    #   더 강하게(score 우위) + 그 region에 **defining 마커가 ON**으로 뜨면 = 부모에 다수로 섞였던 소수가 다른-region
    #   정체로 고립된 것(stomics cl6: 피질 다수 + BLA 19% 소수 → 분할로 BLA blob 고립). 부모 제약을 해제해 자식이 자기
    #   정체로 resolve하게 한다. defining 게이트 + score우위로 leaky cross-region force-fit(옛 cortex→pontine 버그)은 차단
    #   (단일 leaky 마커는 ≥2 on_anchor도 defining도 못 만족). 부모가 region 못 정한 경우(allow=lineage class만)도 동일.
    if allow is not None and rg_score:
        _ina=lambda rg: rg in allow
        _best_in=max((s for rg,s in rg_score.items() if _ina(rg)), default=0.0)
        _esc=[rg for rg,s in rg_score.items() if (not _ina(rg)) and rg_hasdf.get(rg) and s>_best_in]
        if _esc: allow=None        # 제약 해제 → 자식이 다른 region(BLA)로 탈출 가능
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
        # dual-transmitter cross-root term(MHb): root이 달라도 검출 lineage의 전달물질을 R에 가지면 후보 + region-gate 우회
        cross_dual=bool(lin_tx_sym and IDM["root"].get(tid)!=cls_root and tier.get(tid)!="class"
                        and ({str(g).upper() for g in (terms.get(tid,{}).get("R") or [])} & lin_tx_sym))
        if IDM["root"].get(tid)!=cls_root and not cross_dual: continue
        if tier.get(tid)=="class": continue
        if region_of.get(tid,tid) in _absent and not cross_dual: continue   # F-B: 부재 region leaf 제외(non-region-gated 경로 포함)
        if allow is not None and tid not in allow: continue        # parent-constrained: only descendants of restrict_root
        if region_gated and not cross_dual and region_of.get(tid,tid) not in dominant: continue
        # ★ region-group 컨테이너 노드(neocortical/thalamic/… = dominant의 원소 자신)는 main 경쟁에서 제외 —
        #   자기 region-exclusive 마커(Slc30a3/Mef2c)가 적어 ratio=1.0로 자식 층(L4 excl_n8 ratio0.25)을 부당하게
        #   이겨 모든 피질 층을 generic으로 붕괴시킴(2026-06-15). 의도대로 자식 leaf가 하나도 anchor 안 될 때만
        #   아래 fallback으로 surface한다(주석: 'If none anchors, surface the dominant region-group node').
        if region_gated and not cross_dual and tid in dominant: continue
        e=ev(tid)
        # off-panel-defining 가드: KG defining이 전부 off-panel(df_offpanel)이고 독립 exclusive 근거가 약하면
        #   (excl_on<2) panel-blind force-fit → anchor에서 제외 → 아래 region-node fallback으로 부모 region/class
        #   surface(Kisspeptin: Kiss1 off-panel + Prlr 단일 excl → hypothalamic GABAergic으로 하강). excl_on≥2면 통과.
        if e["df_offpanel"] and e["excl_on"]<2: continue
        # ★ REGIME-C 게이트 (df_absent + supporting-only, 2026-06-26): core(R∪DF) defining 마커가 ON-PANEL인데
        #   하나도 ON이 아니고(df_absent) ON된 exclusive가 전부 SUPPORTING(excl_on_core==0)이면 → confident
        #   specific anchor 불가 → region/class floor로 하강. KG가 구분한 defining/required(고유 정체) vs
        #   supporting(공발현)을 엔진이 합쳐 supporting만으로 measured-absent term을 mint하던 버그 차단
        #   (CA2 Pfkp/Sv2b→oligo/cortex 조각, OB-mitral Robo3→피질, FC Notch2/Zfp536→CA1).
        #   ★ Regime-A 보존: defining이 전부 OFF-PANEL이면 df_absent=False(df_onpanel 비어 위 가드 미발동) → supporting
        #     으로 정상 anchor(사용자 지침). ★ Regime-B 보존: 진짜 layer의 quiet defining은 auc_pos(0.65) rescue로
        #     on_pos→df 비어있지 않음→df_absent=False → 영향 0(2026-06-15 reverted hard-demote 회귀 재발 없음).
        #   df_absent인데 core-specific(R∪DF) 1개라도 ON(excl_on_core≥1)이면 통과(정체 근거 존재).
        #   ★ 2026-06-27: df_absent(DF전용)→core_absent(R∪DF)로 일반화. DF가 비고 REQUIRED만 있는 term이
        #   required measured-but-off인데 supporting만으로 통과하던 것(immature granule Prox1/Dcx OFF→Stmn1로
        #   force-fit)을 차단. Regime-A(required 전부 off-panel→_core_set 비어 core_absent=False) 보존,
        #   Regime-B(quiet required/df가 auc_pos 0.65 rescue→excl_on_core≥1→core_absent=False) 보존.
        if e.get("core_absent",False): continue
        # ★ EMPTY-REGION 비대칭 anchor 게이트 (2026-06-18): region_gated면 후보가 이미 dominant_region 안으로
        #   제한돼 cross-region blip이 구조적으로 배제되므로 exclusive 1개로 충분(기존 동작 그대로). 그러나
        #   dominant_region이 비면(region_gated=False) 후보가 계통 전체로 열려 먼 region subtype이 단일 leaky
        #   exclusive 마커 하나(thalamic=Cbln4, OB mitral=Robo3)로 force-fit된다(xenium cl11 피질→thalamic, cl8).
        #   region 미확정 시엔 specific anchor에 exclusive ≥2 OR defining ON 요구 — 단일 비-defining 마커는
        #   cross-region 호출을 못 떠받친다. 진짜 피질 층은 df(Rspo1→L4, Ldb2→L5 ET, Cux2→L2/3)가 ON이라 df 분기로
        #   살아남고(cl11 → L4 IT), defining 없는 thalamic 단일-Cbln4는 차단 → 피질로 정정. region_gated 경로는 불변.
        # ★ df_absent 단일-마커 가드 (2026-07-02): region_gated여도 anchor가 df_absent(on-panel defining이 있으나
        #   전부 OFF)이면서 exclusive가 단 1개(비-defining)면 채택하지 않는다 — region-scoring이 엉뚱한 subcluster를
        #   한 region으로 잘못 잡았을 때(subcortical Slc17a6+/Gad1+ 조각이 Ndst4 하나로 allocortical dominant) 그
        #   region의 자식이 df_absent+단일 exclusive로 clear_top anchor가 되어 region-floor 오라벨(allocortical)을
        #   낳던 문제. 세 허용경로: (a) df_on≥1(quiet defining ON=진짜 층, L4 Rspo1 auc_pos), (b) excl_on≥2(다중
        #   exclusive, L4 Rorb+Whrn), (c) excl_on≥1 & not df_absent(on-panel defining이 아예 없는 off-panel-defining
        #   type=panel-blind 보존). df_absent & df_on==0 & excl_on==1만 차단 → anchors 비면 class_or_unknown로 정직 하강.
        _anchor_ok = ((e["df_on"]>=1 or e["excl_on"]>=2 or (e["excl_on"]>=1 and not e.get("df_absent",False)))
                      if region_gated else (e["excl_on"]>=2 or e["df_on"]>=1))
        # ★ negative veto 완화(2026-06-19, 블로커B): 강한 negative(auc≥0.85=rank-enriched 진짜 반대증거)는 항상 veto.
        #   약한 negative(on_enr-only=ambient/오염 수준)는 positive 증거가 강할 때(excl_on≥2)만 1개 허용 — 공간데이터의
        #   KDE/이웃 bleed로 들어온 단일 약-negative가 강한 anchor를 전면 거부하던 문제(MHb Th 32% → excl_on=3 거부). §4.1 일관.
        _neg_ok = e.get("neg_strong",e["neg_on"])==0 and (e["neg_on"]==0 or e["excl_on"]>=2)
        if _anchor_ok and _neg_ok:
            e["path"]=path_to_root(tid); anchors.append(e)
    if not anchors and region_gated:
        for rg in sorted(dominant, key=lambda r:-rg_score[r]):
            e=ev(rg)
            if e["neg_on"]>0: continue          # region node도 negative leak이면 surface 안 함(Slc17a6+ 클러스터에 DG region 등)
            # ★ Fix A (2026-06-23): region 노드를 floor로 surface하려면 그 region의 OWN exclusive/defining 마커가 ≥1 ON이어야 함.
            #   pan 마커(Slc17a7)만 ON인 generic 클러스터가 region-floor(DG 등)로 force-fit되던 것 차단 — 피질의 가짜
            #   'dentate gyrus glutamatergic neuron'(Prox1/C1ql2=0)·markerless region-floor 제거. region-특이 마커
            #   (DG=Prox1/C1ql2, neocort=Satb2/Slc30a3, thalamic=Tcf7l2)가 하나도 ON 아니면 → surface 안 함 →
            #   anchors 비어 class_or_unknown → lineage class('glutamatergic neuron')로 정직하게 하강.
            #   (leaf anchor 경로·region_gated auc_pos 층마커 보존과 무관: 여기는 'anchor 전무 시 region 노드 fallback'만.)
            if e["excl_on"]<1 and e["df_on"]<1: continue
            e["path"]=path_to_root(rg); anchors.append(e)
    # ★ EMBEDDED-TYPE RESCUE (2026-06-28): 혈관/면역(SMC/endo/pericyte 등) 세포는 조직에 박혀 KDE가 주변
    #   glia/neuron RNA를 voxel에 번뜨림 → 클러스터 절대 프로파일이 주변세포 지배 + off-lineage라 anchor 후보에서
    #   탈락(cl69: Acta2 34× enriched인데 Mbp 절대지배 → oligo, cl103: Cldn5 18× → astro). 고특이 core 마커
    #   (R∪DF, share≤SPEC_REGION_MAX)가 rank(auc≥0.90) + fold(lf≥log2 10 = 10배) 둘 다 강하면 그 term을
    #   anchor로 surface(lineage/neg/absolute 우세 무시). 3중 게이트라 broad/ambient는 미발동 — microglia 3~5×는
    #   fold<10 → floor 유지(정확). agent는 rescued 플래그+rescue_markers 보고 절대-우세 glia 대신 vascular 판정(SKILL §4).
    if auc is not None and lf is not None:
        _RESC_AUC=0.90; _RESC_LF=float(np.log2(5.0)); _anch={a["term_id"] for a in anchors}
        # ★ 2026-06-28 튜닝: fold 임계 10→5 (microglia cl75 특이마커 Tmem119 6×=진짜인데 10× 미달로 놓침;
        #   auc≥0.90 + non-parenchymal 한정이 ambient 배제하므로 5×로 낮춤). + DEFINING 마커는 share 필터 우회
        #   (ependymal Foxj1=DF인데 share=4로 share≤2에 걸려 46×인데도 rescue 안 되던 버그 — DF는 큐레이션-특이).
        # ★ NON-PARENCHYMAL 한정: 조직에 박혀 KDE-smear로 절대-우세를 빼앗기는 건 혈관/면역/수막/맥락총/상의 계열뿐.
        #   neuron(glut/GABA/transmitter)·macroglia(astro/oligo/OPC)는 parenchymal이라 rescue 대상 아님(그쪽 오라벨은
        #   sibling/region 문제로 별도 처리). 이 게이트 없으면 모든 type의 defining 마커가 fold≥10이라 flood됨(검증서 55→).
        _NONPAR=('endothel','pericyte','smooth muscle','leptomening','mural','perivascular','fibroblast',
                 'macrophage','microg','choroid','ependym','arachnoid','vascular','erythro','neutrophil',
                 'lymphocyte','monocyte','mast','tanycyte')
        def _nonpar(_t):
            _rl=lab.get(IDM["root"].get(_t,_t),"").lower(); _tl=lab.get(_t,"").lower()
            return any(k in _rl or k in _tl for k in _NONPAR)
        for tid in terms:
            if tid in _anch or not _nonpar(tid): continue
            _dfset=set(defn.get(tid,()))   # DEFINING은 share 필터 우회(큐레이션-특이; ependymal Foxj1 share=4)
            strong=[g for g in core_map.get(tid,())
                    if (g in _dfset or share.get(g,99)<=SPEC_REGION_MAX) and auc[g]>=_RESC_AUC and lf[g]>=_RESC_LF]
            if strong:
                e=ev(tid); e["rescued"]=True; e["rescue_markers"]=[gname(g) for g in strong][:6]
                e["path"]=path_to_root(tid); anchors.append(e)
    # df_absent(defining 측정+OFF)는 **약한 tiebreak일 뿐** — excl_on/ratio/n_on이 우선. 이렇게 안 하면 DF 없는
    #   generic region node(절대 df_absent 아님)가 정의마커만 quiet한 진짜 층(L4=Rorb/Whrn excl_on2, Rspo1 quiet)을
    #   항상 이겨 피질 층이 전부 generic으로 붕괴함(2026-06-15 회귀). force-fit(mitral)은 ratio로 이미 밀리므로 정렬
    #   강등 불필요. df_absent는 flag(목록·_validate)로만 agent에 전달, 정렬은 동률일 때만 마지막으로 가른다.
    # rescued(embedded vascular/immune)는 절대-우세에 묻히지 않게 상위로 — agent가 보게(최종 판정은 agent, SKILL §4).
    anchors.sort(key=lambda x:(x.get("rescued",False), x["excl_on"], round(x["excl_on"]/max(x["excl_n"],1),2), x["n_on"], not x.get("df_absent",False)), reverse=True)
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
    a2anchor={a["label"]:a for a in anchors}
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
                    # ★ resolve 가드(2026-06-19, 이슈#3·14 FC): resolve 대상의 정체 확인용 **defining이 off-panel
                    #   이거나(df_offpanel) on-panel인데 전부 OFF(df_absent)** 이고, 그 type의 distinct ON이 **1개뿐**이면
                    #   = 약한 단일 비-defining 마커로 단정하는 force-fit → resolve 막고 아래 공통조상 class floor로.
                    #   (FC defining Necab2 off-panel인데 Zfp536/Igfbp4 1개로 FC 단정하던 문제. 진짜 FC는 distinct ≥2라 통과.)
                    def _df_unconf(lbl):
                        an=a2anchor.get(lbl) or {}; return bool(an.get("df_offpanel") or an.get("df_absent"))
                    if td and pd:      action,resolves="split",None
                    elif td and not (_df_unconf(top_label) and len(td)<2):  action,resolves="resolve",top_label
                    elif pd and not (_df_unconf(par) and len(pd)<2):        action,resolves="resolve",par
                    else:
                        # ★ 공유마커만 ON, 양쪽 고유마커 OFF(또는 off-panel) → 둘 중 하나로 단정 불가.
                        #   Unknown이 아니라 두 멤버의 **deepest 공통 조상 CLASS**로 정직하게 귀착(2026-06-18, 사용자 요청).
                        #   confusion-pair = "이 type들이 나올 것으로 예상하고 split한" 맥락이므로(=기대한 celltype),
                        #   그 안에서 공유마커만이면 공통 region/class가 정답(FC vs CA2 → hippocampal pyramidal neuron).
                        #   FC처럼 변별 defining이 off-panel이면 여기로 와서 CA2 단정 대신 상위 class가 됨.
                        _pa=path_to_root(tt) if tt else []; _pb=path_to_root(pt) if pt else []
                        _common=None
                        for _x,_y in zip(_pa,_pb):
                            if _x==_y: _common=_x
                            else: break
                        action,resolves="unknown",_common      # resolves_to=공통 조상 class(없으면 None→Unknown 폴백)
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
    cand=[]   # (tid, region, wk[gene idx]) — mixture 후보
    for tid in terms:
        if IDM["root"].get(tid)!=cls_root or tier.get(tid)=="class": continue
        if region_of.get(tid,tid) in _absent: continue   # F-B: 부재 region은 mixture 후보에서도 제외
        if allow is not None and tid not in allow: continue
        if region_gated and region_of.get(tid,tid) not in dominant: continue
        exc=[g for g in disc.get(tid,()) if share.get(g,99)<=SPEC_REGION_MAX]
        if not exc: continue
        wk=[g for g in exc if on_anchor(g) or (auc is not None and auc[g]>=WEAK_AUC)]
        if not wk or any(on_anchor(g) for g in neg.get(tid,())): continue
        cand.append((tid, region_of.get(tid,tid), wk))
    # ★ EMPTY-REGION mixture 정리 — KG-region grouping (발현 기반, 공간 無; 2026-06-18): ⚠️ 여기서 'region'은 **KG 계층의
    #   region-group**(neocortical/thalamic = region_of, 그래프 구조)이고 판정은 **마커 발현**(on_anchor/auc)만 쓴다 — 세포
    #   xy좌표를 안 쓰므로 scRNA-seq에서도 동일 작동(dominant_region·graph-descent와 같은 원리). 'spatial coherence'(15-NN
    #   공간 이웃 비율)와 무관. region_gated면 위 region-gate가 이미 cross-region을 제거하고 diluted 층을 느슨히(1-마커)
    #   잡는다(2µm 의도 보존). 그러나 region 미확정(region_gated=False)이면 mixture가 계통 전체로 열려 단일 leaky 마커의
    #   cross-region 잡음(thalamic 2개가 Cbln4 하나 공유, OB=Robo3, 중뇌=Ntrk3)이 섞인다. 판별: 진짜 under-clustered
    #   mixture는 **같은 KG-region-group에 distinct exclusive 마커 ≥2개**가 발현 ON(피질=Rspo1+Ldb2…); 잡음은 *서로 다른
    #   region에 흩어진 단일 마커* 또는 *형제끼리 1개 마커 공유*(Cbln4)다. region별 distinct ON exclusive 마커 union을
    #   세어 ≥2인 region의 subtype만 남긴다 — 희석된 진짜 층(cl12 L5 IT/ET/L5b, 마커 distinct)은 살고, Cbln4 공유
    #   thalamic·고립 단일마커는 제거. (마커 개수 OR df 기준은 희석 층을 죽여 cl12 회귀 → 폐기.)
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
    """★ BIMODAL-SPLIT backstop (entry-trigger gap, 2026-06-23). The cluster-level mixture
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

# ★ 2026-07-19: state 판정은 **KG의 tier 가 유일한 정본**이다.
#   예전에는 라벨 키워드(_DEV_STATE_KW: developmental/neuroblast/radial glia/immature/reactive…)로도
#   state 를 추정했다. 그 결과 KG 가 tier='class'/'subclass' 로 명시한 **정상 세포형 38개**가 이름
#   때문에 top 후보에서 빠졌다(Radial glia, Neuroblast, Immature B cell / neutrophil / platelet /
#   cardiomyocyte, ISP thymocyte, immature enterocyte, immature Sertoli cell …). 성숙 단계는 골수·흉선
#   같은 조직에서 다수를 차지하는 실재 세포라 이 억제는 체계적 미검출이었다. 예외 목록이
#   ("immature granule",) 하나뿐이었던 것도 같은 오탐을 개별 땜질하던 흔적.
#
#   이 단순화의 전제는 KG 쪽 정리가 선행됐다는 것이다(2026-07-19, 순서대로 수행):
#     1) 세포인데 disease_context 로 막혀 있던 2건 → 질병을 OBSERVED_IN_DISEASE 엣지로 옮기고
#        프로퍼티 제거 (reapply_disease_context_cleanup.py)
#     2) 발생 시기 200건 → disease_context 에서 `developmental_stage` 로 분리, slim-index 는 `DS`
#        로 노출 (reapply_developmental_stage_axis.py). 발생은 질병이 아니며, 태아 데이터에서는
#        radial glia 가 정답이어야 한다. 성체/태아 판단은 DS 를 근거로 **expert agent** 가 한다.
#     3) tier='state' 오분류 158건 감별 (reapply_state_vs_celltype.py)
#   그 결과 "라벨은 state 같은데 KG 는 정체 + 비정상 맥락" 인 위험군이 0 이 되어 키워드가 불필요해졌다.
#   질병은 여기서 보지 않는다 — disease 는 필터가 아니라 힌트이고(§7), 질병 샘플의 정상 세포
#   (cDC1/Treg/naive CD4 T cell in breast cancer 등 233건)는 정상 라벨로 나와야 한다.
def _is_state_term(label, tier):
    """STATE 판정 = KG 의 tier 가 'state' 인가. 라벨·질병 맥락은 보지 않는다.
    state 는 base-agnostic 조건(proliferating/IFN-stimulated/hypoxic)이며 라벨 뒤에 ', <state>' 로
    이어붙는다(STATE-AXIS). 세포 이름을 가진 것은 state 가 아니라 Term 이다."""
    return tier == "state"
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
                      "R":list(t.get("R") or []),"S":list(t.get("S") or []),"DF":list(t.get("DF") or []),
                      "A":list(t.get("A") or []),"MP":list(t.get("MP") or [])}
    return out
_MIM_SPEC_SHARE_MAX=3   # 흉내 프로그램으로 인정할 마커의 최대 gene-degree(share). pan 마커(CD3E/PTPRC…) 배제.
def _mimicry_terms(idx):
    """lineage-mimicry/전이 term = MIMICS_PHENOTYPE(MP)와 identity-anchor(A)를 둘 다 가진 term.
    tier와 무관(mimetic TEC는 tier=class, EMT는 tier=state) — 정의 속성은 MP+A 보유다.
    이 term들로 _profile_leaves가 leaf별 transition_candidates(anchor 잔존 + 흉내 프로그램 present)를 만들어,
    '흉내 대상 lineage로 force-fit하지 말고 origin+state로 판정'을 evidence-strong하게 surface한다
    (STATE_TRANSITION_DESIGN §엔진규칙, anchor-tiebreaker).
    ★ Qspec = 흉내 프로그램의 SPECIFIC 마커만(share≤MAX) — pan-lineage 마커(CD3D/CD52/IRF8…)가
      모든 그 lineage 클러스터에서 전이를 거짓발동시키는 것을 막는다(VALID-EVIDENCE 원칙과 동일).
    반환 {tid:{L,D,A,MP,R,S,DF,P,Qspec}}."""
    terms=((idx or {}).get("terms",{}) or {})
    def _gm(t): return [str(x) for x in (list(t.get("R") or [])+list(t.get("S") or [])+list(t.get("DF") or []))]
    share={}                                            # gene -> #terms listing it (R+S+DF)
    for t in terms.values():
        for g in set(x.upper() for x in _gm(t)): share[g]=share.get(g,0)+1
    def _anc_markers(tid, seen=None):                   # origin(조상) 마커 — 흉내 프로그램에서 제외할 origin lineage 마커
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
        # 흉내 프로그램 = 자기 마커 중 (anchor 아님) AND (low-share 특이) AND (origin 조상 마커 아님), 중복제거
        Qspec=_dedup([g for g in _gm(t) if g.upper() not in _A and share.get(g.upper(),99)<=_MIM_SPEC_SHARE_MAX and g.upper() not in _anc])
        out[tid]={"L":t.get("L",tid),"D":(t.get("D") or "normal"),"A":A,"MP":MP,
                  "R":list(t.get("R") or []),"S":list(t.get("S") or []),"DF":list(t.get("DF") or []),
                  "P":list(t.get("P") or []),"Qspec":Qspec}
    # ★ StateAxis 기반 transition (2026-07-26): mimics+anchor_markers 를 가진 StateAxis(sa)도 transition
    #   후보로 흡수한다. StateAxis 재구조화로 EMT/EndMT/dedifferentiation 등이 Term-level A/MP 에서 sa 로
    #   옮겨졌으므로, 여기서 통합해 leaf 판정 시 false-split 가드를 유지한다(상피 base + mesenchymal program
    #   = EMT state 이지 상피/mesenchymal 혼합 아님 → split 금지). StateAxis.markers 는 이미 큐레이션된
    #   program signature 이므로 share 필터 없이 (anchor 제외) 그대로 Qspec 으로 쓴다.
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
        Qspec=_dd([g for g in prog if g.upper() not in _A])          # 흉내 program = anchor 제외 program 마커
        if not Qspec: continue
        out["SA:"+str(name)]={"L":str(name),"D":"normal","A":_dd(A),
                  "MP":[{"lineage":mim,"program":str(name)}],
                  "R":[],"S":[],"DF":[],"P":[],"Qspec":Qspec,
                  "of":[str(x) for x in (st.get("of") or [])]}       # origin lineages (STATE_OF 대상)
    return out
def _state_terms(idx):
    """Module의 모든 state term(tier=state 또는 dev/reactive 키워드)을 추출.
    ★ disease는 필터가 아니라 힌트(2026-06-22): --disease 미지정이어도 이 term들을 leaf별 disease_candidates로
    항상 surface해, 마커가 우세하면 agent가 그 state로 판정할 수 있게 한다(SKILL §7). disease 지정 시 _disease_terms와
    교차해 hint=True 표시. 반환 {tid:{L,D,R,S,DF}}."""
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
    dzm=M.get("dz",{})                                   # tid -> disease_context (state 판정이 KG를 신뢰하도록)
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
    # ★ DEFINING-MARKER CANDIDACY (2026-06-23). CASE = a type whose discriminative identity rests on a
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
    # ★ DOMINANT broad-DEFINING marker (2026-06-23). A region-BROAD defining marker (share>SPEC_SHARE_MAX —
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
        # share≤SPEC_SHARE_MAX 변별마커 + 이 term의 DOMINANT broad-DEFINING 마커(_df_dom; region_share 무관).
        #   broad-DF로 정의되는 타입(Sst+ interneuron)이 spec_pass/spec_ok 통과하도록. low-share DF는 이미
        #   disc에 포함됨. leaky/비dominant broad-DF는 auc<auc_hi라 미포함 → spec_cov 부풀리기로 오발 차단.
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
    #   rescue it if `hp`/supporting clearly show the type IS present.
    if top is not None and top.get("tier")!="class" and not top["spec_pass"]:
        top=None
    # ranked: valid-evidence terms (spec_pass) surface first, then dominant-lineage, then decisiveness
    ranked=(sorted(coh, key=lambda c:(c["spec_pass"], c["_root"]==dom_root, c["dec"], c["wspec"]), reverse=True)[:5]
            or sorted(cand, key=lambda c:(c["spec_pass"], c["dec"], c["wspec"], c["n_on"]), reverse=True)[:5])
    if top is not None and top not in ranked: ranked=[top]+ranked[:4]
    # ★ STATE VISIBILITY (2026-06-22, disease=hint-not-filter): state/disease terms are excluded from the
    #   chimera/`top` core (above) to keep their broad markers from force-fitting normal cells. But if a state
    #   term has its OWN valid evidence (n_on>=2, no specific-neg leak, spec_pass), it must be VISIBLE so the
    #   agent can call it when its markers dominate — even with --disease omitted. Surface strong state terms in
    #   `ranked` (flagged is_state) WITHOUT making them auto-`top` (decision stays with the agent, §7).
    for c in cand: c["is_state"]=_is_state_term(c["label"], c["tier"])
    # ★ SPECIFICITY GATE (2026-07-02): state가 ranked에 promotable로 오르려면 자기 '고유(specific) 마커'가 실제 ON
    #   이어야 함(spec_on≥1, spec_n≥1). 공유 reactive 마커(Gfap 등)만 켜진 조각이 disease state로 뜨던 것 차단
    #   (DAA/DAM FP). spec_n==0(panel-blind)의 trivial spec_pass도 여기선 불인정 — 근거가 측정돼야 승격.
    state_strong=sorted([c for c in cand if c["is_state"] and c["n_on"]>=2 and c["neg"]==0 and c["spec_pass"]
                         and c.get("spec_on",0)>=1 and c.get("spec_n",0)>=1
                         and c not in ranked], key=lambda c:(c["dec"], c["wspec"], c["n_on"]), reverse=True)
    if state_strong: ranked=ranked+state_strong[:2]
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
    """★ MUTUAL-EXCLUSIVITY mixture (2026-06-24, CASE+RULE, 하드코딩 없음).
    CASE: leaf가 한 lineage CLASS에 머물렀는데(class_floor/ambiguous), 그 lineage의 sibling subtype 둘이
      MUTUALLY EXCLUSIVE — X의 positive(R/DF) 마커가 Y의 NEGATIVE 마커이고 그 역도 성립. 한 세포는 동시에
      가질 수 없는 마커쌍(CD4 vs CD8: CD4SP R=CD4·neg=CD8 / CD8SP R=CD8·neg=CD4)이라, 클러스터가 X+Y- 세포와
      X-Y+ 세포의 혼합이면 cluster-평균으론 둘 다 '낮게 present(서로 희석)'로 보인다 → MEAN-기반 top/mixture가
      못 잡고 class_floor로 떨어진다(한쪽이 강한 게 아니라 둘 다 낮은데 비슷 = 혼합의 신호).
    RULE: 상호배타 쌍을 KG 구조에서 자동 도출(X.pos ∩ Y.neg ≠∅ AND Y.pos ∩ X.neg ≠∅, same-lineage sibling).
      leaf의 PER-CELL로 각 세포를 X-side(X마커+ Y마커-) / Y-side(Y+ X-)로 분류 →
        · 양쪽 배타집단이 둘 다 ≥split_frac → verdict 'split' (진짜 X+Y 혼합 → needs_subclustering)
        · 한쪽만 ≥split_frac        → 'dominant:<label>' (우세 subtype 호출; 반대편 부재가 증거 — CD4 dropout이어도
                                       CD8-부재로 CD4SP 확정)
        · 어느 쪽도 부족(연속체/dropout) → 미발동.
    [[_bimodal_split]](low-share 양성 마커만)과 보완: 여기선 reciprocal pos↔neg를 써 공유도 높은 lineage
    마커(CD4/CD8)도 잡는다. 순수 ADDITIVE(기존 트리거 불변). 반환 [{pair,markers,frac_exclusive,verdict}] 또는 None."""
    terms=octx.get("terms",{}); lab=IDM["lab"]; root=IDM["root"]; tier=IDM["tier"]
    cls=IDM["lin_class"].get(lineage_label)
    if cls is None: return None
    cls_root=root.get(cls,cls)
    sibs=[t for t in terms if root.get(t)==cls_root and tier.get(t)!="class"
          and (terms[t].get("D") or "normal")=="normal"]
    U=lambda xs:{str(g).upper() for g in (xs or [])}
    # exclusive 변별자 = DEFINING(DF, '단독 충분' curated 마커)만. required(R)는 RAG1 같은 maturation-stage
    #   공유마커를 포함해 lineage-배타가 아닌 거짓 쌍(CD8 vs RAG1-immature)을 만든다 → DF로 제한해 진짜 lineage 변별만.
    pos={t:U(terms[t].get("DF")) for t in sibs}
    neg={t:U(terms[t].get("N")) for t in sibs}
    pairs=[]
    for i in range(len(sibs)):
        for j in range(i+1,len(sibs)):
            X,Y=sibs[i],sibs[j]
            mx=pos[X]&neg[Y]; my=pos[Y]&neg[X]            # X-defining이 Y-negative, 상호 → 상호배타 쌍
            if mx and my: pairs.append((X,Y,sorted(mx),sorted(my)))
    if not pairs: return None
    if hasattr(Xleaf,"toarray"): Xleaf=Xleaf.toarray()
    Xleaf=np.asarray(Xleaf)
    if Xleaf.shape[0]<min_cells: return None
    gidx={str(g).upper():i for i,g in enumerate(list(genes_arr))}
    out=[]
    for X,Y,mx,my in pairs:
        xi=[gidx[g] for g in mx if g in gidx]; yi=[gidx[g] for g in my if g in gidx]
        if not xi or not yi: continue                     # 한쪽 마커가 off-panel이면 배타 판정 불가
        # ★ ENRICHMENT 기반 ON (2026-06-26): 단순 검출(>0)은 sparse SSAM KDE에서 노이즈(trace bleed)도 ON으로
        #   세어 가짜 mixture를 만든다(cl17: Wfs1/Prox1 절대평균 0.001인데 19/20% '검출'→가짜 CA1+DG split).
        #   hi_thr(전역 gmu+gsd, fon의 ON 정의와 동일)가 주어지면 그 임계 초과만 ON으로 — 노이즈 제거.
        if hi_thr is not None:
            xon=(Xleaf[:,xi]>=np.asarray(hi_thr)[xi]).any(axis=1); yon=(Xleaf[:,yi]>=np.asarray(hi_thr)[yi]).any(axis=1)
        else:
            xon=(Xleaf[:,xi]>0).any(axis=1); yon=(Xleaf[:,yi]>0).any(axis=1)
        fX=float((xon&~yon).mean()); fY=float((~xon&yon).mean()); fBoth=float((xon&yon).mean())
        # ★ 진짜 상호배타 가드: 이중양성(둘 다 발현) 세포가 배타집단보다 드물어야 한다. 마커가 자주 공존하면
        #   (CCR7 naive vs CD4: naive CD4는 CCR7+CD4+) 구조상 pos∩neg라도 진짜 배타 아님 → 제외.
        if fBoth >= 0.5*min(fX, fY): continue
        # SPLIT만: 양쪽 배타집단이 둘 다 충분 = 진짜 X+Y 혼합. 한쪽만(발달 연속체/단일정체)이면 미발동
        #   — mixture가 아니라 '어느 쪽 단일'이며 정상 scoring 영역(dominant 남발=노이즈).
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
    """leaf별 profile. up/dn 무제한(AUC 임계만), hp=50.
    identity_evidence = density(vf_norm) + specificity_peak + lineage_markers_present + housekeeping_share + cross_lineage."""
    # ★ stratified subsample (2026-06-17 속도): 전체 cell의 AUC rank 계산(rankdata axis=0)이 650k에서 ~11분으로
    #   파이프라인 최대 병목. 마커는 population 신호라 leaf별 충분한 cell이면 AUC가 안정 → profile_cap 초과 시 leaf별
    #   층화 subsample(작은 leaf는 전량 유지, 큰 leaf만 비례 축소). 전역 gmu/gsd는 init이 FULL 데이터로 계산해 넘기므로
    #   absolute identity 기준은 불변. 결정성 위해 seed 고정. profile_cap=0이면 full(정확도 최우선).
    leaf_arr=np.asarray(leaf_arr); _n_full=X.shape[0]
    # ★ TRUE leaf 크기(subsample 전) 보존 → entry["n"]은 실제 cell 수로 리포트(agent의 절대-수 판단·희귀타입 sanity
    #   보존). AUC/pin/mean 등 통계는 subsample로 계산하되, 보고되는 n/n_out만 TRUE. (finalize gate는 state의 TRUE
    #   cluster_paths를 별도로 써서 무관.)
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
        print(f"[profile] stratified subsample {len(_sel)}/{_n_full} cells (cap={profile_cap}, MINKEEP={_MINKEEP}) — AUC/rank 가속", flush=True)
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
    IDM=None; _gpin_arr=_ghi_arr=None
    if gmu is not None and gsd is not None:
        IDM=_build_ident_maps(octx, genes_arr)
        gmu_arr=np.asarray(gmu,dtype=np.float64); gsd_arr=np.asarray(gsd,dtype=np.float64)
        sumHI=M@((X>=(gmu_arr+gsd_arr)).astype(np.float64))   # (K, G) leaf별 high세포 수
        if adaptive_floor:   # ★ 데이터-유도 floor용 전역 baseline (reference-free): 전역 검출률/강발현률 (1회)
            _gpin_arr=np.asarray(totBin,dtype=np.float64).ravel()/max(n_cells,1)
            _ghi_arr=np.asarray((X>=(gmu_arr+gsd_arr)).sum(axis=0),dtype=np.float64).ravel()/max(n_cells,1)
            print(f"[adaptive-floor] ON (F={af_F}, Nmin={af_Nmin}) — region-gated positive floor = min(fixed, ambient×F); "
                  f"median gpin={np.median(_gpin_arr):.4f} ghi={np.median(_ghi_arr):.4f}", flush=True)

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
        # region_share(영역 인식 변별도, _build_ident_maps와 동일)를 심볼 키로 변환해 넘김 → candidate_subtypes의
        #   변별 판정을 graph_evidence anchor와 통일. IDM 없으면 None(→ pan_disc 폴백).
        _rs_sym=({_U(IDM["genes"][g]):c for g,c in (IDM.get("region_share") or {}).items() if 0<=g<len(IDM.get("genes") or [])}
                 if IDM is not None else None)
        cands, sig_targets=_candidate_subtypes(octx.get("terms",{}), ogt, up, octx, parent_up_auc=parent_up_auc, region_share=_rs_sym)
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
        _tn=_true_n.get(path, ni)                              # TRUE leaf 크기(subsample 무관) — 리포트용
        entry={"n":_tn,"n_out":_n_full-_tn,"n_sub":ni,"up":up,"dn":dn,"hp":hp,"ontology_gene_table":ogt,
               "identity_evidence":identity, "lineage_scan":lscan, "candidate_subtypes":cands,
               "signal_targets_suggested":sig_targets, "subtype_groups":subtype_groups,
               "flags":{"multi_lineage":multi,"layer_mixing":layer_mix,"pan_class_only":bool(pan_class),"background_score":bgscore},
               "sibling_contrast":sib}
        if no<MIN_N_OUT: entry["warn"]="n_out_too_small"
        # absolute identity_scores (전역 기준, 조각화 불변) — Leiden+absolute 판단의 핵심 근거
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
            # (구 Step 0.5 cluster-level compartment(neuropil/fiber tract) emission은 2026-07-21 제거 —
            #  process는 clustering-based 경로에서 라벨하지 않는다. 사용자가 명시 요청하는 scoring-based
            #  annotation(score-profile/apply/finalize)이 유일한 process 탐지 경로다.)
            # DF-confirmed terms in this leaf (a DEFINING marker is fold-ON: log2fold>=1.5 AND detected>=25%).
            # Used by the finalize split-review gate (§8.1) + DF_MULTI_SUBTYPE to flag splits with a real 2nd identity.
            # ★ LINEAGE-GATED (2026-06-17): leaf의 dominant lineage(graph_evidence.lineage)에 속한 term만 df_present에
            #   넣는다. 이유: 공유 마커 DEFINING(CD4=CD4 T cell DF인데 골수계 mono/macro/DC에도 발현)이 lineage 무관하게
            #   잡히면, 골수계 클러스터가 "CD4 T cell"을 df_present에 넣어 DF_MULTI_SUBTYPE/gate를 오염시킨다. lineage
            #   게이트로 골수계 오염은 막고, T-lineage 내 CD4(DF)+CD8B(DF) 공존(=진짜 CD4/CD8 혼합)은 그대로 감지된다.
            #   (mwb에서도 glut leaf의 ambient oligo DF 같은 cross-lineage 가짜 split 신호가 덤으로 제거됨.) lineage가
            #   None(no_lineage)이면 게이트 미적용(보수적 fallback). DF-override 앵커링은 이미 dom_root 게이트라 무관.
            _defn=IDM.get("defn",{}); _lab=IDM.get("lab",{}); _root=IDM.get("root",{})
            _lin_root=IDM.get("lin_class",{}).get((entry.get("graph_evidence") or {}).get("lineage"))
            _dfp=[tid for tid,gs in _defn.items() if any(lf[g]>=1.5 and pin[g]>=0.25 for g in gs)]
            if _lin_root is not None:
                _lr=_root.get(_lin_root,_lin_root); _dfp=[tid for tid in _dfp if _root.get(tid)==_lr]
            entry["df_present"]=sorted({_lab.get(tid,tid) for tid in _dfp})
            # ★ BIMODAL-SPLIT backstop (2026-06-23): per-cell anti-correlated sibling programs that the
            #   cluster-level mixture detector missed (balanced 50/50 mix of sibling-region subtypes —
            #   arterial/venous EC, alveolar/adventitial fibroblast). Only when NO subtype confidently
            #   anchored (verdict≠clear_top) AND mixture_subtypes empty (the case it missed). Region-
            #   independent, so additive (existing triggers unchanged); finalize gate consumes it.
            _ge_b=entry.get("graph_evidence") or {}
            if (_ge_b.get("lineage") and not _ge_b.get("mixture_subtypes")
                    and _ge_b.get("verdict") in ("ambiguous_tie","class_or_unknown","confusion_unresolved")):
                _bim=_bimodal_split(X[np.asarray(leaf_arr)==path], IDM, octx, _ge_b.get("lineage"), genes_arr)
                if _bim: entry["bimodal_subtypes"]=_bim
            # ★ MUTUAL-EXCLUSIVITY mixture (2026-06-24) — 상호배타 마커쌍(CD4/CD8 등)의 per-cell 혼합 감지.
            #   lineage가 정해졌으면 항상 평가(verdict 무관) — class_floor로 떨어진 same-lineage 아형 혼합/우세를
            #   구제한다. ADDITIVE(KG reciprocal pos↔neg로 자동 도출, 하드코딩 없음).
            _exm_lin=(entry.get("identity_scores") or {}).get("dominant_lineage") or _ge_b.get("lineage")
            if _exm_lin:                                    # dominant_lineage 우선(graph lineage는 transmitter gate로 흔들림)
                _Xl=X[np.asarray(leaf_arr)==path]
                _hi=(gmu_arr+gsd_arr) if (gmu is not None and gsd is not None) else None
                _exm=_exclusive_mixture(_Xl, IDM, octx, _exm_lin, genes_arr, hi_thr=_hi)
                if _exm: entry["exclusive_mixture"]=_exm
            # ★ CONTEXT-AXIS (§4.3, 2026-06-21) — DETERMINISTIC zonation/layer subtyping. lineage가 module 선언
            #   ca축의 applies_to면 pole별 context_markers 점수화 → mixed(≥2 pole 존재)면 부모에서 분할대상,
            #   sub-leaf면 dominant_pole→suggested_subtype 제안. lineage는 graph_evidence가 확정(여기선 불변).
            _ge_ca=entry.get("graph_evidence") or {}
            _ca_lin=_ge_ca.get("lineage")
            # region/anchor-aware 매칭용 라벨: dominant_region + anchor 라벨 + anchor 조상 path 라벨
            _ca_extra=set(_ge_ca.get("dominant_region") or [])
            for _an in (_ge_ca.get("anchors") or []):
                if _an.get("label"): _ca_extra.add(_an["label"])
                for _pid in (_an.get("path") or []):
                    _pl=IDM["lab"].get(_pid) if IDM else None
                    if _pl: _ca_extra.add(_pl)
            _ca=_context_axis_eval(_ca_lin, octx.get("ca") or [], fon, lf, pin, gidx_U, extra_labels=_ca_extra)
            if _ca:
                entry["context_axis"]=_ca
                # flag: ≥2 pole 존재(mixed) → 부모면 §4.3대로 subcluster; dominant_pole 있으면 그 subtype 확정대상
                entry["flags"]["context_axis_mixed"]=any(a.get("mixed") for a in _ca)
                entry["flags"]["context_axis_pole"]=next((a.get("dominant_pole") for a in _ca if a.get("dominant_pole")), None)
            # ★ STATE-AXIS (base-agnostic cell states): each State judged independently (ENRICHED program
            #   markers), any number active → appended to the label as ', <state>' at finalize. Evidence for
            #   the agent; distinct from disease_candidates (those are per-base disease/injury state Terms).
            _sax=_state_axis_eval(octx.get("sa") or [], fon, lf, pin, gidx_U)
            if _sax:
                entry["state_axis"]=_sax
            # ★ disease_candidates — disease는 필터가 아니라 힌트(2026-06-22). _is_state_term이 정상 scoring
            #   코어(top/coh/chimera)에서 state를 빼므로(broad 마커 force-fit 방지) 여기서 별도 노출하되, --disease
            #   미지정이어도 module의 모든 state term(st_terms)을 항상 surface한다 → 마커가 우세하면 agent가 그 state로
            #   판정 가능(SKILL §7). --disease 매칭분(dz_terms)은 hint=True로 표시·우선정렬. 같은 ON 정의 사용.
            _src = st_terms if st_terms else dz_terms
            if _src:
                # ★ SPECIFICITY GATE (2026-07-02, disease-state FP fix): 병 상태를 raw ON 개수가 아니라 '고유+enriched'
                #   마커로 판정한다. Gfap/Serpina3n/Vim/Cd44 같은 pan-reactive 마커는 여러 state가 공유(high share)
                #   하거나 ambient(약발현)로 켜져, 마커 목록이 긴 state가 '개수'로 1등이 되어 정상 조직에서 disease
                #   state로 승격되던 문제(DAA/DAM 가짜양성; 건강 뇌 23셀→AD astrocyte). n_specific_on = ON 마커 중
                #   low-share(≤3, 그 상태에 비교적 고유) AND rank-enriched(auc≥0.85, ambient 아님)인 것 수. 0이면
                #   공유/ambient 마커뿐 → 정상 대응(reactive/fibrous astro)으로 fallback해야 한다(SKILL §2.5). surface는
                #   유지(진짜 state 미검출 방지), 승격 판정만 n_specific_on으로.
                _shmap=(IDM.get("region_share") or IDM.get("share") or {}) if IDM else {}
                dzc=[]
                for _tid,_info in _src.items():
                    ons=[]; spons=[]; tot=0
                    _rdf={str(g).upper() for g in (_info.get("R",[])+_info.get("DF",[]))}  # identity-정의 마커(supporting 제외)
                    for _g in (_info.get("R",[])+_info.get("S",[])+_info.get("DF",[])):
                        _gi=gidx_U.get(str(_g).upper())
                        if _gi is None: continue
                        tot+=1
                        if fon[_gi]>=0.30 or (lf[_gi]>=1.5 and pin[_gi]>=0.25):
                            ons.append(_g)
                            # 고유 증거 = REQUIRED/DEFINING 중 low-share(≤3) + enriched(auc≥0.85)인 것만.
                            #   supporting은 공유·corroborative라 불인정(Ctsb/Gsn/Hspb1 같은 KG-저share 일반 stress
                            #   마커가 supporting으로 게이트를 뚫는 것 방지). DAA required=Gfap/Serpina3n(둘 다 high-share)
                            #   → spons=0 → promotable=False → 정상 astro로 fallback.
                            if str(_g).upper() in _rdf and _shmap.get(_gi,99)<=3 and auc is not None and auc[_gi]>=0.85:
                                spons.append(_g)
                    if ons:
                        dzc.append({"term_id":_tid,"label":_info["L"],"D":_info["D"],
                                    "on_markers":ons,"n_on":len(ons),"n_markers":tot,
                                    "specific_markers":spons,"n_specific_on":len(spons),
                                    "promotable":bool(spons),
                                    "hint":bool(dz_terms and _tid in dz_terms)})
                dzc.sort(key=lambda d:(-int(d["hint"]), -d["n_specific_on"], -d["n_on"]))
                if dzc: entry["disease_candidates"]=dzc[:8]
            # ★ transition_candidates — anchor-tiebreaker (STATE_TRANSITION_DESIGN §엔진규칙, evidence-strong).
            #   lineage-mimicry term(MP+A 보유, tier 무관)에 대해: origin identity-anchor(A) 마커 잔존 present
            #   AND 흉내 프로그램(R/S/DF의 비-anchor 마커) 동시 present면 → 이 leaf는 흉내 대상 lineage가 아니라
            #   'origin lineage + 전이상태'. 흉내 lineage로의 force-fit/chimera 오판 방지를 위해 최상위 신호로 surface.
            if mt_terms:
                def _on_g(_g):
                    _gi=gidx_U.get(str(_g).upper())
                    if _gi is None: return False
                    return fon[_gi]>=0.30 or (lf[_gi]>=1.5 and pin[_gi]>=0.25)
                # ★ origin-lineage 게이트: transition term의 origin lineage가 leaf의 dominant lineage와 같을 때만
                #   발동(흉내 대상이 아니라 origin이 일치해야 '그 origin의 전이상태'다). 다르면 cross-lineage 거짓발동
                #   (mTEC term이 endothelial/T 클러스터에 발동) 차단. leaf lineage 미정(None)이면 게이트 미적용(보수적).
                _geL=(entry.get("graph_evidence") or {}).get("lineage") if IDM is not None else None
                _leaf_root=IDM["root"].get(IDM.get("lin_class",{}).get(_geL)) if (IDM is not None and _geL) else None
                trc=[]
                for _tid,_info in mt_terms.items():
                    if _leaf_root is not None and IDM is not None:
                        _troot=IDM["root"].get(_tid)
                        if _troot is not None and _troot!=_leaf_root: continue   # origin lineage ≠ leaf lineage
                    anchor_on=[g for g in _info.get("A",[]) if _on_g(g)]
                    if not anchor_on: continue                       # origin 잔존 없음 → 미발동(진짜 흉내 lineage일 수 있음)
                    spec_on=[g for g in _info.get("Qspec",[]) if _on_g(g)]   # 흉내 lineage SPECIFIC 마커(pan 제외)
                    if len(spec_on)<2: continue                      # ★ 특이 흉내 프로그램 ≥2 ON일 때만 발동(pan-마커 거짓발동 차단)
                    trc.append({"term_id":_tid,"label":_info["L"],"D":_info["D"],
                                "anchor_retained":anchor_on,"acquired_on":spec_on[:8],
                                "mimics":[{"lineage":m.get("lineage"),"program":m.get("program","")} for m in _info.get("MP",[])],
                                "n_anchor":len(anchor_on),"n_acquired":len(spec_on)})
                if trc:
                    trc.sort(key=lambda d:(-d["n_anchor"], -d["n_acquired"]))
                    entry["transition_candidates"]=trc[:6]
                    _mlin=sorted({m["lineage"] for d in trc for m in d["mimics"] if m.get("lineage")})
                    entry["transition_note"]=("identity-anchor 잔존 → transition_candidates는 origin lineage+전이상태다. "
                        "흉내 대상 lineage("+", ".join(_mlin)+")로 순수 판정하지 말 것; anchor 마커가 origin 정체성 증거.")
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

def _check_input_matrix(adata, force=False):
    """init 진입 게이트: 표준 annotator는 adata.X만 읽는다(--layer 없음). 두 전제를
    검증한다 — X=log-normalized(정수 raw count 아님), var_names=gene symbol(Ensembl ID 아님).
    둘 다 위반해도 파이프라인은 조용히 완주하지만 라벨은 쓰레기가 된다(마커축 통계가 lognorm을
    가정, 마커 매칭이 심볼 기준). 그래서 여기서 fail-fast. --force로 무시 가능(비표준 입력 자인)."""
    import re as _re
    X=adata.X
    n=adata.n_obs
    # ---- 표본 추출(대형 h5ad 전체를 densify하지 않도록 앞 N행만) --------------------
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
        problems.append(f"var_names가 gene symbol이 아니라 Ensembl ID로 보입니다(표본 {ens}/{len(vn)}개, 예: '{ex}'). "
                        f"마커는 심볼(EPCAM, Snap25)로 매칭하므로 Ensembl ID면 마커가 하나도 안 잡힙니다. "
                        f"adata.var_names를 gene symbol로 바꾸세요.")
    # ---- (2) X = log-normalized? (정수 raw count 감지) ---------------------------
    finite=Xs[np.isfinite(Xs)]
    if finite.size:
        if np.any(finite < 0):
            problems.append("X에 음수가 있습니다 — log-normalized가 아닌 정규화로 보입니다"
                            "(z-scored/scaled, 또는 Pearson residual 등). "
                            "annotator는 비음수 log-normalized X를 기대합니다 — scale/residual 이전 단계를 사용하세요.")
        else:
            nz=finite[finite>0]
            allint = nz.size>0 and np.allclose(nz, np.round(nz))
            xmax=float(finite.max())
            # log-normalized X는 log1p 결과라 사실상 절대 all-integer가 아니다 → all-integer면 raw count.
            # (max 임계에 의존하지 않는다: 저심도 데이터도 걸러야 하고, max>1 은 binary mask 오탐만 배제.)
            if allint and xmax>1:
                problems.append(f"X가 정수 raw count로 보입니다(비영 값이 전부 정수, max={xmax:.0f}). "
                                f"annotator는 log-normalized X를 기대합니다(count는 마커축 통계를 왜곡). "
                                f"sc.pp.normalize_total(adata, target_sum=1e4) 후 sc.pp.log1p(adata)를 적용하세요.")
    if problems:
        msg="\n".join(f"   • {p}" for p in problems)
        if force:
            print(f"[init][WARN:INPUT_MATRIX] 입력 전제 위반(--force로 강행):\n{msg}", flush=True)
        else:
            sys.exit(f"[init][ERROR:INPUT_MATRIX] 입력 h5ad가 annotator 전제를 위반합니다:\n{msg}\n"
                     f"   → 고친 뒤 다시 실행하거나, 비표준 입력임을 인지하고 강행하려면 --force. "
                     f"→ caller는 사용자에게 위 문제를 그대로 전달할 것.")

# ===================================================================== init
def cmd_init(a):
    os.makedirs(a.out, exist_ok=True)
    print(f"[init] loading {a.h5ad} (cluster_col={a.cluster_col})", flush=True)
    adata=ad.read_h5ad(a.h5ad)
    if a.cluster_col not in adata.obs:
        sys.exit(f"[init] obs에 cluster 컬럼 '{a.cluster_col}' 없음. 사용가능: {list(adata.obs.columns)}")
    _check_input_matrix(adata, force=getattr(a,"force_input",False))   # X=lognorm / var_names=symbol 전제 검증
    cps=adata.obs[a.cluster_col].astype(str).values.copy()
    # 사용자가 준 cluster/annotation 컬럼의 모든 라벨이 leaf(=annotation 대상). clustering-based
    #   annotation은 사용자 클러스터링(외부/노트북)을 그대로 입력받는다.
    leaves=sorted(set(cps.tolist()),
                  key=lambda s:[int(x) for x in s.split(".")] if s.replace('.','').isdigit() else [10**9,s])
    # ★ subcluster 표현공간(use_rep): subcluster의 neighbors를 통합 임베딩(scVI 등)에서 재계산.
    #   미지정('auto')이면 obsm에서 통합 임베딩 자동탐지; 'recompute'/'none'이면 raw 대신 normalize+log1p+PCA fallback.
    #   (예전 동작=raw count PCA는 stress/library-size에 지배돼 정체성 분할 실패 → 더 이상 기본 아님.)
    rep=a.use_rep
    if rep in (None,"auto",""):
        rep=next((c for c in ("X_scVI","X_scANVI","X_scanvi","X_pca_harmony","X_harmony","X_pca","X_PCA")
                  if c in adata.obsm), None)
    elif rep in ("recompute","none","raw","RECOMPUTE"):
        rep=None
    elif rep not in adata.obsm:
        sys.exit(f"[init] --use-rep '{rep}' 가 obsm에 없음. 사용가능 obsm: {list(adata.obsm)} "
                 "(또는 --use-rep recompute 로 normalize+log1p+PCA 재계산).")
    print(f"[init] subcluster representation: {rep or 'recompute (normalize_total+log1p+PCA)'}", flush=True)
    # ★ subcluster metric/n_neighbors는 **초기 클러스터링과 반드시 같아야** 한다(다르면 조용히 틀어짐).
    #   입력 h5ad를 만든 클러스터링(사용자 노트북/외부)이 uns['cellman_cluster_params']에 metric/n_neighbors를
    #   남겨두면 여기서 그대로 상속한다 — 없으면 기본(euclidean/15). --subcluster-metric/-n-neighbors로 직접
    #   지정하면 uns를 이깁니다(외부에서 다른 설정으로 클러스터링해 온 h5ad용).
    _cp={}
    try:
        _raw=adata.uns.get("cellman_cluster_params")
        _cp=json.loads(_raw) if isinstance(_raw,str) else (dict(_raw) if _raw else {})
    except Exception: _cp={}
    sc_metric=getattr(a,"sc_metric",None) or _cp.get("metric") or "euclidean"
    sc_n_neighbors=int(getattr(a,"sc_n_neighbors",None) or _cp.get("n_neighbors") or 15)
    if _cp and not getattr(a,"sc_metric",None):
        print(f"[init] uns['cellman_cluster_params']에서 초기 클러스터링 설정 상속: "
              f"metric={sc_metric} n_neighbors={sc_n_neighbors} (backend={_cp.get('backend')})", flush=True)
    if rep is None:
        print(f"[init] subcluster feature: log-normalize + metric={sc_metric} + n_neighbors={sc_n_neighbors}", flush=True)
    st={"h5ad_path":os.path.abspath(a.h5ad),"cluster_col":a.cluster_col,
        "cluster_paths":cps.tolist(),"leaves":leaves,"round":1,"parent_of":{p:None for p in leaves},
        "output":a.output,"disease":a.disease,"use_rep":rep,
        "sc_metric":sc_metric,"sc_n_neighbors":sc_n_neighbors,
        # ★ adaptive-floor도 persist(2026-07-02): init에서 한 번 --adaptive-floor면 모든 subcluster 라운드가 상속
        #   (subcluster-metric과 동일 방식) — 라운드마다 재지정 불필요.
        "adaptive_floor":bool(getattr(a,"adaptive_floor",False)),"af_F":float(getattr(a,"af_F",3.0)),"af_Nmin":int(getattr(a,"af_Nmin",10))}  # disease/use_rep/sc_*/adaptive_floor persist(모든 round 공유)
    _save_state(a.out, st)
    # ontology URL 자동 구성: --ontology-url 직접지정 > (species/organ + 기본/지정 서버) ; 로컬은 --ontology.
    url=a.ontology_url
    if not url and not a.ontology:
        if not (a.species and a.organ):
            sys.exit("[init] --species/--organ 가 필요합니다(또는 --ontology-url / 로컬 --ontology). "
                     "서버는 자동으로 기본 Cell-Ontology 서버에 연결됩니다.")
        url=f"{_resolve_server(a.server_url)}/module/{a.species}/{a.organ}/slim-index"
        print(f"[init] ontology server: {_resolve_server(a.server_url)} (module {a.species}/{a.organ})", flush=True)
    idx=_fetch_slim_index(a.out, url, a.ontology)
    if not idx or not _ontology_gene_set(idx):
        sys.exit("[init] FATAL: ontology(slim-index) 비어있음 — 서버/ module 확인. ontology-blind 금지.")
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
    dz_terms=_disease_terms(idx, a.disease); st_terms=_state_terms(idx); mt_terms=_mimicry_terms(idx)
    if mt_terms: print(f"[init] mimicry/transition terms {len(mt_terms)} (MP+anchor) → leaf별 transition_candidates(anchor-tiebreaker, §state-transition)", flush=True)
    print(f"[init] state terms {len(st_terms)} → leaf별 disease_candidates로 항상 surface (disease=힌트; 판정은 agent §7)"
          + (f"; --disease='{a.disease}' 매칭 {len(dz_terms)}개 hint" if a.disease else "; --disease 미지정(모든 state 고려)"), flush=True)
    # ★ F-B: anatomical region-presence 게이트 (전체 데이터 1회 계산 → persist). 부재 region(소뇌/뇌교 등)을
    #   anchor/dominant/floor 후보에서 제외해 force-fit 차단. brain-anat 매칭 region-group만 영향(비-brain module 무영향).
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
    # ★ Unknown re-pool 모드: 현 라운드 Unknown-결정 leaf들의 셀을 cross-parent로 모아 하나의 synthetic 부모
    #   U{round}로 Leiden 재클러스터. 흩어진 동일-type Unknown 조각이 재응집 → 다음 라운드 재판정. 최종 라운드까지
    #   진짜 Unknown인 것만 finalize에서 'Unknown' 라벨. (사용자 2026-06-27: Unknown은 마지막 라운드 전까진 종착 아님.)
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
    rep=st.get("use_rep")                              # ★ subcluster 표현공간(init에서 결정). None이면 normalize+log1p+PCA 재계산.
    new_all=[]; parent_prof={}
    for path in paths:
        if repool:
            mask=np.isin(cps, unk_leaves); n=int(mask.sum())   # cross-parent Unknown 풀
            print(f"[repool] pooling {n} cells from {len(unk_leaves)} Unknown leaves -> {path}", flush=True)
        else:
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
        nn=min(int(st.get("sc_n_neighbors",15)),sub.n_obs-1)  # ★ init에서 persist된 n_neighbors(미지정 15)
        # ★ use_rep 있으면 통합 임베딩(scVI 등) subset에서 neighbors 재계산(정체성 보존, top-level과 일관).
        #   없으면 raw 대신 normalize_total+log1p+PCA 후 neighbors(raw-count PCA의 library-size/stress 지배 회피).
        use_rep=rep if (rep and rep in sub.obsm) else None
        # ★ 초기 클러스터링과 일관된 metric (init에서 persist). recompute 경로에서만 적용.
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
        # repool이 실제 분리(new_all)했으면 원래 Unknown leaf들을 U-subleaf로 대체; 분리 안 됐으면(단일 blob) 그대로 둠.
        leaves=([p for p in st["leaves"] if p not in unk_leaves]+new_all) if new_all else st["leaves"]
        if not new_all: print("[repool] pooled Unknown did not split (single blob) — Unknown leaves unchanged.", flush=True)
    else:
        leaves=[p for p in st["leaves"] if p not in parent_prof]+new_all
    st["leaves"]=leaves; st["cluster_paths"]=cps.tolist(); st["round"]+=1; _save_state(a.out,st)
    idx=_load_ontology_index(a.out); ogs=_ontology_gene_set(idx); prot=_ontology_protected(idx); octx=_onto_ctx(idx)
    dz_terms=_disease_terms(idx, st.get("disease")); st_terms=_state_terms(idx); mt_terms=_mimicry_terms(idx)   # state는 항상 surface(disease=힌트), sub-leaf도 동일; mt=transition anchor-tiebreaker
    X=np.asarray(adata.X.todense() if hasattr(adata.X,"todense") else adata.X)
    vf,low=_vf_setup(adata)
    # ★ F-B: init에서 persist한 부재 region-group 로드(부분집합에선 직접계산 불가하므로 재사용)
    _arp=os.path.join(a.out,"_region_presence.json")
    absent_rg=set(json.load(open(_arp)).get("absent_region_groups",[])) if os.path.exists(_arp) else set()
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
        # ★ 부모가 단일 region일 때만 그 region으로 제약(2026-06-18, 사용자 지침 + 기존 주석 의도 일치). 부모가
        #   **상위 region 2개 이상 혼합**(dominant_region≥2)이면 단일 region으로 고정하지 말고 **lineage class로만**
        #   제약 → 자식이 두 region 어디로든 resolve 가능(피질 arc + BLA 덩어리가 한 부모에 섞인 stomics cl6 같은
        #   경우, 자식이 amygdalar로 탈출 가능). dr[0] 단일 고정은 소수 region(BLA)을 다수 region(neocortical)에
        #   가두는 버그였음. (siblings within one region stay findable: CA1 inside CA3, both hippocampal pyramidal.)
        if len(dr)==1 and dr[0] in _lab2tid: return _lab2tid[dr[0]]
        lin=ge.get("lineage")
        return _lab2tid.get(lin)                                  # 0개 or ≥2개(혼합) → lineage class (cross-lineage만 차단)
    profiles={}
    for parent,(mask,sub_paths) in parent_prof.items():
        sl=sorted(set(sub_paths.tolist()))
        p_up={r[0]:r[1] for r in (prior.get(parent,{}).get("up") or [])}   # 부모 up(=뇌 전체 대비 AUC)
        rr=_restrict_for(parent)
        pr=_profile_leaves(X[mask], genes, sub_paths, sl, ogs, prot, octx,
                           vf[mask] if vf is not None else None, low, parent_up_auc=p_up, gmu=gmu, gsd=gsd, restrict_root=rr, dz_terms=dz_terms, st_terms=st_terms, mt_terms=mt_terms, profile_cap=getattr(a,"profile_cap",0), auc_pos=getattr(a,"auc_pos",0.65), absent_rg=absent_rg, adaptive_floor=(bool(st.get("adaptive_floor",False)) or getattr(a,"adaptive_floor",False)), af_F=float(st.get("af_F",getattr(a,"af_F",3.0))), af_Nmin=int(st.get("af_Nmin",getattr(a,"af_Nmin",10))))
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
    _NONP=('endothel','pericyte','smooth muscle','leptomening','mural','perivascular','fibroblast',
           'macrophage','microg','choroid','ependym','arachnoid','vascular','erythro','neutrophil',
           'lymphocyte','monocyte','mast','tanycyte')   # non-parenchymal(embedded) 라벨 키워드
    def _isnonpar(s): return any(k in (s or '').lower() for k in _NONP)
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
        # ★ advisory 2종 (2026-06-28) — 강제 아님: 근거 제시 + agent 재판단 유도(helper는 결정 불변).
        _ge=p.get("graph_evidence") or {}
        # A) RESCUE_IGNORED: 엔진이 embedded vascular/immune anchor를 rescue했는데 라벨이 parenchymal이면
        #    절대-우세 glia(KDE smear)에 휩쓸렸을 수 있음 → §4.1b로 재판단.
        _resc=[an for an in (_ge.get("anchors") or []) if an.get("rescued")]
        if dec=="celltype" and _resc and not _isnonpar(d.get("label","")):
            _r=_resc[0]
            add(lf,"RESCUE_IGNORED","warn",
                f"엔진이 embedded vascular/immune anchor '{_r.get('label')}'을 rescue함(특이마커 {_r.get('rescue_markers')} 강하게 enriched) — 그런데 라벨은 parenchymal '{d.get('label')}'. glia/neuron 신호는 KDE smear일 수 있음.",
                "§4.1b: rescued 특이 vascular/immune 마커가 강하면 절대-우세 glia 대신 그 type으로 재판단(진짜 근거 있으면 유지).")
        # B) NO_ANCHOR_SPECIFIC: 엔진이 positive anchor 0(class_or_unknown)인데 specific으로 라벨하면
        #    마커 근거 없는 region/subtype force-fit 가능(allocortex/subiculum 등) → class/Unknown 재고.
        if dec=="celltype":
            _verd=_ge.get("verdict"); _linc=str(_ge.get("lineage") or "").strip().lower()
            _ll=str(d.get("label","")).strip().lower()
            _is_proc_term=((idx.get((d.get("term_id") or "").strip(),{}) or {}).get("T") in _PROC_TIERS)
            if (_verd=="class_or_unknown" and _ll and _ll!=_linc and _ll not in ("unknown","background")
                    and not _is_proc_term):   # scoring-based process 라벨은 force-fit 검사 면제
                # ★ 라벨 고유 마커(R∪DF)가 클러스터 강한 up에 하나도 없을 때만 flag (astrocyte Aqp4 등 마커 ON이면 정당 → 면제).
                _tid=(d.get("term_id") or "").strip(); _term=idx.get(_tid) or {}
                _mk={_U(g) for g in ((_term.get("R") or [])+(_term.get("DF") or []))}
                if _mk and not (_mk & us):   # 고유 마커 전무 = force-fit 의심
                    add(lf,"NO_ANCHOR_SPECIFIC","warn",
                        f"엔진 positive anchor 0(class_or_unknown) + specific '{d.get('label')}'의 고유 마커가 클러스터에 하나도 강하게 ON 아님 — region/subtype force-fit 의심.",
                        f"고유 마커 ON이면 유지; 아니면 lineage class('{_ge.get('lineage')}') 또는 Unknown으로 재판단(§3 positive-evidence).")
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
            # ★ CONTEXT-AXIS enforcement (§4.3, 2026-06-21): 엔진이 context_axis(zonation/layer)를 계산했는데
            #   bare lineage class로 종결했으면 — pole subtype을 놓친 것. dominant_pole 있으면 그 subtype으로
            #   relabel, 없고 mixed(≥2 pole)면 subcluster 요구. (helper는 결정 불변 — check로 보고만; suggested_subtype을
            #   엔진이 이미 계산해줘 agent의 재결정은 기계적.) lineage class 라벨로 종결한 경우만 발동(진짜 subtype 라벨은 면제).
            _cax=p.get("context_axis") or []
            _calin=str((p.get("graph_evidence") or {}).get("lineage") or "").strip().lower()
            _is_top=("." not in lf)                          # top-level parent (subcluster 안 된 원클러스터)
            _nleaf=int(p.get("n",0) or 0)
            for _ax in _cax:
                # 1) ★ 큰 top-level 부모에 양극이 *실질* 공존(mixed) → 한 극으로 라벨했든 class로 뒀든, 먼저 분할해
                #    양 zone을 분리해야(소수 극 매몰 방지). sub-leaf(이미 분할됨)는 면제 — lean-label이 맞다.
                if _ax.get("mixed") and _is_top and _nleaf>=1500:
                    add(lf,"CONTEXT_AXIS_SPLIT","warn",
                        f"context axis '{_ax['axis']}' 양극 실질 공존(poles_present={_ax['poles_present']}) + 큰 top-level 클러스터(n={_nleaf})인데 '{d.get('label')}'로 미분할 종결 — 한 극 라벨은 소수 극을 묻는다. §4.3: subcluster해 양 zone 분리.",
                        "needs_subclustering(reason_code=context_axis)로 분할 → 각 sub-leaf를 dominant_pole subtype으로 라벨(양극 회수).")
                # 2) bare lineage class로 종결했는데 dominant_pole 있음 → 그 pole subtype으로 relabel
                elif _lab_l==_calin and _ax.get("dominant_pole") and _ax.get("suggested_subtype"):
                    add(lf,"CONTEXT_AXIS_SUBTYPE","warn",
                        f"context axis '{_ax['axis']}'의 dominant_pole={_ax['dominant_pole']}(context 우세)인데 bare class '{d.get('label')}'로 종결 — §4.3: '{_ax['suggested_subtype']}'로 라벨해야.",
                        f"label을 '{_ax['suggested_subtype']}'로 변경(reason_code=context_axis, confidence={_ax.get('pole_confidence') or 'low'}). 내재 off-panel 마커가 아니라 pole context로 판정.")
            # ★ over-split 일반 규칙 (2026-06-23): "magnitude-only subtype" = subtype 라벨인데 부모 class를 구별하는
            #   *고유* 변별마커가 ON이 아님 → 공유마커의 발현 *크기*만으로의 분할 → 부모 class로 collapse. 하드코딩 0
            #   (KG 계층+데이터로만 분기), 모든 module 적용. 진짜 subtype(L4=Rorb·CA3=Grik4·Opalin+ oligo)은 고유마커 ON이라 유지.
            #   #1=일반 subtype(term-마커 differential), #2=context-axis pole(cross-leaf 반대극 존재). 둘로 나눈 이유:
            #   liver zonation pole(ambient context로 분리)은 per-leaf differential로 판정 불가 → 반대극 공존으로만 진짜/noise 구분.
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
                    _diff=_Smk-_Pmk                                       # 부모와 공유 안 되는 고유 변별마커
                    _ogtv={_U(g):v for g,v in (p.get("ontology_gene_table") or {}).items()}
                    _diff_op={g for g in _diff if not ((_ogtv.get(g) or {}).get("panel") is False)}  # on-panel만(off-panel은 미지)
                    if _diff_op and not (_diff_op & us):                 # 고유마커 있는데 enriched(up auc≥0.6)인 게 0
                        _Cl=idx[_Pids[0]].get("L") or _Pids[0]
                        add(lf,"DIFFERENTIAL_UNSUPPORTED","warn",
                            f"'{_S.get('L')}'(subtype)의 부모 '{_Cl}' 대비 고유 변별마커 {sorted(_diff_op)} 중 enriched(up)인 게 없음 — 공유마커의 레벨 차이만으로의 분할(magnitude-only subtype, over-split 의심).",
                            f"부모 class '{_Cl}'로 라벨 변경. 진짜 subtype이면 고유마커가 ON이어야 하며, 같은 정체의 형제는 부모로 병합된다.")
            # ── #2 CONTEXT_AXIS_LONELY_POLE (context-axis pole) ──
            if _lab_l in _ca_pole_subs:
                _all_lab={str((dd or {}).get("label","")).strip().lower() for dd in decisions.values()}
                for _ax in (p.get("context_axis") or []):
                    _psubs={str(po.get("subtype")).strip().lower() for po in (_ax.get("poles") or []) if po.get("subtype")}
                    if _lab_l not in _psubs: continue
                    _others=_psubs-{_lab_l}
                    if _others and not (_others & _all_lab):             # 반대 극이 어떤 클러스터에도 없음 → 외톨이 pole
                        _Pp=idx.get(tid,{}).get("P") or []
                        _Cl=idx.get(_Pp[0],{}).get("L") if _Pp else None
                        add(lf,"CONTEXT_AXIS_LONELY_POLE","warn",
                            f"context axis '{_ax['axis']}'의 '{d.get('label')}' pole로 라벨됐으나 반대 극({sorted(_others)})이 어떤 클러스터에도 없음 — 축이 실제로 분리 못 한 단일-극 noise lean(진짜 zonation이면 양극이 모두 떠야).",
                            f"부모 class '{_Cl or 'parent'}'로 라벨 변경(반대 극이 실제로 나타나면 그때 pole 분할). cross-leaf 판정 — liver(양극 공존)는 유지됨.")
            # ★ PROCESS (tier='process' — neuropil / fiber tract 등)은 celltype이 아니라 scoring-based
            #   annotation이 다루는 조직 process다. clustering-based 결정 JSON에는 원칙적으로 나오지 않지만,
            #   혹시 process term_id가 실려 있으면 celltype 검사(REQUIRED_ABSENT_ONPANEL 등)를 건너뛴다(continue).
            #   process 탐지는 score-profile/apply/finalize로 하라는 안내를 warn으로 남긴다.
            _is_proc=((idx.get(tid,{}) or {}).get("T") in _PROC_TIERS)
            if _is_proc:
                add(lf,"PROCESS_IN_CLUSTERING","warn",
                    f"'{d.get('label')}'은 tier='process'(조직 process) — clustering-based annotation은 이를 라벨하지 않는다.",
                    "process(neuropil/fiber tract 등)는 scoring-based annotation(score-profile→score-apply→score-finalize)으로 찾으세요.")
                continue                                          # process → celltype 검사 skip
            elif not tid or (idx and tid not in idx):
                add(lf,"BAD_TERM_ID","error",
                    f"celltype인데 term_id가 비었거나 온톨로지에 없음(term_id='{tid}').",
                    "유효한 term_id로 고치거나 결정을 바꾸세요. helper가 임의로 Unknown 처리하지 않습니다. (조직 구획이면 label='neuropil' 사용 가능.)")
            # ★ 고유 필수명찰 검사(REQUIRED_ABSENT_ONPANEL): 이 term의 비-pan required 중 on-panel인 게
            # 있으면 적어도 하나는 present여야. UBC=Eomes(on-panel)인데 absent면 보조마커(Grm1) 오배정 →
            # flag. CA1=required 전부 off-panel → 검사 제외(보조명찰 Fibcd1로 식별, §5 panel-blind).
            if tid in idx:
                t_=idx[tid]; ogt_l={_U(g):v for g,v in (p.get("ontology_gene_table") or {}).items()}
                up_l={_U(r[0]):r[1] for r in (p.get("up") or [])}
                # ★ GLIAL_MISLABEL 백스톱 (2026-06-16): up(vs-rest) top8이 glial 마커 우세(≥3)+뉴런/pan-neuronal 마커 0인데
                #   뉴런 term으로 라벨 → fiber-tract/혈관/미세아교/뇌막 오염 조각을 부모-제약 때문에 뉴런 subtype으로
                #   force-fit한 것(~19k cell 버그). 선택 term 자신이 glial이면 면제. up은 vs-rest라 ambient(non-enriched)
                #   myelin엔 안 걸림(cl9 L6b: up=Cplx3/Nxph3이라 안 뜸). 진짜 뉴런은 up이 정체마커라 glial<3.
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
                # 면제: 선택 term의 자기 마커가 up에 있으면(진짜 그 뉴런인데 glial bleed) skip.
                _own_in_up=len({_U(g) for g in (t_.get('R') or [])+(t_.get('S') or [])+(t_.get('DF') or [])} & set(_ut))>=1
                if len(_ng)>=3 and len(_nm)==0 and not _own_in_up and not _chosen_glial:
                    # ★ Fix B (2026-06-23): up의 glial 마커를 sub-family로 분류해 '어떤' glia인지 제안.
                    #   agent가 vascular/oligo 우세인데 microglia로 과특정하던 문제(microglia 75 = Stab1/Sparc 등
                    #   vascular-mixed인데 Cx3cr1 near-noise를 보고 microglia 확정) 방지. 순수 메시지(결정 불변).
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
                    _hint=(f"up이 가리키는 우세 glial sub-family = **{_dom}** ({_fam_str}). 이 sub-family의 celltype으로 재라벨하세요. "
                           "다른 sub-family 마커가 near-noise(절대발현 ~0)면 그쪽으로 확정하지 마세요 — 예: vascular/oligo 마커가 up에 우세한데 "
                           "Cx3cr1/Tmem119만 rank-ON(절대 ~0)이면 microglia로 과특정 금지, 우세 sub-family 또는 그 계통 class로.")
                    add(lf,"GLIAL_MISLABEL","error",   # error=finalize 차단(robust: glial up은 풍부/뚜렷, own-marker·pan-neuronal 면제로 오발동≈0)
                        f"up(vs-rest) top8이 glial 마커 우세({_ng}) + 뉴런/pan-neuronal/자기-정체 마커 0인데 '{t_.get('L')}'(뉴런)로 라벨 — fiber-tract/혈관/glia 오염 조각 가능성.",
                        _hint+" (graph_evidence가 부모-제약으로 뉴런만 보였어도 up이 glial이면 glia. 진짜 뉴런이면 up의 뉴런 정체/transmitter 근거 제시.)")
                # ★ INTERNEURON_MISLABEL 백스톱 (2026-06-16): up(vs-rest) top6에 GABA transmitter(Gad1/2/Slc32a1)가 ENRICHED
                #   + interneuron subclass 마커(Pvalb/Sst/Vip/Erbb4/Pthlh…) ≥1인데 라벨 term이 glut(자기 마커에 GABA
                #   transmitter 없음) → 피질 glut에 박힌 인터뉴런 조각이 ambient Slc17a7로 lineage=glut 오판돼 L2/3 IT 등
                #   으로 force-fit된 것(B2). 진짜 glut 뉴런은 Gad가 vs-rest enriched될 수 없으므로 오발동≈0. GABAergic
                #   term(자기 R에 Gad 보유)은 _is_glut_term=False라 면제.
                _IN_M={'PVALB','SST','VIP','NPY','ERBB4','PTHLH','RELN','CCK','SNCG','CALB2','CXCL14','HTR3A','ADARB2','LHX6','CNR1','KCNC1','KCNC2','KCNC3'}
                _GABA_TX={'GAD1','GAD2','SLC32A1'}
                _term_mk={_U(g) for g in (t_.get('R') or [])+(t_.get('S') or [])+(t_.get('DF') or [])}
                _gaba_up=[g for g in _ut[:6] if g in _GABA_TX]; _in_up=[g for g in _ut[:6] if g in _IN_M]
                # warn(error 아님): L4/L5는 실제 Pvalb 인터뉴런이 조밀해 ambient로 ≥2 뜰 수 있어 강제 재라벨은 위험 →
                #   agent가 판단(명확한 Pvalb-FS/CGE 조각만 재라벨). GABA transmitter enriched + 인터뉴런 subclass ≥2.
                _is_glut_term=not (_term_mk & (_GABA_TX|{'SLC6A5'}))
                if _gaba_up and len(_in_up)>=2 and _is_glut_term and not _chosen_glial:
                    add(lf,"INTERNEURON_MISLABEL","warn",
                        f"up(vs-rest) top6에 GABA transmitter({_gaba_up}) + 인터뉴런 마커({_in_up})가 enriched인데 '{t_.get('L')}'(glut)로 라벨 — ambient Slc17a7로 lineage 오판된 피질 GABA 인터뉴런 조각 가능성.",
                        "up이 인터뉴런 우세하면 cortical GABAergic interneuron(Pvalb+/Sst+/Vip+ 등)으로 재라벨. L4/L5처럼 glut 정체가 분명하고 Gad가 ambient면 유지(진짜 glut 근거 제시).")
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
                        f"엔진이 surface한 anchor('{_topa}')를 쓰거나, 이 term의 OWN 고유마커가 명확히 우세함을 hp/supporting으로 입증하세요.")
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
                    "specific으로 확정하지 마세요 → lineage 분명하면 class-floor(subclass_resolved=false), 불분명하면 Unknown. hp/supporting에서 그 type 마커가 실제 우세하면 그때만 유지(§5 spec_pass).")
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
                        _cls=cu.get("resolves_to")   # 공통 조상 class (helper가 계산)
                        add(lf,"CONFUSION_UNRESOLVED","warn",
                            f"'{chosen}'와 짝({partners})이 공유 마커로만 anchor, 양쪽 고유 마커 모두 OFF/off-panel — 마커로 분리 불가({pair}).",
                            (f"→ 둘 중 하나로 단정 말고 공통 상위 class '{_cls}'로 결정하세요(예: FC 변별마커 off-panel → CA2 단정 대신 '{_cls}'). "
                             f"discriminator(위치 등)가 분명하면 그 type 유지." if _cls else
                             "→ Unknown으로 결정하세요(둘 중 하나로 단정 금지)."))
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
    _full_idx=_load_ontology_index(out) or {}
    idx=_full_idx.get("terms",{})
    # ★ ContextAxis 탈출구(2026-06-22): 안 갈라지는 mixture가 선언된 공간축(zonation/layer) 연속체면 무한 재분할
    #   대신 축 pole로 라벨하고 종료. ca축의 pole subtype 라벨 집합을 미리 만든다(소문자).
    _CA=_full_idx.get("ca") or []
    def _axis_pole_labels(ax):
        out_s=set()
        for p in (ax.get("poles") or []):
            for v in (p.get("subtypes") or {}).values():
                if v: out_s.add(str(v).strip().lower())
        return out_s
    def _axis_match(lineage, match_labels):
        """lineage/region/anchor 라벨이 어떤 ca축의 applies_to에 맞으면 그 축 반환(region/anchor-aware, _context_axis_eval과 동일 원리)."""
        cl={str(lineage).strip().lower()} | {str(x).strip().lower() for x in match_labels if x}
        for ax in _CA:
            if ax.get("lineage") and str(ax["lineage"]).strip().lower()!=str(lineage).strip().lower(): continue
            if any(str(a).strip().lower() in cl for a in (ax.get("applies_to") or [])): return ax
        return None
    lab2D={}                                                            # label→disease_context(첫 매칭)
    for t in idx.values(): lab2D.setdefault(t.get("L"),t.get("D"))
    ncnt=Counter(st.get("cluster_paths") or [])
    cands=[]; axis_cont=[]; converged=[]
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
        # mixture subtype별 region-group
        _mix_regs=[(m, idx.get(_reg_of(_lab2tid[m["label"]]),{}).get("L")) for m in _mix if _lab2tid.get(m.get("label"))]
        if _dom:
            _mix_inreg=[m for m,r in _mix_regs if r in _dom]
        else:
            # ★ dominant_region 비었을 때(2026-06-17): 피질 laminar 혼합 클러스터는 layer가 여러 region-exclusive
            #   마커를 약하게 공유해 region-scoring이 dominant_region을 못 정하는 경우가 많다(L2/3·L4·L5가 다
            #   neocortical인데 _dom=[]). 그러면 옛 'mixture가 dominant_region 안' 조건이 전부 탈락 → 피질이 split
            #   후보로 안 올라와 레이어가 generic으로 붕괴. 대신 mixture subtype들이 **같은 region-group ≥2개**로
            #   모이면(예: 모두 neocortical) 그 region의 진짜 혼합으로 보고 surface. cross-region ambient(CA1 안의
            #   L2/3 단발)는 같은 region 2개를 못 채워 제외 → 오발동 가드 유지.
            _rc=Counter(r for m,r in _mix_regs if r)
            _common={r for r,c in _rc.items() if c>=2}
            _mix_inreg=[m for m,r in _mix_regs if r in _common]
        mixture_trigger=_lin_ok and ((_cu.get("action")=="split") or (len(_mix_inreg)>=2))
        # ★ BIMODAL backstop trigger (2026-06-23): per-cell anti-correlated sibling programs (≥2) the
        #   mean-based mixture detector missed (balanced sibling-region mix). Region-independent, so it
        #   needs no _mix_inreg region check; _lin_ok still guards cross-lineage ambient.
        _bim=p.get("bimodal_subtypes") or []
        bimodal_trigger=_lin_ok and len(_bim)>=2
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
            # ★ REGION-FLOOR descend backstop (2026-07-15): when the leaf is REGION-ESTABLISHED (dominant_region
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
            if len(strong)<2 and not mixture_trigger and not bimodal_trigger: continue   # ≥2 자식(region-floor: 그 region 고유 subtype) OR graph_evidence 혼합 OR per-cell bimodal
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
            if len(strong)<1 and not mixture_trigger and not bimodal_trigger: continue   # 형제 ≥1개 OR graph_evidence 혼합 OR per-cell bimodal
            split_kind="siblings"
        # k = 분리될 distinct type 추정: DF-confirmed 형제(strong) ∪ graph_evidence.mixture_subtypes(희석 공존) ∪ bimodal.
        if not strong and _mix_inreg: strong=[m.get("label","") for m in _mix_inreg if m.get("label")]  # 혼합 trigger인데 DF-자식 없으면 in-region mixture 라벨
        if not strong and _bim: strong=[m.get("label","") for m in _bim if m.get("label")]   # bimodal-only trigger → per-cell 분리된 sibling 라벨
        nexp=max(len(strong), len(_mix), len(_bim), 2)              # split 후보면 최소 2-way
        depth=lf.count(".")                                          # 분할 깊이 = 이미 시도한 retry 수
        mix_labs={str(x).strip().lower() for x in strong} | {str((m.get("label") or "")).strip().lower() for m in _mix}
        mix_labs.discard("")
        # ★① ContextAxis 탈출구: 이미 1번+ 쪼갰는데(depth≥1) 여전히 같은 mixture고, 그게 선언된 공간축의 pole들이면
        #   = 연속체라 더 쪼개도 안 갈라짐 → 재분할 후보에서 빼고 axis_continuum으로 종료(caller가 context_axis pole로 라벨).
        _anchor_labs=[a.get("label") for a in (_ge.get("anchors") or []) if a.get("label")]
        _ax=_axis_match(_ge.get("lineage"), list(_dom)+_anchor_labs+[d.get("label","")])
        if depth>=1 and _ax is not None:
            _poles=_axis_pole_labels(_ax)
            if mix_labs and mix_labs<=_poles:                        # mixture가 전부 그 축의 pole subtype → 연속체
                axis_cont.append({"cluster_path":lf,"n_cells":n,"label":d.get("label",""),
                                  "axis":_ax.get("name"),"poles":sorted(_poles),
                                  "action":"label by context_axis pole (stop re-splitting — spatial continuum won't separate)"})
                continue
        # ★② Δ-separation 종료: 이미 쪼갠 leaf(depth≥1)인데 mixture가 부모와 동일(분리 개선 Δ≈0)이면 수렴 → 재분할 무익 → 종료(class_floor).
        if depth>=1:
            _par=lf.rsplit(".",1)[0]
            _pmix={str((m.get("label") or "")).strip().lower() for m in ((prof.get(_par,{}).get("graph_evidence") or {}).get("mixture_subtypes") or [])}
            _pmix.discard("")
            if _pmix and mix_labs and mix_labs==_pmix:               # 부모와 똑같은 혼합 → split이 분리 못 함 → 종료
                converged.append({"cluster_path":lf,"reason":"mixture unchanged vs parent (Δ separation ≈ 0)"})
                continue
        # ★③ γ-escalation: 재시도(depth)마다 Leiden resolution을 ×1.5 상향(상한 2.0) — 첫 시도 낮게, 안 갈라지면 올려 수렴.
        start_res=round(min(2.0, _res_for(nexp) * (1.5 ** depth)), 2)
        cands.append({"cluster_path":lf,"n_cells":n,"label":d.get("label",""),"term_id":tid,
                      "subclass_resolved":resolved,"split_kind":split_kind,
                      "density_pct":dens.get("pct"),"low_frac":lfrac,"split_depth":depth,
                      "reason_code":d.get("reason_code",""),"resolvable_children":strong[:8],
                      "n_expected":nexp,"start_resolution":start_res})
    cands.sort(key=lambda c:-c["n_cells"])
    # ★ FLOORED-OVER-ANCHOR backstop (2026-07-15): the agent floored a leaf to a CLASS/region-group ancestor
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
    # ★ DF-PRESENT SUBTYPE backstop (2026-07-15): a leaf floored to a class/group whose `df_present` names a
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
    # ★ MUST-DESCEND — the un-bypassable "always descend before floor" enforcement (2026-07-15). The other
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
            "floored_over_anchor":floored_over,                     # ★ RELABEL(분할 아님): clear_top+excl_on≥2 subtype anchor를 조상으로 floor한 leaf
            "df_present_subtype":df_present_sub,                    # ★ RELABEL: df_present(특이 defining fold-ON) same-lineage subtype이 ambient에 묻혀 floor된 leaf
            "must_descend":must_descend,                            # ★ UN-BYPASSABLE: never-descended 대형 class/region floor → descend 강제(--accept-large 무효, --force만)
            "axis_continuum":axis_cont,                              # ★① 연속체: 재분할 말고 context_axis pole로 라벨·종료
            "converged_stop":converged,                             # ★② Δ-separation 종료: 부모와 동일 혼합 → 재분할 무익
            "max_split_depth":max_depth,"large_min_n":large_min_n,
            "note":"resolution은 *예측*이 아니라 *수렴*이다(γ↔#클러스터는 데이터 의존, 닫힌 공식 없음 — Traag/CPM). "
                   "start_resolution=expected #type로 스케일(0.25·k, 하한 0.1·상한 2.0; k=2→0.5,4→1.0,6→1.5; 옛 0.12·n·상한0.5 폐기). "
                   "낮게 시작(과분할 노이즈 회피), 잔여 혼합(SHOULD_SPLIT_MIXTURE/DF_MULTI_SUBTYPE)이면 γ×1.5 재분할로 수렴↑; "
                   "구별 안 되는 자식(split_no_gain)은 같은 라벨끼리 병합·종결. per-cluster는 cand.start_resolution. "
                   "★종료/탈출(무한루프 방지): start_resolution은 split_depth로 γ×1.5^depth 자동 상향(재시도=depth, 상한 depth≤max_split_depth + γ≤2.0); "
                   "depth≥1 leaf가 부모와 동일 혼합이면 converged_stop(분리 Δ≈0→종료); 안 갈라지는 혼합이 선언된 ContextAxis 연속체면 "
                   "axis_continuum(재분할 대신 §4.3 pole로 라벨·종료). 이 둘은 split_candidates에서 빠져 HOLD 안 일으킴(현 라벨로 확정)."}

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
    # ★ 무한루프 방지 종료(재분할 안 함, 현 라벨로 확정): ContextAxis 연속체 / Δ-separation 수렴
    for c in (review.get("axis_continuum") or []):
        print(f"[finalize] axis-exit: {c['cluster_path']} '{c['label']}' = '{c['axis']}' 연속체 → 재분할 중단, context_axis pole로 종결.", flush=True)
    for c in (review.get("converged_stop") or []):
        print(f"[finalize] converged: {c['cluster_path']} {c['reason']} → 재분할 중단.", flush=True)
    # ★ MUST-DESCEND HARD BLOCK (2026-07-15): never-descended 대형 class/region floor는 `--accept-large`로 우회
    #   못 한다 — descend를 실제로 시도해야만 finalize 가능(정말 안 갈라지면 그때 sub-leaf가 depth≥1이라 이 블록에서
    #   빠짐). "init→judge→floor→finalize(round 1, descend 0회)" 실패를 원천 차단. 정말 강행하려면 --force + rationale.
    md=review.get("must_descend") or []
    if md and not a.force:
        print(f"[finalize][BLOCK] descend를 한 번도 안 한 대형 class/region floor {len(md)}개(합 {sum(c['n_cells'] for c in md)} cell). "
              f"h5ad 미작성 — **--accept-large로 우회 불가**(descend 필수).\n"
              f"  floor는 'descend해서 안 갈라졌을 때만' 획득합니다(SKILL R8). 이들은 aggregate에서 subtype이 희석돼 안 보일 뿐,\n"
              f"  descend하면 나옵니다(예: astrocyte→Fibrous는 whole-cluster Gfap auc0.53이나 split 후 0.99).\n"
              f"   • 각 leaf를 `descend --node '<그 leaf의 class/region 라벨>'` → 재판정 → finalize. 정말 안 갈라지면 그때 자동 floor.\n"
              f"   • 정당한 이유로 강행: --force (+ rationale에 근거). 상세: {rp}", flush=True)
        for c in md[:40]:
            print(f"  [descend!] {c['cluster_path']} n={c['n_cells']} '{c['label']}' — 이 노드의 KG 자식으로 descend 필요", flush=True)
        sys.exit(3)
    # ★ FLOORED-OVER-ANCHOR HOLD (2026-07-15): RELABEL, 분할 아님. 엔진이 clear_top + excl_on≥2(자기 고유
    #   exclusive 마커 ON, 강한 neg 누출 없음)로 특이 subtype을 지목했는데 agent가 그 조상(class/region-group)으로
    #   floor한 leaf. 재클러스터링 없이 라벨만 anchor로 바꾸면 되므로 과분할 위험 0. dense Stereo-seq에서 agent가
    #   descend 과분할을 겁내 L5/L6/subiculum 등을 generic으로 덮는 회귀를 막는다.
    fova=review.get("floored_over_anchor") or []
    if fova and not a.accept_large and not a.force:
        print(f"[finalize][HOLD] clear_top+excl_on≥2 특이 anchor를 조상으로 floor한 leaf {len(fova)}개 감지(합 {sum(c['n_cells'] for c in fova)} cell). h5ad 미작성.\n"
              f"  이건 분할이 아니라 RELABEL 문제입니다 — 엔진이 이미 특이 subtype(자기 고유 exclusive 마커 ≥2 ON)을 clear_top으로 지목했는데 상위 class/region으로 덮였습니다.\n"
              f"   • 각 leaf의 결정을 anchor subtype으로 교체(label+term_id)하거나, floor할 정당한 이유(negative 누출/confusion 등)를 rationale에 남기고 finalize ... --accept-large.\n"
              f"  (재클러스터 불필요 — 결정 JSON의 label/term_id만 수정 후 finalize 재실행.) 상세: {rp}", flush=True)
        for c in fova[:40]:
            print(f"  [relabel?] {c['cluster_path']} n={c['n_cells']} '{c['floored_label']}' → '{c['anchor_label']}' "
                  f"(excl_on={c['excl_on']}: {', '.join(c['excl_markers'][:4])})", flush=True)
        sys.exit(3)
    # ★ DF-PRESENT SUBTYPE HOLD (2026-07-15): same-lineage subtype whose SPECIFIC defining marker is fold-ON
    #   (df_present) but was buried under a cross-lineage ambient anchor → RELABEL, not split.
    dfs=review.get("df_present_subtype") or []
    if dfs and not a.accept_large and not a.force:
        print(f"[finalize][HOLD] df_present에 특이 defining marker가 fold-ON인 same-lineage subtype이 있는데 상위 class/group으로 floor된 leaf {len(dfs)}개 감지(합 {sum(c['n_cells'] for c in dfs)} cell). h5ad 미작성.\n"
              f"  RELABEL 문제 — subtype의 고유 정체(특이 defining 마커 ≥3× 발현)는 감지됐으나, tiny 파편서 cross-lineage ambient(혈관/면역)가 top anchor를 가로채 라벨이 안 됐습니다.\n"
              f"   • 그 subtype으로 라벨 교체하거나(defining 마커 실측 확인 후), 오염이 진짜 지배적이면 rationale에 남기고 finalize ... --accept-large.\n"
              f"  (재클러스터 불필요 — label/term_id만 수정.) 상세: {rp}", flush=True)
        for c in dfs[:40]:
            print(f"  [relabel?] {c['cluster_path']} n={c['n_cells']} '{c['floored_label']}' → '{c['subtype_label']}' "
                  f"(specific defining fold-ON: {', '.join(c['specific_df'][:4])})", flush=True)
        sys.exit(3)
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
    # ★ 컬럼명(2026-06-25): 표준 annotation = `cellman_annotation`; 사용자 상호작용(post-standard, SKILL §3.5
    #   interactive phase) 결과 = `cellman_free_annotation` (--column으로 caller가 지정). free finalize는 이미
    #   쓰인 annotated h5ad를 읽어 표준 컬럼을 보존한 채 free 컬럼만 추가한다.
    col=getattr(a,"column",None) or "cellman_annotation"
    out_h5ad=a.output or st.get("output") or os.path.join(a.out, re.sub(r"\.h5ad$","",os.path.basename(st["h5ad_path"]))+"_annotated.h5ad")
    src=out_h5ad if os.path.exists(out_h5ad) else st["h5ad_path"]   # 기존 annotated가 있으면 이전 컬럼 보존
    adata=ad.read_h5ad(src)
    cps=np.array(st["cluster_paths"], dtype=object)
    ann=np.full(adata.n_obs,"",dtype=object)
    # ★ STATE-AXIS label synthesis: append base-agnostic active cell states as ", <state>" (multiple allowed)
    #   → "CD8 T cell, proliferating, interferon-stimulated". Source per leaf = the agent's explicit
    #   dec["states"] override if set, else the engine's `state_axis` (ENRICHED program markers) from the
    #   latest profile. Only for `celltype` decisions; states already named in the base label are dropped.
    #   Modules without a state axis (sa empty) get nothing → fully additive.
    _state_of={}
    for _r in range(1, int(st.get("round",1))+1):
        _pf=os.path.join(a.out, f"cluster_gene_profiles_round{_r}.json")
        if os.path.exists(_pf):
            try:
                for _p,_e in json.load(open(_pf)).items():
                    _sx=[s.get("name") for s in (_e.get("state_axis") or []) if s.get("name")]
                    if _sx: _state_of[_p]=_sx
            except Exception: pass
    # ★ STATE LINEAGE GATE (2026-07-09): a lineage-restricted state (M2=myeloid, EMT/mesenchymal=epithelial,
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
    def _synth(path, base, dec):
        base_s=str(base); _bl=base_s.strip().lower()
        _is_proc=((_gterms.get((dec.get("term_id") or "").strip(),{}) or {}).get("T") in _PROC_TIERS)
        if ((dec.get("decision") or "")!="celltype" or base_s in ("","Background","Unknown")
                or _is_proc):   # scoring-based process 라벨엔 state 미부착
            return base_s
        states=dec.get("states")
        if states is None: states=_state_of.get(path, [])
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
    adata.obs[col]=ann                            # ★ 핵심 컬럼만 추가(helper 강등/변경 없음)
    adata.write_h5ad(out_h5ad)
    print(f"[finalize] wrote {out_h5ad} (input copy + obs['{col}'])", flush=True)
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
    sfx="_free" if "free" in col else ""           # free finalize는 표준 산출물을 덮지 않게 별도 이름
    csv_path=os.path.join(a.out,f"expert_judgment_annotation{sfx}.csv")
    pd.DataFrame(rows).to_csv(csv_path,index=False)
    print(f"[finalize] wrote {csv_path}", flush=True)
    # ★ annotation_rationale.md (SKILL §3.5) — 클러스터별 '어떻게 결정됐는가' 자연어 서술:
    #   결정 경로 + 근거(ON 마커·기각 후보) + 뒷받침 PMID(선택 term의 KG grounding=slim `GM`). English.
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
def _helper_cli():
    p=argparse.ArgumentParser(); sp=p.add_subparsers(dest="cmd"); sp.required=True
    pi=sp.add_parser("init"); pi.add_argument("--h5ad",required=True); pi.add_argument("--cluster-col",dest="cluster_col",required=True)
    pi.add_argument("--out",required=True)
    pi.add_argument("--species",default=None); pi.add_argument("--organ",default=None)
    pi.add_argument("--server-url",dest="server_url",default=None)   # 보통 불필요(기본 서버 자동연결)
    pi.add_argument("--ontology-url",dest="ontology_url",default=None)  # 전체 URL 직접지정(고급)
    pi.add_argument("--ontology",default=None); pi.add_argument("--output",default=None)
    pi.add_argument("--force",dest="force_input",action="store_true",
                    help="입력 행렬 검증(X=lognorm / var_names=symbol)을 경고로 낮추고 강행. 비표준 입력임을 인지한 경우만.")
    pi.add_argument("--disease",default=None,help="입력 데이터의 질병 컨텍스트(예: 'brain cancer'). 그 질병의 state term을 leaf별 disease_candidates로 surface(§7).")
    pi.add_argument("--use-rep",dest="use_rep",default="auto",
                    help="subcluster의 neighbors를 계산할 표현공간(obsm key, 예 X_scVI). "
                         "기본 'auto'=obsm에서 통합 임베딩 자동탐지(X_scVI>scANVI>harmony>pca). "
                         "'recompute'=normalize_total+log1p+PCA 재계산. 통합 임베딩이 있으면 그걸 쓰는 게 권장(정체성 분할 보존).")
    # ★ subcluster metric/n_neighbors: 초기 클러스터링(cluster_vectors)과 일관되게 맞추는 입력.
    #   subcluster recompute는 항상 log-normalize(+PCA).
    #   use_rep(통합 임베딩)이 잡히면 metric은 무시되고 그 임베딩+euclidean 사용(기존 동작).
    pi.add_argument("--subcluster-metric",dest="sc_metric",default=None,
                    help="subcluster kNN metric(correlation|euclidean). 미지정시 correlation(log-normalize 기본).")
    pi.add_argument("--subcluster-n-neighbors",dest="sc_n_neighbors",type=int,default=None,
                    help="subcluster kNN 이웃 수(n_neighbors). 미지정시 30(기존 기본). 작을수록 미세구조 분할↑, 클수록 그래프 부드러움.")
    pi.add_argument("--auc-pos",dest="auc_pos",type=float,default=0.65,
                    help="region-gated 안에서 positive 레이어/subtype 마커 ON 판정 auc bar (0.85=기존 strict, 0.65=quiet 레이어 복원)")
    pi.add_argument("--profile-cap",dest="profile_cap",type=int,default=150000,
                    help="AUC/rank 프로파일을 계산할 최대 cell 수(stratified subsample; 0=무제한). 650k 전체 rank는 ~11분이 "
                         "병목 → 150k cap이면 ~2분(마커는 population 신호라 정확도 보존, 희귀 클러스터는 전량 유지, 전역 "
                         "gmu/gsd는 full 데이터 그대로). 정확도 최우선이면 0(full).")
    pi.add_argument("--adaptive-floor",dest="adaptive_floor",action="store_true",
                    help="(opt-in) sparse 플랫폼용 데이터-유도 검출 floor: region-gated positive anchor에서 fon/pin 문턱을 "
                         "min(고정, gpin/ghi×F)로 낮춤(reference-free). 0.5µm SSAM의 저카피 defining(Rspo1→L4) 회수용. "
                         "scRNA/10µm엔 불필요(끄면 기존과 완전 동일). lineage/negative/region-scoring은 불변.")
    pi.add_argument("--af-F",dest="af_F",type=float,default=3.0,help="adaptive-floor 전역대비 배수 F (기본 3)")
    pi.add_argument("--af-Nmin",dest="af_Nmin",type=int,default=10,help="adaptive-floor 최소 검출 세포수 가드 (기본 10)")
    pi.set_defaults(f=cmd_init)
    ps=sp.add_parser("subcluster"); ps.add_argument("--out",required=True); ps.add_argument("--paths",default=None)
    ps.add_argument("--resolution",default="1.0")   # 수동 fallback만(옛 0.2는 under-split). 실제론 finalize start_resolution(_res_for=0.25·k)이 덮어씀
    ps.add_argument("--gpu",default=None); ps.add_argument("--ontology",default=None)
    # ★ Unknown carry-forward (2026-06-27): --repool-unknown이면 --paths 대신 현 라운드 Unknown-결정 leaf들의 셀을
    #   전부 모아(cross-parent) 하나의 풀로 Leiden 재클러스터한다. 흩어진 동일-type Unknown 조각이 다시 뭉쳐 식별됨.
    ps.add_argument("--repool-unknown",dest="repool_unknown",action="store_true")
    ps.add_argument("--decisions",default=None,help="--repool-unknown일 때 현 라운드 _annotate_decisions.json (Unknown leaf 식별용)")
    ps.add_argument("--profile-cap",dest="profile_cap",type=int,default=150000)   # init과 동일(부모<cap이면 무영향)
    ps.add_argument("--auc-pos",dest="auc_pos",type=float,default=0.65)            # init과 동일
    ps.add_argument("--adaptive-floor",dest="adaptive_floor",action="store_true")  # init과 동일(라운드마다 명시 필요)
    ps.add_argument("--af-F",dest="af_F",type=float,default=3.0); ps.add_argument("--af-Nmin",dest="af_Nmin",type=int,default=10)
    ps.set_defaults(f=_hcmd_subcluster)
    # signal_split 제거(Leiden+absolute paradigm): 강제배정 오염 방지. 분할은 subcluster(Leiden) 전용.
    pc=sp.add_parser("check"); pc.add_argument("--out",required=True); pc.add_argument("--decisions",required=True); pc.set_defaults(f=cmd_check)
    pf=sp.add_parser("finalize"); pf.add_argument("--out",required=True); pf.add_argument("--decisions",required=True)
    pf.add_argument("--output",default=None); pf.add_argument("--force",action="store_true")
    pf.add_argument("--accept-large",dest="accept_large",action="store_true")        # 대형 leaf 분할 리뷰 통과(user가 NO)
    pf.add_argument("--max-split-depth",dest="max_split_depth",type=int,default=1)    # 분할 깊이 상한(무한루프 방지)
    pf.add_argument("--large-min-n",dest="large_min_n",type=int,default=1500)         # '대형' 기준 셀 수
    pf.add_argument("--column",default="cellman_annotation",                          # 표준=cellman_annotation
                    help="obs 컬럼명. 표준 annotation=cellman_annotation(기본); 사용자 상호작용 결과=cellman_free_annotation.")
    pf.set_defaults(f=cmd_finalize)
    a=p.parse_args(); a.f(a)


# ===================================================================================
# annotate-mode additions — marker-axis split (subcluster) + KG descent (descend)
# ===================================================================================
# ---------------------------------------------------------------- candidate marker axis (KG-aware)
def _term_full_markers(terms, tid):
    """term tid 의 FULL signature = 자신 ∪ 모든 조상(IS_A P-chain) 의 DF∪R (정규화 delta 복원).
    supporting/negative 제외(변별축은 정체 마커 DF∪R 로 구성; supporting 은 광역 공발현이라 분리 신호 아님)."""
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
    """slim-index 에서 node_label term 의 **직속 자식** 라벨들(P 에 node 의 tid 가 든 term).
    class/region-floor 하강(descend)의 후보 = 그 node 의 KG 자식(region 그룹 / subtype)."""
    nl = (node_label or "").strip().lower()
    tid = next((k for k, t in terms.items() if (t.get("L") or "").strip().lower() == nl), None)
    if tid is None: return []
    return sorted({t.get("L") for t in terms.values() if tid in (t.get("P") or []) and t.get("L")})

def _candidate_axis(terms, cand_tids, var_up):
    """cand 별 FULL 마커(_term_full_markers) → 변별축.
    제거: (a) 패널 밖, (b) **모든** cand 공유(=공통 게이트, 변별 아님)만. 측정-광역(global_on) 필터도,
    share 필터도 쓰지 않음 — 정규화·큐레이션된 KG 라 DF∪R 자체가 특이 신호이고, 광역-컷은 국소 변별자까지
    버릴 수 있어 제거함. 반환 (axis_genes[list], per_cand{tid:set})."""
    per = {tid: {g for g in _term_full_markers(terms, tid) if g.upper() in var_up} for tid in cand_tids}
    ncand = len(per)
    from collections import Counter
    cnt = Counter(g for gs in per.values() for g in gs)              # cand 별 1회씩(교집합 판정용)
    axis = [g for g, c in cnt.items() if not (ncand >= 2 and c >= ncand)]   # 전 cand 공유(공통 게이트)만 제거
    return sorted(axis), per

def _marker_axis_leiden(sub, axis_genes, gUp, resolution, nn, backend, mod):
    """sub(AnnData) 를 axis_genes 만 subset → centering-free 90pct 스케일(0 보존) → kNN → Leiden.
    ★ NO-SIGNAL GUARD (2026-07-09): 축 유전자를 하나도 안 켠 세포(all-zero row)는 클러스터에서 제외하고
    **'u'(unassigned)로 반환** → caller가 parent(bare class)로 floor. 이유: (1) all-zero 세포가 다수면
    kNN 그래프가 붕괴해 세포당 1클러스터로 degenerate(oligo 26504); (2) region 마커를 하나도 안 켠 세포를
    노이즈로 어느 region sub-leaf에 강제배정하면 그 region 라벨을 받아 **공간 leak**(thalamic이 피질에 산포)이
    난다. region 마커를 실제로 켠 세포만 region으로 가르고, 안 켠 세포는 정직하게 bare class로.
    반환 labels[str]('u'=미배정) (분리 실패/신호부족 시 None)."""
    cols = [gUp[g.upper()] for g in axis_genes if g.upper() in gUp]
    if len(cols) < 2 or sub.n_obs < 3: return None
    Xa = np.asarray(sub.X.todense() if hasattr(sub.X, "todense") else sub.X)[:, cols].astype(float)
    has_sig = (Xa > 0).any(axis=1)                                    # 축 신호가 있는 세포만 클러스터
    n_sig = int(has_sig.sum())
    if n_sig < 3: return None                                         # 축 신호 세포가 너무 적음 → 분리 불가(parent 유지)
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
    # --candidates: {path:[labels]} (per-path) 또는 [labels](전 path 공통). 없는 path → PCA fallback(annotate-mode 동일).
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
            print(f"[skip] {path} n={n}<3: 분할 연산 불가.", flush=True); continue
        sub = adata[mask].copy(); res = res_of(path)
        cands = [c for c in cand_of(path)]
        tids = [lab2tid.get(c.strip().lower()) for c in cands]; tids = [t for t in tids if t]
        lbls = None; mode = "pca"
        if len(tids) >= 2:                                            # ── MARKER-AXIS split (v2 핵심) ──
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
    # ── re-profile: helper.py 재사용(annotate-mode 판정 substrate 그대로) ──
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
    """class/region-floor leaf 를 KG node 의 **직속 자식(region→subtype)으로 무조건** marker-axis 하강.
    kg-mode 식 계층 하강 + annotate-mode 판정. graph_evidence 의 flat-multiplicity 를 **기다리지 않는다** —
    region-혼합 floor 에서 region 축으로 먼저 쪼개 sub-leaf 를 region-농축시키면, 재판정이 비로소
    dominant_region/anchor 를 잡아 region/subtype 을 해상(층 마커가 희석에서 풀림). 자식 후보를 marker
    axis 로 써서 subcluster 코어에 그대로 위임. 반복(node→children→그 child node→…)은 agent 가 SKILL 대로.

    사용: helper.py descend --out O --paths '["cl3"]' --node "glutamatergic neuron"
      → node 의 자식(neocortical/thalamic/hippocampal… glut)을 marker-axis 후보로 subcluster.
    STOP 조건(agent): 하강해도 sub-leaf 가 전부 부모와 같은 class 로 재판정되면 = 진짜 floor → 멈춘다."""
    idx = _load_ontology_index(a.out); terms = (idx or {}).get("terms", {})
    kids = _children_of(terms, a.node)
    if len(kids) < 2:
        print(f"[descend] node '{a.node}' has <2 KG children in slim-index — nothing to descend."); return
    paths = json.loads(a.paths)
    a.candidates = json.dumps({p: kids for p in paths})       # KG 자식 = marker-axis 후보 → subcluster 코어 재사용
    print(f"[descend] node '{a.node}' → {len(kids)} KG children as marker-axis candidates: {kids[:6]}{'…' if len(kids)>6 else ''}", flush=True)
    cmd_subcluster(a)

# ---------------------------------------------------------------- scoring-based helpers
def _score_ctx_for(a):
    """slim-index를 받아 _score_targets_ctx(idx, targets)를 돌린다. --targets(JSON list, 선택)=임의 KG class.
    반환 (idx, ctx, targets)."""
    import json
    url=None if a.ontology else f"{_resolve_server(a.server_url)}/module/{a.species}/{a.organ}/slim-index"
    idx=_fetch_slim_index(a.out, url, a.ontology)
    targets=None
    if getattr(a,"targets",None):
        targets=json.loads(a.targets) if isinstance(a.targets,str) else a.targets
    return idx, _score_targets_ctx(idx, targets), targets


def _restrict_mask(A, restrict, column):
    """--restrict: obs[column] 값이 주어진 라벨 집합에 속하는 벡터만 스코어링 대상으로 (예: neuron들 안에서
    neuropil을 score). None/미지정 = 전 벡터. 반환: 전체길이 bool 마스크(None=전체)."""
    import numpy as np, json
    if not restrict: return None
    want={str(x).strip() for x in (json.loads(restrict) if isinstance(restrict,str) else restrict)}
    if column not in A.obs:
        raise SystemExit(f"[score] --restrict 인데 obs['{column}'] 없음. 있는 것: {list(A.obs.columns)}")
    m=np.isin(A.obs[column].astype(str).to_numpy(), list(want))
    if not m.any():
        raise SystemExit(f"[score] --restrict {sorted(want)} 에 해당하는 벡터가 obs['{column}']에 없음.")
    return m


def cmd_score_profile(a):
    """scoring-based annotation의 **판단 재료**를 산출한다. 라벨을 쓰지 않는다 — REAL 판정은 에이전트가 한다.
    대상 = --targets(임의 KG class 라벨) 또는 기본 tier='process' class. --restrict로 현 annotation 라벨
    subset에 한정 스코어링(예: neuron들 안에서 neuropil). 대상이 없으면 applicable=false."""
    import anndata as ad, json, os
    _idx, ctx, targets = _score_ctx_for(a)
    os.makedirs(a.out, exist_ok=True)
    dst=os.path.join(a.out,"_score_profile.json")
    if not ctx or not ctx.get("classes"):
        json.dump({"applicable":False,
                   "reason":(f"--targets {targets} 가 module {a.species}/{a.organ}의 어떤 term에도 매칭 안 됨"
                             if targets else f"module {a.species}/{a.organ}에 tier='process' term 없음")},
                  open(dst,"w"), ensure_ascii=False, indent=1)
        print(f"[score] 대상 class 없음 → applicable=false. -> {dst}", flush=True); return
    A=ad.read_h5ad(a.h5ad)
    rmask=_restrict_mask(A, getattr(a,"restrict",None), a.column)
    recs=_proc_scores(A, ctx, a.cand_pct, a.neg_pct, restrict=rmask)
    prof={"applicable":True,"h5ad":a.h5ad,"n_vectors":int(A.n_obs),"module":f"{a.species}_{a.organ}",
          "targets":(targets or "tier=process (default)"),
          "restrict":{"column":a.column,
                      "labels":(json.loads(a.restrict) if getattr(a,"restrict",None) else None),
                      "n_restricted":(int(rmask.sum()) if rmask is not None else int(A.n_obs))},
          "params":{"cand_pct":a.cand_pct,"neg_pct":a.neg_pct,
                    "rule":"후보=pos_score 상위 cand_pct%; 제거=neg_score 상위 neg_pct%. "
                           "score=표준화 composite(유전자별 z→평균→재표준화). "
                           "AND/OR(k-of-n) 금지: 수율이 유전자 개수에 지수적으로 끌려감. "
                           "검출(>0) 금지: blur 때문에 모든 마커가 ~100% '발현'됨."},
          "classes":[{k:v for k,v in r.items() if not k.startswith("_")} for r in recs]}
    json.dump(prof, open(dst,"w"), ensure_ascii=False, indent=1)
    print(f"[score] module={prof['module']} n={A.n_obs} targets={prof['targets']} "
          f"restrict={prof['restrict']['n_restricted']} classes={[r['label'] for r in recs]}", flush=True)
    for r in recs:
        if not r.get("usable"): print(f"  {r['label']:16s} 사용불가: {r['reason']}", flush=True); continue
        rm=r.get("removal")
        print(f"  {r['label']:16s} 후보 {r['pct_candidate']:5.2f}%  "
              + (f"제거 {rm['observed_pct']:5.2f}% (우연 {rm['chance_pct']:.0f}%, 배수 {rm['ratio']:.2f})  "
                 if rm else "제거축 없음  ")
              + f"-> {r['pct_final']:5.2f}%", flush=True)
        if r["pos_missing"] or r["neg_missing"]:
            print(f"     ! 패널 누락  pos={r['pos_missing']} neg={r['neg_missing']}", flush=True)
    print(f"-> {dst}", flush=True)


def cmd_score_apply(a):
    """에이전트의 REAL 판정을 obs['cellman_process']에 기록한다(어떤 대상 class인지; keep 벡터만, 나머지 '').
    --decisions {"<class label>": true|false}  true = 그 class를 REAL로 인정. score-finalize가 이를 KG typed
    subtype으로 세분화해 최종 라벨을 --column에 쓴다. (background/cellman_class 파티션 없음 — 2026-07-21.)"""
    import anndata as ad, numpy as np, json, os, pandas as pd, collections
    _idx, ctx, targets = _score_ctx_for(a)
    if not ctx or not ctx.get("classes"):
        raise SystemExit("[score-apply] 대상 class 없음 — score-profile을 먼저 확인하세요.")
    A=ad.read_h5ad(a.h5ad)
    dec=json.loads(open(a.decisions).read()) if os.path.exists(a.decisions) else json.loads(a.decisions)
    rmask=_restrict_mask(A, getattr(a,"restrict",None), a.column)
    recs=_proc_scores(A, ctx, a.cand_pct, a.neg_pct, restrict=rmask)
    if "cellman_class" in A.obs: del A.obs["cellman_class"]      # 옛 파티션 컬럼 정리
    lab=np.array([""]*A.n_obs, dtype=object)
    for r in recs:
        if not r.get("usable"): continue
        if not dec.get(r["label"]):
            print(f"  [skip] {r['label']}: 에이전트가 REAL로 인정 안 함", flush=True); continue
        m=r["_mask"] & (lab=="")          # 먼저 인정된 class 보호(겹치면 첫 class 우선)
        lab[m]=r["label"]
    A.obs["cellman_process"]=pd.Categorical(lab)       # 어떤 대상 class인지 (neuropil / fiber tract / …; '' 아님)
    A.uns["cellman_score"]=json.dumps(
        {"params":{"cand_pct":a.cand_pct,"neg_pct":a.neg_pct},
         "targets":(targets or "tier=process (default)"),
         "restrict":{"column":a.column,
                     "n_restricted":(int(rmask.sum()) if rmask is not None else int(A.n_obs))},
         "decisions":dec,
         "classes":[{k:v for k,v in r.items() if not k.startswith("_")} for r in recs]},
        ensure_ascii=False)
    A.write_h5ad(a.h5ad)
    print("[score-apply] obs['cellman_process'] =", dict(collections.Counter(lab)), flush=True)
    print(f"  -> {a.h5ad}", flush=True)


def cmd_score_finalize(a):
    """score-apply가 쓴 obs['cellman_process'](어떤 대상 class)를 KG typed subtype으로 세분화해 최종 라벨을
    obs[--column](기본 cellman_annotation)에 쓴다 — process 벡터만 채우고 나머지 값은 그대로 둔다.
    세분화 = subtype 델타 composite(표준화)의 per-vector argmax가 ENRICHED(>0)면 그 typed subtype, 아니면
    bare class. (background 병합 없음 — 2026-07-21.)"""
    import anndata as ad, numpy as np, pandas as pd, json, os, collections
    _idx, ctx, targets = _score_ctx_for(a)
    A=ad.read_h5ad(a.h5ad)
    if "cellman_process" not in A.obs:
        raise SystemExit("[score-finalize] obs['cellman_process'] 없음 — score-apply를 먼저 돌리세요.")
    plab=A.obs["cellman_process"].astype(str).to_numpy()
    col=a.column
    if col in A.obs:
        ann=A.obs[col].astype(object).to_numpy()
    else:
        ann=np.array([""]*A.n_obs, dtype=object)
        print(f"[score-finalize] obs['{col}'] 없음 → 새로 만들고 process 벡터만 채움(나머지 '').", flush=True)
    X=A.X; X=np.asarray(X.todense()) if hasattr(X,"todense") else np.asarray(X)
    um={str(g).upper():i for i,g in enumerate(A.var_names)}
    def zc(genes):
        gi=[um[g] for g in genes if g in um]
        if not gi: return None
        return np.column_stack([(X[:,i]-X[:,i].mean())/(X[:,i].std()+1e-9) for i in gi]).mean(1)
    subtype=np.array([""]*A.n_obs, dtype=object)
    cls_children={c["label"]:c.get("children",[]) for c in (ctx.get("classes") or [])}
    for cl_label in set(plab):
        if not cl_label: continue
        mask=(plab==cl_label)
        cand=[]
        for k in cls_children.get(cl_label,[]):
            z=zc([g.upper() for g in k.get("delta",[])])
            if z is not None: cand.append((k["label"],z))
        if cand:
            M=np.column_stack([z for _,z in cand]); names=[n for n,_ in cand]
            best=M.argmax(1); bestval=M.max(1)
            for i in np.where(mask)[0]:
                subtype[i]= names[best[i]] if bestval[i]>0 else cl_label   # ENRICHED면 typed, 아니면 bare
        else:
            subtype[mask]=cl_label
        ann[mask]=subtype[mask]
    A.obs[col]=pd.Categorical([str(x) for x in ann])
    A.write_h5ad(a.h5ad)
    pcount={k:v for k,v in collections.Counter(subtype[plab!=""]).items()}
    print(f"[score-finalize] obs['{col}']에 process 세분화 기록: {pcount}", flush=True)
    print(f"  -> {a.h5ad}", flush=True)


# ================================================================ annotate-mode CLI (single entry)
def main():
    """SINGLE entry point for annotate-mode. subcluster/descend = v2 marker-axis (this file);
    init/check/finalize = the inlined helper engine (_helper_cli), run IN-PROCESS (no subprocess)."""
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: helper.py {init|descend|subcluster|check|finalize|score-profile|score-apply|score-finalize} ...\n"
              "  [clustering-based annotation — 기본]\n"
              "  init | check | finalize = the inlined annotate engine (사용자 지정 cluster 컬럼을 입력받음)\n"
              "  descend    = v2 marker-axis descent into a KG node's children (region->subtype, unconditional)\n"
              "  subcluster = v2 marker-axis split by EXPLICIT named candidates (PCA fallback when 0/1 candidate)\n"
              "  [scoring-based annotation — 'score 기반으로 찾아' / process(neuropil·fiber tract) 탐지 전용]\n"
              "  score-profile  = positive-후보/negative-배제 판단재료 (라벨 안 씀; --targets 임의 KG class)\n"
              "  score-apply    = 에이전트 REAL 판정을 obs['cellman_process']에 기록\n"
              "  score-finalize = process를 KG typed subtype으로 세분화해 obs[--column]에 기록")
        return
    cmd, rest = argv[0], argv[1:]
    if cmd in ("score-profile", "score-apply", "score-finalize"):
        ap = argparse.ArgumentParser(prog=f"helper.py {cmd}")
        ap.add_argument("--h5ad", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--species", required=True)
        ap.add_argument("--organ", required=True)
        ap.add_argument("--server-url", dest="server_url", default=None)
        ap.add_argument("--ontology", default=None)
        # --targets: 스코어링할 KG class 라벨 JSON list(선택). 미지정 시 module의 tier='process' class가 기본 대상.
        ap.add_argument("--targets", default=None,
                        help='스코어링 대상 KG class 라벨 JSON list(예: \'["neuropil","fiber tract"]\'). '
                             "미지정=tier='process' class 기본.")
        # --column: (profile/apply) --restrict 기준 컬럼 · (finalize) 최종 라벨을 쓸 컬럼(기본 cellman_annotation).
        ap.add_argument("--column", default="cellman_annotation")
        if cmd in ("score-profile", "score-apply"):
            ap.add_argument("--cand-pct", dest="cand_pct", type=float, default=_PROC_CAND_PCT)
            ap.add_argument("--neg-pct", dest="neg_pct", type=float, default=_PROC_NEG_PCT)
            # --restrict: obs[--column] 값이 이 라벨들에 속하는 벡터만 스코어링(예: neuron들 안에서 neuropil).
            ap.add_argument("--restrict", default=None,
                            help='스코어링을 한정할 현 annotation 라벨 JSON list(예: \'["glutamatergic neuron","GABAergic neuron"]\').')
        if cmd == "score-apply":
            ap.add_argument("--decisions", required=True)   # {"<class label>": true|false}
        a = ap.parse_args(rest)
        {"score-profile":cmd_score_profile,"score-apply":cmd_score_apply,
         "score-finalize":cmd_score_finalize}[cmd](a)
    elif cmd in ("subcluster", "descend"):
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
        print(f"[helper] unknown command '{cmd}' — use init|subcluster|descend|check|finalize|score-profile|score-apply|score-finalize")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
