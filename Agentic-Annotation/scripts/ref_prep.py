#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ref_prep.py — reference-guided annotate 모드의 ★온톨로지 가지치기★ (신규, helper.py 불변).

annotate-mode는 reference 없이 de novo로 celltype을 찾는다. 이 스크립트는 "정답지(reference)"가
있을 때, KG(slim-index)를 그 정답지 서브트리로 **가지치기**해서 helper.py가 `--ontology`로 받을 수
있는 *제약된 slim-index* 한 개를 만든다. 그 인덱스를 helper init에 주입하면 graph-descent·
subcluster mixture·merge·finalize 등 helper의 모든 로직이 **그대로 reference-제약**된다(코드 추가 0).

핵심 규칙(사용자 확정):
  1. 'term 제약만' — 유전자(마커)는 KG 것을 그대로 유지. reference의 발현은 안 씀(라벨 집합만).
  2. keep = R(정답지 term) ∪ descendants(R)(자식까지 전부 줌 → 신호 섞이면 subcluster로 세분화)
            ∪ ancestors(R)(root까지 — lineage root-R 게이트/region-grouping/path 보존에 필수).
  3. strict — 정답지 밖 KG term은 제거 → 매칭 안 되는 클러스터는 자연히 Unknown.

각 kept term에 'ref' 태그: anchor(정답지 직접) / child(정답지의 후손) / ancestor(정답지 위 조상).
helper는 알 수 없는 키(ref / 최상위 reference)를 무시하므로 안전. ref_postcheck.py가 이 태그로
ABOVE_REFERENCE 진단과 coverage 리포트를 만든다.

사용 예:
  # 서버에서 wing fetch + 라벨 리스트 정답지
  python ref_prep.py --species mouse --organ whole_brain --server-url http://10.125.208.41:8088 \
      --reference ref_labels.json --out-index restricted_index.json --out-report _reference_report.json
  # reference h5ad의 어노테이션 컬럼에서 정답지 추출
  python ref_prep.py --species human --organ thymus --server-url http://10.125.208.41:8088 \
      --reference-h5ad ref.h5ad --reference-col cell_type --out-index restricted_index.json
그 다음:
  python helper.py init --h5ad input.h5ad --cluster-col leiden --out {base}/annotation_out \
      --ontology restricted_index.json        # ← species/organ 불필요(로컬 인덱스가 서버 우회)
"""
import argparse, json, os, sys, csv, re

DEFAULT_ONTOLOGY_SERVER = "http://localhost:8088"   # helper.py와 동일 기본값(배포 시 env로 override)

def _resolve_server(server_url=None):
    return (server_url or os.environ.get("CELLONTO_SERVER_URL") or DEFAULT_ONTOLOGY_SERVER).rstrip("/")

def _norm(s):
    """라벨 정규화: 소문자 + 괄호주석 제거 + 공백 단일화."""
    return re.sub(r"\s+", " ", re.sub(r"\s*\(.*?\)", "", str(s))).strip().lower()

# ----------------------------------------------------------------- fetch full index
def _fetch_full_index(args):
    """전체 slim-index 확보. 우선순위: --ontology-url > --full-ontology(로컬) > species/organ+server."""
    if args.full_ontology and os.path.exists(args.full_ontology):
        idx = json.load(open(args.full_ontology))
        if "terms" not in idx or not isinstance(idx["terms"], dict):
            sys.exit(f"[ref_prep] --full-ontology 파일이 slim-index 형식이 아님(terms dict 없음): {args.full_ontology}")
        print(f"[ref_prep] full slim-index loaded from {args.full_ontology} ({len(idx['terms'])} terms)", flush=True)
        return idx
    if args.ontology_url:
        url = args.ontology_url
    else:
        if not (args.species and args.organ):
            sys.exit("[ref_prep] --species/--organ 또는 --ontology-url 또는 --full-ontology 중 하나가 필요합니다.")
        url = f"{_resolve_server(args.server_url)}/wing/{args.species}/{args.organ}/slim-index"
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            idx = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            sys.exit(f"[ref_prep][ERROR:WING_NOT_FOUND] 서버에 해당 wing이 없습니다 ({url}, HTTP 404). "
                     f"species/organ 확인 또는 그 wing 빌드를 요청하세요.")
        sys.exit(f"[ref_prep][ERROR:SERVER_HTTP_{e.code}] 온톨로지 서버 오류 {e.code} ({url}).")
    except urllib.error.URLError as e:
        sys.exit(f"[ref_prep][ERROR:SERVER_UNREACHABLE] 서버 연결 불가 ({url}; {e.reason}). "
                 f"--server-url / CELLONTO_SERVER_URL 확인.")
    if not idx or not idx.get("terms"):
        sys.exit(f"[ref_prep][ERROR:WING_EMPTY] wing 응답했으나 term이 비어있음 ({url}).")
    print(f"[ref_prep] full slim-index fetched from server ({url}) — {len(idx['terms'])} terms", flush=True)
    return idx

# ----------------------------------------------------------------- parse reference
def _parse_reference(args):
    """정답지 → 라벨/term_id 문자열 리스트(중복제거, 원본순서 보존)."""
    items = []
    if args.reference_h5ad:
        if not os.path.exists(args.reference_h5ad):
            sys.exit(f"[ref_prep] --reference-h5ad 없음: {args.reference_h5ad}")
        if not args.reference_col:
            sys.exit("[ref_prep] --reference-h5ad 사용 시 --reference-col(어노테이션 컬럼) 필요.")
        import anndata as ad
        adata = ad.read_h5ad(args.reference_h5ad, backed="r")
        if args.reference_col not in adata.obs:
            sys.exit(f"[ref_prep] obs에 '{args.reference_col}' 없음. 사용가능: {list(adata.obs.columns)}")
        vals = adata.obs[args.reference_col].astype(str).values
        for v in vals:
            if v and v.lower() not in ("nan", "none", ""):
                items.append(v)
        print(f"[ref_prep] reference labels from h5ad column '{args.reference_col}'", flush=True)
    elif args.reference:
        if not os.path.exists(args.reference):
            sys.exit(f"[ref_prep] --reference 없음: {args.reference}")
        if args.reference.lower().endswith(".json"):
            raw = json.load(open(args.reference))
            if isinstance(raw, dict):
                raw = raw.get("labels") or raw.get("terms") or raw.get("celltypes") or []
            items = [str(x) for x in raw]
        else:  # CSV/TSV — --reference-col 컬럼(미지정 시 첫 컬럼)의 모든 값
            with open(args.reference, newline="") as f:
                rdr = csv.reader(f, delimiter="\t" if args.reference.lower().endswith(".tsv") else ",")
                rows = list(rdr)
            if not rows:
                sys.exit(f"[ref_prep] --reference 파일이 비었음: {args.reference}")
            header = rows[0]
            col = 0
            if args.reference_col and args.reference_col in header:
                col = header.index(args.reference_col); body = rows[1:]
            elif args.reference_col:
                sys.exit(f"[ref_prep] CSV에 컬럼 '{args.reference_col}' 없음. 헤더: {header}")
            else:
                body = rows[1:] if any(not _looks_label(c) for c in header) else rows  # 헤더 추정
            for r in body:
                if col < len(r) and r[col].strip():
                    items.append(r[col].strip())
    else:
        sys.exit("[ref_prep] --reference(리스트 파일) 또는 --reference-h5ad(+--reference-col) 중 하나 필요.")
    # 중복제거(순서보존)
    seen = set(); uniq = []
    for x in items:
        k = x.strip()
        if k and k.lower() not in seen:
            seen.add(k.lower()); uniq.append(k)
    if not uniq:
        sys.exit("[ref_prep] 정답지에서 라벨을 하나도 추출하지 못했습니다.")
    return uniq

def _looks_label(s):
    return bool(re.search(r"[A-Za-z]", str(s)))

# ----------------------------------------------------------------- resolve labels -> term_ids
def _resolve_terms(full_idx, ref_items, alias=None):
    """라벨/term_id → term_id 집합. 매칭: term_id 직접 > 라벨 exact(정규화) > alias 맵.
    반환: (matched_tids set, mapping[ref_item]->tid|None, unmatched[ref_item])."""
    terms = full_idx["terms"]
    by_norm = {}
    for tid, t in terms.items():
        by_norm.setdefault(_norm(t.get("L", "")), tid)  # 첫 등장 우선
    alias = {(_norm(k)): v for k, v in (alias or {}).items()}
    matched = set(); mapping = {}; unmatched = []
    for it in ref_items:
        tid = None
        if it in terms:                         # 이미 term_id
            tid = it
        elif _norm(it) in by_norm:              # 라벨 정규화 exact
            tid = by_norm[_norm(it)]
        elif _norm(it) in alias:                # 수동 alias (라벨 또는 id)
            av = alias[_norm(it)]
            tid = av if av in terms else by_norm.get(_norm(av))
        mapping[it] = tid
        if tid:
            matched.add(tid)
        else:
            unmatched.append(it)
    return matched, mapping, unmatched

# ----------------------------------------------------------------- prune
def _children_index(terms):
    ch = {}
    for tid, t in terms.items():
        for p in dict.fromkeys(t.get("P") or []):       # P 중복부모 dedup
            if p in terms and p != tid:
                ch.setdefault(p, []).append(tid)
    return ch

def _descendants(roots, ch):
    seen = set(); stack = list(roots)
    while stack:
        x = stack.pop()
        for c in ch.get(x, []):
            if c not in seen:
                seen.add(c); stack.append(c)
    return seen      # roots 자신은 미포함

def _ancestors(nodes, terms):
    seen = set(); stack = list(nodes)
    while stack:
        x = stack.pop()
        for p in dict.fromkeys((terms.get(x, {}).get("P") or [])):
            if p in terms and p not in seen and p != x:
                seen.add(p); stack.append(p)
    return seen      # nodes 자신은 미포함

def _prune(full_idx, R):
    terms = full_idx["terms"]
    ch = _children_index(terms)
    desc = _descendants(R, ch)
    anc = _ancestors(R, terms)
    keep = set(R) | desc | anc
    kept_terms = {}
    tag_count = {"anchor": 0, "child": 0, "ancestor": 0}
    for tid in keep:
        t = dict(terms[tid])                       # 얕은 복제(원본 불변)
        if tid in R:
            tag = "anchor"
        elif tid in desc:
            tag = "child"
        else:
            tag = "ancestor"
        t["ref"] = tag
        tag_count[tag] += 1
        kept_terms[tid] = t
    out = {k: v for k, v in full_idx.items() if k != "terms"}   # species/organ/cp/oq/ca 보존
    out["terms"] = kept_terms
    out["reference"] = {
        "anchors": sorted(R),
        "anchor_labels": sorted(terms[t].get("L", t) for t in R),
        "n_terms_before": len(terms),
        "n_terms_after": len(kept_terms),
        "tag_count": tag_count,
    }
    return out, tag_count

# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="reference-guided annotate: KG slim-index 가지치기")
    # full index source
    ap.add_argument("--species", default=None)
    ap.add_argument("--organ", default=None)
    ap.add_argument("--server-url", dest="server_url", default=None)
    ap.add_argument("--ontology-url", dest="ontology_url", default=None)
    ap.add_argument("--full-ontology", dest="full_ontology", default=None, help="로컬 전체 slim-index json(서버 대신)")
    # reference source
    ap.add_argument("--reference", default=None, help="정답지 라벨/term_id 리스트 파일(.json/.csv/.tsv)")
    ap.add_argument("--reference-h5ad", dest="reference_h5ad", default=None, help="정답지를 추출할 reference h5ad")
    ap.add_argument("--reference-col", dest="reference_col", default=None, help="reference 어노테이션 컬럼(또는 CSV 컬럼명)")
    ap.add_argument("--alias", default=None, help="수동 라벨매핑 json {ref_label: kg_label_or_id}")
    # outputs
    ap.add_argument("--out-index", dest="out_index", required=True, help="제약된 slim-index 출력 경로(→ helper --ontology)")
    ap.add_argument("--out-report", dest="out_report", default=None, help="매칭/미매칭 리포트 json 경로")
    args = ap.parse_args()

    full = _fetch_full_index(args)
    ref_items = _parse_reference(args)
    alias = json.load(open(args.alias)) if (args.alias and os.path.exists(args.alias)) else None
    R, mapping, unmatched = _resolve_terms(full, ref_items, alias)
    if not R:
        sys.exit(f"[ref_prep][ERROR] 정답지 {len(ref_items)}개 중 KG에 매칭된 term이 0개입니다. "
                 f"라벨이 wing의 canonical_label과 일치하는지, 올바른 species/organ인지 확인하세요. 미매칭: {unmatched[:20]}")
    pruned, tag_count = _prune(full, R)

    with open(args.out_index, "w") as f:
        json.dump(pruned, f, separators=(",", ":"), ensure_ascii=False)
    report = {
        "n_reference_items": len(ref_items),
        "n_matched": len(R),
        "matched": {it: mapping[it] for it in ref_items if mapping.get(it)},
        "unmatched": unmatched,
        "tag_count": tag_count,
        "n_terms_before": pruned["reference"]["n_terms_before"],
        "n_terms_after": pruned["reference"]["n_terms_after"],
        "species": full.get("species"), "organ": full.get("organ"),
    }
    if args.out_report:
        with open(args.out_report, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"[ref_prep] DONE → {args.out_index}", flush=True)
    print(f"[ref_prep] matched {len(R)}/{len(ref_items)} reference items; "
          f"keep {tag_count['anchor']} anchor + {tag_count['child']} child + {tag_count['ancestor']} ancestor "
          f"= {pruned['reference']['n_terms_after']}/{pruned['reference']['n_terms_before']} terms", flush=True)
    if unmatched:
        print(f"[ref_prep][WARN] 미매칭 {len(unmatched)}개(KG에 없음 → build-mode 필요할 수 있음): {unmatched[:20]}", flush=True)

if __name__ == "__main__":
    main()
