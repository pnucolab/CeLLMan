#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ref_postcheck.py — reference-guided annotate 모드의 ★진단/리포트★ (신규, helper.py 불변).

helper.py의 check/finalize는 reference를 모른다(제약된 인덱스만 받았을 뿐). 이 스크립트는 reference
태그(ref_prep.py가 박은 anchor/child/ancestor)를 이용해 helper가 못 하는 2가지를 *추가로* 한다:

  [check 단계]  ABOVE_REFERENCE — celltype이 'ancestor'(정답지보다 거친 조상)에 정지한 경우를 경고.
                정답지 해상도에 못 닿았다는 뜻 → 전문가가 재판정(자식이 해소되면 자식으로, 정말 안 되면
                유지). helper의 `_annotate_violations.json`에 **additive 병합**(원본 항목 보존)하여 기존
                expert agent가 그대로 읽게 한다. (별도 `_reference_violations.json`도 남김.)

  [finalize 후] coverage — 정답지(anchor) 대비 회수율 리포트(`_reference_coverage.json`):
                hit(정답지 그대로) / refined(자식으로 세분화 = cellman 장점) / above(조상에 머묾) /
                missing(데이터에 없던 기대 타입) / unknown·background(정답지 밖). claim 3(recovery) 지표.

ref 밖 term은 ref_prep가 이미 제거했으므로 celltype이 그걸 가리키는 일은 없다(있으면 helper가 이미
BAD_TERM_ID로 잡음). 여기서는 index에 있으나 ref 태그가 ancestor인 경우만 advisory로 처리.

사용:
  # check 직후(helper check가 _annotate_violations.json 쓴 뒤)
  python ref_postcheck.py --out {annotation_out} --mode check
  # finalize 후(최종 라벨 확정)
  python ref_postcheck.py --out {annotation_out} --mode coverage
"""
import argparse, json, os, sys

def _load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default

def _ref_tag(terms, tid):
    t = terms.get(tid)
    return t.get("ref") if t else None

def _nearest_anchor(terms, tid):
    """tid에서 P를 타고 올라가 처음 만나는 anchor term_id(없으면 None). 자식→정답지 매핑용."""
    seen = set(); stack = [tid]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        t = terms.get(x)
        if not t:
            continue
        if t.get("ref") == "anchor":
            return x
        for p in dict.fromkeys(t.get("P") or []):
            if p in terms:
                stack.append(p)
    return None

def _decisions_iter(decisions):
    for path, d in decisions.items():
        if isinstance(d, dict):
            yield path, d

def cmd_check(out):
    terms = (_load(os.path.join(out, "_ontology_index.json")) or {}).get("terms", {})
    decisions = _load(os.path.join(out, "_annotate_decisions.json"), {})
    ref_viol = []
    for path, d in _decisions_iter(decisions):
        if d.get("decision") != "celltype":
            continue
        tid = d.get("term_id") or ""
        if not tid:                       # neuropil 등 term 없는 celltype은 제외
            continue
        if tid not in terms:
            ref_viol.append({"cluster_path": path, "check": "OUT_OF_REFERENCE", "severity": "warn",
                             "message": f"term_id '{tid}'가 제약된 온톨로지에 없음(정답지 밖).",
                             "suggestion": "정답지 안의 type으로 재판정하거나 Unknown."})
            continue
        if _ref_tag(terms, tid) == "ancestor":
            anchors = sorted({terms[t].get("L", t) for t in terms if terms[t].get("ref") == "anchor"})
            ref_viol.append({"cluster_path": path, "check": "ABOVE_REFERENCE", "severity": "warn",
                             "message": (f"'{d.get('label')}'은 정답지보다 거친 상위 class입니다(정답지 해상도 미달). "
                                         "이 class의 정답지(anchor) 자식이 이 클러스터에서 해소되면 그 자식으로 재판정하세요."),
                             "suggestion": "자식 anchor의 고유 마커가 ON이면 그 type으로; 정말 해소 불가면 현 class 유지(advisory)."})
    json.dump(ref_viol, open(os.path.join(out, "_reference_violations.json"), "w"), ensure_ascii=False, indent=1)
    # helper의 violations에 additive 병합(원본 보존; 기존 expert가 읽도록). 중복(path,check) 방지.
    vpath = os.path.join(out, "_annotate_violations.json")
    base = _load(vpath, [])
    if isinstance(base, list):
        have = {(v.get("cluster_path"), v.get("check")) for v in base if isinstance(v, dict)}
        merged = list(base) + [v for v in ref_viol if (v["cluster_path"], v["check"]) not in have]
        json.dump(merged, open(vpath, "w"), ensure_ascii=False, indent=1)
        print(f"[ref_postcheck] ABOVE_REFERENCE/OUT_OF_REFERENCE {len(ref_viol)}건 → "
              f"_reference_violations.json + _annotate_violations.json(+{len(merged)-len(base)} 병합)", flush=True)
    else:
        print(f"[ref_postcheck] ABOVE_REFERENCE/OUT_OF_REFERENCE {len(ref_viol)}건 → _reference_violations.json "
              f"(주의: _annotate_violations.json 형식 비표준이라 병합 생략)", flush=True)
    if not ref_viol:
        print("[ref_postcheck] reference 위반 없음(모든 celltype이 정답지 anchor 또는 자식 해상도).", flush=True)

def cmd_coverage(out):
    idx = _load(os.path.join(out, "_ontology_index.json")) or {}
    terms = idx.get("terms", {})
    # 최종 라벨 소스: finalize 후엔 h5ad지만, 결정 JSON으로 동일하게 집계 가능(같은 라벨=같은 어노테이션).
    decisions = _load(os.path.join(out, "_annotate_decisions.json"), {})
    anchors = {t for t in terms if terms[t].get("ref") == "anchor"}
    anchor_label = {t: terms[t].get("L", t) for t in anchors}

    hit, refined, above, unknown, background, other = [], [], [], [], [], []
    assigned_anchor = set()
    for path, d in _decisions_iter(decisions):
        dec = d.get("decision"); tid = d.get("term_id") or ""; lab = d.get("label", "")
        rec = {"cluster_path": path, "label": lab, "term_id": tid}
        if dec == "Unknown":
            unknown.append(rec); continue
        if dec == "Background":
            background.append(rec); continue
        if dec != "celltype":
            other.append({**rec, "decision": dec}); continue   # needs_subclustering 등(중간상태)
        if not tid or tid not in terms:
            other.append({**rec, "decision": dec}); continue
        tag = _ref_tag(terms, tid)
        if tag == "anchor":
            assigned_anchor.add(tid); hit.append(rec)
        elif tag == "child":
            a = _nearest_anchor(terms, tid)
            if a:
                assigned_anchor.add(a)
            refined.append({**rec, "parent_anchor": (anchor_label.get(a) if a else None)})
        elif tag == "ancestor":
            above.append(rec)
        else:
            other.append({**rec, "decision": dec})
    missing = sorted(anchor_label[t] for t in anchors if t not in assigned_anchor)
    cov = {
        "n_reference_anchors": len(anchors),
        "n_anchors_assigned": len(assigned_anchor),
        "recovery_rate": round(len(assigned_anchor) / len(anchors), 3) if anchors else None,
        "counts": {"hit": len(hit), "refined_to_child": len(refined), "above_reference": len(above),
                   "unknown": len(unknown), "background": len(background), "other": len(other)},
        "missing_anchors": missing,                 # 데이터에서 안 잡힌 기대 타입
        "hit": hit, "refined_to_child": refined, "above_reference": above,
        "unknown": unknown, "background": background,
    }
    json.dump(cov, open(os.path.join(out, "_reference_coverage.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[ref_postcheck] reference coverage → _reference_coverage.json", flush=True)
    print(f"  anchors assigned {len(assigned_anchor)}/{len(anchors)} "
          f"(recovery {cov['recovery_rate']}); hit={len(hit)} refined={len(refined)} "
          f"above_ref={len(above)} unknown={len(unknown)} bg={len(background)}", flush=True)
    if missing:
        print(f"  missing(데이터에 없던 기대 타입) {len(missing)}: {missing[:20]}", flush=True)
    if above:
        print(f"  above_reference(정답지 해상도 미달) {len(above)}: {[r['label'] for r in above][:10]}", flush=True)

def main():
    ap = argparse.ArgumentParser(description="reference-guided annotate 진단/coverage")
    ap.add_argument("--out", required=True, help="annotation_out 디렉토리(helper와 동일)")
    ap.add_argument("--mode", choices=["check", "coverage", "both"], default="both")
    args = ap.parse_args()
    if not os.path.exists(os.path.join(args.out, "_ontology_index.json")):
        sys.exit(f"[ref_postcheck] {args.out}/_ontology_index.json 없음 — helper init을 먼저 실행했는지 확인.")
    if args.mode in ("check", "both"):
        cmd_check(args.out)
    if args.mode in ("coverage", "both"):
        cmd_coverage(args.out)

if __name__ == "__main__":
    main()
