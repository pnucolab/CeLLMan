#!/usr/bin/env python
"""kg_helper.py — annotate-KG-mode 백엔드 (Ontology-Gated Descent).

annotate-mode(leiden-only splitting) 와 별개인 새 mode. 핵심 차이:
  - 분할 = **OGD** (KG 계층을 게이트로: 부모 required=IN 게이트, 형제 defining=OUT 배제,
    자식 defining/required=변별). leiden 은 lineage 판정에만.
  - 강제배정 오염(annotate-mode 가 signal_split 을 버린 이유)을 **형제 경쟁 게이트 +
    centering-free scaling + Unknown/__other__** 로 차단.
  - identity 판정은 agent(SKILL)의 몫. helper 는 프로파일/OGD/경로만 계산.

경로 ID = annotate-mode 와 동일 `{parent}.{child}` (예 0.12.5).

commands:
  init     --h5ad --out [--cluster-col leiden] [--organ .. --species ..]
             leiden 로드 → 각 클러스터 lineage(class root) 판정(+mixed flag) → 프로파일 출력.
  descend  --out --paths '["3","7"]'
             지정 클러스터를 그 KG 노드의 자식으로 OGD 분할 → 새 경로 → 프로파일 출력.
  finalize --out --decisions <json> [--output <h5ad>]
             {path: celltype|Unknown} 를 obs['cellman_standard'] 로 저장.
"""
import argparse, json, os
import numpy as np
import scanpy as sc
from neo4j import GraphDatabase
from collections import Counter

NEO4J_URL = os.environ.get("CELLMAN_NEO4J", "neo4j://localhost:7687")
NEO4J_AUTH = ("neo4j", os.environ.get("CELLMAN_NEO4J_PW", "cellman2026"))

# ============================================================ KG 접근
class KG:
    def __init__(self, organ, species):
        self.d = GraphDatabase.driver(NEO4J_URL, auth=NEO4J_AUTH)
        self.organ, self.species = organ, species

    def q(self, cy, **kw):
        with self.d.session() as s:
            return s.run(cy, organ=self.organ, species=self.species, **kw).data()

    def lineage_roots(self):
        """class root(= class 부모 없는 최상위 class) = lineage 후보. 게이트 = required(없으면 defining)."""
        rows = self.q("""MATCH (t:Term {organ:$organ,species:$species}) WHERE t.tier='class'
            AND NOT (t)-[:IS_A]->(:Term {organ:$organ,tier:'class'})
            OPTIONAL MATCH (t)-[:HAS_REQUIRED_MARKER]->(r:Gene)
            OPTIONAL MATCH (t)-[:HAS_DEFINING_MARKER]->(d:Gene)
            RETURN t.label_original AS lab, collect(DISTINCT r.symbol) AS req,
                   collect(DISTINCT d.symbol) AS df""")
        out = {}
        for x in rows:
            req = [g for g in x["req"] if g]; df = [g for g in x["df"] if g]
            out[x["lab"]] = {"gate": req or df, "df": df}
        return out

    def gate_for(self, node):
        """하강 노드의 (IN 게이트=조상 required, OUT 배제=형제 defining, 변별=자식 df/req)."""
        ing = self.q("""MATCH (t:Term {organ:$organ,species:$species}) WHERE t.label_original=$node
            MATCH (t)-[:IS_A*0..6]->(a:Term {organ:$organ}) MATCH (a)-[:HAS_REQUIRED_MARKER]->(g:Gene)
            RETURN collect(DISTINCT g.symbol) AS r""", node=node)
        ing = ing[0]["r"] if ing else []
        outg = self.q("""MATCH (t:Term {organ:$organ,species:$species}) WHERE t.label_original=$node
            MATCH (t)-[:IS_A]->(p)<-[:IS_A]-(sib) WHERE sib<>t AND sib.organ=$organ
            MATCH (sib)-[:HAS_DEFINING_MARKER]->(g:Gene) RETURN collect(DISTINCT g.symbol) AS r""", node=node)
        outg = outg[0]["r"] if outg else []
        disc = self.q("""MATCH (t:Term {organ:$organ,species:$species}) WHERE t.label_original=$node
            MATCH (ch:Term)-[:IS_A]->(t) WHERE ch.organ=$organ
            OPTIONAL MATCH (ch)-[:HAS_DEFINING_MARKER]->(d:Gene)
            OPTIONAL MATCH (ch)-[:HAS_REQUIRED_MARKER]->(rq:Gene)
            RETURN ch.label_original AS ch, ch.tier AS tier, collect(DISTINCT d.symbol) AS df,
                   collect(DISTINCT rq.symbol) AS req ORDER BY ch""", node=node)
        for d in disc:
            d["df"] = [g for g in d["df"] if g]; d["req"] = [g for g in d["req"] if g]
        return ing, outg, disc

# ============================================================ scaling / score
def _dense(a, g):
    v = a[:, g].X
    return np.asarray(v.todense()).ravel() if hasattr(v, "todense") else np.asarray(v).ravel()

class Expr:
    """centering-free 99-percentile scaling 캐시 (0 보존)."""
    def __init__(self, a):
        self.a = a; self.cache = {}
    def col(self, g):
        if g not in self.a.var_names: return None
        if g not in self.cache:
            c = _dense(self.a, g)
            nz = c[c > 0]                                   # sparse 마커 폭발 방지: nonzero 기준 스케일
            scale = np.percentile(nz, 90) if nz.size else 1.0
            self.cache[g] = c / (scale + 1e-9)
        return self.cache[g]
    def mat(self, genes):
        cols = [(g, self.col(g)) for g in genes]
        cols = [(g, c) for g, c in cols if c is not None]
        if not cols: return None, []
        return np.column_stack([c for _, c in cols]), [g for g, _ in cols]

# ============================================================ OGD
def ogd_split(E, kg, node, mask):
    """node 의 자식들로 OGD 분할. 형제 경쟁 게이트(절대 임계 아닌 상대 비교).
    반환 (label[n]: 자식라벨|'Unknown'|'__other__', parent_ok[n])."""
    ing, outg, disc = kg.gate_for(node)
    n = E.a.n_obs
    inM, _ = E.mat(ing)
    in_ok = (inM.max(1) > 0) if inM is not None else np.ones(n, bool)      # 조상 required 발현
    outM, _ = E.mat([g for g in outg if g not in ing])
    sib_max = outM.max(1) if outM is not None else np.zeros(n)            # 형제 계통 최강 마커
    names = [d["ch"] for d in disc]
    scores = np.zeros((n, max(len(disc), 1)))
    for j, d in enumerate(disc):
        gs = d["df"] or [g for g in d["req"] if g not in ing]
        Ms, _ = E.mat(gs)
        if Ms is not None: scores[:, j] = Ms.mean(1)
    child_max = scores.max(1)
    parent_ok = mask & in_ok & (child_max >= sib_max) & (child_max >= 0.05)
    lab = np.array(["__other__"] * n, dtype=object)
    lab[mask & in_ok & (child_max < 0.05)] = "Unknown"   # 계통 안이나 변별 신호 없음
    if len(disc):
        best = scores.argmax(1)
        lab[parent_ok] = np.array(names)[best][parent_ok]
    return lab, parent_ok

# ============================================================ lineage 판정
def lineage_call(E, idx, roots):
    """클러스터(idx)의 lineage = root class 게이트 평균 발현 argmax. mixed = 2등도 강함."""
    scores = {}
    for lab, mk in roots.items():
        M, gg = E.mat(mk["gate"])
        scores[lab] = float(M[idx].mean()) if M is not None else 0.0
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top, sec = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0.0))
    mixed = sec[1] >= 0.6 * top[1] and sec[1] > 0.1
    return top[0], mixed, [{"lineage": l, "score": round(s, 3)} for l, s in ranked[:5]]

# ============================================================ 프로파일
def profile_cluster(E, mask, kg, node):
    """agent 판정용: 크기 + 게이트아웃 비율 + 자식 변별 발현율 + top enriched 마커."""
    a = E.a; idx = np.where(mask)[0]
    prof = {"size": int(mask.sum()), "node": node}
    if node is not None:
        ing, outg, disc = kg.gate_for(node)
        # 자식 변별 마커별 클러스터 발현율 (부모 대비 어느 자식으로 갈지)
        ch_prof = []
        for d in disc:
            gs = d["df"] or d["req"]
            on = []
            for g in gs:
                c = E.col(g)
                if c is not None:
                    on.append([g, round(float((c[idx] > 0).mean()), 2), round(float(c[idx].mean()), 3)])
            ch_prof.append({"child": d["ch"], "tier": d["tier"], "markers_on": on})
        prof["children"] = ch_prof
    # top enriched (전체 대비) — KG 밖 신호 감지용
    gsum = np.asarray(a.X.mean(0)).ravel()
    lsum = np.asarray(a.X[idx].mean(0)).ravel()
    enr = (lsum + 1e-9) / (gsum + 1e-9)
    order = np.argsort(-enr * (lsum > 0.05))[:15]
    prof["top_enriched"] = [[str(a.var_names[i]), round(float(enr[i]), 2), round(float(lsum[i]), 3)]
                            for i in order if lsum[i] > 0.05]
    return prof

# ============================================================ state
def _sp(out): return os.path.join(out, "_kg_state.json")
def load_state(out):
    with open(_sp(out)) as f: return json.load(f)
def save_state(out, st):
    os.makedirs(out, exist_ok=True)
    json.dump(st, open(_sp(out), "w"), ensure_ascii=False, separators=(",", ":"))

# ============================================================ commands
def cmd_init(a):
    ad = sc.read_h5ad(a.h5ad)
    E = Expr(ad); kg = KG(a.organ, a.species)
    roots = kg.lineage_roots()
    cps = ad.obs[a.cluster_col].astype(str).values.copy()
    leaves = sorted(set(cps), key=lambda x: (len(x), x))
    node_of, lineage_of, rep = {}, {}, []
    for lf in leaves:
        idx = np.where(cps == lf)[0]
        lin, mixed, ranked = lineage_call(E, idx, roots)
        node_of[lf] = lin if not mixed else None       # mixed → 하강 보류(leiden 재분할 대상)
        lineage_of[lf] = {"lineage": lin, "mixed": mixed, "ranked": ranked}
        prof = profile_cluster(E, cps == lf, kg, None if mixed else lin)
        prof.update({"path": lf, "lineage": lin, "mixed_lineage": mixed, "lineage_ranked": ranked})
        rep.append(prof)
    st = {"h5ad": os.path.abspath(a.h5ad), "cluster_col": a.cluster_col,
          "organ": a.organ, "species": a.species, "round": 1,
          "cluster_paths": cps.tolist(), "leaves": leaves,
          "node_of": node_of, "lineage_of": lineage_of, "parent_of": {lf: None for lf in leaves},
          "decisions": {}}
    save_state(a.out, st)
    json.dump({"round": 1, "n_clusters": len(leaves),
               "mixed_lineage": [lf for lf in leaves if lineage_of[lf]["mixed"]], "clusters": rep},
              open(os.path.join(a.out, "profile_round1.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[init] {ad.n_obs} cells, {len(leaves)} leiden clusters | "
          f"mixed-lineage {sum(v['mixed'] for v in lineage_of.values())} | profile_round1.json")

def cmd_descend(a):
    st = load_state(a.out)
    ad = sc.read_h5ad(st["h5ad"]); E = Expr(ad); kg = KG(st["organ"], st["species"])
    cps = np.array(st["cluster_paths"], dtype=object)
    paths = json.loads(a.paths)
    new_leaves = list(st["leaves"]); rep = []
    for p in paths:
        node = st["node_of"].get(p)
        if node is None:
            print(f"[descend] {p}: node 미정(mixed lineage) — leiden 재분할 필요, skip"); continue
        mask = cps == p
        lab, pok = ogd_split(E, kg, node, mask)
        # 자식 라벨별로 새 경로 부여
        children = kg.gate_for(node)[2]
        cidx = {d["ch"]: i for i, d in enumerate(children)}
        assigned = 0
        for ch, i in cidx.items():
            m = (lab == ch)
            if m.sum() == 0: continue
            npath = f"{p}.{i}"
            cps[m] = npath
            st["node_of"][npath] = ch
            st["parent_of"][npath] = p
            if npath not in new_leaves: new_leaves.append(npath)
            assigned += int(m.sum())
            rep.append({**profile_cluster(E, m, kg, ch), "path": npath, "child": ch,
                        "parent_path": p, "parent_node": node})
        # Unknown / __other__ 는 그대로 부모 경로에 남김 (agent 가 후속 판단)
        if p in new_leaves and (cps == p).sum() == 0:
            new_leaves.remove(p)
        print(f"[descend] {p} ({node}): assigned {assigned}, "
              f"Unknown {int((lab=='Unknown').sum())}, other {int((lab=='__other__').sum())}")
    st["cluster_paths"] = cps.tolist(); st["leaves"] = new_leaves; st["round"] = st["round"] + 1
    save_state(a.out, st)
    json.dump({"round": st["round"], "clusters": rep},
              open(os.path.join(a.out, f"profile_round{st['round']}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[descend] round {st['round']} | {len(new_leaves)} leaves | profile_round{st['round']}.json")

def cmd_finalize(a):
    st = load_state(a.out)
    ad = sc.read_h5ad(st["h5ad"])
    dec = json.load(open(a.decisions)) if os.path.exists(a.decisions) else json.loads(a.decisions)
    cps = np.array(st["cluster_paths"], dtype=object)
    final = np.array(["Unknown"] * ad.n_obs, dtype=object)
    for p in set(cps):
        lab = dec.get(p) or dec.get(str(p)) or st["node_of"].get(p) or "Unknown"
        final[cps == p] = lab
    import pandas as pd
    ad.obs["cellman_standard"] = pd.Categorical(final)
    ad.obs["cellman_kg_path"] = pd.Categorical(cps)
    out = a.output or st["h5ad"].replace(".h5ad", "_kg_annotated.h5ad")
    ad.write(out)
    print(f"[finalize] cellman_standard 저장 → {out}")
    for lab, c in Counter(final).most_common(): print(f"  {c:7d}  {lab}")

def cmd_split_lineage(a):
    """mixed-lineage 클러스터를 leiden 으로 세분 → 각 서브의 lineage 재판정 (계획 1번).
    단일 lineage 로 갈라지면 그 노드로 하강 가능해짐; 여전히 mixed 면 다시 split 대상."""
    st = load_state(a.out)
    ad = sc.read_h5ad(st["h5ad"]); E = Expr(ad); kg = KG(st["organ"], st["species"])
    roots = kg.lineage_roots()
    cps = np.array(st["cluster_paths"], dtype=object)
    paths = json.loads(a.paths); new_leaves = list(st["leaves"]); rep = []
    for p in paths:
        mask = cps == p
        sub = ad[mask].copy()
        if "X_pca" not in sub.obsm:
            sc.pp.pca(sub, n_comps=min(30, sub.n_vars - 1))
        sc.pp.neighbors(sub, use_rep="X_pca")
        sc.tl.leiden(sub, resolution=a.resolution, key_added="_sub", flavor="igraph", n_iterations=2, directed=False)
        base = np.where(mask)[0]
        for sl in sorted(sub.obs["_sub"].unique(), key=int):
            sidx = base[sub.obs["_sub"].values == sl]
            npath = f"{p}.{sl}"
            lin, mixed, ranked = lineage_call(E, sidx, roots)
            cps[sidx] = npath
            st["node_of"][npath] = None if mixed else lin
            st["lineage_of"][npath] = {"lineage": lin, "mixed": mixed, "ranked": ranked}
            st["parent_of"][npath] = p
            if npath not in new_leaves: new_leaves.append(npath)
            rep.append({**profile_cluster(E, cps == npath, kg, None if mixed else lin),
                        "path": npath, "lineage": lin, "mixed_lineage": mixed, "lineage_ranked": ranked})
        if p in new_leaves: new_leaves.remove(p)
        print(f"[split-lineage] {p} → {sub.obs['_sub'].nunique()} leiden sub")
    st["cluster_paths"] = cps.tolist(); st["leaves"] = new_leaves; st["round"] = st["round"] + 1
    save_state(a.out, st)
    json.dump({"round": st["round"], "still_mixed": [r["path"] for r in rep if r["mixed_lineage"]], "clusters": rep},
              open(os.path.join(a.out, f"profile_linsplit_r{st['round']}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[split-lineage] {len(new_leaves)} leaves | still-mixed {sum(r['mixed_lineage'] for r in rep)}")

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    pi = sp.add_parser("init"); pi.add_argument("--h5ad", required=True); pi.add_argument("--out", required=True)
    pi.add_argument("--cluster-col", default="leiden", dest="cluster_col")
    pi.add_argument("--organ", required=True); pi.add_argument("--species", required=True)
    ps = sp.add_parser("split-lineage"); ps.add_argument("--out", required=True); ps.add_argument("--paths", required=True)
    ps.add_argument("--resolution", type=float, default=0.5)
    pd_ = sp.add_parser("descend"); pd_.add_argument("--out", required=True); pd_.add_argument("--paths", required=True)
    pf = sp.add_parser("finalize"); pf.add_argument("--out", required=True)
    pf.add_argument("--decisions", required=True); pf.add_argument("--output", default=None)
    a = ap.parse_args()
    {"init": cmd_init, "split-lineage": cmd_split_lineage, "descend": cmd_descend, "finalize": cmd_finalize}[a.cmd](a)

if __name__ == "__main__":
    main()
