#!/usr/bin/env python3
"""annotate-mode expert agent md 생성 (read-only 공개판).

사용 (Agentic-Annotation/tools/ 에서):
  python generate_annotate_experts.py --server http://localhost:8088          # manifest 전체
  python generate_annotate_experts.py --wing mouse:whole_brain --wing human:lung [--server ...]

서버 미지정 시 --wing 으로 준 것만 생성(오프라인). 생성된 expert md는 ../agents/ 에 쓴다.
"""
import argparse, os, json, urllib.request

HERE=os.path.dirname(os.path.abspath(__file__))                  # Agentic-Annotation/tools
TEMPLATE=os.path.join(HERE,"annotate-expert-template.md")
AGENTS_DIR=os.path.abspath(os.path.join(HERE,"..","agents"))     # Agentic-Annotation/agents (output)
GENE_CONV={"human":"UPPERCASE (e.g. CD3D, FOXP3, GFAP)","mouse":"Title-case (e.g. Cd3d, Foxp3, Gfap)"}

def render(species, organ, server):
    t=open(TEMPLATE).read()
    for k,v in {"{SPECIES}":species,"{ORGAN}":organ,"{WING}":f"{species}_{organ}",
                "{GENE_CONVENTION}":GENE_CONV.get(species,""),
                "{SERVER}":server or "${CELLONTO_SERVER_URL}"}.items():
        t=t.replace(k,v)
    return t

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--server",default=None)
    ap.add_argument("--wing",action="append",default=[],help="species:organ (반복가능)")
    a=ap.parse_args()
    wings=[]
    if a.wing:
        for w in a.wing:
            sp,og=w.split(":",1); wings.append((sp,og))
    elif a.server:
        d=json.load(urllib.request.urlopen(a.server.rstrip("/")+"/wings", timeout=30))
        wings=[(w["species"],w["organ"]) for w in d["wings"]]
    else:
        ap.error("--server 또는 --wing 중 하나 필요")
    os.makedirs(AGENTS_DIR, exist_ok=True)
    n=0
    for sp,og in wings:
        open(os.path.join(AGENTS_DIR,f"{sp}-{og}-expert.md"),"w").write(render(sp,og,a.server)); n+=1
    print(f"[gen] annotate expert {n}개 생성 → {AGENTS_DIR}")

if __name__=="__main__":
    main()
