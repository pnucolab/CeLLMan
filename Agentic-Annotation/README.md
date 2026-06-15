# Agentic-Annotation (공개 플러그인)

scRNA/spatial 클러스터링 h5ad를 expert-judgment로 celltype 주석(+공간 도메인)하는 Claude Code 플러그인.
온톨로지 데이터·neo4j·build는 포함하지 않음 — 실행 시 **Cell-Ontology 서버에서 slim-index를 fetch**한다.

## 설치
1. deps: `pip install -r requirements.txt` (사용자 venv; GPU면 rapids-singlecell 추가 권장, 없으면 scanpy CPU 자동).
2. 서버: `config/server.example.env` 참고해 `CELLONTO_SERVER_URL` 설정.
3. expert agent: 예시 `mouse-whole_brain`, `human-lung` 동봉. 다른 wing은
   `python tools/generate_annotate_experts.py --server $CELLONTO_SERVER_URL` 로 생성
   (서버 manifest의 가용 wing만).

## 사용 (caller에게)
`CALLER_PROMPT_GUIDE.md`의 형식으로 명령. 예:
```
mouse-whole_brain-expert agent를 annotate 모드로 호출해서 my_clusters.h5ad 를 annotation 해줘.
[UNNAMED annotate params] species/organ/input_h5ad/cluster_col/disease/python/gpu_device...
```
→ 출력 `{input}_annotated.h5ad`(입력 전체 복사 + `obs['celltype_annotation']`).

## 구조
- `skills/annotate-mode/SKILL.md` — celltype 주석 rubric(4지선다).
- `skills/domain-mode/SKILL.md` — 공간 도메인 주석(이웃 조성→region).
- `agents/{species}-{organ}-expert.md` — annotate 전용·read-only expert(예시 2개; 템플릿·생성기는 tools/).
- `tools/` — expert agent 템플릿 + 생성기(`generate_annotate_experts.py`).
- `scripts/helper.py` — annotate 실행기(init/subcluster/finalize; 서버 slim-index fetch; GPU/CPU 분기).
- `scripts/domain_helper.py` — domain 실행기.
- `CALLER_PROMPT_GUIDE.md` — 명령(호출) 형식. (파이프라인 설계 스펙은 비공개 상위 `UNNAMED/ANNOTATION_PIPELINE.md`.)

## 백엔드
subcluster는 GPU(rapids-singlecell) 있으면 사용, 없으면 scanpy CPU 자동. **백엔드별로 leiden 결과(라벨)가 다를 수 있음.**

## 안 들어있는 것 (= Cell-Ontology 비공개)
ontology 데이터·neo4j·build 파이프라인·문헌수집·MemPalace. 사용자는 read-only로 서버 slim-index만 받는다.

## 프레임워크 통합 (Claude Code / Codex / 기타)
`agents/`·`skills/`는 **특정 프레임워크에 묶이지 않은 일반 markdown 프롬프트 파일**이다(.claude 안에 두지 않음).
- **Claude Code**: 자동 인식을 원하면 `agents/`→`.claude/agents/`, `skills/`→`.claude/skills/`로 심볼릭링크/복사
  (또는 플러그인 설치 스크립트). caller가 expert agent를 subagent로 호출.
- **Codex / 다른 LLM**: caller(사용자의 LLM)가 `skills/annotate-mode/SKILL.md`와 해당
  `agents/{species}-{organ}-expert.md`를 프롬프트로 읽어 동일 절차 수행. helper.py/domain_helper.py는
  어느 환경이든 사용자 venv의 python으로 실행.
즉 판단(LLM)과 실행기(helper.py)가 분리돼 있어 **모델·프레임워크 교체 가능**.
