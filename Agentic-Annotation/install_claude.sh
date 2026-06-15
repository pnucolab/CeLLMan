#!/usr/bin/env bash
# (선택) Claude Code 편의 부트스트랩.
# 이 플러그인의 skills/agents를 <target>/.claude 에 심볼릭링크해서 `/annotate-mode` 슬래시 +
# {species}-{organ}-expert subagent를 네이티브로 쓸 수 있게 한다.
#
# 등록은 "필수"가 아니다 — 어떤 LLM이든 "skills/annotate-mode/SKILL.md를 읽고 따라줘"로 동작한다.
# 이 스크립트는 Claude Code에서 슬래시 명령 편의를 원할 때만 실행하면 된다.
#
# 사용:
#   bash Agentic-Annotation/install_claude.sh            # 기본: 상위(=상위 repo root)에 .claude 생성
#   bash Agentic-Annotation/install_claude.sh /path/dir  # 특정 작업 디렉터리에 생성
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"                 # .../Agentic-Annotation
TARGET="${1:-$(cd "$HERE/.." && pwd)}"                # 기본: Agentic-Annotation의 상위(=repo root)
mkdir -p "$TARGET/.claude/skills" "$TARGET/.claude/agents"

# skills (디렉터리 단위 링크: 각 폴더에 SKILL.md)
for s in annotate-mode domain-mode; do
  [ -d "$HERE/skills/$s" ] && ln -sfn "$HERE/skills/$s" "$TARGET/.claude/skills/$s"
done

# agents (expert md 전부 링크)
n=0
for a in "$HERE"/agents/*-expert.md; do
  [ -e "$a" ] || continue
  ln -sfn "$a" "$TARGET/.claude/agents/$(basename "$a")"; n=$((n+1))
done

echo "[install_claude] linked → $TARGET/.claude"
echo "  skills:  $(ls "$TARGET/.claude/skills" 2>/dev/null | tr '\n' ' ')"
echo "  agents:  ${n}개 expert"
echo "이후 Claude Code를 '$TARGET'에서 열면 /annotate-mode + expert subagent 사용 가능."
echo "(해제: $TARGET/.claude 의 해당 심볼릭링크 삭제)"
