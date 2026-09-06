# `cellman-kg` — MCP server (the default lookup path)

Read-only MCP access to the Cell-Ontology **module vocabulary**: turn a user's free-text tissue into
the canonical `(species, organ)` the Knowledge Graph actually serves, before annotation starts. The
skill and the expert agent reach the KG through these tools by default; register it with your host as
part of installing CeLLMan.

**It wraps the helper rather than reimplementing it**, so there is one implementation and two surfaces,
always resolved against the same server. That also means **nothing breaks if you skip registration** —
the skill falls back to the very command the server would have run:

```bash
{python} {HELPER} modules [--species mouse]
{python} {HELPER} resolve-module --species mouse --tissue "mammary gland"
```

Same code, same answer, one extra shell call. Register it when you want the lookup as a native tool
(no Bash permission needed), or when you need to put authentication in front of the KG.

---

## Tools

| tool | returns |
|---|---|
| `resolve_module(tissue, species?)` | `status` = `exact` (proceed) / `candidates` (**ask the user**; each carries `match_type` = `synonym`, `parent_of_subtype`, `umbrella_children`, `ambiguous`, `token`, plus `can_serve_together`) / `none` (do not annotate), and the `server_url` it resolved against |
| `list_modules(species?)` | every served wing with `n_terms` and `last_updated`, plus `server_url` |
| `get_term(species, organ, term_id)` | one Term as the KG stores it — for "why did you call it that?" after a run |

Nothing here can write the ontology, run an annotation, or read an h5ad.

## Requirements

`python3` only — the server itself is dependency-free. It shells out to `scripts/helper.py`, which
does need the annotate dependencies, so **point `CELLMAN_PYTHON` at the virtual environment you
installed `requirements.txt` into**. Leaving it unset makes the server use its own interpreter, which
usually lacks `anndata` and fails with a message saying exactly that.

| variable | meaning | default |
|---|---|---|
| `CELLMAN_PYTHON` | interpreter that can run `helper.py` | the interpreter running the server (usually wrong — set it) |
| `CELLMAN_HELPER` | absolute path to `CeLLMan/scripts/helper.py` | `../scripts/helper.py` next to this file |
| `CELLONTO_SERVER_URL` | the KG server | passed through to `helper.py`, which owns the precedence |

## Register — Claude Code

```bash
claude mcp add cellman-kg \
  --env CELLMAN_PYTHON=/path/to/.venv/bin/python \
  -- python3 /abs/path/to/CeLLMan/mcp/cellman_kg_mcp.py
```

or commit it to the project as `.mcp.json`:

```json
{
  "mcpServers": {
    "cellman-kg": {
      "command": "python3",
      "args": ["/abs/path/to/CeLLMan/mcp/cellman_kg_mcp.py"],
      "env": { "CELLMAN_PYTHON": "/path/to/.venv/bin/python" }
    }
  }
}
```

## Register — Codex

In `~/.codex/config.toml`:

```toml
[mcp_servers.cellman-kg]
command = "python3"
args = ["/abs/path/to/CeLLMan/mcp/cellman_kg_mcp.py"]
env = { CELLMAN_PYTHON = "/path/to/.venv/bin/python" }
```

## Check it works

```bash
CELLMAN_PYTHON=/path/to/.venv/bin/python python3 mcp/cellman_kg_mcp.py <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"resolve_module","arguments":{"tissue":"mouse, mammary gland"}}}
EOF
```

Three JSON lines come back; the third should propose `breast` with `match_type: "synonym"`.

## Pass `server_url` on

`resolve_module` and `list_modules` report the `server_url` they used. Hand that value to
`helper.py init --server-url` — that is what keeps the module the caller confirmed and the wing the
engine fetches on the **same** KG. It matters in practice: the built-in default and a deployment's
`CELLONTO_SERVER_URL` are frequently different addresses.

## A note on the alias tables

`resolve_module`'s synonym / parent / umbrella / ambiguous tables live in `scripts/helper.py`, marked
as a client-side stopgap. The canonical home for them is the KG server (`api/server.py` already keeps
a legacy organ alias). Once `/modules` publishes `synonyms` / `parent` / `children` per wing, those
tables should be deleted and the server's answer used instead — no change is needed here or in the
skills, because both surfaces read the same helper. Until then, every proposal is intersected with what
`/modules` actually serves, so a stale entry can never name a module that does not exist.
