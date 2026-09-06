#!/usr/bin/env python3
"""cellman-kg — a READ-ONLY MCP server over the Cell-Ontology module vocabulary.

WHAT IT IS FOR. Before annotation starts, the caller has to turn the user's free-text tissue into the
canonical `(species, organ)` module the Knowledge Graph actually serves, and it has to know WHICH
server that answer came from. This exposes exactly that, as tools, so a host that speaks MCP (Claude
Code, Codex) can do it without shelling out.

WHAT IT IS NOT. It cannot annotate, cannot write the ontology, and cannot reach the h5ad. Three
read-only lookups, nothing else. `build-mode` writes the KG; the annotate side never does.

WHY IT SHELLS OUT TO helper.py INSTEAD OF REIMPLEMENTING THE LOOKUP. The alias tables (synonym /
parent-of-subtype / umbrella / ambiguous) and the server-address precedence
(--server-url > $CELLONTO_SERVER_URL > built-in default) live in `scripts/helper.py`. Duplicating them
here is how the caller and the engine would drift apart again — the exact failure this whole layer
exists to prevent. So: one implementation, two surfaces. `resolve_module` and `list_modules` return the
`server_url` they resolved against, and the agent is expected to pass that value to
`helper.py init --server-url`, which is what pins the LLM and the engine to the same KG.

WHY NO MCP SDK. Three tools over stdio JSON-RPC 2.0. Implementing `initialize` / `tools/list` /
`tools/call` directly keeps this file dependency-free, so it runs under any python3 even when the
annotate environment (numpy / anndata / scanpy) is a different interpreter — see CELLMAN_PYTHON.

CONFIGURATION (environment)
  CELLMAN_PYTHON       python that can run helper.py (i.e. the venv holding anndata/scanpy/numpy).
                       Default: the interpreter running this server — usually WRONG, so set it.
  CELLMAN_HELPER       absolute path to CeLLMan/scripts/helper.py. Default: ../scripts/helper.py
                       relative to this file.
  CELLONTO_SERVER_URL  the KG server. Passed straight through to helper.py, which owns the precedence.

COST. Each tool call spawns helper.py, whose module-level imports (anndata, scipy) cost ~2.5-3 s. A run
makes one or two of these calls, against a 30-minute annotation — irrelevant, and worth it for having
a single implementation.
"""
import json, os, subprocess, sys, urllib.request, urllib.error, urllib.parse

NAME = "cellman-kg"
VERSION = "0.1.0"
DEFAULT_PROTOCOL = "2025-06-18"
_HERE = os.path.dirname(os.path.abspath(__file__))
_server_url_cache = None


class ToolError(Exception):
    """A failure to report to the model as tool output (isError), not as a protocol error."""


# ----------------------------------------------------------------- helper.py bridge
def _helper_path():
    p = os.environ.get("CELLMAN_HELPER")
    if p:
        if not os.path.exists(p):
            raise ToolError(f"CELLMAN_HELPER points at a file that does not exist: {p}")
        return p
    cand = os.path.join(os.path.dirname(_HERE), "scripts", "helper.py")
    if os.path.exists(cand):
        return cand
    raise ToolError("cannot locate CeLLMan/scripts/helper.py — set CELLMAN_HELPER to its absolute path.")


def _run_helper(args, timeout=90):
    cmd = [os.environ.get("CELLMAN_PYTHON") or sys.executable, _helper_path()] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ToolError(f"helper.py timed out after {timeout}s: {' '.join(args)}")
    except OSError as e:
        raise ToolError(f"could not run helper.py ({e}). Check CELLMAN_PYTHON / CELLMAN_HELPER.")
    out = r.stdout or ""
    if r.returncode != 0:
        msg = (r.stderr or out).strip() or f"helper.py exited {r.returncode}"
        if "ModuleNotFoundError" in msg:
            msg += ("\n(the interpreter running helper.py lacks its dependencies — point CELLMAN_PYTHON "
                    "at the virtual environment that has anndata/scanpy/numpy)")
        raise ToolError(msg)
    i = out.find("{")
    if i < 0:
        raise ToolError(f"helper.py returned no JSON:\n{out.strip() or r.stderr.strip()}")
    try:
        return json.loads(out[i:])
    except json.JSONDecodeError as e:
        raise ToolError(f"helper.py returned malformed JSON ({e}):\n{out[i:][:400]}")


def _server_url():
    """The KG address helper.py itself resolves to — never decided independently here."""
    global _server_url_cache
    if _server_url_cache is None:
        _server_url_cache = _run_helper(["modules"]).get("server_url")
    return _server_url_cache


# ----------------------------------------------------------------- tools
def t_list_modules(args):
    a = ["modules"]
    if args.get("species"):
        a += ["--species", str(args["species"])]
    return _run_helper(a)


def t_resolve_module(args):
    tissue = args.get("tissue")
    if not tissue:
        raise ToolError("'tissue' is required.")
    a = ["resolve-module", "--tissue", str(tissue)]
    if args.get("species"):
        a += ["--species", str(args["species"])]
    return _run_helper(a)


def t_get_term(args):
    for k in ("species", "organ", "term_id"):
        if not args.get(k):
            raise ToolError(f"'{k}' is required.")
    q = urllib.parse.quote
    url = (f"{_server_url()}/module/{q(str(args['species']), safe='')}/"
           f"{q(str(args['organ']), safe='')}/term/{q(str(args['term_id']), safe='')}")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise ToolError(f"HTTP {e.code} for {url}" + (" — no such term in that module." if e.code == 404 else ""))
    except urllib.error.URLError as e:
        raise ToolError(f"cannot reach the ontology server ({url}; {e.reason}).")


TOOLS = [
    {
        "name": "resolve_module",
        "description": (
            "Turn a free-text tissue into the canonical Cell-Ontology module(s) the KG actually serves. "
            "Run this BEFORE helper.py init. Returns status='exact' (an exact module name — proceed "
            "silently, never second-guess it), 'candidates' (ASK THE USER; each candidate carries "
            "match_type: synonym | parent_of_subtype | umbrella_children | ambiguous | token, and "
            "can_serve_together says whether 'use all of them' is a valid answer), or 'none' (the KG has "
            "no such module — do not annotate from guesswork). It also returns server_url: pass that "
            "exact value to helper.py init --server-url so the engine reads the same KG."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tissue": {"type": "string",
                           "description": "the user's own wording, e.g. 'mammary gland', 'cerebellum', "
                                          "'mouse, large intestine'"},
                "species": {"type": "string",
                            "description": "mouse | human (aliases such as 'Mus musculus' accepted). May "
                                           "be omitted when tissue is given as 'mouse, brain'."},
            },
            "required": ["tissue"],
        },
    },
    {
        "name": "list_modules",
        "description": (
            "The canonical (species, organ) list the Cell-Ontology server serves, with n_terms and "
            "last_updated per wing, plus the server_url they came from. Use it to show coverage or to "
            "check a module the user named explicitly."),
        "inputSchema": {
            "type": "object",
            "properties": {"species": {"type": "string", "description": "optional filter, e.g. mouse"}},
        },
    },
    {
        "name": "get_term",
        "description": (
            "One ontology Term as the KG stores it (markers, tier, parents, grounding). Read-only. "
            "Useful after annotation, when the user asks why a cluster got a label — not needed during "
            "a run, since the whole module is already in {out}/_ontology_index.json."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "species": {"type": "string"},
                "organ": {"type": "string", "description": "canonical module name, e.g. large_intestine"},
                "term_id": {"type": "string", "description": "e.g. MOUSE_KIDNEY:0000012"},
            },
            "required": ["species", "organ", "term_id"],
        },
    },
]
HANDLERS = {"resolve_module": t_resolve_module, "list_modules": t_list_modules, "get_term": t_get_term}


# ----------------------------------------------------------------- stdio JSON-RPC
def _send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _result(rid, result):
    _send({"jsonrpc": "2.0", "id": rid, "result": result})


def _error(rid, code, message):
    _send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})


def _handle(msg):
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params") or {}
    if rid is None:                                   # notification — never answered
        return
    if method == "initialize":
        proto = params.get("protocolVersion")
        _result(rid, {"protocolVersion": proto if isinstance(proto, str) else DEFAULT_PROTOCOL,
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": NAME, "version": VERSION}})
    elif method == "ping":
        _result(rid, {})
    elif method == "tools/list":
        _result(rid, {"tools": TOOLS})
    elif method == "tools/call":
        name = params.get("name")
        fn = HANDLERS.get(name)
        if fn is None:
            _error(rid, -32602, f"unknown tool '{name}'")
            return
        try:
            payload = fn(params.get("arguments") or {})
            text = json.dumps(payload, indent=1)
            is_err = False
        except ToolError as e:
            text, is_err = str(e), True
        except Exception as e:                        # never kill the server on one bad call
            text, is_err = f"{type(e).__name__}: {e}", True
        _result(rid, {"content": [{"type": "text", "text": text}], "isError": is_err})
    else:
        _error(rid, -32601, f"method not found: {method}")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue                                  # not addressable: no id to answer with
        try:
            _handle(msg)
        except Exception as e:                        # protocol-level guard
            if msg.get("id") is not None:
                _error(msg["id"], -32603, f"internal error: {e}")
            print(f"[cellman-kg] internal error: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
