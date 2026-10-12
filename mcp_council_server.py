#!/usr/bin/env python
"""
MCP SERVER: COUNCIL-L99 (JSON-RPC 2.0 STDIO)
Exposes the multi-agent deliberation and validation tools via the Model Context Protocol (MCP).
Compatible with Claude Desktop, Antigravity, and any MCP client over stdio.
"""

import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from council_deliberate import deliberate
from deepseek_oracle import (
    consult_deepseek_oracle,
    extract_clean_code,
    inspect_local_targets,
    validate_python_ast,
)
from persistent_brain import search_memories, store_lesson

TOOLS = [
    {
        "name": "council_deliberate_all",
        "description": "Convenes the multi-agent council (DeepSeek-R1 8B, Llama 3.1 8B, Qwen 2.5 Coder 7B) for structured consensus with persistent vector memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Problem, architecture, or directive to deliberate"}
            },
            "required": ["topic"]
        }
    },
    {
        "name": "deepseek_delegate",
        "description": "Delegates file inspection (read-only) or AST-validated code synthesis to the secondary oracle/local models.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "directive": {"type": "string", "description": "Task or problem to resolve"},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Optional file paths for read-only inspection"},
                "write_out": {"type": "string", "description": "Optional path where AST-validated code should be written"}
            },
            "required": ["directive"]
        }
    },
    {
        "name": "persistent_memory",
        "description": "Recalls or stores learned lessons in SQLite vector memory (nomic-embed-text embeddings).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["recall", "store"]},
                "query_or_lesson": {"type": "string"},
                "key_tag": {"type": "string"}
            },
            "required": ["action", "query_or_lesson"]
        }
    },
    {
        "name": "code_audit_and_verify",
        "description": "Passes a target Python file through strict AST syntax parsing and static analysis (Bandit SAST & Ruff linter).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to Python file to verify"}
            },
            "required": ["file_path"]
        }
    }
]


def handle_tool_call(name: str, args: dict) -> dict:
    if name == "council_deliberate_all":
        topic = args.get("topic", "")
        res = deliberate(topic)
        return {"content": [{"type": "text", "text": json.dumps(res, indent=2, ensure_ascii=False)}]}

    elif name == "deepseek_delegate":
        directive = args.get("directive", "")
        files = args.get("files", [])
        write_out = args.get("write_out")
        full_p = directive
        if files:
            inspected = inspect_local_targets(files)
            if inspected:
                full_p += f"\n\n[INSPECTED CONTEXT]:\n{inspected}"

        res = consult_deepseek_oracle(full_p)
        text = res.get("content", "")
        if write_out and res.get("success"):
            code = extract_clean_code(text)
            ok, msg = validate_python_ast(code)
            if ok:
                p = Path(write_out)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(code, encoding="utf-8")
                text += f"\n\n[FILE WRITTEN]: {write_out} ({msg})"
            else:
                text += f"\n\n[AST REJECTED]: {msg}"
        return {"content": [{"type": "text", "text": text}]}

    elif name == "persistent_memory":
        action = args.get("action")
        ql = args.get("query_or_lesson", "")
        tag = args.get("key_tag", "lesson")
        if action == "store":
            store_lesson(tag, ql)
            return {"content": [{"type": "text", "text": f"Lesson '{tag}' stored in vector memory."}]}
        else:
            mems = search_memories(ql, top_k=3)
            return {"content": [{"type": "text", "text": json.dumps(mems, indent=2, ensure_ascii=False)}]}

    elif name == "code_audit_and_verify":
        fp = args.get("file_path", "")
        path = Path(fp)
        if not path.exists():
            return {"content": [{"type": "text", "text": f"File not found: {fp}"}], "isError": True}

        code = path.read_text(encoding="utf-8", errors="replace")
        ok_ast, msg_ast = validate_python_ast(code)

        # Run ruff check if installed in current python environment
        ruff_res = "ruff not installed (install via pip install ruff)"
        try:
            r = subprocess.run([sys.executable, "-m", "ruff", "check", str(path)], capture_output=True, text=True, timeout=10, check=False)
            if r.returncode == 0:
                ruff_res = "Clean (0 issues)"
            elif "No module named ruff" in r.stderr:
                ruff_res = "ruff not installed (install via pip install ruff)"
            else:
                ruff_res = r.stdout or r.stderr
        except FileNotFoundError:
            ruff_res = "ruff not installed (install via pip install ruff)"
        except (subprocess.SubprocessError, OSError) as e:
            ruff_res = f"ruff execution error: {e}"

        # Run bandit check if installed in current python environment
        bandit_res = "bandit not installed (install via pip install bandit)"
        try:
            b = subprocess.run([sys.executable, "-m", "bandit", "-q", str(path)], capture_output=True, text=True, timeout=10, check=False)
            if b.returncode == 0:
                bandit_res = "Clean (0 vulnerabilities)"
            elif "No module named bandit" in b.stderr:
                bandit_res = "bandit not installed (install via pip install bandit)"
            else:
                bandit_res = b.stdout or b.stderr
        except FileNotFoundError:
            bandit_res = "bandit not installed (install via pip install bandit)"
        except (subprocess.SubprocessError, OSError) as e:
            bandit_res = f"bandit execution error: {e}"

        report = {
            "file": str(path),
            "ast_syntax": msg_ast,
            "ast_passed": ok_ast,
            "ruff_linter": ruff_res.strip(),
            "bandit_sast": bandit_res.strip()
        }
        return {"content": [{"type": "text", "text": json.dumps(report, indent=2)}]}

    return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


def run_stdio_server():
    """Implements standard JSON-RPC 2.0 stdio loop for MCP protocol."""
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")

            if method == "initialize":
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "council-l99-mcp", "version": "1.0.0"}
                    }
                }
            elif method == "tools/list":
                res = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
            elif method == "tools/call":
                params = req.get("params", {})
                t_name = params.get("name")
                t_args = params.get("arguments", {})
                tool_res = handle_tool_call(t_name, t_args)
                res = {"jsonrpc": "2.0", "id": req_id, "result": tool_res}
            else:
                res = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError as e:
            err_res = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()
        except Exception as e:  # noqa: BLE001
            err_res = {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(e)}}
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
