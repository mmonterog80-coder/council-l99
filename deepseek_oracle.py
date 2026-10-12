#!/usr/bin/env python
"""
DEEPSEEK ORACLE & AST CODE VALIDATOR — COUNCIL L99
Provides secondary technical synthesis, code extraction, and strict AST syntax validation.
Executes code generation and verification either via optional cloud API or local Ollama models.
"""

import argparse
import ast
import io
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, io.UnsupportedOperation):
        pass

try:
    from persistent_brain import format_memory_context, store_lesson
except ImportError:
    format_memory_context = None
    store_lesson = None

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")
CLOUD_MODELS = ["deepseek-chat", "deepseek-reasoner"]

SYSTEM_PROMPT_ORACLE = (
    "You are the technical advisor and code auditor for Council L99. "
    "Rules: Deliver valid, production-ready Python code. Verify AST compatibility. "
    "Minimize token consumption with concise, actionable output."
)


def get_api_key() -> str | None:
    """Retrieves optional DeepSeek API key from standard environment variables."""
    k = os.environ.get("DEEPSEEK_API_KEY")
    return k.strip() if k and k.strip() else None


def validate_python_ast(code_str: str) -> tuple[bool, str]:
    """Validates that generated Python code parses successfully with ast.parse."""
    try:
        ast.parse(code_str)
        return True, "AST syntax check passed."
    except SyntaxError as e:
        return False, f"AST SyntaxError: line {e.lineno}, column {e.offset}: {e.msg}"


def extract_clean_code(markdown_text: str) -> str:
    """Extracts raw Python code blocks from markdown formatted LLM outputs."""
    lines = markdown_text.splitlines()
    in_block = False
    extracted = []

    for line in lines:
        if line.strip().startswith("```python") or line.strip().startswith("```py"):
            in_block = True
            continue
        elif line.strip().startswith("```") and in_block:
            in_block = False
            continue
        if in_block:
            extracted.append(line)

    if not extracted:
        # If no markdown block found, return cleaned original text
        return markdown_text.strip()
    return "\n".join(extracted)


def inspect_local_targets(file_paths: list[str]) -> str:
    """Read-only inspection of target files for code context analysis."""
    outputs = []
    for p in file_paths:
        path = Path(p)
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                outputs.append(f"=== File: {path.name} ===\n{content[:2000]}")
            except OSError as e:
                outputs.append(f"=== File: {path.name} (Error reading: {e}) ===")
    return "\n\n".join(outputs)


def consult_deepseek_oracle(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT_ORACLE,
    inject_memory: bool = True,
    temperature: float = 0.1,
    max_tokens: int = 2048,
    timeout: int = 60
) -> dict[str, Any]:
    """
    Queries DeepSeek Cloud if API key is provided; otherwise falls back to local Ollama models.
    """
    mem_block = format_memory_context(prompt, top_k=2) if (inject_memory and format_memory_context) else ""
    enriched_prompt = f"{mem_block}\n\n[TASK]:\n{prompt}" if mem_block else prompt
    api_key = get_api_key()
    t0 = time.time()

    # 1. Cloud API execution if key is present
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        for candidate_model in CLOUD_MODELS:
            payload = {
                "model": candidate_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enriched_prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            try:
                resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=timeout)
                elapsed = round(time.time() - t0, 2)
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]["message"]
                    content = choice.get("content", "").strip()
                    usage = data.get("usage", {})
                    if content:
                        return {
                            "success": True,
                            "model": f"{candidate_model} (Cloud API)",
                            "elapsed_sec": elapsed,
                            "content": content,
                            "usage": usage
                        }
            except (requests.RequestException, KeyError, ValueError):
                continue

    # 2. Local Ollama fallback if cloud key is not set or network fails
    for local_mod in ["qwen2.5-coder:7b", "qwen2.5-coder:1.5b"]:
        try:
            r = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": local_mod,
                    "prompt": enriched_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": max_tokens}
                },
                timeout=timeout
            )
            if r.status_code == 200:
                txt = r.json().get("response", "").strip()
                if txt:
                    return {
                        "success": True,
                        "model": f"{local_mod} (Local Ollama)",
                        "elapsed_sec": round(time.time() - t0, 2),
                        "content": txt,
                        "usage": {"total_tokens": 0}
                    }
        except (requests.RequestException, KeyError, ValueError):
            continue

    return {
        "success": False,
        "error": "Neither Cloud API nor local Ollama models were reachable.",
        "elapsed_sec": round(time.time() - t0, 2),
        "content": ""
    }


def main():
    parser = argparse.ArgumentParser(description="Council L99 Oracle & Code Validator")
    parser.add_argument("query", nargs="?", help="Task directive or problem to solve")
    parser.add_argument("--file", nargs="+", help="Optional local files to inspect (read-only)")
    parser.add_argument("--write-out", help="Destination path for AST-validated code output")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum response tokens")
    args = parser.parse_args()

    full_prompt = args.query or ""
    if args.file:
        inspected = inspect_local_targets(args.file)
        if inspected:
            full_prompt += f"\n\n[INSPECTED CODE CONTEXT]:\n{inspected}"

    if not full_prompt.strip():
        print("Usage: python deepseek_oracle.py \"Directive\" [--file path] [--write-out dest.py]")
        sys.exit(1)

    res = consult_deepseek_oracle(full_prompt, max_tokens=args.max_tokens)
    if not res["success"]:
        print(f"[ERROR] {res['error']}")
        sys.exit(1)

    print(f"\n### OUTPUT ({res['model']}) ###")
    print(res["content"])
    print(f"Elapsed: {res['elapsed_sec']}s")

    if args.write_out:
        code = extract_clean_code(res["content"])
        ok_ast, msg_ast = validate_python_ast(code)
        if ok_ast:
            out_path = Path(args.write_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(code, encoding="utf-8")
            print(f"[WRITTEN] File saved to {out_path} ({msg_ast})")
        else:
            print(f"[REJECTED] Code did not pass AST validation: {msg_ast}")


if __name__ == "__main__":
    main()
