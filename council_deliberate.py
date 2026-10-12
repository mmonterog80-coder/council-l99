#!/usr/bin/env python
"""
COUNCIL L99 — MULTI-MODEL LOCAL DELIBERATION PIPELINE
Orchestrates sequential multi-agent review across specialized models running on Ollama.
Manages VRAM dynamically with keep_alive=0 and injects episodic vector context.
"""

import argparse
import io
import os
import sqlite3
import sys
import time

import requests

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, io.UnsupportedOperation):
        pass

try:
    from persistent_brain import format_memory_context, record_deliberation
except ImportError:
    format_memory_context = None
    record_deliberation = None

try:
    from deepseek_oracle import consult_deepseek_oracle
except ImportError:
    consult_deepseek_oracle = None

OLLAMA_API = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434/api/generate")

ORDERED_COUNCIL = [
    "deepseek-r1:8b",
    "llama3.1:8b",
    "qwen2.5-coder:7b",
]

ROLES = {
    "deepseek-r1:8b": {
        "title": "Adversarial Code & Logic Auditor (DeepSeek-R1 8B)",
        "num_predict": 300,
        "system": (
            "You are the Adversarial Auditor for Council L99. Attack the problem looking for "
            "edge cases, potential vulnerabilities, state regressions, and architectural traps."
        )
    },
    "llama3.1:8b": {
        "title": "UX & Business Constraints Evaluator (Llama 3.1 8B)",
        "num_predict": 250,
        "system": (
            "You are the Business & Operational Evaluator. Assess practical usability, operational constraints, "
            "clarity, and human interface considerations."
        )
    },
    "qwen2.5-coder:7b": {
        "title": "Technical Systems Architect (Qwen 2.5 Coder 7B)",
        "num_predict": 350,
        "system": (
            "You are the Technical Architect. Review data structures, typing, algorithms, "
            "performance, and maintainable implementation patterns."
        )
    }
}


def unload_model_ram(model_name: str):
    """Frees model VRAM immediately after its turn (keep_alive=0)."""
    try:
        requests.post(OLLAMA_API, json={"model": model_name, "keep_alive": 0}, timeout=5)
    except requests.RequestException:
        pass


def query_local_model(model: str, accumulated_prompt: str, timeout: int = 120) -> str:
    """Queries a local model via Ollama and releases VRAM immediately."""
    config = ROLES.get(model, {
        "title": model,
        "num_predict": 250,
        "system": "You are a specialized technical assistant."
    })

    payload = {
        "model": model,
        "prompt": accumulated_prompt,
        "system": config["system"],
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0.2,
            "num_ctx": 2048,
            "num_predict": config["num_predict"]
        }
    }
    t0 = time.time()
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=timeout)
        elapsed = round(time.time() - t0, 2)
        unload_model_ram(model)
        if resp.status_code == 200:
            res_text = resp.json().get("response", "").strip()
            if not res_text:
                thinking = resp.json().get("thinking", "").strip()
                if "</think>" in thinking:
                    res_text = thinking.split("</think>")[-1].strip()
                elif thinking:
                    res_text = thinking
            res_text = res_text.replace("<think>\n</think>", "").strip()
            return f"{res_text}\n\n*(Model: {model} | Elapsed: {elapsed}s)*"
        return f"[HTTP {resp.status_code} Error from {model}]"
    except requests.RequestException as e:
        unload_model_ram(model)
        return f"[Exception calling {model} after {round(time.time() - t0, 1)}s: {e}]"


def deliberate(topic: str, models: list[str] | None = None, call_oracle: bool = False) -> dict[str, str]:
    """
    Executes collegiate multi-agent review across local models with persistent context.
    """
    t_start = time.time()
    active_models = models or list(ORDERED_COUNCIL)
    results: dict[str, str] = {}

    mem_context = ""
    if format_memory_context:
        try:
            mem_context = format_memory_context(topic, top_k=3)
        except (requests.RequestException, sqlite3.Error, KeyError):
            mem_context = ""

    if mem_context:
        results["memory_rag"] = mem_context

    accumulated_context = f"{mem_context}\n\n[PRIMARY DIRECTIVE]:\n{topic}\n" if mem_context else f"[PRIMARY DIRECTIVE]:\n{topic}\n"

    for mod in active_models:
        title = ROLES.get(mod, {}).get("title", mod)
        prompt_turn = (
            f"{accumulated_context}\n"
            f"[ROLE DIRECTIVE FOR {mod}]: Provide your structured assessment (up to 5 key points) "
            f"building upon previous feedback without duplicating points."
        )
        opinion = query_local_model(mod, prompt_turn)
        results[mod] = opinion
        summary_slice = opinion[:800]
        accumulated_context += f"\n--- Feedback from {title} ---\n{summary_slice}\n"

    # Optional synthesis via secondary oracle
    if call_oracle and consult_deepseek_oracle:
        prompt_synthesis = (
            f"{accumulated_context}\n\n"
            "[SYNTHESIS MISSION]: Synthesize the collegiate deliberation into an actionable execution plan."
        )
        oracle_res = consult_deepseek_oracle(prompt_synthesis, inject_memory=False)
        if oracle_res.get("success"):
            results["oracle_synthesis"] = (
                f"{oracle_res['content']}\n\n"
                f"*(Synthesizer: {oracle_res['model']} | Elapsed: {oracle_res['elapsed_sec']}s)*"
            )

    elapsed_total = round(time.time() - t_start, 2)
    if record_deliberation:
        try:
            synthesis = results.get("oracle_synthesis") or list(results.values())[-1]
            record_deliberation(topic, synthesis, list(results.keys()), elapsed_total)
        except (sqlite3.Error, OSError):
            pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Council L99 Multi-Model Deliberation")
    parser.add_argument("topic", nargs="?", help="Topic, architecture, or directive to deliberate")
    parser.add_argument("--models", nargs="+", help="Specific models to invoke")
    parser.add_argument("--oracle", action="store_true", help="Invoke secondary oracle for final synthesis")
    args = parser.parse_args()

    if not args.topic:
        print("Usage: python council_deliberate.py \"Directive to deliberate\" [--oracle]")
        sys.exit(1)

    print(f"\nDeliberating: {args.topic}\n" + "=" * 60)
    findings = deliberate(args.topic, models=args.models, call_oracle=args.oracle)

    for mod, text in findings.items():
        title = ROLES.get(mod, {}).get("title", mod)
        print(f"\n### {title} ###")
        print(text)
        print("-" * 60)


if __name__ == "__main__":
    main()
