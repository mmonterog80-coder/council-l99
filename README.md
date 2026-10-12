# Council L99 — Local Multi-Agent Orchestration (MCP)

Multi-agent orchestration pipeline running open models on Ollama, exposed via the Model Context Protocol (MCP) with persistent vector memory and automated AST code verification.

> **Status:** Working prototype.

---

## Overview

Council L99 is a local-first multi-agent system designed to review technical tasks, architecture proposals, and code implementations. It coordinates specialized local models sequentially, injects relevant past context via SQLite vector search, and validates generated code syntax with AST parsing before writing.

### Core Components
- **MCP Server (`mcp_council_server.py`):** Standard JSON-RPC 2.0 stdio server providing tools for deliberation, memory retrieval, and on-demand code validation (`ast.parse`, Bandit SAST, Ruff linter) intended for MCP clients (not tested with specific clients).
- **Sequential Deliberation (`council_deliberate.py`):** Coordinates specialized models on Ollama (Adversarial Auditor, Business/UX Evaluator, Technical Architect) with dynamic VRAM management (`keep_alive=0`).
- **Persistent Vector Memory (`persistent_brain.py`):** Stores architectural axioms and deliberation summaries in SQLite using cosine similarity on local embeddings (`nomic-embed-text`).
- **Synthesis & AST Oracle (`deepseek_oracle.py`):** Provides code generation and technical synthesis via optional DeepSeek Cloud API or local Ollama models (`qwen2.5-coder:7b`/`1.5b`), verifying AST parsing before writing code to disk.

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │  MCP Client (Desktop / IDE)      │
                    └────────────────┬────────────────┘
                                     │ JSON-RPC (stdio)
                    ┌────────────────▼────────────────┐
                    │      mcp_council_server.py      │
                    └───────┬───────────────┬─────────┘
                            │               │
            ┌───────────────▼────────┐  ┌───▼─────────────────────┐
            │  council_deliberate.py │  │    persistent_brain.py  │
            │  (Sequential Review)   │  │ (SQLite + Embeddings)   │
            └───────────────┬────────┘  └───┬─────────────────────┘
                            │               │
                            ▼               ▼
                   Local Ollama Server (127.0.0.1:11434)
              - deepseek-r1:8b  - llama3.1:8b  - nomic-embed-text
              - qwen2.5-coder:7b
```

---

## Model Execution & Hybrid Oracle Disclosure

- **Default Execution (100% Local):** All agent turns and vector embeddings execute locally on Ollama (`http://127.0.0.1:11434`). No data leaves the local machine.
- **Optional External Oracle (`DEEPSEEK_API_KEY`):** When the `DEEPSEEK_API_KEY` environment variable is explicitly provided, `deepseek_oracle.py` can optionally consult DeepSeek Cloud API for secondary synthesis and review. If this key is absent or undefined, or if the cloud request fails, the oracle automatically falls back to local Ollama inference (`qwen2.5-coder:7b`).

---

## Safeguards & Code Verification

1. **Static Code Verification Tool:** The `code_audit_and_verify` tool validates a given Python file with `ast.parse()`, Bandit SAST, and the Ruff linter.
2. **AST Parsing on Code Output:** When `deepseek_oracle.py` writes code via `--write-out`, it verifies that the extracted code passes `ast.parse()` before saving to disk.
3. **Static Analysis & SAST Audit Status:** Bandit static analysis reports 3 Low-severity findings in `mcp_council_server.py` due to invoking the `ruff` and `bandit` CLI tools via `subprocess.run` (without `shell=True`). Ruff linter reports 0 issues.

---

## Setup & Running

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running locally
- Required local models:
  ```bash
  ollama pull deepseek-r1:8b
  ollama pull llama3.1:8b
  ollama pull qwen2.5-coder:7b
  ollama pull qwen2.5-coder:1.5b
  ollama pull nomic-embed-text
  ```

### Installation

```bash
# Clone the repository
git clone https://github.com/mmonterog80-coder/council-l99.git
cd council-l99

# Install dependencies
pip install requests numpy ruff bandit

# Optional: Set environment variables in your shell (.env files are not loaded automatically)
# PowerShell:
$env:DEEPSEEK_API_KEY="your_optional_cloud_key"
# Linux / macOS:
export DEEPSEEK_API_KEY="your_optional_cloud_key"
```

### Run Deliberation CLI

```bash
python council_deliberate.py "Evaluate offline synchronization design for mobile inspection app"
```

### Connect to MCP Client

Add the server to your `claude_desktop_config.json` or `mcp_config.json`:

```json
{
  "mcpServers": {
    "council-l99": {
      "command": "python",
      "args": ["/absolute/path/to/council-l99/mcp_council_server.py"]
    }
  }
}
```

---

## Verification & Testing Status

### What Was Tested (Live Hardware & Smoke Verified):
- Successful import of all 4 modules (`mcp_council_server`, `council_deliberate`, `deepseek_oracle`, `persistent_brain`).
- Dynamic SQLite schema creation (`init_memory_db()`).
- Live vector memory storage and cosine similarity retrieval with `nomic-embed-text` on local Ollama (`persistent_brain.py` / `persistent_memory`).
- Live sequential multi-model deliberation on local Ollama across all 3 council models (`deepseek-r1:8b`, `llama3.1:8b`, `qwen2.5-coder:7b`) with dynamic per-model VRAM release (`keep_alive=0`).
- JSON-RPC 2.0 stdio handshake (`initialize`, `tools/list`, and `tools/call` for `code_audit_and_verify`).
- Automated AST syntax parsing (`ast.parse`) and static analysis invocation (`code_audit_and_verify` with Ruff and Bandit).

### What Was NOT Tested:
- Tool invocation of `council_deliberate_all` via MCP protocol from external IDE clients.
- Direct Cloud synthesis tool call (`deepseek_delegate`) via MCP protocol.
- Vector memory recall via MCP protocol (`persistent_memory`).
- Cloud fallback behavior without API key when calling MCP delegate.

### Deliberation Output Notes:
- **Token Limits & Truncation:** `council_deliberate.py` enforces a `num_predict` parameter (250–300 tokens) per model to guarantee fast response times on local GPU hardware; verbose responses may stop mid-sentence if they exceed this limit.
- **Reasoning Tokens:** Reasoning models such as `deepseek-r1:8b` stream internal Chain-of-Thought deliberation tokens directly in standard output before providing their final structured feedback.

### Optional / External Features:
- Optional Cloud synthesis (`deepseek_delegate`) requires active external internet connectivity and a valid user-provided `DEEPSEEK_API_KEY`.

---

## License

Copyright (c) 2026 Miguel Montero. All rights reserved. Source published for portfolio review.
