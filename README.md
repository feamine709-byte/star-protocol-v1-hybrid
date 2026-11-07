# Star Protocol V1: Experimental Hybrid Local Yes/No Gatekeeper for Antigravity & MCP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Runtime: ONNX INT8](https://img.shields.io/badge/Runtime-ONNX%20INT8-green.svg)](https://onnxruntime.ai/)
[![X (Twitter)](https://img.shields.io/badge/Connect-%40AMiNE__ER7-black.svg?logo=x)](https://x.com/AMiNE_ER7)

> **An experimental, lightweight hybrid edge gatekeeper designed to dynamically prune and filter Model Context Protocol (MCP) tools and Agentic Skills locally before forwarding requests to massive cloud LLMs.**

---

## 📖 The Story & Problem Statement

In modern agentic ecosystems—such as **Google Antigravity**, **Claude Code**, and **Cursor**—AI agents are granted access to dozens or hundreds of MCP tools, APIs, and agent skills. 

However, sending the full tool manifests and JSON schemas on every conversation turn causes:
1. **Context Bloat:** Thousands of redundant tokens consumed on system instructions and tool schemas.
2. **Escalating Costs:** Inflated token usage per interaction, especially on high-tier frontier models.
3. **Inference Latency & Hallucinations:** Large models become slower and more prone to selecting incorrect tools when overwhelmed with massive, irrelevant tool libraries.

### The Hybrid Solution: Local Yes/No Gatekeeper

**Star Protocol V1** was created as a pioneering experiment in **Hybrid Edge-to-Cloud AI Routing**. 

Instead of streaming everything directly to the cloud LLM, an ultra-fast local edge model (**Laya INT8 ONNX**, <60ms) acts as an intelligent front-line firewall:

```
[ User Prompt ]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│    Star Protocol V1: Local Yes/No Gatekeeper (<60ms)       │
│    (Laya INT8 ONNX Model + Dynamic Schema Tokenizer)        │
└─────────────────────────────────────────────────────────────┘
       │
       ├───► [Pure Conceptual / Explanation] ──► 🛡️ 100% Tools Pruned (Zero Overhead)
       │
       ├───► [Pure Algorithm / Code Logic]   ──► ⚡ Coding Tools Only
       │
       └───► [Action / Execution Query]     ──► 🎯 Only Matching MCP Tools Passed
                                                 (Host Tools Safely Whitelisted)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│         Upstream Cloud LLM (Gemini / Claude / GPT)          │
│               Only Receives The Exact Tools Needed           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Capabilities

- **Lightweight INT8 Quantized Model:** Runs locally on CPU via ONNX Runtime with minimal memory footprint.
- **Three-Way Classification Gate:**
  - `no_tool` (Class 0): Theoretical or conceptual questions. All MCP schemas are stripped to preserve 100% prompt cache and eliminate tool overhead.
  - `coding_skill` (Class 1): Pure code logic and algorithms without requiring terminal commands.
  - `mcp_tool` (Class 2): System-level actions, file operations, or API calls requiring specific external MCP tools.
- **Host Tools Immunity:** Critical built-in agent actions (`Bash`, `Read`, `Edit`, `run_command`, etc.) are unconditionally whitelisted and protected from being pruned.
- **Dynamic Tool & Skill Discovery (`scanner.py`):** Automatically scans and parses active MCP configs and `SKILL.md` definitions across Antigravity, Claude Code, and Cursor environments.
- **Confidence Fail-Open Strategy:** If the model's confidence falls below the calibrated threshold (`CONFIDENCE_FAIL_OPEN_THRESHOLD = 75.0%`), it safely fails open to ensure zero disruption to the user's agent workflows.

---

## 🔬 Experimental Results & Challenges

### Promising Findings in Antigravity
During live testing in the **Google Antigravity** environment, Star Protocol V1 demonstrated significant token reductions:
- Eliminated up to **80–95% of tool schema token bloat** during non-execution turns.
- Preserved prompt cache stability through deterministic schema ordering.
- Kept the agent responsive by filtering out unneeded tools prior to sending requests upstream.

### Challenges Encountered
While the concept proved effective, experimental deployment revealed real-world hurdles:
1. **Local Training Data Distribution:** Edge cases and ambiguous user prompts required significantly more diverse and specialized domain training data to achieve near-perfect classification accuracy.
2. **Python Runtime & Latency Ceilings:** While raw ONNX tensor inference executed in under 20ms, Python's runtime overhead, garbage collection, and GIL (Global Interpreter Lock) introduced latency jitter when operating as an in-flight unbuffered streaming reverse proxy.

### 🚀 The Evolution to V2.5 (Rust)
The insights and lessons gathered from this V1 Python experiment directly shaped the architecture of **Star Protocol V2.5**, which has been rewritten from the ground up in **Rust** to achieve microsecond-level proxy routing, true non-blocking async concurrency, and zero-overhead memory safety.

---

## 📂 Repository Structure

```
.
├── cli.py                 # Interactive & benchmark CLI interface
├── laya_model.py          # Core INT8 ONNX inference engine & schema filter
├── scanner.py             # Automatic MCP and Agentic Skill discovery scanner
├── installed_tools.json   # Cached registry of discovered MCP tools & schemas
├── pyproject.toml         # Package configuration and dependencies
├── LICENSE                # Official MIT License
├── README.md              # Project documentation
└── model/                 # Local INT8 ONNX Gatekeeper weights & tokenizer
    ├── config.json
    ├── model_int8.onnx     # Quantized local ONNX model (tracked via Git LFS)
    ├── rl_agent_config.json
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Git LFS (for cloning the quantized ONNX model weights)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/feamine709-byte/star-protocol-v1-hybrid.git
   cd star-protocol-v1-hybrid
   git lfs pull
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Run tool discovery scanner:**
   ```bash
   python scanner.py
   ```

4. **Launch the CLI gatekeeper:**
   ```bash
   # Single query test
   python cli.py start -q "Explain the concept of quantum computing"

   # Single query with MCP execution requirement
   python cli.py start -q "Process a refund for customer stripe_cus_123"

   # Interactive testing session
   python cli.py start
   ```

---

## 🤝 Calling the Open Source Community

We are open-sourcing Star Protocol V1 to invite AI researchers, agent developers, and systems engineers to collaborate:
- **Model Fine-Tuning & Datasets:** Help contribute dataset samples for edge-case tool intent classification.
- **Quantization & Distillation:** Experiment with smaller, specialized model architectures (e.g., MobileBERT, MiniLM, custom INT4/INT8 classifiers).
- **Benchmarking:** Measure token savings and latency impacts across diverse MCP server setups.

Feel free to fork this repository, submit Pull Requests, or open an Issue with your benchmark findings!

---

## 💬 Connect & Discussion

I would love to hear your thoughts, feedback, and ideas:

- **Author:** AMiNE ER7
- **X (Twitter):** [@AMiNE_ER7](https://x.com/AMiNE_ER7)
- **GitHub:** [@feamine709-byte](https://github.com/feamine709-byte)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
