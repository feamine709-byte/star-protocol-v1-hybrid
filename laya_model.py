#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laya INT8 ONNX Routing & Dynamic MCP Schema Shield Engine (Enterprise Grade)
- Host Tools Immunity: Built-in agent tools (Bash, Read, Edit, Write, Grep, Glob, etc.) are NEVER stripped.
- 100% Dynamic Schema Tokenization, Morphological Synonym Bridging & IDF Weighting.
- Anthropic Prompt Cache Preservation: Deterministic tool sorting (Host tools alphabetically, then MCP tools alphabetically)
  and session-level stability cache (`session_tool_cache`).
"""

import os
import sys
import re
import math
import time
import json
import threading
from collections import OrderedDict, Counter
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["USE_TF"] = "0"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
TOOLS_CACHE = os.path.join(BASE_DIR, "installed_tools.json")

# Calibrated operational thresholds
CONFIDENCE_FAIL_OPEN_THRESHOLD = 75.0
HYBRID_EXEC_PROB_THRESHOLD = 12.0
MAX_LATENCY_BUDGET_MS = 60.0
PROMPT_CACHE_MAX_SIZE = 256
SESSION_CACHE_MAX_SIZE = 128

CLASS_MAPPING = {
    0: {"name": "no_tool", "desc": "شرح نظري / مفاهيمي"},
    1: {"name": "coding_skill", "desc": "كود برمجي نقي وخوارزميات"},
    2: {"name": "mcp_tool", "desc": "تنفيذ أوامر طرفية وأدوات MCP"}
}

# ==============================================================================
# 1. HOST TOOLS IMMUNITY WHITELIST (Built-in Claude Code, Cursor & Agent Tools)
# ==============================================================================
HOST_TOOLS_WHITELIST = frozenset({
    "Bash", "bash", "execute_bash", "terminal", "run_command", "run_terminal_cmd",
    "Read", "read", "read_file", "view_file",
    "Edit", "edit", "edit_file", "MultiEdit", "multi_edit", "str_replace_editor",
    "Write", "write", "write_file", "write_to_file",
    "Grep", "grep", "ripgrep",
    "Glob", "glob", "file_search",
    "LS", "ls", "DirectoryListing", "list_dir",
    "Task", "task", "dispatch_agent",
    "TodoWrite", "todo_write", "TodoRead", "todo_read",
    "NotebookRead", "notebook_read", "NotebookEdit", "notebook_edit",
    "WebFetch", "web_fetch", "WebSearch", "web_search"
})
HOST_TOOLS_LOWER = frozenset(t.lower() for t in HOST_TOOLS_WHITELIST)

# Low-information generic verbs/nouns in tool schemas
GENERIC_TOOL_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "when", "use", "using",
    "get", "set", "list", "create", "update", "delete", "remove", "add", "read",
    "write", "fetch", "call", "run", "execute", "tool", "tools", "server", "mcp",
    "api", "data", "item", "items", "info", "all", "new", "by", "in", "to", "of",
    "on", "is", "are", "can", "will", "into", "over", "via", "name", "type", "id",
    "object", "string", "number", "integer", "boolean", "array", "optional", "required"
})

# ==============================================================================
# 2. MORPHOLOGICAL & ACTION-VERB SYNONYM GROUPS (Zero-RAM Semantic Bridge)
# ==============================================================================
MORPHOLOGICAL_SYNONYM_GROUPS = (
    frozenset({"pay", "payment", "refund", "charge", "invoice", "bill", "billing", "subscription", "customer", "checkout", "payout", "balance"}),
    frozenset({"db", "database", "sql", "query", "table", "schema", "postgres", "postgresql", "sqlite", "mysql", "bigquery", "migration", "row", "record", "select"}),
    frozenset({"repo", "repository", "git", "commit", "branch", "pr", "pull", "push", "merge", "rebase", "diff", "clone", "stash", "checkout"}),
    frozenset({"browser", "page", "webpage", "screenshot", "navigate", "click", "dom", "selector", "devtools", "headless", "url"}),
    frozenset({"docker", "container", "image", "dockerfile", "compose", "kubectl", "kubernetes", "pod", "cluster"}),
    frozenset({"binary", "decompile", "disassemble", "reverse", "assembly", "elf", "macho", "firmware", "frida", "gdb", "rizin", "objdump", "xref"}),
    frozenset({"channel", "message", "thread", "chat", "notify", "notification", "workspace"}),
    frozenset({"ticket", "sprint", "cycle", "backlog", "roadmap", "kanban", "issue", "assignee"})
)

# Pre-compiled lookup map: token -> frozenset of morphological peers
TOKEN_TO_SYNONYM_PEERS = {}
for _grp in MORPHOLOGICAL_SYNONYM_GROUPS:
    for _tok in _grp:
        TOKEN_TO_SYNONYM_PEERS[_tok] = _grp

# Irregular technical stem canonicalization
IRREGULAR_STEM_MAP = {
    "paid": "pay",
    "paying": "pay",
    "payments": "payment",
    "refunded": "refund",
    "refunding": "refund",
    "charged": "charge",
    "charging": "charge",
    "invoices": "invoice",
    "invoicing": "invoice",
    "queries": "query",
    "querying": "query",
    "queried": "query",
    "tables": "table",
    "databases": "database",
    "repositories": "repository",
    "repos": "repo",
    "commits": "commit",
    "committed": "commit",
    "committing": "commit",
    "branches": "branch",
    "merged": "merge",
    "merging": "merge",
    "screenshots": "screenshot",
    "containers": "container",
    "binaries": "binary"
}

THEORETICAL_PREFIXES = (
    "explain ", "explain:", "what is ", "what are ", "what's ", "whats ",
    "difference between ", "the difference between ", "compare ", "comparison between ",
    "how does ", "how do ", "why does ", "why do ", "why is ", "define ", "definition of ",
    "اشرح ", "ما هو ", "ما هي ", "الفرق بين ", "ما الفرق بين ", "وضح ", "قارن بين "
)

STRICT_IMPERATIVE_EXEC_CUES = (
    "run ", "execute ", "deploy ", "take a screenshot", "push ", "pull ", "commit ",
    "open ", "inspect ", "check ", "create a ", "refund ", "charge ", "query ",
    "شغل ", "نفذ ", "افحص ", "ارفع ", "جرب "
)

EXEC_ACTION_CUES = (
    "run ", "execute", "check ", "inspect", "open ", "push", "pull", "commit", "deploy",
    "test it", "run test", "npm ", "pytest", "docker ", "kubectl", "curl ", "terminal",
    "take a screenshot", "browse", "search the web", "search ", "query ", "fetch ", "scroll ",
    "refund", "invoice", "charge", "payment", "pay ", "channel", "sprint", "migration", "sql",
    "شغل", "نفذ", "افحص", "شوف ليا", "دير ", "ارفع", "افتح", "جرب", "تأكد من"
)

CODE_ACTION_CUES = (
    "write ", "implement", "create a function", "create a class", "refactor", "code ",
    "function", "algorithm", "component", "def ", "class ", "اكتب", "صاوب", "دالة",
    "كود", "خوارزمية", "صلح الكود", "عدّل"
)


def _stem_token(tok: str) -> str:
    """
    Enhanced morphological stemmer for technical identifiers and action verbs.
    Maps irregular forms and strips suffixes (-ies, -tion, -ing, -ed, -es, -s).
    """
    if not tok:
        return ""
    if tok in IRREGULAR_STEM_MAP:
        return IRREGULAR_STEM_MAP[tok]
    if len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("es") and len(tok) > 4 and not tok.endswith(("ches", "shes", "sses", "xes")):
        return tok[:-1]
    if tok.endswith("s") and len(tok) > 3 and not tok.endswith(("ss", "us", "is", "os")):
        return tok[:-1]
    if tok.endswith("ing") and len(tok) > 5:
        base = tok[:-3]
        if len(base) >= 3 and base[-1] == base[-2]:
            base = base[:-1]
        return IRREGULAR_STEM_MAP.get(base, base)
    if tok.endswith("ed") and len(tok) > 4:
        base = tok[:-2]
        return IRREGULAR_STEM_MAP.get(base, base)
    return tok


def _tokenize_identifier(text: str) -> list:
    """
    Dynamically splits snake_case, kebab-case, camelCase, PascalCase, and natural text
    into normalized tokens and morphological stems.
    """
    if not text:
        return []
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(text))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"[^a-zA-Z0-9\u0600-\u06FF]+", " ", s).lower()

    tokens = []
    for raw in s.split():
        if len(raw) >= 2:
            tokens.append(raw)
            stemmed = _stem_token(raw)
            if stemmed != raw and len(stemmed) >= 2:
                tokens.append(stemmed)
    return tokens


def expand_morphological_synonyms(tokens: set) -> set:
    """
    Expands a set of stem tokens with their morphological synonym peers
    (e.g. `pay` <-> `refund` <-> `charge`, `db` <-> `sql` <-> `query` <-> `table`,
    `repo` <-> `git` <-> `commit` <-> `pr`).
    """
    expanded = set()
    for tok in tokens:
        peers = TOKEN_TO_SYNONYM_PEERS.get(tok)
        if peers:
            expanded.update(peers)
        stemmed = _stem_token(tok)
        if stemmed != tok:
            peers_stem = TOKEN_TO_SYNONYM_PEERS.get(stemmed)
            if peers_stem:
                expanded.update(peers_stem)
    return expanded - set(tokens)


def extract_tool_metadata(tool_obj: dict) -> tuple:
    """
    Extracts (tool_name, description, schema_text) from either Anthropic or OpenAI tool format.
    """
    if not isinstance(tool_obj, dict):
        return "", "", ""
    if "function" in tool_obj and isinstance(tool_obj["function"], dict):
        fn = tool_obj["function"]
        t_name = str(fn.get("name", ""))
        t_desc = str(fn.get("description", ""))
        params = fn.get("parameters", {}) or {}
    else:
        t_name = str(tool_obj.get("name", ""))
        t_desc = str(tool_obj.get("description", ""))
        params = tool_obj.get("input_schema", {}) or {}

    prop_parts = []
    if isinstance(params, dict):
        props = params.get("properties", {})
        if isinstance(props, dict):
            for pk, pv in props.items():
                prop_parts.append(str(pk))
                if isinstance(pv, dict) and pv.get("description"):
                    prop_parts.append(str(pv["description"]))

    return t_name, t_desc, " ".join(prop_parts)


def parse_mcp_server_from_tool_name(tool_name: str) -> str:
    """
    Extracts the MCP server namespace if the tool follows `mcp__<server>__<tool>` or `mcp_<server>_<tool>`.
    """
    t_low = tool_name.lower()
    if t_low.startswith("mcp__"):
        parts = t_low.split("__")
        if len(parts) >= 3 and parts[1]:
            return parts[1]
    if t_low.startswith("mcp_"):
        parts = t_low.split("_")
        if len(parts) >= 3 and parts[1]:
            return parts[1]
    return ""


class LayaEngine:
    r"""
    Laya INT8 ONNX Routing & Dynamic MCP Shield Engine:
    - Preserves all Host tools (100% immunity).
    - Prunes only external MCP tools using genuine ONNX softmax probabilities,
      morphological synonym bridging, and dynamic IDF schema matching.
    - Guarantees deterministic tool ordering and session stability for Anthropic Prompt Cache preservation.
    """
    CHARTER_MODE = "TRANSPARENT_BACKGROUND_PROXY"

    def __init__(self, model_path: str = MODEL_DIR):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"[!] Model directory not found: {model_path}")

        self.model_path = model_path
        self.onnx_path = os.path.join(model_path, "model_int8.onnx")
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"[!] Quantized ONNX model not found: {self.onnx_path}")

        self._prompt_cache = OrderedDict()
        self._cache_max_size = PROMPT_CACHE_MAX_SIZE
        self.session_tool_cache = OrderedDict()
        self._session_cache_max_size = SESSION_CACHE_MAX_SIZE
        self._infer_lock = threading.Lock()

        # 1. Native Rust Tokenizer
        tok_json = os.path.join(model_path, "tokenizer", "tokenizer.json")
        if not os.path.exists(tok_json):
            tok_json = os.path.join(model_path, "tokenizer.json")
        self.fast_tokenizer = Tokenizer.from_file(tok_json)
        self.fast_tokenizer.enable_truncation(max_length=128)

        # 2. ONNX Runtime Session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = True
        try:
            sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        except Exception:
            pass
        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        # 3. Dynamic Catalog & IDF Index
        self.tools_catalog = self._load_or_scan_tools()
        self._build_in_memory_index()

        # 4. Warm-up ONNX Runtime
        for wq in ("warmup query one", "run git status", "write python code"):
            enc = self.fast_tokenizer.encode(wq)
            self.session.run(None, {
                "input_ids": np.array([enc.ids], dtype=np.int64),
                "attention_mask": np.array([enc.attention_mask], dtype=np.int64)
            })

    def _load_or_scan_tools(self) -> dict:
        if not os.path.exists(TOOLS_CACHE):
            from scanner import ToolDiscoveryScanner
            return ToolDiscoveryScanner().scan_all()
        try:
            with open(TOOLS_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            from scanner import ToolDiscoveryScanner
            return ToolDiscoveryScanner().scan_all()

    def _build_entity_profile(self, entity_id: str, info: dict, is_mcp: bool = True) -> dict:
        weighted_tokens = {}

        # 1. Server/Skill ID & Name tokens (Weight: 5.0)
        id_toks = _tokenize_identifier(entity_id) + _tokenize_identifier(info.get("name", ""))
        for t in id_toks:
            if t not in GENERIC_TOOL_STOPWORDS:
                weighted_tokens[t] = max(weighted_tokens.get(t, 0.0), 5.0)

        # 2. Sub-tools names (Weight: 3.2)
        sub_tools = info.get("sub_tools", []) or []
        sub_tool_phrases = set()
        for st in sub_tools:
            st_clean = str(st).lower().strip()
            sub_tool_phrases.add(st_clean)
            sub_tool_phrases.add(st_clean.replace("_", " ").replace("-", " "))
            for t in _tokenize_identifier(st_clean):
                if t not in GENERIC_TOOL_STOPWORDS:
                    weighted_tokens[t] = max(weighted_tokens.get(t, 0.0), 3.2)
                else:
                    weighted_tokens[t] = max(weighted_tokens.get(t, 0.0), 0.3)

        # 3. Category, Description & Keywords (Weight: 1.8)
        cat_str = info.get("category", "")
        desc_str = info.get("description", "")
        kw_list = info.get("keywords", []) or []
        meta_toks = (
            _tokenize_identifier(cat_str)
            + _tokenize_identifier(desc_str)
            + _tokenize_identifier(" ".join(str(k) for k in kw_list))
        )
        for t in meta_toks:
            if t not in GENERIC_TOOL_STOPWORDS:
                weighted_tokens[t] = max(weighted_tokens.get(t, 0.0), 1.8)
            else:
                weighted_tokens[t] = max(weighted_tokens.get(t, 0.0), 0.2)

        # 4. Morphological synonym expansion (Weight: 1.6)
        synonym_peers = expand_morphological_synonyms(set(weighted_tokens.keys()))
        for syn_tok in synonym_peers:
            if syn_tok not in weighted_tokens:
                weighted_tokens[syn_tok] = 1.6

        return {
            "id": entity_id,
            "is_mcp": is_mcp,
            "category": cat_str or ("mcp_server" if is_mcp else "skill"),
            "weighted_tokens": weighted_tokens,
            "sub_tool_phrases": sub_tool_phrases,
            "sub_tools_set": {str(s).lower() for s in sub_tools}
        }

    def _build_in_memory_index(self):
        self.mcp_servers = self.tools_catalog.get("mcp_servers", {})
        self.skills = self.tools_catalog.get("skills", {})
        self.all_mcp_ids = list(self.mcp_servers.keys())
        self.all_skill_ids = list(self.skills.keys())
        self.total_entities = len(self.all_mcp_ids) + len(self.all_skill_ids)

        self.total_sub_tools = sum(
            s.get("tools_count", len(s.get("sub_tools", [])))
            for s in self.mcp_servers.values()
        )

        self.mcp_profiles = {}
        self.skill_profiles = {}
        self.all_registered_sub_tools = set()
        doc_freq = Counter()

        for srv_id, srv_info in self.mcp_servers.items():
            prof = self._build_entity_profile(srv_id, srv_info, is_mcp=True)
            self.mcp_profiles[srv_id] = prof
            self.all_registered_sub_tools.update(prof["sub_tools_set"])
            for tok in prof["weighted_tokens"]:
                doc_freq[tok] += 1

        for sk_id, sk_info in self.skills.items():
            prof = self._build_entity_profile(sk_id, sk_info, is_mcp=False)
            self.skill_profiles[sk_id] = prof
            for tok in prof["weighted_tokens"]:
                doc_freq[tok] += 1

        total_docs = max(1, self.total_entities)
        self.token_idf = {
            tok: round(1.0 + math.log((total_docs + 1.0) / (df + 1.0)), 4)
            for tok, df in doc_freq.items()
        }

    def register_dynamic_mcp_server(self, server_id: str, server_schema: dict):
        with self._infer_lock:
            self.mcp_servers[server_id] = server_schema
            self.tools_catalog.setdefault("mcp_servers", {})[server_id] = server_schema
            self._build_in_memory_index()
            self._prompt_cache.clear()

    def clear_session_cache(self, session_id: str = None):
        with self._infer_lock:
            if session_id is None:
                self.session_tool_cache.clear()
            else:
                self.session_tool_cache.pop(session_id, None)

    def is_host_tool(self, tool_name: str) -> bool:
        """
        Returns True if `tool_name` is a built-in Host/Agent tool (Bash, Read, Edit, Write, Grep, Glob, etc.)
        and must NEVER be stripped.
        """
        if not tool_name:
            return False
        t_clean = str(tool_name).strip()
        t_low = t_clean.lower()

        if t_clean in HOST_TOOLS_WHITELIST or t_low in HOST_TOOLS_LOWER:
            return True

        if t_low.startswith("mcp__") or t_low.startswith("mcp_"):
            return False

        if t_low in self.all_registered_sub_tools:
            return False
        for srv_id in self.all_mcp_ids:
            s_low = srv_id.lower()
            if t_low.startswith(f"{s_low}_") or t_low.startswith(f"{s_low}-"):
                return False

        return True

    def is_mcp_tool(self, tool_name: str) -> bool:
        return not self.is_host_tool(tool_name)

    def sort_tools_deterministically(self, tools_list: list) -> list:
        """
        Sorts tools deterministically to preserve Anthropic's 5-minute ephemeral prefix Prompt Cache:
        1. Host tools sorted alphabetically by name first.
        2. Active MCP tools sorted alphabetically by name second.
        """
        def _sort_key(tool_obj: dict):
            t_name, _, _ = extract_tool_metadata(tool_obj)
            host_flag = 0 if self.is_host_tool(t_name) else 1
            return (host_flag, t_name.lower(), t_name)

        return sorted(tools_list, key=_sort_key)

    def score_mcp_relevance(
        self,
        prompt_tokens: set,
        name_tokens: set,
        desc_tokens: set,
        server_ns: str = "",
        dynamic_req_idf: dict = None
    ) -> float:
        """
        Scores relevance between `prompt_tokens` and an MCP tool's schema tokens using:
        1. Exact stem matches weighted by IDF (`name_tokens` = 4.0x, `desc_tokens` = 1.8x).
        2. Morphological synonym group matches (`expand_morphological_synonyms` = 2.2x IDF)
           bridging zero-stem pairs like `pay` <-> `refund` <-> `charge`, `db` <-> `sql` <-> `table`,
           and `repo` <-> `git` <-> `commit` <-> `pr`.
        """
        score = 0.0
        idf_map = dynamic_req_idf or {}
        prompt_synonyms = expand_morphological_synonyms(prompt_tokens)

        for p_tok in prompt_tokens:
            idf = max(self.token_idf.get(p_tok, 1.5), idf_map.get(p_tok, 1.5))
            if p_tok in name_tokens:
                score += 4.0 * idf
            elif p_tok in desc_tokens:
                score += 1.8 * idf

        for syn_tok in prompt_synonyms:
            idf = max(self.token_idf.get(syn_tok, 1.4), idf_map.get(syn_tok, 1.4))
            if syn_tok in name_tokens:
                score += 2.2 * idf
            elif syn_tok in desc_tokens:
                score += 1.1 * idf

        if server_ns and (server_ns in prompt_tokens or server_ns in prompt_synonyms):
            score += 8.0

        return round(score, 3)

    def _score_profile_against_prompt(self, profile: dict, prompt_lower: str, prompt_tokens: set) -> float:
        score = 0.0
        w_toks = profile["weighted_tokens"]
        prompt_synonyms = expand_morphological_synonyms(prompt_tokens)

        for p_tok in prompt_tokens:
            if p_tok in w_toks:
                idf = self.token_idf.get(p_tok, 1.8)
                score += w_toks[p_tok] * idf

        for syn_tok in prompt_synonyms:
            if syn_tok in w_toks:
                idf = self.token_idf.get(syn_tok, 1.5)
                score += w_toks[syn_tok] * 0.65 * idf

        entity_id_clean = profile["id"].lower().replace("-", " ").replace("_", " ")
        if len(entity_id_clean) >= 3 and (profile["id"].lower() in prompt_lower or entity_id_clean in prompt_lower):
            score += 10.0

        for phrase in profile["sub_tool_phrases"]:
            if len(phrase) >= 5 and " " in phrase and phrase in prompt_lower:
                score += 7.5

        return round(score, 3)

    def predict_intent(self, prompt: str) -> dict:
        """Runs genuine INT8 ONNX inference with calibrated softmax probabilities (no clamping)."""
        t0 = time.perf_counter()
        enc = self.fast_tokenizer.encode(prompt)
        ort_inputs = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64)
        }
        logits = self.session.run(None, ort_inputs)[0][0]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        pred_class = int(np.argmax(logits))
        confidence = float(probs[pred_class]) * 100.0

        return {
            "class_id": pred_class,
            "category": CLASS_MAPPING[pred_class]["name"],
            "description": CLASS_MAPPING[pred_class]["desc"],
            "confidence": round(confidence, 2),
            "latency_ms": round(latency_ms, 2),
            "probabilities": {
                CLASS_MAPPING[i]["name"]: round(float(probs[i]) * 100, 2)
                for i in range(len(CLASS_MAPPING))
            }
        }

    def _analyze_domains_in_memory(self, prompt: str) -> dict:
        t_mem0 = time.perf_counter()
        p_lower = prompt.lower()
        p_clean = p_lower.strip().lstrip("؟?!.,:;\"'")
        prompt_tokens = set(_tokenize_identifier(prompt))

        mcp_scores = []
        for srv_id, prof in self.mcp_profiles.items():
            sc = self._score_profile_against_prompt(prof, p_lower, prompt_tokens)
            if sc >= 3.0:
                mcp_scores.append((srv_id, sc, prof["category"]))

        mcp_scores.sort(key=lambda x: x[1], reverse=True)

        targeted_mcp = []
        matched_labels = []
        if mcp_scores:
            top_score = mcp_scores[0][1]
            cutoff = max(3.2, top_score * 0.4)
            for srv_id, sc, cat in mcp_scores:
                if sc >= cutoff:
                    targeted_mcp.append(srv_id)
                    if cat not in matched_labels:
                        matched_labels.append(cat)

        skill_scores = []
        for sk_id, prof in self.skill_profiles.items():
            sc = self._score_profile_against_prompt(prof, p_lower, prompt_tokens)
            if sc >= 3.2:
                skill_scores.append((sk_id, sc))
        skill_scores.sort(key=lambda x: x[1], reverse=True)
        targeted_skills = [sk_id for sk_id, _ in skill_scores[:6]]

        p_exec = p_lower
        for _alg in ("binary search", "depth-first search", "breadth-first search", "search, and", "query parameters", "neighbor query", "range minimum query"):
            p_exec = p_exec.replace(_alg, "")

        has_code_cue = any(c in p_lower for c in CODE_ACTION_CUES)

        starts_with_theory = (
            p_clean.startswith(THEORETICAL_PREFIXES)
            or "difference between " in p_clean[:35]
            or "الفرق بين " in p_clean[:35]
        )
        has_strict_imperative = any(imp in p_exec for imp in STRICT_IMPERATIVE_EXEC_CUES)
        is_theoretical_query = starts_with_theory and not has_strict_imperative and not has_code_cue

        if is_theoretical_query:
            has_exec_cue = False
        else:
            has_exec_cue = any(c in p_exec for c in EXEC_ACTION_CUES)

        is_compound_hybrid = has_exec_cue and has_code_cue
        is_multi_step = len(targeted_mcp) >= 2 and len(matched_labels) >= 2 and not is_theoretical_query

        lookup_ms = (time.perf_counter() - t_mem0) * 1000.0

        return {
            "matched_domains": matched_labels,
            "matched_labels": matched_labels,
            "targeted_mcp": targeted_mcp,
            "targeted_skills": targeted_skills,
            "mcp_scores": {s[0]: s[1] for s in mcp_scores[:5]},
            "has_exec_cue": has_exec_cue,
            "has_code_cue": has_code_cue,
            "is_theoretical_query": is_theoretical_query,
            "is_compound_hybrid": is_compound_hybrid,
            "is_multi_step": is_multi_step,
            "in_memory_lookup_ms": round(lookup_ms, 4)
        }

    def _store_in_cache(self, cache_key: str, result: dict) -> dict:
        if cache_key:
            self._prompt_cache[cache_key] = result.copy()
            if len(self._prompt_cache) > self._cache_max_size:
                self._prompt_cache.popitem(last=False)
        out = result.copy()
        out["cache_hit"] = False
        return out

    def route_and_shield(self, prompt: str, use_cache: bool = True) -> dict:
        t_total0 = time.perf_counter()
        cache_key = prompt.strip() if prompt else ""

        if use_cache and cache_key and cache_key in self._prompt_cache:
            cached = self._prompt_cache[cache_key].copy()
            lookup_ms = round((time.perf_counter() - t_total0) * 1000.0, 4)
            cached["latency_ms"] = lookup_ms
            cached["in_memory_lookup_ms"] = lookup_ms
            cached["cache_hit"] = True
            return cached

        with self._infer_lock:
            if use_cache and cache_key and cache_key in self._prompt_cache:
                t_hit0 = time.perf_counter()
                cached = self._prompt_cache[cache_key].copy()
                lookup_ms = round((time.perf_counter() - t_hit0) * 1000.0, 4)
                cached["latency_ms"] = lookup_ms
                cached["in_memory_lookup_ms"] = lookup_ms
                cached["cache_hit"] = True
                return cached

            t_infer0 = time.perf_counter()
            intent = self.predict_intent(prompt)
            mem_info = self._analyze_domains_in_memory(prompt)

            class_id = intent["class_id"]
            conf = intent["confidence"]
            mcp_prob = intent["probabilities"]["mcp_tool"]

            # 1. Explicit conceptual/theoretical query -> ZERO_TOOL_SHIELD
            if mem_info["is_theoretical_query"]:
                total_ms = round((time.perf_counter() - t_infer0) * 1000.0, 2)
                res = {
                    "proxy_role": self.CHARTER_MODE,
                    "protocol_mode": "ZERO_TOOL_SHIELD",
                    "is_theoretical_query": True,
                    "fail_open_triggered": False,
                    "needs_tool": False,
                    "category": "no_tool",
                    "description": CLASS_MAPPING[0]["desc"],
                    "active_mcp_servers": [],
                    "active_skills": [],
                    "active_tools_list": [],
                    "shielded_tools_count": self.total_entities,
                    "confidence": conf,
                    "latency_ms": total_ms,
                    "in_memory_lookup_ms": mem_info["in_memory_lookup_ms"],
                    "action": f"🛡️ [حجب تام] ({conf}%): حجب جميع الـ {self.total_entities} أداة ومهارة MCP"
                }
                return self._store_in_cache(cache_key, res) if use_cache else res

            # 2. Strong dynamic schema match with execution cue -> route to matched MCP tool(s)
            top_dyn_score = max(mem_info["mcp_scores"].values()) if mem_info["mcp_scores"] else 0.0
            has_strong_schema_match = mem_info["has_exec_cue"] and not mem_info["has_code_cue"] and top_dyn_score >= 6.0
            if has_strong_schema_match and class_id != 2:
                class_id = 2
                intent["category"] = "mcp_tool"

            is_low_confidence = (conf < CONFIDENCE_FAIL_OPEN_THRESHOLD) and not has_strong_schema_match
            is_ambiguous_hybrid = (
                mem_info["is_compound_hybrid"]
                or (class_id != 2 and mcp_prob >= HYBRID_EXEC_PROB_THRESHOLD and mem_info["has_exec_cue"] and not has_strong_schema_match)
                or (class_id == 1 and mem_info["has_exec_cue"] and not mem_info["has_code_cue"] and not has_strong_schema_match)
            )
            fail_open_triggered = is_low_confidence or is_ambiguous_hybrid

            # Path A: FAIL-OPEN SAFE MODE
            if fail_open_triggered:
                candidate_mcp = list(mem_info["targeted_mcp"])
                candidate_skills = list(mem_info["targeted_skills"])

                if not candidate_mcp:
                    candidate_mcp = self.all_mcp_ids[:3]

                shielded_mcp = [m for m in self.all_mcp_ids if m not in candidate_mcp]
                shielded_skills = [s for s in self.all_skill_ids if s not in candidate_skills]
                shielded_count = len(shielded_mcp) + len(shielded_skills)
                total_ms = round((time.perf_counter() - t_infer0) * 1000.0, 2)

                reason_tag = (
                    f"ثقة النموذج ({conf}% < 75%)"
                    if is_low_confidence and not is_ambiguous_hybrid
                    else f"طلب هجين/مركب (كود + تنفيذ، ثقة {conf}%)"
                )

                res = {
                    "proxy_role": self.CHARTER_MODE,
                    "protocol_mode": "FAIL_OPEN_SAFE_MODE",
                    "is_theoretical_query": False,
                    "fail_open_triggered": True,
                    "needs_tool": True,
                    "category": intent["category"],
                    "description": f"شك آمن ({reason_tag}) ⬅️ تمرير الأدوات المشتبه بها دون عرقلة",
                    "active_mcp_servers": candidate_mcp,
                    "active_skills": candidate_skills[:5],
                    "active_tools_list": candidate_mcp + candidate_skills[:5],
                    "shielded_tools_count": shielded_count,
                    "confidence": conf,
                    "latency_ms": total_ms,
                    "in_memory_lookup_ms": mem_info["in_memory_lookup_ms"],
                    "action": (
                        f"🔓 [Fail-Open الآمن] ({reason_tag}): تمرير {len(candidate_mcp)} خادم MCP "
                        f"({candidate_mcp}) + {len(candidate_skills[:5])} مهارة، مع حجب {shielded_count} أداة بعيدة!"
                    )
                }
                return self._store_in_cache(cache_key, res) if use_cache else res

            # Path B: High-Confidence Conceptual Query -> ZERO_TOOL_SHIELD
            if class_id == 0:
                total_ms = round((time.perf_counter() - t_infer0) * 1000.0, 2)
                res = {
                    "proxy_role": self.CHARTER_MODE,
                    "protocol_mode": "ZERO_TOOL_SHIELD",
                    "is_theoretical_query": True,
                    "fail_open_triggered": False,
                    "needs_tool": False,
                    "category": "no_tool",
                    "description": intent["description"],
                    "active_mcp_servers": [],
                    "active_skills": [],
                    "active_tools_list": [],
                    "shielded_tools_count": self.total_entities,
                    "confidence": conf,
                    "latency_ms": total_ms,
                    "in_memory_lookup_ms": mem_info["in_memory_lookup_ms"],
                    "action": f"🛡️ [حجب تام] ثقة عالية ({conf}%): حجب جميع الـ {self.total_entities} أداة ومهارة (0 توكنز مهدرة)"
                }
                return self._store_in_cache(cache_key, res) if use_cache else res

            # Path C: High-Confidence Pure Code Query -> PURE_CODE_PASSTHROUGH
            if class_id == 1:
                total_ms = round((time.perf_counter() - t_infer0) * 1000.0, 2)
                res = {
                    "proxy_role": self.CHARTER_MODE,
                    "protocol_mode": "PURE_CODE_PASSTHROUGH",
                    "is_theoretical_query": False,
                    "fail_open_triggered": False,
                    "needs_tool": False,
                    "category": "coding_skill",
                    "description": intent["description"],
                    "active_mcp_servers": [],
                    "active_skills": [],
                    "active_tools_list": [],
                    "shielded_tools_count": self.total_entities,
                    "confidence": conf,
                    "latency_ms": total_ms,
                    "in_memory_lookup_ms": mem_info["in_memory_lookup_ms"],
                    "action": f"🛡️ [كود نقي] ثقة عالية ({conf}%): حجب كافة خوادم MCP مع بقاء أدوات المضيف الأساسية"
                }
                return self._store_in_cache(cache_key, res) if use_cache else res

            # Path D: Executive MCP Tool Routing (Multi-step or Single-tool Precision)
            if mem_info["is_multi_step"] and len(mem_info["targeted_mcp"]) >= 2:
                active_mcp = mem_info["targeted_mcp"]
                active_skills = mem_info["targeted_skills"][:4]
                shielded_mcp = [m for m in self.all_mcp_ids if m not in active_mcp]
                shielded_count = len(shielded_mcp) + (len(self.all_skill_ids) - len(active_skills))
                total_ms = round((time.perf_counter() - t_infer0) * 1000.0, 2)

                res = {
                    "proxy_role": self.CHARTER_MODE,
                    "protocol_mode": "MULTI_STEP_TARGETED_TOOLSET",
                    "is_theoretical_query": False,
                    "fail_open_triggered": False,
                    "needs_tool": True,
                    "category": "mcp_tool",
                    "description": f"مهمة متسلسلة ({' + '.join(mem_info['matched_labels'])})",
                    "active_mcp_servers": active_mcp,
                    "active_skills": active_skills,
                    "active_tools_list": active_mcp + active_skills,
                    "shielded_tools_count": shielded_count,
                    "confidence": conf,
                    "latency_ms": total_ms,
                    "in_memory_lookup_ms": mem_info["in_memory_lookup_ms"],
                    "action": f"🔗 [مهارات متسلسلة Targeted Toolset]: السماح بـ {active_mcp} معاً، وحجب {shielded_count} أداة بعيدة!"
                }
                return self._store_in_cache(cache_key, res) if use_cache else res

            best_tool_id = (
                mem_info["targeted_mcp"][0]
                if mem_info["targeted_mcp"]
                else self._fallback_keyword_match(prompt)
            )
            best_tool_info = self.mcp_servers.get(best_tool_id, {})
            shielded_count = max(0, self.total_entities - 1)
            total_ms = round((time.perf_counter() - t_infer0) * 1000.0, 2)

            res = {
                "proxy_role": self.CHARTER_MODE,
                "protocol_mode": "SINGLE_TOOL_PRECISION",
                "is_theoretical_query": False,
                "fail_open_triggered": False,
                "needs_tool": True,
                "category": "mcp_tool",
                "description": intent["description"],
                "selected_tool": {
                    "id": best_tool_id,
                    "name": best_tool_info.get("name", best_tool_id),
                    "category": best_tool_info.get("category", "mcp_server"),
                    "tools_count": best_tool_info.get("tools_count", len(best_tool_info.get("sub_tools", []))),
                    "sub_tools": best_tool_info.get("sub_tools", [])[:10]
                },
                "active_mcp_servers": [best_tool_id],
                "active_skills": [],
                "active_tools_list": [best_tool_id],
                "shielded_tools_count": shielded_count,
                "confidence": conf,
                "latency_ms": total_ms,
                "in_memory_lookup_ms": mem_info["in_memory_lookup_ms"],
                "action": f"🎯 [توجيه حصري دقيق]: تفعيل [{best_tool_id}] فقط وحجب {shielded_count} أداة أخرى تلقائياً!"
            }
            return self._store_in_cache(cache_key, res) if use_cache else res

    def _fallback_keyword_match(self, prompt: str) -> str:
        prompt_tokens = set(_tokenize_identifier(prompt))
        p_lower = prompt.lower()
        best_srv = self.all_mcp_ids[0] if self.all_mcp_ids else "default"
        max_sc = -1.0
        for sid, prof in self.mcp_profiles.items():
            sc = self._score_profile_against_prompt(prof, p_lower, prompt_tokens)
            if sc > max_sc:
                max_sc = sc
                best_srv = sid
        return best_srv

    def filter_request_tools(
        self,
        prompt: str,
        tools_list: list,
        route_res: dict = None,
        session_id: str = None,
        required_tool_names: set = None
    ) -> dict:
        if not tools_list:
            return {
                "kept_tools": [],
                "stripped_tools": [],
                "route_result": route_res or self.route_and_shield(prompt)
            }

        if route_res is None:
            route_res = self.route_and_shield(prompt)

        required_lower = {str(n).lower() for n in (required_tool_names or set())}
        session_unlocked_lower = set()
        if session_id and session_id in self.session_tool_cache:
            session_unlocked_lower = set(self.session_tool_cache[session_id])

        # Partition incoming tools into immune Host tools vs candidate MCP tools
        host_tools = []
        mcp_tools = []
        for tool_obj in tools_list:
            t_name, _, _ = extract_tool_metadata(tool_obj)
            if self.is_host_tool(t_name):
                host_tools.append(tool_obj)
            else:
                mcp_tools.append(tool_obj)

        # Compute dynamic IDF across incoming MCP tools' schemas
        req_doc_freq = Counter()
        parsed_mcp_entries = []
        for tool_obj in mcp_tools:
            t_name, t_desc, t_schema = extract_tool_metadata(tool_obj)
            srv_ns = parse_mcp_server_from_tool_name(t_name)
            name_toks = [t for t in _tokenize_identifier(t_name) if t not in GENERIC_TOOL_STOPWORDS]
            desc_toks = [t for t in _tokenize_identifier(t_desc + " " + t_schema) if t not in GENERIC_TOOL_STOPWORDS]
            all_toks = set(name_toks + desc_toks)
            for tok in all_toks:
                req_doc_freq[tok] += 1
            parsed_mcp_entries.append({
                "tool_obj": tool_obj,
                "name": t_name,
                "name_low": t_name.lower(),
                "server_ns": srv_ns,
                "name_toks": set(name_toks),
                "desc_toks": set(desc_toks),
                "all_toks": all_toks
            })

        num_mcp = max(1, len(parsed_mcp_entries))
        dynamic_req_idf = {
            tok: 1.0 + math.log((num_mcp + 1.0) / (df + 1.0))
            for tok, df in req_doc_freq.items()
        }

        prompt_tokens = {t for t in _tokenize_identifier(prompt) if t not in GENERIC_TOOL_STOPWORDS}
        prot_mode = route_res.get("protocol_mode", "")
        category = route_res.get("category", "")
        fail_open_active = route_res.get("fail_open_triggered", False) or prot_mode == "FAIL_OPEN_SAFE_MODE"
        is_zero_tool_shield = (prot_mode == "ZERO_TOOL_SHIELD" or category == "no_tool") and not fail_open_active
        is_conceptual_or_pure_code = (
            is_zero_tool_shield
            or prot_mode == "PURE_CODE_PASSTHROUGH"
            or category == "coding_skill"
        )

        has_exec_intent = (not is_zero_tool_shield) and (
            category == "mcp_tool"
            or any(c in prompt.lower() for c in EXEC_ACTION_CUES)
        )

        active_catalog_servers = {s.lower() for s in route_res.get("active_mcp_servers", [])}
        active_catalog_sub_tools = set()
        for srv_id in route_res.get("active_mcp_servers", []):
            prof = self.mcp_profiles.get(srv_id)
            if prof:
                active_catalog_sub_tools.update(prof["sub_tools_set"])

        server_group_max_score = Counter()
        for entry in parsed_mcp_entries:
            sc = self.score_mcp_relevance(
                prompt_tokens=prompt_tokens,
                name_tokens=entry["name_toks"],
                desc_tokens=entry["desc_toks"],
                server_ns=entry["server_ns"],
                dynamic_req_idf=dynamic_req_idf
            )
            if (
                not is_conceptual_or_pure_code
                and (entry["name_low"] in active_catalog_sub_tools or (entry["server_ns"] and entry["server_ns"] in active_catalog_servers))
            ):
                sc = round(sc + 6.0, 3)
            entry["score"] = sc
            if entry["server_ns"]:
                server_group_max_score[entry["server_ns"]] = max(server_group_max_score[entry["server_ns"]], sc)

        kept_mcp_tools = []
        newly_unlocked_names = set()

        for entry in parsed_mcp_entries:
            t_low = entry["name_low"]
            srv_ns = entry["server_ns"]

            # Rule A: Historical tool_use in conversation history MUST be kept to avoid Anthropic 400 error
            if t_low in required_lower:
                kept_mcp_tools.append(entry["tool_obj"])
                newly_unlocked_names.add(t_low)
                continue

            # Rule B: Previously unlocked tool in the same session stays active for Prompt Cache stability
            if t_low in session_unlocked_lower:
                kept_mcp_tools.append(entry["tool_obj"])
                continue

            # Rule C: During conceptual / pure code queries, strip all non-historical, non-session MCP tools
            if is_conceptual_or_pure_code and not (has_exec_intent and server_group_max_score and max(server_group_max_score.values()) >= 8.0):
                continue

            # Rule D: Dynamic activation of relevant MCP tools (by direct tool score or matched server namespace)
            srv_score = server_group_max_score.get(srv_ns, 0.0) if srv_ns else 0.0
            top_overall_score = max((e["score"] for e in parsed_mcp_entries), default=0.0)
            cutoff = max(4.0, top_overall_score * 0.45)

            if entry["score"] >= cutoff or (srv_ns and srv_score >= max(6.0, top_overall_score * 0.6)):
                kept_mcp_tools.append(entry["tool_obj"])
                newly_unlocked_names.add(t_low)

        if session_id and newly_unlocked_names:
            with self._infer_lock:
                existing = self.session_tool_cache.setdefault(session_id, set())
                existing.update(newly_unlocked_names)
                if len(self.session_tool_cache) > self._session_cache_max_size:
                    self.session_tool_cache.popitem(last=False)

        combined_kept = self.sort_tools_deterministically(host_tools + kept_mcp_tools)
        kept_obj_ids = {id(t) for t in combined_kept}
        stripped_tools = [t for t in tools_list if id(t) not in kept_obj_ids]

        return {
            "kept_tools": combined_kept,
            "stripped_tools": stripped_tools,
            "kept_host_tools_count": len(host_tools),
            "kept_mcp_tools_count": len(kept_mcp_tools),
            "route_result": route_res
        }
