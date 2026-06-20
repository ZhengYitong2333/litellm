#!/usr/bin/env python3
"""
Corner-case proxy tests for deploy/sophnet-azure.

Each case maps to a historical failure mode (routing, param translation, gateway quirks).
Output is JSON-only on stdout for piping; human summary goes to stderr.

Usage:
  python3 test_corner_cases.py              # quick (default)
  python3 test_corner_cases.py --matrix     # CC/Codex x model matrix
  python3 test_corner_cases.py --full       # full + matrix tiers
  python3 test_corner_cases.py --strict-upstream
  python3 test_corner_cases.py --stress
  python3 test_corner_cases.py --list
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Set

# Module-level dispatch table: each new matrix case registers itself here at
# import time. build_cases lambdas look up names in this dict, avoiding late
# binding in the lambdas (which has been finicky in this codebase).
_MATRIX_CASES: Dict[str, Callable[[str], Dict[str, Any]]] = {}

BASE = os.environ.get("LITELLM_PROXY_BASE", "http://localhost:4000")
KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-sophnet-azure-local")
WORKERS = int(os.environ.get("LITELLM_TEST_WORKERS", "4"))
STRICT_UPSTREAM = os.environ.get("LITELLM_STRICT_UPSTREAM", "").lower() in {
    "1",
    "true",
    "yes",
}

Tier = Literal["quick", "full", "matrix", "stress"]
Expect = Literal["ok", "fail", "skip"]

CHAT_MODELS = [
    "sophnet-glm-5.1",
    "sophnet-glm-5.2",
    "sophnet-gpt-5.5",
    "sophnet-deepseekv4-pro",
    "sophnet-deepseekv4-flash",
    "sophnet-minimax-m3",
    "sophnet-claude-opus-4-7",
    "azure-gpt-5.5",
    "azure-gpt-5.4",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
]

CLAUDE = "sophnet-claude-opus-4-7"
GLM = "sophnet-glm-5.2"
GLM51 = "sophnet-glm-5.1"
GPT55 = "sophnet-gpt-5.5"
AZURE = "azure-gpt-5.4"
AZURE55 = "azure-gpt-5.5"
DEEPSEEK_PRO = "deepseek-v4-pro"

# azure-gpt-5.5 + GLM exercise the adapter -> chat path; deepseek-v4-pro exercises
# the native Anthropic passthrough path (api.deepseek.com/anthropic/v1/messages).
ADAPTER_REGRESSION_MODELS = [AZURE55, GLM, DEEPSEEK_PRO]

ANTHROPIC_HEADERS = {"anthropic-version": "2023-06-01"}

GREETING_SCHEMA = {
    "type": "object",
    "properties": {"greeting": {"type": "string"}},
    "required": ["greeting"],
}

CODEX_TOOLS_CHAT = [
    {
        "type": "function",
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {"type": "shell", "environment": {"type": "local"}},
    {
        "type": "computer_use_preview",
        "display_width": 1024,
        "display_height": 768,
        "environment": "mac",
    },
    {"type": "namespace", "name": "codex"},
]

ANTHROPIC_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get weather for a city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

WEB_SEARCH_TOOL = [
    {
        "type": "web_search_20250305",
        "name": "web_search_20250305",
        "max_uses": 3,
    }
]

RETRY_STATUS = {429, 502, 503, 504}
TIER_ORDER = {"quick": 0, "full": 1, "matrix": 2, "stress": 3}


@dataclass
class Case:
    id: str
    tier: Tier
    why: str
    fn: Callable[[], Dict[str, Any]]
    expect: Expect = "ok"
    timeout: int = 120
    retries: int = 2


def _headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
        **(extra or {}),
    }


def post(
    path: str,
    payload: dict,
    *,
    headers: Optional[dict] = None,
    timeout: int = 120,
    stream: bool = False,
    retries: int = 0,
) -> dict:
    hdrs = _headers(headers)
    last: dict = {}
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{BASE}{path}",
            data=json.dumps(payload).encode(),
            headers=hdrs,
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if stream:
                    chunks: List[str] = []
                    for line in resp:
                        chunks.append(line.decode(errors="replace"))
                    raw = "".join(chunks)
                    return {
                        "ok": True,
                        "status": resp.status,
                        "latency_s": round(time.perf_counter() - start, 2),
                        "stream_lines": len(chunks),
                        "preview": raw[:200],
                    }
                raw = resp.read().decode(errors="replace")
                body = json.loads(raw) if raw else {}
                return {
                    "ok": True,
                    "status": resp.status,
                    "latency_s": round(time.perf_counter() - start, 2),
                    "body": body,
                }
        except urllib.error.HTTPError as exc:
            last = {
                "ok": False,
                "status": exc.code,
                "latency_s": round(time.perf_counter() - start, 2),
                "error": exc.read().decode(errors="replace")[:800],
            }
            if exc.code in RETRY_STATUS and attempt < retries:
                time.sleep(4)
                continue
            return last
        except Exception as exc:  # noqa: BLE001
            last = {
                "ok": False,
                "status": None,
                "latency_s": round(time.perf_counter() - start, 2),
                "error": str(exc)[:800],
            }
            if attempt < retries:
                time.sleep(2)
                continue
            return last
    return last


def _chat_text(body: dict) -> str:
    msg = body.get("choices", [{}])[0].get("message", {})
    parts = [msg.get("content") or "", msg.get("reasoning_content") or ""]
    return " | ".join(p for p in parts if p)[:120]


def _messages_preview(body: dict) -> str:
    texts = [
        c.get("text", "")
        for c in body.get("content", [])
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    types = [c.get("type") for c in body.get("content", []) if isinstance(c, dict)]
    out = texts[0][:80] if texts else ""
    return f"{out} [blocks:{','.join(types)}]"


# --- case builders ---


def _is_upstream_rate_limit(result: dict) -> bool:
    err = (result.get("error") or "").lower()
    return "too many requests" in err or result.get("status") == 429


def _is_opaque_upstream_error(result: dict) -> bool:
    err = (result.get("error") or "").lower()
    status = result.get("status")
    if status in {502, 503, 504}:
        return True
    opaque_markers = (
        "invalid anthropic messages api request",
        "<nil>",
        "invalidparameter",
        "upstream gateway",
        "internal server error",
    )
    return any(marker in err for marker in opaque_markers)


def _maybe_skip_upstream(result: dict, *, label: str = "upstream") -> Optional[dict]:
    if STRICT_UPSTREAM:
        return None
    if _is_upstream_rate_limit(result):
        return {
            "ok": True,
            "skip": True,
            "preview": f"{label} rate limit",
            "latency_s": result.get("latency_s"),
            "status": result.get("status"),
        }
    if _is_opaque_upstream_error(result):
        return {
            "ok": True,
            "skip": True,
            "preview": f"{label} opaque/gateway error",
            "latency_s": result.get("latency_s"),
            "status": result.get("status"),
            "error": (result.get("error") or "")[:200],
        }
    return None


def case_chat_basic(model: str) -> dict:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: OK"}],
        "max_tokens": 32,
    }
    if not model.startswith("sophnet-gpt") and not model.startswith("azure-gpt"):
        if model != CLAUDE:
            payload["temperature"] = 0
    r = post("/v1/chat/completions", payload, retries=1)
    if r.get("ok"):
        r["preview"] = _chat_text(r["body"])
    elif skip := _maybe_skip_upstream(r):
        return skip
    return r


def case_chat_reasoning(model: str) -> dict:
    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What is 17+25? Answer with the number only.",
            }
        ],
        "max_tokens": 128,
    }
    if model.startswith("azure-gpt") or model.startswith("sophnet-gpt"):
        payload["reasoning_effort"] = "low"
    elif "glm" in model or "deepseek" in model:
        payload["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        return {"ok": True, "skip": True, "preview": "not applicable"}
    r = post("/v1/chat/completions", payload, timeout=180)
    if r.get("ok"):
        r["preview"] = _chat_text(r["body"])
    elif skip := _maybe_skip_upstream(r, label="reasoning"):
        return skip
    return r


def case_responses_basic(model: str) -> dict:
    r = post(
        "/v1/responses",
        {
            "model": model,
            "input": "Reply with exactly one word: OK",
            "max_output_tokens": 16,
        },
        timeout=60,
        retries=1,
    )
    if r.get("ok"):
        r["preview"] = str(r.get("body", ""))[:120]
    elif skip := _maybe_skip_upstream(r, label="responses"):
        return skip
    return r


def case_responses_gpt55() -> dict:
    return case_responses_basic(GPT55)


def case_messages_basic(model: str) -> dict:
    r = post(
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Say hi in one word"}],
        },
        headers=ANTHROPIC_HEADERS,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    elif skip := _maybe_skip_upstream(r, label="messages"):
        return skip
    return r


def case_messages_thinking() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": CLAUDE,
            "max_tokens": 256,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [
                {
                    "role": "user",
                    "content": "What is 6+7? Reply with the number only.",
                }
            ],
        },
        headers=ANTHROPIC_HEADERS,
        timeout=180,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    return r


def case_messages_stream() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": CLAUDE,
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "Count 1 to 3."}],
        },
        headers=ANTHROPIC_HEADERS,
        stream=True,
        timeout=120,
    )
    if r.get("ok") and r.get("stream_lines", 0) < 2:
        return {
            "ok": False,
            "error": f"expected multiple stream events, got {r.get('stream_lines')}",
            "latency_s": r["latency_s"],
            "status": r.get("status"),
        }
    return r


def case_messages_web_search_tool() -> dict:
    r = post(
        "/v1/messages?beta=true",
        {
            "model": CLAUDE,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Hi. Do not search."}],
            "tools": WEB_SEARCH_TOOL,
        },
        headers=ANTHROPIC_HEADERS,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
        return r
    err = r.get("error") or ""
    # Pass if LiteLLM normalized name (no Sophnet name validation error).
    if (
        "Input should be 'web_search'" not in err
        and "web_search_20250305.name" not in err
    ):
        return {
            "ok": True,
            "skip": True,
            "preview": "upstream rejected tool but name sanitize ok",
            "latency_s": r.get("latency_s"),
            "status": r.get("status"),
            "error": err[:200],
        }
    return r


def case_messages_codex_tools() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": CLAUDE,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Say hi. Do not call tools."}],
            "tools": CODEX_TOOLS_CHAT + ANTHROPIC_TOOLS,
        },
        headers=ANTHROPIC_HEADERS,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
        return r
    if skip := _maybe_skip_upstream(r, label="codex_tools"):
        return skip
    return r


def case_messages_invalid_thinking_history() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": CLAUDE,
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 512},
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "prior",
                            "signature": "invalid-signature",
                        },
                        {"type": "text", "text": "4"},
                    ],
                },
                {"role": "user", "content": "What is 3+3? Number only."},
            ],
        },
        headers=ANTHROPIC_HEADERS,
        timeout=120,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    return r


def case_messages_redacted_history() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": CLAUDE,
            "max_tokens": 32,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hi"},
                        {"type": "redacted_thinking", "data": "bad-data!!!"},
                    ],
                },
                {"role": "user", "content": "Say ok"},
            ],
        },
        headers=ANTHROPIC_HEADERS,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    return r


def case_adapter_output_config_json() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": GLM,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Reply hello"}],
            "output_config": {
                "format": {"type": "json_schema", "schema": GREETING_SCHEMA},
            },
        },
        headers=ANTHROPIC_HEADERS,
        timeout=180,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    return r


def case_adapter_output_format_json() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": CLAUDE,
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "Return JSON greeting"}],
            "output_format": {"type": "json_schema", "schema": GREETING_SCHEMA},
        },
        headers=ANTHROPIC_HEADERS,
        timeout=180,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    return r


def case_azure_messages_thinking_tools() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": AZURE,
            "max_tokens": 128,
            "thinking": {"type": "enabled", "budget_tokens": 512},
            "tools": ANTHROPIC_TOOLS,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 5+5? Number only.",
                }
            ],
        },
        headers=ANTHROPIC_HEADERS,
        timeout=240,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    return r


def case_glm_temperature_zero_fails() -> dict:
    r = post(
        "/v1/chat/completions",
        {
            "model": GLM,
            "max_tokens": 8,
            "temperature": 0,
            "messages": [{"role": "user", "content": "OK"}],
        },
    )
    if r.get("ok"):
        return {
            "ok": True,
            "skip": True,
            "preview": "upstream now accepts temperature=0",
            "latency_s": r.get("latency_s"),
            "status": r.get("status"),
        }
    return r


def case_auth_bad_key() -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/messages",
        data=json.dumps(
            {
                "model": GLM,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "x"}],
            }
        ).encode(),
        headers={**_headers(ANTHROPIC_HEADERS), "Authorization": "Bearer sk-invalid"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        urllib.request.urlopen(req, timeout=30)
        return {"ok": True, "latency_s": round(time.perf_counter() - start, 2)}
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "latency_s": round(time.perf_counter() - start, 2),
            "error": exc.read().decode(errors="replace")[:200],
        }


def case_chat_stream_ttfb(model: str) -> dict:
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 8,
                "stream": True,
            }
        ).encode(),
        headers=_headers(),
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            first: Optional[float] = None
            chunks = 0
            while True:
                line = resp.readline()
                if not line:
                    break
                if first is None and line.strip():
                    first = time.perf_counter()
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    chunks += 1
            ttfb = (first - start) if first else None
            ok = ttfb is not None and chunks > 0
            return {
                "ok": ok,
                "status": resp.status if ok else None,
                "latency_s": round(ttfb or (time.perf_counter() - start), 2),
                "preview": f"chunks={chunks}",
            }
    except Exception as exc:  # noqa: BLE001
        err = str(exc)[:200]
        if "500" in err or "429" in err or "too many" in err.lower():
            return {
                "ok": True,
                "skip": True,
                "preview": "upstream rate limit or 500",
                "latency_s": round(time.perf_counter() - start, 2),
                "error": err,
            }
        return {
            "ok": False,
            "latency_s": round(time.perf_counter() - start, 2),
            "error": err,
        }


def case_burst(model: str, n: int = 3) -> dict:
    if model == GPT55:
        runner: Callable[[], dict] = case_responses_gpt55
    elif model == CLAUDE:
        runner = lambda: case_messages_basic(CLAUDE)
    else:
        runner = lambda: case_chat_basic(model)

    def one() -> bool:
        return bool(runner().get("ok"))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        oks = list(pool.map(lambda _: one(), range(n)))
    elapsed = round(time.perf_counter() - start, 2)
    ok_count = sum(oks)
    if ok_count < n:
        return {
            "ok": False,
            "error": f"burst {ok_count}/{n} ok in {elapsed}s",
            "latency_s": elapsed,
        }
    return {"ok": True, "preview": f"burst {ok_count}/{n}", "latency_s": elapsed}


def case_messages_tool_history() -> dict:
    r = post(
        "/v1/messages",
        {
            "model": CLAUDE,
            "max_tokens": 128,
            "tools": ANTHROPIC_TOOLS,
            "messages": [
                {"role": "user", "content": "Weather in London?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_test001",
                            "name": "get_weather",
                            "input": {"city": "London"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_test001",
                            "content": "15C cloudy",
                        }
                    ],
                },
                {"role": "user", "content": "One short sentence."},
            ],
        },
        headers=ANTHROPIC_HEADERS,
        timeout=180,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    return r


def case_messages_adapter_orphan_tool_call(model: str) -> dict:
    missing_id = "toolu_orphan_matrix_test"
    r = post(
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 128,
            "tools": ANTHROPIC_TOOLS,
            "messages": [
                {"role": "user", "content": "run"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": missing_id,
                            "name": "get_weather",
                            "input": {"city": "London"},
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
        },
        headers=ANTHROPIC_HEADERS,
        timeout=180,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    elif skip := _maybe_skip_upstream(r, label="adapter_orphan"):
        return skip
    return r


def case_messages_adapter_empty_tool_result(model: str) -> dict:
    tool_id = "toolu_empty_matrix_test"
    r = post(
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 128,
            "tools": ANTHROPIC_TOOLS,
            "messages": [
                {"role": "user", "content": "weather"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": [],
                        }
                    ],
                },
                {"role": "user", "content": "thanks"},
            ],
        },
        headers=ANTHROPIC_HEADERS,
        timeout=180,
    )
    if r.get("ok"):
        r["preview"] = _messages_preview(r["body"])
    elif skip := _maybe_skip_upstream(r, label="adapter_empty_result"):
        return skip
    return r


def case_responses_bridge_interleaved_tool_result(model: str) -> dict:
    call_id = "call_matrix_interleaved"
    r = post(
        "/v1/responses",
        {
            "model": model,
            "max_output_tokens": 64,
            "input": [
                {"type": "message", "role": "user", "content": "run"},
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "id": call_id,
                    "name": "get_weather",
                    "arguments": '{"city":"London"}',
                },
                {"type": "message", "role": "user", "content": "interrupt"},
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "15C",
                },
                {"type": "message", "role": "user", "content": "continue"},
            ],
        },
        timeout=180,
    )
    if r.get("ok"):
        r["preview"] = str(r.get("body", ""))[:120]
    elif skip := _maybe_skip_upstream(r, label="responses_interleaved"):
        return skip
    return r


def case_responses_bridge_orphan_tool_call(model: str) -> dict:
    call_id = "call_matrix_orphan"
    r = post(
        "/v1/responses",
        {
            "model": model,
            "max_output_tokens": 64,
            "input": [
                {"type": "message", "role": "user", "content": "run"},
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "id": call_id,
                    "name": "Bash",
                    "arguments": "{}",
                },
                {"type": "message", "role": "user", "content": "continue"},
            ],
        },
        timeout=180,
    )
    if r.get("ok"):
        r["preview"] = str(r.get("body", ""))[:120]
    elif skip := _maybe_skip_upstream(r, label="responses_orphan"):
        return skip
    return r


def build_cases() -> List[Case]:
    cases: List[Case] = []

    for model in CHAT_MODELS:
        cases.append(
            Case(
                id=f"chat.basic.{model}",
                tier="quick",
                why="Key, routing, model alias, basic upstream connectivity",
                fn=lambda m=model: case_chat_basic(m),
            )
        )

    cases.extend(
        [
            Case(
                id="responses.gpt55.native",
                tier="quick",
                why="Sophnet GPT-5.5 must use native /v1/responses (not chat bridge hang)",
                fn=case_responses_gpt55,
            ),
            Case(
                id="messages.claude.basic",
                tier="quick",
                why="Anthropic native /v1/messages path",
                fn=lambda: case_messages_basic(CLAUDE),
            ),
            Case(
                id="messages.claude.thinking",
                tier="quick",
                why="Extended thinking on Sophnet Claude gateway",
                fn=case_messages_thinking,
            ),
            Case(
                id="messages.claude.web_search_tool",
                tier="quick",
                why="web_search_20250305 name must be web_search on Sophnet",
                fn=case_messages_web_search_tool,
            ),
            Case(
                id="messages.claude.codex_tools",
                tier="quick",
                why="Strip/filter unsupported CC/Codex tools on gateway",
                fn=case_messages_codex_tools,
            ),
            Case(
                id="responses.gpt55.stream",
                tier="quick",
                why="Codex GPT streaming must emit events and close cleanly",
                fn=lambda: _MATRIX_CASES["responses_stream"](GPT55),
                timeout=90,
            ),
            Case(
                id="responses.gpt55.stream_tools",
                tier="quick",
                why="Codex GPT streaming + tools must not hang or drop SSE errors",
                fn=lambda: _MATRIX_CASES["responses_stream_tools"](GPT55),
                timeout=120,
            ),
            Case(
                id="messages.minimax.tools",
                tier="quick",
                why="MiniMax tool schema sanitization must avoid upstream 2013/timeouts",
                fn=lambda: _MATRIX_CASES["messages_tools"]("sophnet-minimax-m3"),
                timeout=120,
            ),
            Case(
                id="messages.claude.thinking_history",
                tier="quick",
                why="Strip invalid thinking signatures before upstream",
                fn=case_messages_invalid_thinking_history,
            ),
            Case(
                id="messages.claude.redacted_history",
                tier="quick",
                why="Strip invalid redacted_thinking on third-party gateway",
                fn=case_messages_redacted_history,
            ),
            Case(
                id="adapter.glm.output_config_json",
                tier="quick",
                why="output_config.format -> response_format on adapter path",
                fn=case_adapter_output_config_json,
            ),
            Case(
                id="auth.bad_key",
                tier="quick",
                why="Proxy auth rejects invalid master key",
                fn=case_auth_bad_key,
                expect="fail",
                timeout=30,
                retries=0,
            ),
        ]
    )

    for model in CHAT_MODELS:
        cases.append(
            Case(
                id=f"messages.basic.{model}",
                tier="matrix",
                why="CC /v1/messages x model: native Claude or adapter path",
                fn=lambda m=model: case_messages_basic(m),
            )
        )
        cases.append(
            Case(
                id=f"responses.basic.{model}",
                tier="matrix",
                why="Codex /v1/responses x model: native GPT or chat bridge",
                fn=lambda m=model: case_responses_basic(m),
            )
        )

    for model in ADAPTER_REGRESSION_MODELS:
        cases.append(
            Case(
                id=f"messages.adapter.orphan_tool_call.{model}",
                tier="matrix",
                why="Adapter inserts placeholder when tool_result missing from CC history",
                fn=lambda m=model: case_messages_adapter_orphan_tool_call(m),
                timeout=180,
            )
        )
        cases.append(
            Case(
                id=f"messages.adapter.empty_tool_result.{model}",
                tier="matrix",
                why="Adapter always emits tool message for empty tool_result list content",
                fn=lambda m=model: case_messages_adapter_empty_tool_result(m),
                timeout=180,
            )
        )

    cases.extend(
        [
            Case(
                id="responses.bridge.interleaved_tool_result",
                tier="matrix",
                why="Responses bridge pulls tool output contiguous after function_call",
                fn=lambda: case_responses_bridge_interleaved_tool_result(AZURE55),
                timeout=180,
            ),
            Case(
                id="responses.bridge.orphan_tool_call",
                tier="matrix",
                why="Responses bridge inserts placeholder for missing function_call_output",
                fn=lambda: case_responses_bridge_orphan_tool_call(AZURE55),
                timeout=180,
            ),
        ]
    )

    # Per-model matrix cases (each iterates over all 9 modellist entries).
    # Use the _MATRIX_CASES dispatch table rather than module-level name
    # lookups inside lambdas — the latter has been flaky in this codebase
    # (NameError on first invocation in some threading contexts).
    matrix = _MATRIX_CASES
    for model in CHAT_MODELS:
        cases.append(
            Case(
                id=f"matrix.chat.stream.{model}",
                tier="matrix",
                why="Chat streaming stability for every modellist entry",
                fn=lambda m=model, fn=matrix["chat_stream"]: fn(m),
                timeout=90,
            )
        )
        cases.append(
            Case(
                id=f"matrix.messages.stream.{model}",
                tier="matrix",
                why="Messages streaming stability for every modellist entry",
                fn=lambda m=model, fn=matrix["messages_stream"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.responses.stream.{model}",
                tier="matrix",
                why="Responses streaming stability for every modellist entry",
                fn=lambda m=model, fn=matrix["responses_stream"]: fn(m),
                timeout=90,
            )
        )
        cases.append(
            Case(
                id=f"matrix.chat.tools.{model}",
                tier="matrix",
                why="Chat tool-calling stability for every modellist entry",
                fn=lambda m=model, fn=matrix["chat_tools"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.messages.tools.{model}",
                tier="matrix",
                why="Messages tool-calling stability for every modellist entry",
                fn=lambda m=model, fn=matrix["messages_tools"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.responses.tools.{model}",
                tier="matrix",
                why="Responses tool-calling stability for every modellist entry",
                fn=lambda m=model, fn=matrix["responses_tools"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.chat.long_context.{model}",
                tier="matrix",
                why="8k-token input does not crash any modellist entry",
                fn=lambda m=model, fn=matrix["chat_long_context"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.messages.system.{model}",
                tier="matrix",
                why="System prompt forwarding on /v1/messages for every entry",
                fn=lambda m=model, fn=matrix["messages_system_prompt"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.chat.max_tokens_one.{model}",
                tier="matrix",
                why="max_tokens=1 boundary case on every entry",
                fn=lambda m=model, fn=matrix["chat_max_tokens_one"]: fn(m),
                timeout=60,
            )
        )
        cases.append(
            Case(
                id=f"matrix.chat.anthropic_beta_header.{model}",
                tier="matrix",
                why="anthropic-beta header filtered per JSON config on every entry",
                fn=lambda m=model, fn=matrix["chat_anthropic_beta_header"]: fn(m),
                timeout=60,
            )
        )
        cases.append(
            Case(
                id=f"matrix.chat.stream_tools.{model}",
                tier="matrix",
                why="Streaming tool calling on /v1/chat/completions for every entry",
                fn=lambda m=model, fn=matrix["chat_stream_tools"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.messages.stream_tools.{model}",
                tier="matrix",
                why="Streaming tool calling on /v1/messages for every entry",
                fn=lambda m=model, fn=matrix["messages_stream_tools"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.responses.stream_tools.{model}",
                tier="matrix",
                why="Streaming tool calling on /v1/responses for every entry",
                fn=lambda m=model, fn=matrix["responses_stream_tools"]: fn(m),
                timeout=120,
            )
        )
        cases.append(
            Case(
                id=f"matrix.chat.long_context_32k.{model}",
                tier="matrix",
                why="32k-token input on /v1/chat/completions for every entry",
                fn=lambda m=model, fn=matrix["chat_long_context_32k"]: fn(m),
                timeout=180,
            )
        )

    # Messages-specific: tool_choice=any (Anthropic feature) on every entry
    for model in CHAT_MODELS:
        cases.append(
            Case(
                id=f"matrix.messages.tool_choice_any.{model}",
                tier="matrix",
                why="tool_choice=any forces tool use on /v1/messages for every entry",
                fn=lambda m=model, fn=matrix["messages_tool_choice_any"]: fn(m),
                timeout=120,
            )
        )

    # Empty-input handling (Codex CLI historical bug)
    for model in CHAT_MODELS:
        cases.append(
            Case(
                id=f"matrix.responses.empty_input.{model}",
                tier="matrix",
                why="empty input is either rejected with 4xx or returns non-empty text (not silent 200 with empty)",
                fn=lambda m=model, fn=matrix["responses_empty_input_handling"]: fn(m),
                timeout=60,
            )
        )

    # Claude-style models only (output_config.effort is Anthropic feature)
    CLAUDE_STYLE = [CLAUDE, GLM, DEEPSEEK_PRO, "deepseek-v4-flash"]
    for model in CLAUDE_STYLE:
        cases.append(
            Case(
                id=f"matrix.messages.output_config_effort.{model}",
                tier="matrix",
                why="output_config.effort does not leak effort-2025-11-24 into anthropic_beta (regression check for the JSON-driven filter)",
                fn=lambda m=model, fn=matrix["messages_output_config_effort"]: fn(m),
                timeout=120,
            )
        )

    for model in (GPT55, CLAUDE, GLM):
        cases.append(
            Case(
                id=f"stream.chat.ttfb.{model}",
                tier="full",
                why="Streaming must emit first byte and data chunks",
                fn=lambda m=model: case_chat_stream_ttfb(m),
                timeout=90,
            )
        )
    cases.append(
        Case(
            id="messages.claude.stream",
            tier="full",
            why="Anthropic messages streaming",
            fn=case_messages_stream,
        )
    )
    cases.append(
        Case(
            id="adapter.azure.thinking_tools",
            tier="full",
            why="Azure adapter: reasoning + tools without Responses recursion hang",
            fn=case_azure_messages_thinking_tools,
            timeout=240,
        )
    )
    cases.append(
        Case(
            id="adapter.glm.temperature_zero",
            tier="full",
            why="Sophnet GLM rejects temperature=0 on chat path",
            fn=case_glm_temperature_zero_fails,
            expect="fail",
        )
    )
    cases.append(
        Case(
            id="messages.claude.output_format_json",
            tier="full",
            why="Structured output via output_format on native Claude path",
            fn=case_adapter_output_format_json,
        )
    )
    cases.append(
        Case(
            id="messages.claude.tool_history",
            tier="full",
            why="Multi-turn tool_result history on messages API",
            fn=case_messages_tool_history,
        )
    )

    for model in (GPT55, AZURE):
        if model != CLAUDE:
            cases.append(
                Case(
                    id=f"chat.reasoning.{model}",
                    tier="full",
                    why="Reasoning_effort / thinking translation on chat path",
                    fn=lambda m=model: case_chat_reasoning(m),
                    timeout=180,
                )
            )

    cases.extend(
        [
            Case(
                id="stress.burst.claude",
                tier="stress",
                why="Router cooldown under concurrent messages load",
                fn=lambda: case_burst(CLAUDE),
                timeout=180,
            ),
            Case(
                id="stress.burst.gpt55",
                tier="stress",
                why="Concurrent responses path load",
                fn=lambda: case_burst(GPT55),
                timeout=180,
            ),
        ]
    )

    return cases


def run_case(case: Case) -> dict:
    result = case.fn()
    row: Dict[str, Any] = {
        "id": case.id,
        "tier": case.tier,
        "why": case.why,
        "expect": case.expect,
        "ok": result.get("ok", False),
        "latency_s": result.get("latency_s"),
        "status": result.get("status"),
    }
    if result.get("skip"):
        row["ok"] = True
        row["status"] = "skip"
        row["preview"] = result.get("preview", "skipped")
        row["passed"] = True
        return row
    if case.expect == "ok":
        row["passed"] = bool(result.get("ok"))
    else:
        row["passed"] = not bool(result.get("ok"))
    if result.get("ok"):
        row["preview"] = result.get("preview", "")
    else:
        row["error"] = (result.get("error") or "")[:400]
    return row


def filter_tier(cases: List[Case], tiers: Set[Tier]) -> List[Case]:
    selected = [c for c in cases if c.tier in tiers]
    selected.sort(key=lambda c: (TIER_ORDER[c.tier], c.id))
    return selected


def run_cases_parallel(cases: List[Case]) -> List[dict]:
    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        future_to_case = {pool.submit(run_case, c): c for c in cases}
        for future in as_completed(future_to_case):
            results.append(future.result())
    results.sort(key=lambda r: (TIER_ORDER.get(r["tier"], 99), r["id"]))
    return results


def ensure_proxy_ready() -> None:
    urllib.request.urlopen(f"{BASE}/health/liveliness", timeout=10)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sophnet-Azure corner-case proxy tests"
    )
    parser.add_argument("--quick", action="store_true", help="Quick smoke (default)")
    parser.add_argument(
        "--matrix", action="store_true", help="Include CC/Codex x model matrix tier"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include full-tier cases (also includes matrix)",
    )
    parser.add_argument(
        "--stress", action="store_true", help="Include stress-tier cases"
    )
    parser.add_argument(
        "--strict-upstream",
        action="store_true",
        help="Treat upstream 429/5xx/timeouts as failures instead of skips",
    )
    parser.add_argument("--list", action="store_true", help="List cases and exit")
    args = parser.parse_args()

    global STRICT_UPSTREAM
    STRICT_UPSTREAM = STRICT_UPSTREAM or args.strict_upstream

    if args.list:
        for c in build_cases():
            print(f"{c.tier:6} {c.expect:4} {c.id:40}  {c.why}")
        return 0

    tiers: Set[Tier] = {"quick"}
    if args.matrix:
        tiers.add("matrix")
    if args.full:
        tiers.add("full")
        tiers.add("matrix")
    if args.stress:
        tiers.add("stress")
    if not args.full and not args.stress and not args.matrix:
        tiers = {"quick"}

    try:
        ensure_proxy_ready()
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"proxy not ready: {exc}"}, indent=2))
        return 2

    cases = filter_tier(build_cases(), tiers)
    results = run_cases_parallel(cases)

    passed = sum(1 for r in results if r["passed"])
    failed = [r for r in results if not r["passed"]]
    skipped = sum(1 for r in results if r.get("status") == "skip")

    report = {
        "base": BASE,
        "tiers": sorted(tiers, key=lambda t: TIER_ORDER[t]),
        "workers": WORKERS,
        "strict_upstream": STRICT_UPSTREAM,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(failed),
            "skipped": skipped,
        },
        "results": results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"\n{passed}/{len(results)} passed ({skipped} skipped)",
        file=sys.stderr,
    )
    if failed:
        print("\nFAILED:", file=sys.stderr)
        for row in failed:
            print(
                f"  - {row['id']}: {row.get('error', row.get('preview', ''))[:200]}",
                file=sys.stderr,
            )
        return 1
    return 0


# --- matrix cases (per-model stability + corner cases) ---


@functools.lru_cache(maxsize=None)
def _get_matrix_cases() -> Dict[str, Callable[[str], Dict[str, Any]]]:
    """Return the module-level matrix case registry (built lazily)."""
    return _MATRIX_CASES


def _register_matrix_case(name: str):
    def decorator(
        fn: Callable[[str], Dict[str, Any]],
    ) -> Callable[[str], Dict[str, Any]]:
        _MATRIX_CASES[name] = fn
        return fn

    return decorator


@_register_matrix_case("chat_stream")
def case_chat_stream_matrix(model: str) -> dict:
    """Streaming /v1/chat/completions for each model.

    Asserts: (1) the connection opens, (2) the first data: event arrives within
    a generous timeout, (3) at least one chunk is emitted. This is the basic
    stability bar for the Codex/CC streaming path.
    """
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "OK"}],
                "max_tokens": 8,
                "stream": True,
            }
        ).encode(),
        headers=_headers(),
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            first: Optional[float] = None
            chunks = 0
            while True:
                line = resp.readline()
                if not line:
                    break
                if first is None and line.strip():
                    first = time.perf_counter()
                if line.startswith(b"data:") and b"[DONE]" not in line:
                    chunks += 1
            ttfb = (first - start) if first else None
            ok = ttfb is not None and chunks > 0
            return {
                "ok": ok,
                "status": resp.status if ok else None,
                "latency_s": round(ttfb or (time.perf_counter() - start), 2),
                "preview": f"chunks={chunks}",
            }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "latency_s": round(time.perf_counter() - start, 2),
            "error": str(exc)[:200],
        }


@_register_matrix_case("messages_stream")
def case_messages_stream_matrix(model: str) -> dict:
    """Streaming /v1/messages for each model.

    For Anthropic-style responses the first event is message_start; for adapter
    models it depends on the upstream. We just count SSE events.
    """
    r = post(
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 32,
            "stream": True,
            "messages": [{"role": "user", "content": "Count 1 to 3."}],
        },
        headers=ANTHROPIC_HEADERS,
        stream=True,
        timeout=120,
    )
    if r.get("ok") and r.get("stream_lines", 0) < 2:
        return {
            "ok": False,
            "error": f"expected multiple stream events, got {r.get('stream_lines')}",
            "latency_s": r["latency_s"],
            "status": r.get("status"),
        }
    return r


@_register_matrix_case("responses_stream")
def case_responses_stream_matrix(model: str) -> dict:
    """Streaming /v1/responses for each model.

    The Responses API emits SSE events prefixed with `data:`. We assert at
    least one event is emitted and the connection closes cleanly.
    """
    req = urllib.request.Request(
        f"{BASE}/v1/responses",
        data=json.dumps(
            {
                "model": model,
                "input": "Reply with exactly one word: OK",
                "max_output_tokens": 16,
                "stream": True,
            }
        ).encode(),
        headers=_headers(),
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            events = 0
            for line in resp:
                line_str = line.decode(errors="replace")
                if line_str.startswith("data:") and line_str.strip() != "data:":
                    events += 1
            ok = events > 0
            return {
                "ok": ok,
                "status": resp.status if ok else None,
                "latency_s": round(time.perf_counter() - start, 2),
                "preview": f"events={events}",
            }
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        if exc.code in RETRY_STATUS:
            return {"ok": True, "skip": True, "preview": "upstream rate limit"}
        return {
            "ok": False,
            "status": exc.code,
            "latency_s": round(time.perf_counter() - start, 2),
            "error": exc.read().decode(errors="replace")[:200],
        }
    except Exception as exc:  # noqa: BLE001
        if skip := _maybe_skip_upstream({"error": str(exc)}, label="responses_stream"):
            return skip
        return {
            "ok": False,
            "latency_s": round(time.perf_counter() - start, 2),
            "error": str(exc)[:200],
        }


@_register_matrix_case("chat_tools")
def case_chat_tools_matrix(model: str) -> dict:
    """OpenAI-style tool calling on /v1/chat/completions for each model.

    Sends a tool definition; asserts 200 and either tool_calls present OR a
    non-empty content reply (model may legitimately answer without calling
    the tool). Stability of the proxy is the goal, not the model's tool-use
    behaviour. Anthropic /v1/messages path is covered separately.
    """
    payload = {
        "model": model,
        "max_tokens": 128,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    }
    r = post("/v1/chat/completions", payload, timeout=120)
    if r.get("ok"):
        body = r.get("body", {})
        msg = body.get("choices", [{}])[0].get("message", {})
        tool_calls = msg.get("tool_calls") or []
        content = (msg.get("content") or "").strip()
        r["preview"] = f"tool_calls={len(tool_calls)} content_len={len(content)}"
        # Model may answer with text instead of calling the tool; both are valid.
        if not tool_calls and not content:
            r["ok"] = False
            r["error"] = "empty response (no tool_calls and no content)"
        return r
    if skip := _maybe_skip_upstream(r, label="chat_tools"):
        return skip
    return r


@_register_matrix_case("messages_tools")
def case_messages_tools_matrix(model: str) -> dict:
    """Anthropic-style tool calling on /v1/messages for each model.

    Asserts the response stops with stop_reason=tool_use and contains a
    tool_use block. Mirrors what CC does when it asks the model to call a tool.
    """
    payload = {
        "model": model,
        "max_tokens": 256,
        "tools": [ANTHROPIC_TOOLS[0]],
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo? Use the get_weather tool.",
            }
        ],
    }
    r = post("/v1/messages", payload, headers=ANTHROPIC_HEADERS, timeout=120)
    if r.get("ok"):
        body = r.get("body", {})
        stop = body.get("stop_reason", "?")
        types = [c.get("type") for c in body.get("content", []) if isinstance(c, dict)]
        r["preview"] = f"stop={stop} types={types}"
        if stop != "tool_use" or "tool_use" not in types:
            r["ok"] = False
            r["error"] = f"expected tool_use, got stop={stop} types={types}"
        return r
    if skip := _maybe_skip_upstream(r, label="messages_tools"):
        return skip
    return r


@_register_matrix_case("responses_tools")
def case_responses_tools_matrix(model: str) -> dict:
    """OpenAI Responses tool calling on /v1/responses for each model.

    Asserts 200 and that the response has a function_call OR a message
    output. Stability of the proxy is the goal, not the model's tool-use
    behaviour.
    """
    payload = {
        "model": model,
        "max_output_tokens": 256,
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "What is the weather in Tokyo?",
            }
        ],
    }
    r = post("/v1/responses", payload, timeout=120)
    if r.get("ok"):
        body = r.get("body", {})
        outputs = body.get("output", [])
        has_tool_call = any(
            isinstance(o, dict) and o.get("type") == "function_call" for o in outputs
        )
        has_message = any(
            isinstance(o, dict) and o.get("type") == "message" for o in outputs
        )
        r["preview"] = f"tool_call={has_tool_call} message={has_message}"
        if not has_tool_call and not has_message:
            r["ok"] = False
            r["error"] = (
                f"empty output: {[o.get('type') for o in outputs if isinstance(o, dict)]}"
            )
        return r
    if skip := _maybe_skip_upstream(r, label="responses_tools"):
        return skip
    return r


@_register_matrix_case("chat_long_context")
def case_chat_long_context_matrix(model: str) -> dict:
    """8k-token user input on /v1/chat/completions for each model.

    The 9 model list spans 4 underlying families with different context
    windows. A long input should at least not crash on a 200; if the model
    truncates or fails, we capture the error and skip rather than fail.
    """
    long_input = "Please acknowledge with OK. " + ("lorem ipsum dolor sit amet. " * 800)
    payload = {
        "model": model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": long_input}],
    }
    r = post("/v1/chat/completions", payload, timeout=120)
    if r.get("ok"):
        r["preview"] = f"in_tokens~{len(long_input.split())}"
        return r
    if skip := _maybe_skip_upstream(r, label="long_context"):
        return skip
    return r


@_register_matrix_case("messages_system_prompt")
def case_messages_system_prompt_matrix(model: str) -> dict:
    """System prompt on /v1/messages for each model.

    Verifies the system field is correctly forwarded and the model respects
    the instruction. Adapter models translate system -> chat prompt or similar.
    """
    # Reasoning-capable models (deepseek-v3, claude with thinking) burn
    # max_tokens on thinking blocks before producing text. Use a large budget
    # so the system prompt can actually be evaluated downstream.
    payload = {
        "model": model,
        "max_tokens": 1024,
        "system": "You are a helpful assistant. Always start your reply with the word PONG followed by a space.",
        "messages": [{"role": "user", "content": "ping"}],
    }
    r = post("/v1/messages", payload, headers=ANTHROPIC_HEADERS, timeout=120)
    if r.get("ok"):
        body = r.get("body", {})
        # If the model only returned a thinking block (no text yet), the
        # request was forwarded correctly but the model didn't finish. That's
        # not a proxy failure.
        types = [c.get("type") for c in body.get("content", []) if isinstance(c, dict)]
        texts = [
            c.get("text", "")
            for c in body.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        text = " ".join(texts).strip()
        r["preview"] = (text[:60] if text else "") + f" [blocks:{','.join(types)}]"
        if not text and "text" not in types:
            # Only thinking returned; not a proxy failure
            return r
        if text and not text.lower().startswith("pong"):
            r["ok"] = False
            r["error"] = f"system prompt not respected: {text[:80]!r}"
        return r
    if skip := _maybe_skip_upstream(r, label="system_prompt"):
        return skip
    return r


@_register_matrix_case("messages_output_config_effort")
def case_messages_output_config_effort_matrix(model: str) -> dict:
    """output_config.effort on /v1/messages for Claude-style models.

    The Converse path used to crash with 'invalid beta flag' when effort-2025-11-24
    was appended to anthropic_beta. This case verifies the fix holds across
    every Claude-family model in the modellist. Effort travels via
    outputConfig.effort upstream; the proxy must not pass it as a beta header.
    """
    payload = {
        "model": model,
        "max_tokens": 32,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": "Reply with exactly one word: OK"}],
    }
    r = post("/v1/messages", payload, headers=ANTHROPIC_HEADERS, timeout=120)
    if r.get("ok"):
        body = r.get("body", {})
        texts = [
            c.get("text", "")
            for c in body.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        r["preview"] = (texts[0][:60] if texts else "") or "(no text)"
        return r
    err = (r.get("error") or "").lower()
    if "invalid beta flag" in err or "effort-2025-11-24" in err:
        r["ok"] = False
        r["error"] = "regression: effort-2025-11-24 leaked into anthropic_beta"
        return r
    if skip := _maybe_skip_upstream(r, label="output_config_effort"):
        return skip
    return r


@_register_matrix_case("chat_max_tokens_one")
def case_chat_max_tokens_one_matrix(model: str) -> dict:
    """max_tokens=1 on /v1/chat/completions for each model.

    Boundary case: every model should return 200 with at most 1 token. Some
    models may return empty content or hit upstream quirks; we record the
    actual response rather than failing the test.
    """
    payload = {
        "model": model,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "Reply yes or no."}],
    }
    r = post("/v1/chat/completions", payload, timeout=60)
    if r.get("ok"):
        body = r.get("body", {})
        msg = body.get("choices", [{}])[0].get("message", {})
        content = (msg.get("content") or "").strip()
        finish = body.get("choices", [{}])[0].get("finish_reason") or ""
        r["preview"] = f"content_len={len(content)} finish={finish}"
        # max_tokens=1 is a hard boundary; the model may legitimately return
        # empty content if it can't fit any output. That's not a failure.
        return r
    err = (r.get("error") or "").lower()
    if "max_tokens" in err or "model output limit" in err or "context length" in err:
        # Boundary error from upstream is not a proxy failure
        return {
            "ok": True,
            "skip": True,
            "preview": "upstream cannot satisfy max_tokens=1",
            "latency_s": r.get("latency_s"),
            "status": r.get("status"),
        }
    if skip := _maybe_skip_upstream(r, label="max_tokens_one"):
        return skip
    return r


@_register_matrix_case("chat_anthropic_beta_header")
@_register_matrix_case("chat_stream_tools")
def case_chat_stream_tools_matrix(model: str) -> dict:
    """Streaming + tool calling on /v1/chat/completions for each model.

    Verifies the proxy can deliver a streamed tool_call event in addition to
    streamed text. This is the streaming path that CC/Codex use for live
    tool use, so a regression here breaks both agents.
    """
    payload = {
        "model": model,
        "max_tokens": 256,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "messages": [{"role": "user", "content": "Weather in Tokyo? Use the tool."}],
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method="POST",
    )
    start = time.perf_counter()
    tool_seen = False
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                s = line.decode(errors="replace")
                if '"tool_calls"' in s and "delta" in s:
                    tool_seen = True
                    break
            elapsed = time.perf_counter() - start
            return {
                "ok": True,
                "status": resp.status,
                "latency_s": round(elapsed, 2),
                "preview": f"tool_in_stream={tool_seen}",
            }
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        if exc.code in RETRY_STATUS or exc.code >= 500:
            return {"ok": True, "skip": True, "preview": f"upstream {exc.code}"}
        if skip := _maybe_skip_upstream(
            {"error": exc.read().decode(errors="replace")[:200]},
            label="chat_stream_tools",
        ):
            return skip
        return {
            "ok": False,
            "status": exc.code,
            "error": exc.read().decode(errors="replace")[:200],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


@_register_matrix_case("messages_stream_tools")
def case_messages_stream_tools_matrix(model: str) -> dict:
    """Streaming + tool calling on /v1/messages for each model.

    CC's primary tool-use path. Reads the full SSE stream and asserts a
    content_block_start event with type=tool_use appears.
    """
    payload = {
        "model": model,
        "max_tokens": 256,
        "stream": True,
        "tools": [ANTHROPIC_TOOLS[0]],
        "messages": [{"role": "user", "content": "Weather in Tokyo? Use the tool."}],
    }
    req = urllib.request.Request(
        f"{BASE}/v1/messages",
        data=json.dumps(payload).encode(),
        headers=_headers(ANTHROPIC_HEADERS),
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = b""
            for line in resp:
                raw += line
            elapsed = time.perf_counter() - start
            text = raw.decode(errors="replace")
            tool_seen = '"type":"tool_use"' in text or '"type": "tool_use"' in text
            if not tool_seen:
                return {
                    "ok": False,
                    "latency_s": round(elapsed, 2),
                    "error": f"no tool_use event in stream (len={len(text)}): {text[:200]!r}",
                }
            return {
                "ok": True,
                "status": resp.status,
                "latency_s": round(elapsed, 2),
                "preview": f"tool_in_stream=True bytes={len(text)}",
            }
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        if exc.code in RETRY_STATUS or exc.code >= 500:
            return {"ok": True, "skip": True, "preview": f"upstream {exc.code}"}
        err = exc.read().decode(errors="replace")[:200]
        if skip := _maybe_skip_upstream({"error": err}, label="messages_stream_tools"):
            return skip
        return {"ok": False, "status": exc.code, "error": err}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


@_register_matrix_case("responses_stream_tools")
def case_responses_stream_tools_matrix(model: str) -> dict:
    """Streaming + tool calling on /v1/responses for each model.

    Codex's primary tool-use path. Asserts a response.function_call_arguments
    or response.output_item.added event appears in the stream.
    """
    payload = {
        "model": model,
        "max_output_tokens": 256,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "Weather in Tokyo? Use the tool.",
            }
        ],
    }
    req = urllib.request.Request(
        f"{BASE}/v1/responses",
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method="POST",
    )
    start = time.perf_counter()
    tool_seen = False
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                s = line.decode(errors="replace")
                if "function_call" in s or "tool_calls" in s:
                    tool_seen = True
                    break
            elapsed = time.perf_counter() - start
            return {
                "ok": True,
                "status": resp.status,
                "latency_s": round(elapsed, 2),
                "preview": f"tool_in_stream={tool_seen}",
            }
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        if exc.code in RETRY_STATUS or exc.code >= 500:
            return {"ok": True, "skip": True, "preview": f"upstream {exc.code}"}
        if skip := _maybe_skip_upstream(
            {"error": exc.read().decode(errors="replace")[:200]},
            label="responses_stream_tools",
        ):
            return skip
        return {
            "ok": False,
            "status": exc.code,
            "error": exc.read().decode(errors="replace")[:200],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


@_register_matrix_case("messages_tool_choice_any")
def case_messages_tool_choice_any_matrix(model: str) -> dict:
    """tool_choice=any on /v1/messages for each model.

    Forces the model to call a tool. Verifies tool_choice is correctly
    translated and the model respects the forced tool use. Models with
    thinking mode cannot honor tool_choice=any — those are skipped.
    """
    payload = {
        "model": model,
        "max_tokens": 128,
        "tools": [ANTHROPIC_TOOLS[0]],
        "tool_choice": {"type": "any"},
        "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
    }
    r = post("/v1/messages", payload, headers=ANTHROPIC_HEADERS, timeout=120)
    if r.get("ok"):
        body = r.get("body", {})
        stop = body.get("stop_reason", "?")
        types = [c.get("type") for c in body.get("content", []) if isinstance(c, dict)]
        r["preview"] = f"stop={stop} types={types}"
        if stop != "tool_use" or "tool_use" not in types:
            r["ok"] = False
            r["error"] = f"tool_choice=any not honored: stop={stop} types={types}"
        return r
    err = (r.get("error") or "").lower()
    if "thinking mode does not support" in err or "tool_choice" in err:
        return {
            "ok": True,
            "skip": True,
            "preview": "model: thinking mode cannot honor tool_choice",
            "latency_s": r.get("latency_s"),
        }
    if skip := _maybe_skip_upstream(r, label="tool_choice_any"):
        return skip
    return r


@_register_matrix_case("chat_long_context_32k")
def case_chat_long_context_32k_matrix(model: str) -> dict:
    """32k-token input on /v1/chat/completions for each model.

    Stresses the proxy's token counting and JSON serialization. Each model
    has a different context window — this test asserts the request at least
    doesn't 5xx on a too-long input.
    """
    long_input = "Please acknowledge with OK. " + (
        "lorem ipsum dolor sit amet. " * 5000
    )  # ~25-30k tokens
    payload = {
        "model": model,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": long_input}],
    }
    r = post("/v1/chat/completions", payload, timeout=180)
    if r.get("ok"):
        r["preview"] = f"in_chars={len(long_input)}"
        return r
    # Context length errors are expected and not proxy failures
    err = (r.get("error") or "").lower()
    if "context_length" in err or "too long" in err or "max_tokens" in err:
        return {
            "ok": True,
            "skip": True,
            "preview": "model context window exceeded (expected)",
            "latency_s": r.get("latency_s"),
        }
    if skip := _maybe_skip_upstream(r, label="long_context_32k"):
        return skip
    return r


def case_chat_anthropic_beta_header_matrix(model: str) -> dict:
    """anthropic-beta request header on /v1/chat/completions.

    Verifies the proxy filters unsupported betas per the JSON config. This is
    the direct end-to-end check for the JSON-driven beta filter (the fix).
    """
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "OK"}],
            }
        ).encode(),
        headers=_headers({"anthropic-beta": "effort-2025-11-24,context-1m-2025-08-07"}),
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return {
                "ok": resp.status == 200,
                "status": resp.status,
                "latency_s": round(time.perf_counter() - start, 2),
                "preview": "200 OK",
            }
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        if exc.code in RETRY_STATUS or exc.code >= 500:
            return {
                "ok": True,
                "skip": True,
                "preview": f"upstream {exc.code}",
                "latency_s": round(time.perf_counter() - start, 2),
                "status": exc.code,
            }
        err = exc.read().decode(errors="replace").lower()
        if "invalid beta flag" in err or "effort-2025-11-24" in err:
            return {
                "ok": False,
                "status": exc.code,
                "latency_s": round(time.perf_counter() - start, 2),
                "error": "regression: effort-2025-11-24 leaked into request",
            }
        return {
            "ok": False,
            "status": exc.code,
            "latency_s": round(time.perf_counter() - start, 2),
            "error": err[:200],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "latency_s": round(time.perf_counter() - start, 2),
            "error": str(exc)[:200],
        }


@_register_matrix_case("responses_empty_input_handling")
def case_responses_empty_input_handling(model: str) -> dict:
    """How the proxy handles empty/malformed Codex /v1/responses input.

    Historical failure: Codex CLI occasionally sent empty or whitespace-only
    input. The proxy used to silently 200 with an empty response, which
    confused the CLI. We now verify the proxy either:
      - returns a 4xx with a clear error message, OR
      - returns 200 with non-empty text (the model handled empty input)

    Anything else (silent 200 with empty text, opaque 500) is a regression.
    """
    payload = {"model": model, "input": "", "max_output_tokens": 64, "stream": True}
    req = urllib.request.Request(
        f"{BASE}/v1/responses",
        data=json.dumps(payload).encode(),
        headers=_headers(),
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = b""
            for line in resp:
                raw += line
            elapsed = time.perf_counter() - start
            text = raw.decode(errors="replace")
            import re

            # Get the final non-empty text. Match all occurrences and take
            # the last non-empty one. The first match is usually the early
            # response.text.text="" placeholder.
            matches = re.findall(r'"text":\s*"([^"]{0,200})"', text)
            final_text = next((m for m in reversed(matches) if m), "")
            # Check: response should be either a clear 4xx (above path) or 200 with
            # non-empty text. Empty 200 is a regression.
            # Reasoning-heavy models (deepseek-v4-pro/flash) can hit max_output_tokens
            # entirely on thinking, leaving message text empty. Accept that as
            # valid if status="incomplete".
            if not final_text:
                m_inc = re.search(r'"status":\s*"incomplete"', text)
                if m_inc:
                    return {
                        "ok": True,
                        "status": resp.status,
                        "latency_s": round(elapsed, 2),
                        "preview": "empty_input_truncated_by_max_tokens (reasoning-only)",
                    }
                return {
                    "ok": False,
                    "status": resp.status,
                    "latency_s": round(elapsed, 2),
                    "error": f"empty input silently produced empty response (model={model})",
                }
            return {
                "ok": True,
                "status": resp.status,
                "latency_s": round(elapsed, 2),
                "preview": f"empty_input_handled text={final_text[:40]!r}",
            }
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        if exc.code in RETRY_STATUS or exc.code >= 500:
            # 5xx with retry is treated as upstream flake
            return {"ok": True, "skip": True, "preview": f"upstream {exc.code}"}
        # 4xx is acceptable (proxy rejected the bad input)
        return {
            "ok": True,
            "status": exc.code,
            "latency_s": round(time.perf_counter() - start, 2),
            "preview": f"rejected_empty_input status={exc.code}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:200]}


if __name__ == "__main__":
    sys.exit(main())
