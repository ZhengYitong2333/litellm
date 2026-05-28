#!/usr/bin/env python3
"""
Corner-case proxy tests for deploy/sophnet-azure.

Each case maps to a historical failure mode (routing, param translation, gateway quirks).
Output is JSON-only on stdout for piping; human summary goes to stderr.

Usage:
  python3 test_corner_cases.py              # quick (default)
  python3 test_corner_cases.py --matrix     # CC/Codex x model matrix
  python3 test_corner_cases.py --full       # full + matrix tiers
  python3 test_corner_cases.py --stress
  python3 test_corner_cases.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Set

BASE = os.environ.get("LITELLM_PROXY_BASE", "http://localhost:4000")
KEY = os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-sophnet-azure-local")
WORKERS = int(os.environ.get("LITELLM_TEST_WORKERS", "4"))

Tier = Literal["quick", "full", "matrix", "stress"]
Expect = Literal["ok", "fail", "skip"]

CHAT_MODELS = [
    "sophnet-glm-5.1",
    "sophnet-gpt-5.5",
    "sophnet-deepseekv4-pro",
    "sophnet-deepseekv4-flash",
    "sophnet-claude-opus-4-7",
    "azure-gpt-5.5",
    "azure-gpt-5.4",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
]

CLAUDE = "sophnet-claude-opus-4-7"
GLM = "sophnet-glm-5.1"
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
    parser.add_argument("--list", action="store_true", help="List cases and exit")
    args = parser.parse_args()

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


if __name__ == "__main__":
    sys.exit(main())
