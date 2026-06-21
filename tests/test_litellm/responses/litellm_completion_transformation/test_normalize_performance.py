"""
Benchmark-style tests for Responses→Chat normalize pipeline performance.

Run with ``-s`` to print timings:
  uv run pytest tests/test_litellm/responses/litellm_completion_transformation/test_normalize_performance.py -v -s
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import pytest

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _build_synthetic_codex_history(turns: int) -> List[Dict[str, Any]]:
    """Simulate Codex multi-turn tool history (user → assistant+tool_calls → tool)."""
    messages: List[Dict[str, Any]] = []
    for i in range(turns):
        messages.append({"role": "user", "content": f"step {i}"})
        call_id = f"call_{i:04d}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "arguments": f'{{"cmd": "echo {i}"}}',
                        },
                    }
                ],
            }
        )
        if i % 3 == 0:
            # Interleaved user message before tool result (stress reorder path)
            messages.append({"role": "user", "content": "continue"})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"output {i}",
            }
        )
    return messages


@pytest.mark.parametrize("turns", [50, 100])
def test_normalize_converted_messages_benchmark(
    turns: int, capsys: pytest.CaptureFixture
):
    messages = _build_synthetic_codex_history(turns)
    start = time.perf_counter()
    result = (
        LiteLLMCompletionResponsesConfig._normalize_converted_messages_for_tool_calling(
            messages=messages,
            tools=[],
        )
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(result) >= turns * 2
    tool_messages = [m for m in result if m.get("role") == "tool"]
    assert len(tool_messages) >= turns

    print(
        f"\n[normalize benchmark] turns={turns} messages_in={len(messages)} "
        f"messages_out={len(result)} elapsed_ms={elapsed_ms:.2f}"
    )

    # Generous ceiling — catches accidental regression to multi-second scans.
    assert elapsed_ms < 5000, f"normalize took {elapsed_ms:.0f}ms for {turns} turns"
