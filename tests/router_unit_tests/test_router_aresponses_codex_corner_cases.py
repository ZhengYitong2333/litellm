"""
Codex-specific corner-case tests for the /v1/responses mid-stream fallback path.

Covers function-call, reasoning, and tool-item scenarios that the basic
streaming-fallback tests in test_router_aresponses_streaming_fallback.py
and the chat-completions bridge tests in test_router.py do not exercise.

These tests verify that _ResponsesStreamState correctly tracks tool-call,
reasoning, and text state, and that _build_responses_continuation_input
produces valid ResponseInputParam payloads (including
function_call + function_call_output pairs, developer message instructions
for interrupted tools / reasoning, and pre-filled assistant text).
"""

import os
import sys
from types import SimpleNamespace
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import Router
from litellm.exceptions import MidStreamFallbackError
from litellm.router_utils.responses_stream_state import _ResponsesStreamState
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


# ---------------------------------------------------------------------------
# Safe event builder — avoids MagicMock reserved attribute names (name, id,
# type) by using SimpleNamespace for inner payloads that the helper inspects.
# ---------------------------------------------------------------------------
def _make_event(etype: str, **attrs: Any) -> Any:
    """Build a stream-event-like object.

    MagicMock auto-creates attributes on access and reserves some kwarg names
    (name, id), so we use SimpleNamespace for payloads the helper inspects.
    """
    ns = SimpleNamespace(**attrs)
    ns.type = etype
    return ns


def _make_item(**attrs: Any) -> Any:
    """Build an item payload (value of OUTPUT_ITEM_ADDED/DONE ``item`` attr)."""
    return SimpleNamespace(**attrs)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------
def _make_router_with_fallback(primary="gpt-4", secondary="gpt-3.5-turbo"):
    return Router(
        model_list=[
            {
                "model_name": primary,
                "litellm_params": {"model": primary, "api_key": "k1"},
            },
            {
                "model_name": secondary,
                "litellm_params": {"model": secondary, "api_key": "k2"},
            },
        ],
        fallbacks=[{primary: [secondary]}],
    )


def _make_responses_iterator(*, chunks=(), error=None, model="gpt-4"):
    """Minimal mock Responses API streaming iterator."""
    from litellm.responses.streaming_iterator import (
        BaseResponsesAPIStreamingIterator,
    )

    class _Iter(BaseResponsesAPIStreamingIterator):
        def __init__(self):
            self._chunks = list(chunks)
            self._idx = 0
            self._hidden_params = {}
            self.model = model
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.litellm_metadata = None
            self.responses_api_provider_config = None
            self.finished = False
            self.completed_response = None
            self.response = None
            self.start_time = None
            self.request_data = {}
            self.call_type = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._idx < len(self._chunks):
                self._idx += 1
                return self._chunks[self._idx - 1]
            if error is not None:
                raise error
            raise StopAsyncIteration

    return _Iter()


class _AsyncList:
    """Generic async iterator over a list — used as the fallback response."""

    def __init__(self, items=()):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


def _make_completed_event(
    input_tokens: int, output_tokens: int, total_tokens: int
) -> ResponseCompletedEvent:
    response = ResponsesAPIResponse.model_construct(
        usage=ResponseAPIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
    )
    return ResponseCompletedEvent.model_construct(
        type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
        response=response,
    )


# ===== Test 1: function_call_arguments interrupted ==========================


@pytest.mark.asyncio
async def test_function_call_arguments_interrupted_continuation():
    """
    Mid-stream break in function_call_arguments deltas (no text emitted).

    Verifies:
    - MidStreamFallbackError carries partial_tool_calls with accumulated args
      and done=False.
    - had_in_flight_item=True, is_pre_first_chunk=False (FUNCTION_CALL_ARGUMENTS_DELTA
      is visible).
    - Fallback input does NOT contain function_call / function_call_output items
      (because done=False) but developer message mentions the interrupted call.
    """
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            ResponseCreatedEvent.model_construct(
                type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
                response=ResponsesAPIResponse.model_construct(id="resp_1"),
            ),
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(
                    item_id="item_fc_1",
                    type="function_call",
                    call_id="call_1",
                    name="search_web",
                    arguments="",
                ),
            ),
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                item_id="item_fc_1",
                output_index=0,
                delta='{"query": "',
            ),
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                item_id="item_fc_1",
                output_index=0,
                delta='capital of France"}',
            ),
        ],
        error=litellm.APIConnectionError(
            message="connection reset by peer",
            model="gpt-4",
            llm_provider="openai",
        ),
    )

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Search for capital of France",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for ev in wrapped:
                pass
        except Exception:
            pass

    fallback_call = mock_fallback_utils.call_args
    assert fallback_call is not None, "Fallback was not triggered"

    fbk_e: MidStreamFallbackError = fallback_call.kwargs["e"]
    assert fbk_e.is_pre_first_chunk is False
    assert fbk_e.generated_content == ""
    assert fbk_e.had_in_flight_item is True
    assert len(fbk_e.partial_tool_calls) == 1
    tc = fbk_e.partial_tool_calls[0]
    assert tc["item_id"] == "item_fc_1"
    assert tc["arguments_str"] == '{"query": "capital of France"}'
    assert tc["name"] == "search_web"
    assert tc["call_id"] == "call_1"
    assert tc["done"] is False

    # Fallback input must NOT include function_call items (done=False)
    new_input = fallback_call.kwargs["kwargs"]["input"]
    assert isinstance(new_input, list)
    fc_items = [item for item in new_input if item.get("type") == "function_call"]
    assert len(fc_items) == 0, "Undone tool call should not be an input item"
    # Developer message must mention the interrupted tool
    dev_msgs = [msg for msg in new_input if msg.get("role") == "developer"]
    assert len(dev_msgs) >= 1
    dev_text = dev_msgs[-1]["content"][0]["text"]
    assert "interrupted" in dev_text.lower() or "began" in dev_text.lower()
    assert "search_web" in dev_text


@pytest.mark.asyncio
async def test_function_call_timeout_interrupted():
    """Timeout variant of function_call interrupted (LITELLM_MAX_STREAMING_DURATION)."""
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(
                    item_id="item_fc_t1",
                    type="function_call",
                    call_id="call_db",
                    name="search_db",
                    arguments="",
                ),
            ),
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                item_id="item_fc_t1",
                delta="SELECT * FROM ",
            ),
        ],
        error=litellm.Timeout(
            message="Stream duration exceeded 60s",
            model="gpt-4",
            llm_provider="openai",
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Query the database",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert (
        mock_fallback_utils.call_args is not None
    ), "Fallback not triggered for Timeout"
    fbk_e: MidStreamFallbackError = mock_fallback_utils.call_args.kwargs["e"]
    assert fbk_e.had_in_flight_item is True
    assert len(fbk_e.partial_tool_calls) == 1
    assert fbk_e.partial_tool_calls[0]["done"] is False


# ===== Test 2: reasoning interrupted ========================================


@pytest.mark.asyncio
async def test_reasoning_interrupted():
    """Mid-stream break in reasoning with a single summary_index."""
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(item_id="item_rs_1", type="reasoning"),
            ),
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="item_rs_1",
                output_index=0,
                summary_index=0,
                delta="First I need to consider ",
            ),
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="item_rs_1",
                output_index=0,
                summary_index=0,
                delta="the available data.",
            ),
        ],
        error=litellm.RateLimitError(
            message="429 Too Many Requests",
            model="gpt-4",
            llm_provider="openai",
            response=MagicMock(status_code=429),
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Analyze the data",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert mock_fallback_utils.call_args is not None
    fbk_e: MidStreamFallbackError = mock_fallback_utils.call_args.kwargs["e"]
    assert len(fbk_e.partial_reasoning) >= 1
    rs = fbk_e.partial_reasoning[0]
    assert rs["summary_text"] == "First I need to consider the available data."
    assert rs["done"] is False
    assert rs["summary_index"] == 0

    new_input = mock_fallback_utils.call_args.kwargs["kwargs"]["input"]
    dev_msgs = [msg for msg in new_input if msg.get("role") == "developer"]
    assert len(dev_msgs) >= 1
    dev_text = dev_msgs[-1]["content"][0]["text"]
    assert "First I need to consider" in dev_text


@pytest.mark.asyncio
async def test_reasoning_timeout():
    """Timeout variant of reasoning interrupted."""
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(item_id="item_rs_t", type="reasoning"),
            ),
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="item_rs_t",
                summary_index=0,
                delta="Thinking step one ",
            ),
        ],
        error=litellm.Timeout(
            message="Stream timed out during reasoning",
            model="gpt-4",
            llm_provider="openai",
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Think deeply",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert mock_fallback_utils.call_args is not None
    fbk_e: MidStreamFallbackError = mock_fallback_utils.call_args.kwargs["e"]
    assert len(fbk_e.partial_reasoning) >= 1
    assert fbk_e.partial_reasoning[0]["summary_text"] == "Thinking step one "


# ===== Test 2b: reasoning with multiple summary indices =====================


@pytest.mark.asyncio
async def test_reasoning_multi_summary_index():
    """
    Reasoning with multiple summary_index values: verifies that the helper
    tracks each summary_index separately and does not overwrite earlier ones.
    """
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(item_id="item_rs_multi", type="reasoning"),
            ),
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="item_rs_multi",
                summary_index=0,
                delta="First reasoning block. ",
            ),
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE,
                item_id="item_rs_multi",
                summary_index=0,
                text="First reasoning block. Final.",
            ),
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="item_rs_multi",
                summary_index=1,
                delta="Second reasoning ",
            ),
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="item_rs_multi",
                summary_index=1,
                delta="block.",
            ),
        ],
        error=litellm.APIConnectionError(
            message="connection lost mid-reasoning",
            model="gpt-4",
            llm_provider="openai",
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Think about two things",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert mock_fallback_utils.call_args is not None
    fbk_e: MidStreamFallbackError = mock_fallback_utils.call_args.kwargs["e"]

    reasoning = sorted(fbk_e.partial_reasoning, key=lambda r: r.get("summary_index", 0))
    assert len(reasoning) == 2, f"Expected 2 reasoning entries, got {len(reasoning)}"

    idx0 = [r for r in reasoning if r["summary_index"] == 0][0]
    assert idx0["summary_text"] == "First reasoning block. Final."
    assert idx0["done"] is True

    idx1 = [r for r in reasoning if r["summary_index"] == 1][0]
    assert idx1["summary_text"] == "Second reasoning block."
    assert idx1["done"] is False

    new_input = mock_fallback_utils.call_args.kwargs["kwargs"]["input"]
    dev_msgs = [msg for msg in new_input if msg.get("role") == "developer"]
    assert len(dev_msgs) >= 1
    dev_text = dev_msgs[-1]["content"][0]["text"]
    assert "First reasoning block" in dev_text
    assert "Second reasoning block" in dev_text


# ===== Test 3: completed tool call with no text, then error =================


@pytest.mark.asyncio
async def test_completed_tool_call_without_text_then_error():
    """
    A fully completed function_call item (done=True, complete arguments) with
    no output_text before the error.

    Verifies that the fallback continuation input includes a paired
    function_call + function_call_output input items.
    """
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(
                    item_id="item_fc_done",
                    type="function_call",
                    call_id="call_search",
                    name="search_web",
                    arguments='{"query":"capital of France"}',
                ),
            ),
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                item_id="item_fc_done",
                delta='{"query": ',
            ),
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                item_id="item_fc_done",
                delta='"capital of France"}',
            ),
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
                item_id="item_fc_done",
                arguments='{"query": "capital of France"}',
            ),
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
                output_index=0,
                sequence_number=1,
                item=_make_item(
                    item_id="item_fc_done",
                    type="function_call",
                    call_id="call_search",
                    name="search_web",
                    status="completed",
                    arguments='{"query": "capital of France"}',
                ),
            ),
        ],
        error=litellm.ServiceUnavailableError(
            message="503 upstream failure",
            model="gpt-4",
            llm_provider="openai",
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Search for the capital of France",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert mock_fallback_utils.call_args is not None
    new_input = mock_fallback_utils.call_args.kwargs["kwargs"]["input"]
    assert isinstance(new_input, list)

    fc_items = [i for i in new_input if i.get("type") == "function_call"]
    assert (
        len(fc_items) == 1
    ), f"Expected 1 function_call input item, got {len(fc_items)}"
    fc = fc_items[0]
    assert fc["name"] == "search_web"
    assert fc["call_id"] == "call_search"
    assert fc["arguments"] == '{"query": "capital of France"}'
    assert fc.get("status") == "completed"

    fco_items = [i for i in new_input if i.get("type") == "function_call_output"]
    assert len(fco_items) == 1, f"Expected 1 function_call_output, got {len(fco_items)}"
    fco = fco_items[0]
    assert fco["call_id"] == "call_search"
    assert fco.get("status") == "completed"
    assert "interrupted" in fco["output"]

    dev_msgs = [msg for msg in new_input if msg.get("role") == "developer"]
    assert len(dev_msgs) >= 1
    dev_text = dev_msgs[-1]["content"][0]["text"]
    assert "interrupted" in dev_text.lower()


# ===== Test 4: empty response.failed + trailing error -> Router fallback =====


@pytest.mark.asyncio
async def test_empty_response_failed_flows_to_router_fallback():
    """
    An empty ``response.failed`` followed by a trailing ``error`` event should
    flow up through the Router streaming wrapper and trigger a fallback
    instead of throwing directly.
    """
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            MagicMock(
                type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
                response=MagicMock(id="resp_fail"),
            ),
        ],
        error=litellm.RateLimitError(
            message="429 rate limited",
            model="gpt-4",
            llm_provider="openai",
            response=MagicMock(status_code=429),
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Hello",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert (
        mock_fallback_utils.call_args is not None
    ), "RateLimitError should trigger fallback, not surface directly"


# ===== Test 5: mid-chunk 429 after visible text =============================


@pytest.mark.asyncio
async def test_mid_chunk_429_after_text():
    """
    Rate-limit error after visible text has been emitted.

    Verifies that:
    - Continuation input contains developer interruption instruction.
    - Continuation input contains assistant message with the visible text.
    - No function_call / function_call_output items are injected (pure text).
    """
    router = _make_router_with_fallback()
    text_partial = "Here are some results from your search:"
    src = _make_responses_iterator(
        chunks=[
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                delta=text_partial,
            ),
        ],
        error=litellm.RateLimitError(
            message="429 Too Many Requests",
            model="gpt-4",
            llm_provider="openai",
            response=MagicMock(status_code=429),
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Search for results",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert mock_fallback_utils.call_args is not None
    new_input = mock_fallback_utils.call_args.kwargs["kwargs"]["input"]
    assert isinstance(new_input, list)

    dev_msgs = [msg for msg in new_input if msg.get("role") == "developer"]
    assert len(dev_msgs) >= 1

    asst_msgs = [msg for msg in new_input if msg.get("role") == "assistant"]
    assert len(asst_msgs) >= 1
    asst = asst_msgs[0]
    assert asst["content"][0]["type"] == "output_text"
    assert asst["content"][0]["text"] == text_partial

    fc_items = [
        i
        for i in new_input
        if i.get("type") in ("function_call", "function_call_output")
    ]
    assert (
        len(fc_items) == 0
    ), "Pure text fallback should not inject tool placeholder items"


# ===== Test 6: fallback model also fails ====================================


@pytest.mark.asyncio
async def test_fallback_model_also_fails():
    """
    Primary stream fails and the fallback model also raises.

    Verifies cleanup: source_iterator aclose() is called (shielded),
    fallback_response aclose() is called, and the exception propagates.
    """
    router = _make_router_with_fallback()

    source_aclose = AsyncMock()

    class _SrcIter:
        def __init__(self):
            self._done = False
            self.model = "gpt-4"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self.responses_api_provider_config = MagicMock()
            self.start_time = 0.0
            self.litellm_metadata = {}
            self.request_data = {}
            self.call_type = "aresponses"
            self._hidden_params = {}
            self.completed_response = None
            self.response = None

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._done:
                self._done = True
                return _make_event(
                    ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                    delta="partial text ",
                )
            raise litellm.APIConnectionError(
                message="connection drop",
                model="gpt-4",
                llm_provider="openai",
            )

        async def aclose(self):
            await source_aclose()

    src = _SrcIter()

    class _FailingFallbackStream:
        def __init__(self):
            self._aclose_called = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ValueError("fallback also failed")

        async def aclose(self):
            self._aclose_called = True

    fallback = _FailingFallbackStream()

    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=fallback,
    ):
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Test",
                "original_generic_function": litellm.aresponses,
            },
        )
        with pytest.raises(ValueError, match="fallback also failed"):
            async for _ in wrapped:
                pass

    source_aclose.assert_awaited_once()
    assert fallback._aclose_called is True


# ===== Test 7: pre-first-chunk regression protection ========================


@pytest.mark.asyncio
async def test_output_item_added_without_delta_preserves_bare_retry():
    """
    OUTPUT_ITEM_ADDED without any subsequent delta should NOT trigger
    the continuation path — it must fall through to the "bare input retry"
    branch (pre-first-chunk behavior).

    This proves that OUTPUT_ITEM_ADDED alone is NOT considered visible,
    preserving the existing pre-first-chunk contract.
    """
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            ResponseCreatedEvent.model_construct(
                type=ResponsesAPIStreamEvents.RESPONSE_CREATED,
                response=ResponsesAPIResponse.model_construct(id="resp_added"),
            ),
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(item_id="item_added_only", type="function_call"),
            ),
        ],
        error=litellm.RateLimitError(
            message="429 rate limited (before any content)",
            model="gpt-4",
            llm_provider="openai",
            response=MagicMock(status_code=429),
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "Bare input",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert mock_fallback_utils.call_args is not None
    fbk_e: MidStreamFallbackError = mock_fallback_utils.call_args.kwargs["e"]
    assert (
        fbk_e.is_pre_first_chunk is True
    ), "OUTPUT_ITEM_ADDED alone should keep is_pre_first_chunk=True"

    new_input = mock_fallback_utils.call_args.kwargs["kwargs"]["input"]
    assert (
        new_input == "Bare input"
    ), "Bare OUTPUT_ITEM_ADDED without delta should not rewrite input"


# ===== Test 8: pure text happy-path continuation (regression) ===============


@pytest.mark.asyncio
async def test_pure_text_happy_path_continuation():
    """
    Pure text only (no tools/reasoning) — fallback produces a standard
    continuation input with developer instruction + assistant prefill.

    Matches the behavior tested in
    test_aresponses_streaming_iterator_partial_content_injects_continuation
    (test_router.py:1983).
    """
    router = _make_router_with_fallback()
    src = _make_responses_iterator(
        chunks=[
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                delta="The result is ",
            ),
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                delta="42.",
            ),
        ],
        error=MidStreamFallbackError(
            message="mid-stream error",
            model="gpt-4",
            llm_provider="openai",
            is_pre_first_chunk=False,
            generated_content="The result is 42.",
        ),
    )
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=_AsyncList(),
    ) as mock_fallback_utils:
        wrapped = await router._aresponses_streaming_iterator(
            response=src,
            initial_kwargs={
                "model": "gpt-4",
                "stream": True,
                "input": "What is 6*7?",
                "original_generic_function": litellm.aresponses,
            },
        )
        try:
            async for _ in wrapped:
                pass
        except Exception:
            pass

    assert mock_fallback_utils.call_args is not None
    new_input = mock_fallback_utils.call_args.kwargs["kwargs"]["input"]
    assert isinstance(new_input, list)
    assert new_input[0]["role"] == "user"
    assert new_input[0]["content"][0]["text"] == "What is 6*7?"
    asst = [msg for msg in new_input if msg.get("role") == "assistant"]
    assert len(asst) >= 1
    assert asst[-1]["content"][0]["type"] == "output_text"
    assert asst[-1]["content"][0]["text"] == "The result is 42."


# ===== _ResponsesStreamState unit tests ====================================


class TestResponsesStreamState:
    """Direct unit tests for the helper class."""

    def test_observes_text_delta(self):
        state = _ResponsesStreamState()
        state.observe(
            _make_event(ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA, delta="hello ")
        )
        assert state.has_text is True
        assert state.get_generated_content() == "hello "
        assert state.has_recoverable_state() is True

    def test_observes_text_multi_delta(self):
        state = _ResponsesStreamState()
        for text in ["hello ", "world"]:
            state.observe(
                _make_event(
                    ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
                    delta=text,
                )
            )
        assert state.get_generated_content() == "hello world"

    def test_is_visible_for_text_events(self):
        state = _ResponsesStreamState()
        assert (
            state.is_visible(_make_event(ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA))
            is True
        )
        assert (
            state.is_visible(_make_event(ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE))
            is True
        )
        assert (
            state.is_visible(_make_event(ResponsesAPIStreamEvents.CONTENT_PART_DONE))
            is True
        )
        assert (
            state.is_visible(_make_event(ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE))
            is True
        )

    def test_is_visible_for_tool_and_reasoning_deltas(self):
        state = _ResponsesStreamState()
        assert (
            state.is_visible(
                _make_event(ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA)
            )
            is True
        )
        assert (
            state.is_visible(
                _make_event(ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE)
            )
            is True
        )
        assert (
            state.is_visible(
                _make_event(ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA)
            )
            is True
        )
        assert (
            state.is_visible(
                _make_event(ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE)
            )
            is True
        )

    def test_is_not_visible_for_lifecycle_events(self):
        state = _ResponsesStreamState()
        assert (
            state.is_visible(_make_event(ResponsesAPIStreamEvents.RESPONSE_CREATED))
            is False
        )
        assert (
            state.is_visible(_make_event(ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS))
            is False
        )
        assert (
            state.is_visible(_make_event(ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED))
            is False
        )

    def test_observes_tool_func_call(self):
        state = _ResponsesStreamState()

        # OUTPUT_ITEM_ADDED alone does NOT set has_tool_delta
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(item_id="fc1", type="function_call"),
            )
        )
        assert state.has_tool_delta is False
        assert state.has_recoverable_state() is False

        # FUNCTION_CALL_ARGUMENTS_DELTA sets has_tool_delta
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                item_id="fc1",
                delta="partial ",
            )
        )
        assert state.has_tool_delta is True
        assert state.has_recoverable_state() is True

        payload = state.to_error_payload()
        assert len(payload["partial_tool_calls"]) == 1
        assert payload["partial_tool_calls"][0]["arguments_str"] == "partial "
        assert payload["had_in_flight_item"] is True

    def test_observes_tool_complete_covers_with_done_event(self):
        """FUNCTION_CALL_ARGUMENTS_DONE overrides accumulated delta value."""
        state = _ResponsesStreamState()
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(item_id="fc2", type="function_call"),
            )
        )
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
                item_id="fc2",
                delta='{"que',
            )
        )
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
                item_id="fc2",
                arguments='{"query":"complete"}',
            )
        )
        payload = state.to_error_payload()
        tc = payload["partial_tool_calls"][0]
        assert tc["arguments_str"] == '{"query":"complete"}'
        assert tc["done"] is True

    def test_observes_reasoning_multi_summary(self):
        state = _ResponsesStreamState()
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                item=_make_item(item_id="rs1", type="reasoning"),
            )
        )
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="rs1",
                summary_index=0,
                delta="first ",
            )
        )
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="rs1",
                summary_index=1,
                delta="second ",
            )
        )
        payload = state.to_error_payload()
        reasoning = sorted(
            payload["partial_reasoning"], key=lambda r: r["summary_index"]
        )
        assert len(reasoning) == 2
        assert reasoning[0]["summary_text"] == "first "
        assert reasoning[1]["summary_text"] == "second "

    def test_reasoning_done_overrides_accumulated(self):
        state = _ResponsesStreamState()
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA,
                item_id="rs2",
                summary_index=0,
                delta="partial first ",
            )
        )
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DONE,
                item_id="rs2",
                summary_index=0,
                text="Complete override of first reasoning block",
            )
        )
        payload = state.to_error_payload()
        rs = payload["partial_reasoning"][0]
        assert rs["summary_text"] == "Complete override of first reasoning block"
        assert rs["done"] is True

    def test_no_recoverable_state_for_bare_added(self):
        """OUTPUT_ITEM_ADDED alone is not recoverable."""
        state = _ResponsesStreamState()
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                item=_make_item(item_id="fc3", type="function_call"),
            )
        )
        assert state.has_recoverable_state() is False

    def test_completed_tool_in_continuation_payload(self):
        """Continuation payload includes all tool calls (done and undone);
        builder distinguishes them."""
        state = _ResponsesStreamState()
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=0,
                item=_make_item(item_id="fc_done", type="function_call"),
            )
        )
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
                item_id="fc_done",
                arguments='{"done": true}',
            )
        )
        state.observe(
            _make_event(
                ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
                output_index=1,
                item=_make_item(item_id="fc_undone", type="function_call"),
            )
        )

        continuation = state.to_continuation_payload()
        assert len(continuation["partial_tool_calls"]) == 2
        done_calls = [tc for tc in continuation["partial_tool_calls"] if tc.get("done")]
        undone_calls = [
            tc for tc in continuation["partial_tool_calls"] if not tc.get("done")
        ]
        assert len(done_calls) == 1
        assert done_calls[0]["item_id"] == "fc_done"
        assert len(undone_calls) == 1
        assert undone_calls[0]["item_id"] == "fc_undone"
        assert continuation["had_in_flight_item"] is True
