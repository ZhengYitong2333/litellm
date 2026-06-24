"""
Regression tests for streaming connection pool leak fix.
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import anyio
import httpx
import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx.aiohttp_transport import (
    AiohttpResponseStream,
    LiteLLMAiohttpTransport,
)

# ── aiohttp transport layer tests ──────────────────────────────


@pytest.mark.asyncio
async def test_aiohttp_transport_response_uses_stream_not_content():
    """handle_async_request must use stream= so aclose() propagates to AiohttpResponseStream."""

    class FakeSession:
        closed = False

        def __init__(self):
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        def request(self, **kwargs):
            class Resp:
                status = 200
                headers = {}

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *args):
                    pass

                @property
                def content(self):
                    class C:
                        async def iter_chunked(self, size):
                            yield b"data"

                    return C()

            return Resp()

    transport = LiteLLMAiohttpTransport(client=lambda: FakeSession())  # type: ignore
    response = await transport.handle_async_request(
        httpx.Request("GET", "http://example.com")
    )

    assert isinstance(response.stream, AiohttpResponseStream)


@pytest.mark.asyncio
async def test_aiohttp_response_stream_aclose_releases_connection():
    """AiohttpResponseStream.aclose() must call __aexit__ on the aiohttp response."""
    aexit_called = False

    class MockResponse:
        status = 200
        headers = {}

        @property
        def content(self):
            class C:
                async def iter_chunked(self, size):
                    yield b"data"

            return C()

        async def __aexit__(self, *args):
            nonlocal aexit_called
            aexit_called = True

    stream = AiohttpResponseStream(MockResponse())  # type: ignore
    await stream.aclose()
    assert aexit_called


# ── CustomStreamWrapper.aclose() tests ─────────────────────────


@pytest.mark.asyncio
async def test_aclose_falls_back_to_close():
    """OpenAI's AsyncStream has close() but not aclose(). Must fall back."""
    close_called = False

    class FakeAsyncStream:
        async def close(self):
            nonlocal close_called
            close_called = True

    wrapper = CustomStreamWrapper(
        completion_stream=FakeAsyncStream(),
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    await wrapper.aclose()
    assert close_called


@pytest.mark.asyncio
async def test_aclose_prefers_aclose_over_close():
    """When both aclose() and close() exist, aclose() should be preferred."""
    aclose_called = False
    close_called = False

    class FakeStream:
        async def aclose(self):
            nonlocal aclose_called
            aclose_called = True

        async def close(self):
            nonlocal close_called
            close_called = True

    wrapper = CustomStreamWrapper(
        completion_stream=FakeStream(),
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    await wrapper.aclose()
    assert aclose_called
    assert not close_called


@pytest.mark.asyncio
async def test_aclose_completes_under_cancellation():
    """aclose() must shield cleanup from CancelledError so streams actually close."""
    aclose_completed = False

    class SlowCloseStream:
        async def aclose(self):
            await anyio.sleep(0)
            nonlocal aclose_completed
            aclose_completed = True

    wrapper = CustomStreamWrapper(
        completion_stream=SlowCloseStream(),
        model=None,
        logging_obj=MagicMock(),
        custom_llm_provider=None,
    )

    with anyio.CancelScope() as scope:
        scope.cancel()
        await wrapper.aclose()

    assert aclose_completed


@pytest.mark.asyncio
async def test_custom_stream_wrapper_async_anext_reuses_stream_iterator():
    """Each __anext__ must advance the same underlying async stream.

    OpenAI AsyncStream returns a fresh iterator from __aiter__(); recreating it on
    every __anext__ replays the first chunk forever and pegs CPU at 100%.
    """
    import time

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    stream_chunks = [
        ModelResponseStream(
            id="chatcmpl-a",
            created=1742056047,
            model="gpt-4o",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="a", role="assistant"),
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-b",
            created=1742056047,
            model="gpt-4o",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="b", role="assistant"),
                )
            ],
        ),
    ]

    class FreshIteratorStream:
        def __aiter__(self):
            return _FreshIter(list(stream_chunks))

    class _FreshIter:
        def __init__(self, chunks):
            self._chunks = chunks
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._chunks):
                raise StopAsyncIteration
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk

    wrapper = CustomStreamWrapper(
        completion_stream=FreshIteratorStream(),
        model="gpt-4o",
        logging_obj=Logging(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="completion",
            start_time=time.time(),
            litellm_call_id="12345",
            function_id="1245",
        ),
        custom_llm_provider="bedrock",
    )

    first = await wrapper.__anext__()
    second = await wrapper.__anext__()

    assert first.choices[0].delta.get("content") == "a"
    assert second.choices[0].delta.get("content") == "b"


@pytest.mark.asyncio
async def test_custom_stream_wrapper_async_noop_chunks_do_not_busy_loop():
    class InfiniteNoneAsyncStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            return None

    wrapper = CustomStreamWrapper(
        completion_stream=InfiniteNoneAsyncStream(),
        model="test-model",
        logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )

    result = await asyncio.wait_for(wrapper.__anext__(), timeout=1)

    assert wrapper.sent_last_chunk is True
    assert result.choices[0].finish_reason == "stop"


def test_custom_stream_wrapper_sync_noop_chunks_do_not_busy_loop():
    def infinite_none_stream():
        while True:
            yield None

    wrapper = CustomStreamWrapper(
        completion_stream=infinite_none_stream(),
        model="test-model",
        logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )

    result = wrapper.__next__()

    assert wrapper.sent_last_chunk is True
    assert result.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_custom_stream_wrapper_aclose_after_partial_consume():
    stream_closed = False

    class FakePartialStream:
        def __aiter__(self):
            return self

        def __init__(self):
            self._index = 0

        async def __anext__(self):
            if self._index == 0:
                self._index += 1
                from litellm.types.utils import (
                    Delta,
                    ModelResponseStream,
                    StreamingChoices,
                )

                return ModelResponseStream(
                    id="chatcmpl-test",
                    created=1234567890,
                    model="test-model",
                    object="chat.completion.chunk",
                    choices=[
                        StreamingChoices(
                            finish_reason=None,
                            index=0,
                            delta=Delta(content="hi", role="assistant"),
                        )
                    ],
                )
            await asyncio.sleep(3600)

        async def aclose(self):
            nonlocal stream_closed
            stream_closed = True

    wrapper = CustomStreamWrapper(
        completion_stream=FakePartialStream(),
        model="test-model",
        logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )

    async for _ in wrapper:
        break
    await wrapper.aclose()

    assert stream_closed is True


# ── Router stream_with_fallbacks cleanup tests ──────────────────


@pytest.mark.asyncio
async def test_stream_with_fallbacks_closes_stream_on_generator_close():
    """Closing the FallbackStreamWrapper must aclose() the underlying model_response
    via stream_with_fallbacks' finally block."""
    from litellm.router import Router

    stream_closed = False

    class FakeStream(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=MagicMock(),
                custom_llm_provider="openai",
            )
            self._items = ["chunk1", "chunk2", "chunk3"]
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._index]
            self._index += 1
            return item

        async def aclose(self):
            nonlocal stream_closed
            stream_closed = True

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test",
                    "api_key": "fake",
                },
            }
        ]
    )

    fake_stream = FakeStream()

    # Call _acompletion_streaming_iterator directly so we go through
    # stream_with_fallbacks and its finally block
    result = await router._acompletion_streaming_iterator(
        model_response=fake_stream,
        messages=[{"role": "user", "content": "hi"}],
        initial_kwargs={"model": "test-model"},
    )

    # Consume one chunk then close (simulates client disconnect)
    async for _ in result:
        break
    await result.aclose()

    assert (
        stream_closed
    ), "model_response stream was not closed by stream_with_fallbacks finally block"


@pytest.mark.asyncio
async def test_stream_with_fallbacks_closes_stream_on_normal_completion():
    """stream_with_fallbacks must aclose() model_response even on normal completion."""
    from litellm.router import Router

    stream_closed = False

    class FakeStream(CustomStreamWrapper):
        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=MagicMock(),
                custom_llm_provider="openai",
            )
            self._items = ["chunk1"]
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._index]
            self._index += 1
            return item

        async def aclose(self):
            nonlocal stream_closed
            stream_closed = True

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test",
                    "api_key": "fake",
                },
            }
        ]
    )

    fake_stream = FakeStream()

    result = await router._acompletion_streaming_iterator(
        model_response=fake_stream,
        messages=[{"role": "user", "content": "hi"}],
        initial_kwargs={"model": "test-model"},
    )

    # Exhaust the stream fully
    async for _ in result:
        pass
    await result.aclose()

    assert stream_closed, "model_response stream was not closed after normal completion"


@pytest.mark.asyncio
async def test_stream_with_fallbacks_closes_both_on_fallback_disconnect():
    """When a fallback is triggered and the client disconnects during fallback
    iteration, both model_response and fallback_response must be closed."""
    from litellm.exceptions import MidStreamFallbackError
    from litellm.router import Router

    model_closed = False
    fallback_closed = False

    class FakeModelStream(CustomStreamWrapper):
        """Stream that raises MidStreamFallbackError immediately to trigger fallback."""

        def __init__(self):
            super().__init__(
                completion_stream=None,
                model="test-model",
                logging_obj=MagicMock(),
                custom_llm_provider="openai",
            )
            self.chunks = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise MidStreamFallbackError(
                message="test mid-stream error",
                model="test-model",
                llm_provider="openai",
                generated_content="",
            )

        async def aclose(self):
            nonlocal model_closed
            model_closed = True

    class FakeFallbackStream:
        """Fallback stream that yields chunks."""

        def __init__(self):
            self._items = ["fb1", "fb2", "fb3"]
            self._index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._index >= len(self._items):
                raise StopAsyncIteration
            item = self._items[self._index]
            self._index += 1
            return item

        async def aclose(self):
            nonlocal fallback_closed
            fallback_closed = True

    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "openai/test",
                    "api_key": "fake",
                },
            }
        ]
    )

    fake_model_stream = FakeModelStream()
    fake_fallback_stream = FakeFallbackStream()

    # Mock async_function_with_fallbacks_common_utils to return the fallback stream
    # instead of actually calling through the full fallback machinery
    with patch.object(
        router,
        "async_function_with_fallbacks_common_utils",
        return_value=fake_fallback_stream,
    ):
        result = await router._acompletion_streaming_iterator(
            model_response=fake_model_stream,
            messages=[{"role": "user", "content": "hi"}],
            initial_kwargs={
                "model": "test-model",
                "fallbacks": ["other-model"],
            },
        )

        # Consume one fallback chunk then close (simulates client disconnect)
        async for _ in result:
            break
        await result.aclose()

    assert model_closed, "model_response stream was not closed"
    assert fallback_closed, "fallback_response stream was not closed"
