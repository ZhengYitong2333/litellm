import json
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import (
    ErrorEvent,
    ErrorEventError,
    ResponseFailedEvent,
    ResponsesAPIResponse,
    ResponsesAPIStreamEvents,
)


def _mock_logging_obj() -> Mock:
    logging_obj = Mock(spec=LiteLLMLoggingObj)
    logging_obj.model_call_details = {"litellm_params": {}}
    logging_obj.async_failure_handler = Mock()
    logging_obj.failure_handler = Mock()
    return logging_obj


@pytest.mark.asyncio
async def test_async_responses_stream_error_event_raises_litellm_exception():
    error_chunk = {
        "type": "error",
        "sequence_number": 2,
        "error": {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "message": "Input exceeds the model context window.",
            "param": "input",
        },
    }

    async def mock_aiter_bytes():
        yield f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = _mock_logging_obj()
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=2,
        error=ErrorEventError(
            type="invalid_request_error",
            code="context_length_exceeded",
            message="Input exceeds the model context window.",
            param="input",
        ),
    )
    iterator = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5.4-mini",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with (
        pytest.raises(litellm.ContextWindowExceededError),
        patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async,
        patch("litellm.responses.streaming_iterator.executor") as mock_executor,
    ):
        await iterator.__anext__()

    assert iterator.finished is True
    mock_run_async.assert_called_once()
    mock_executor.submit.assert_called_once()


def test_sync_responses_stream_error_event_raises_litellm_exception():
    error_chunk = {
        "type": "error",
        "sequence_number": 2,
        "error": {
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded",
            "message": "Too many requests.",
            "param": None,
        },
    }

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.iter_bytes.return_value = [
        f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")
    ]
    mock_logging_obj = _mock_logging_obj()
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.return_value = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=2,
        error=ErrorEventError(
            type="rate_limit_error",
            code="rate_limit_exceeded",
            message="Too many requests.",
            param=None,
        ),
    )
    iterator = SyncResponsesAPIStreamingIterator(
        response=mock_response,
        model="gpt-5.4-mini",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with (
        pytest.raises(litellm.RateLimitError),
        patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async,
        patch("litellm.responses.streaming_iterator.executor") as mock_executor,
    ):
        next(iterator)

    assert iterator.finished is True
    mock_run_async.assert_called_once()
    mock_executor.submit.assert_called_once()


def test_responses_stream_error_event_exception_mapping_fallbacks():
    iterator = BaseResponsesAPIStreamingIterator(
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://api.example.test")
        ),
        model="gpt-5.4-mini",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=_mock_logging_obj(),
        custom_llm_provider="openai",
    )

    auth_event = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=1,
        error=ErrorEventError(
            type="authentication_error",
            code="invalid_api_key",
            message="Invalid API key.",
            param=None,
        ),
    )
    default_event = ErrorEvent(
        type=ResponsesAPIStreamEvents.ERROR,
        sequence_number=2,
        error=ErrorEventError(
            type="invalid_request_error",
            code="bad_request",
            message="Bad request.",
            param="input",
        ),
    )

    assert isinstance(
        iterator._exception_from_error_event(auth_event), litellm.AuthenticationError
    )
    assert isinstance(
        iterator._exception_from_error_event(default_event), litellm.BadRequestError
    )


def test_response_failed_event_maps_too_many_requests_to_rate_limit_error():
    iterator = BaseResponsesAPIStreamingIterator(
        response=httpx.Response(
            429, request=httpx.Request("POST", "https://api.example.test")
        ),
        model="sophnet-gpt-5.5",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=_mock_logging_obj(),
        custom_llm_provider="openai",
    )

    failed_event = ResponseFailedEvent(
        type=ResponsesAPIStreamEvents.RESPONSE_FAILED,
        response=ResponsesAPIResponse(
            id="resp_failed",
            created_at=0,
            status="failed",
            model="sophnet-gpt-5.5",
            object="response",
            output=[],
            error={
                "type": "too_many_requests",
                "code": "too_many_requests",
                "message": "Too Many Requests",
            },
        ),
    )

    assert isinstance(
        iterator._exception_from_failed_response_event(failed_event),
        litellm.RateLimitError,
    )


@pytest.mark.asyncio
async def test_async_iterator_raises_before_yielding_response_failed_event():
    failed_chunk = {
        "type": "response.failed",
        "response": {
            "id": "resp_123",
            "created_at": 0,
            "status": "failed",
            "model": "sophnet-gpt-5.5",
            "object": "response",
            "output": [],
            "error": {
                "type": "too_many_requests",
                "code": "too_many_requests",
                "message": "Too Many Requests",
            },
        },
    }
    error_chunk = {
        "type": "error",
        "sequence_number": 2,
        "error": {
            "type": "too_many_requests",
            "code": "too_many_requests",
            "message": "Too Many Requests",
        },
    }

    async def mock_aiter_bytes():
        yield f"data: {json.dumps(failed_chunk)}\n\n".encode("utf-8")
        yield f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = _mock_logging_obj()
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.side_effect = [
        ResponseFailedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_FAILED,
            response=ResponsesAPIResponse(
                id="resp_123",
                created_at=0,
                status="failed",
                model="sophnet-gpt-5.5",
                object="response",
                output=[],
                error={
                    "type": "too_many_requests",
                    "code": "too_many_requests",
                    "message": "Too Many Requests",
                },
            ),
        ),
        ErrorEvent(
            type=ResponsesAPIStreamEvents.ERROR,
            sequence_number=2,
            error=ErrorEventError(
                type="too_many_requests",
                code="too_many_requests",
                message="Too Many Requests",
                param=None,
            ),
        ),
    ]
    iterator = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="sophnet-gpt-5.5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with pytest.raises(litellm.RateLimitError):
        await iterator.__anext__()

    assert iterator.finished is True
    assert iterator.completed_response is not None
    assert iterator.completed_response.type == ResponsesAPIStreamEvents.RESPONSE_FAILED


class _BrokenSyncResponse:
    headers = {}

    def iter_bytes(self):
        yield b": keep-alive\n\n"
        raise httpx.ReadError("stream disconnected")


class _TimeoutSyncResponse:
    headers = {}

    def iter_bytes(self):
        if False:
            yield b""
        raise httpx.ReadTimeout("stream read timed out")


class _BrokenAsyncResponse:
    headers = {}

    async def aiter_bytes(self):
        yield b": keep-alive\n\n"
        raise httpx.ReadError("async stream disconnected")


def test_sync_responses_stream_transport_disconnect_logs_failure():
    iterator = SyncResponsesAPIStreamingIterator(
        response=_BrokenSyncResponse(),
        model="sophnet-gpt-5.5",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=_mock_logging_obj(),
        custom_llm_provider="openai",
    )

    with (
        pytest.raises(httpx.ReadError, match="stream disconnected"),
        patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async,
        patch("litellm.responses.streaming_iterator.executor") as mock_executor,
    ):
        next(iterator)

    assert iterator.finished is True
    mock_run_async.assert_called_once()
    mock_executor.submit.assert_called_once()


def test_sync_responses_stream_transport_timeout_logs_failure():
    iterator = SyncResponsesAPIStreamingIterator(
        response=_TimeoutSyncResponse(),
        model="sophnet-minimax-m3",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=_mock_logging_obj(),
        custom_llm_provider="openai",
    )

    with (
        pytest.raises(httpx.ReadTimeout, match="stream read timed out"),
        patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async,
        patch("litellm.responses.streaming_iterator.executor") as mock_executor,
    ):
        next(iterator)

    assert iterator.finished is True
    mock_run_async.assert_called_once()
    mock_executor.submit.assert_called_once()


@pytest.mark.asyncio
async def test_async_responses_stream_transport_disconnect_logs_failure():
    iterator = ResponsesAPIStreamingIterator(
        response=_BrokenAsyncResponse(),
        model="sophnet-gpt-5.5",
        responses_api_provider_config=Mock(spec=BaseResponsesAPIConfig),
        logging_obj=_mock_logging_obj(),
        custom_llm_provider="openai",
    )

    with (
        pytest.raises(httpx.ReadError, match="async stream disconnected"),
        patch(
            "litellm.responses.streaming_iterator.run_async_function"
        ) as mock_run_async,
        patch("litellm.responses.streaming_iterator.executor") as mock_executor,
    ):
        await iterator.__anext__()

    assert iterator.finished is True
    mock_run_async.assert_called_once()
    mock_executor.submit.assert_called_once()


@pytest.mark.asyncio
async def test_async_iterator_waits_for_error_after_failed_event_without_details():
    failed_chunk = {
        "type": "response.failed",
        "response": {
            "id": "resp_123",
            "created_at": 0,
            "status": "failed",
            "model": "sophnet-gpt-5.5",
            "object": "response",
            "output": [],
            "error": None,
        },
    }
    error_chunk = {
        "type": "error",
        "sequence_number": 2,
        "error": {
            "type": "too_many_requests",
            "code": "too_many_requests",
            "message": "Too Many Requests",
        },
    }

    async def mock_aiter_bytes():
        yield f"data: {json.dumps(failed_chunk)}\n\n".encode("utf-8")
        yield f"data: {json.dumps(error_chunk)}\n\n".encode("utf-8")

    mock_response = Mock()
    mock_response.headers = {}
    mock_response.aiter_bytes = mock_aiter_bytes
    mock_logging_obj = _mock_logging_obj()
    mock_config = Mock(spec=BaseResponsesAPIConfig)
    mock_config.transform_streaming_response.side_effect = [
        ResponseFailedEvent(
            type=ResponsesAPIStreamEvents.RESPONSE_FAILED,
            response=ResponsesAPIResponse(
                id="resp_123",
                created_at=0,
                status="failed",
                model="sophnet-gpt-5.5",
                object="response",
                output=[],
                error=None,
            ),
        ),
        ErrorEvent(
            type=ResponsesAPIStreamEvents.ERROR,
            sequence_number=2,
            error=ErrorEventError(
                type="too_many_requests",
                code="too_many_requests",
                message="Too Many Requests",
                param=None,
            ),
        ),
    ]
    iterator = ResponsesAPIStreamingIterator(
        response=mock_response,
        model="sophnet-gpt-5.5",
        responses_api_provider_config=mock_config,
        logging_obj=mock_logging_obj,
        custom_llm_provider="openai",
    )

    with pytest.raises(litellm.RateLimitError):
        await iterator.__anext__()

    assert iterator.finished is True
    assert iterator.completed_response is not None
    assert iterator.completed_response.type == ResponsesAPIStreamEvents.RESPONSE_FAILED
