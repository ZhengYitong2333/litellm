"""Tests for LiteLLMCompletionTransformationHandler request normalization."""

from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.types.utils import ModelResponse


def test_response_api_handler_drops_client_metadata():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        handler.response_api_handler(
            model="test",
            input="hi",
            responses_api_request={},
            client_metadata={"source": "codex"},
        )

        assert mock_completion.call_count == 1
        assert "client_metadata" not in mock_completion.call_args.kwargs


def test_response_api_handler_drops_acompletion_flag():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        handler.response_api_handler(
            model="test",
            input="hi",
            responses_api_request={},
            acompletion=True,
        )

        assert mock_completion.call_count == 1
        assert "acompletion" not in mock_completion.call_args.kwargs


def test_response_api_handler_skips_nested_responses_bridge():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        handler.response_api_handler(
            model="test",
            input="hi",
            responses_api_request={},
        )

        assert mock_completion.call_count == 1
        assert mock_completion.call_args.kwargs["_skip_responses_api_bridge"] is True


def test_completion_skip_responses_bridge_uses_chat_completion_path():
    response = litellm.completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        custom_llm_provider="openai",
        reasoning_effort="low",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        mock_response="ok",
        _skip_responses_api_bridge=True,
    )

    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_async_response_api_handler_drops_client_metadata():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        await handler.async_response_api_handler(
            litellm_completion_request={"model": "test"},
            request_input="hi",
            responses_api_request={},
            client_metadata={"source": "codex"},
        )

        assert mock_acompletion.call_count == 1
        assert "client_metadata" not in mock_acompletion.call_args.kwargs


@pytest.mark.asyncio
async def test_async_response_api_handler_drops_acompletion_flag():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        await handler.async_response_api_handler(
            litellm_completion_request={"model": "test"},
            request_input="hi",
            responses_api_request={},
            acompletion=True,
        )

        assert mock_acompletion.call_count == 1
        assert "acompletion" not in mock_acompletion.call_args.kwargs


@pytest.mark.asyncio
async def test_async_response_api_handler_skips_nested_responses_bridge():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        await handler.async_response_api_handler(
            litellm_completion_request={"model": "test"},
            request_input="hi",
            responses_api_request={},
        )

        assert mock_acompletion.call_count == 1
        assert mock_acompletion.call_args.kwargs["_skip_responses_api_bridge"] is True
