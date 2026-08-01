from types import SimpleNamespace

import pytest

from litellm.integrations.vision_interception.handler import (
    VisionInterceptionError,
    VisionInterceptionLogger,
)


def _image(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.asyncio
async def test_deployment_hook_skips_requests_without_images():
    logger = VisionInterceptionLogger(
        vision_model="vision-model",
        target_models=["text-only"],
    )
    kwargs = {
        "model": "text-only",
        "messages": [{"role": "user", "content": "plain text"}],
    }

    assert await logger.async_pre_call_deployment_hook(kwargs, None) is None


@pytest.mark.asyncio
async def test_deployment_hook_skips_non_target_model():
    logger = VisionInterceptionLogger(
        vision_model="vision-model",
        target_models=["text-only"],
    )
    kwargs = {
        "model": "vision-capable",
        "messages": [
            {"role": "user", "content": [_image("https://example.com/a.png")]}
        ],
    }

    assert await logger.async_pre_call_deployment_hook(kwargs, None) is None


@pytest.mark.asyncio
async def test_deployment_hook_replaces_images_for_target_model_in_order():
    calls = []

    async def completion_fn(**kwargs):
        calls.append(kwargs)
        return _response("OCR-42")

    logger = VisionInterceptionLogger(
        vision_model="vision-model",
        target_models=["text-only"],
        completion_fn=completion_fn,
    )
    kwargs = {
        "model": "deepseek/deepseek-v4-flash",
        "metadata": {"model_group": "text-only"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    _image("https://example.com/one.png"),
                    {"type": "text", "text": "second"},
                    _image("https://example.com/two.png"),
                ],
            }
        ],
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, None)

    assert result is not None
    assert result["messages"][0]["content"] == [
        {"type": "text", "text": "first"},
        {
            "type": "text",
            "text": "[Image transcription by vision-model]\nOCR-42",
        },
        {"type": "text", "text": "second"},
        {
            "type": "text",
            "text": "[Image transcription by vision-model]\nOCR-42",
        },
    ]
    assert len(calls) == 2
    assert all(call["model"] == "vision-model" for call in calls)
    assert all(call["_vision_interception_internal"] is True for call in calls)
    assert kwargs["messages"][0]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_deployment_hook_raises_when_vision_call_fails():
    async def completion_fn(**kwargs):
        raise ValueError("upstream unavailable")

    logger = VisionInterceptionLogger(
        vision_model="vision-model",
        target_models=["text-only"],
        completion_fn=completion_fn,
    )
    kwargs = {
        "model": "text-only",
        "messages": [
            {"role": "user", "content": [_image("https://example.com/a.png")]}
        ],
    }

    with pytest.raises(VisionInterceptionError, match="vision-model"):
        await logger.async_pre_call_deployment_hook(kwargs, None)


@pytest.mark.asyncio
async def test_deployment_hook_uses_fallback_vision_model_after_primary_failure():
    calls = []

    async def completion_fn(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "primary-vision-model":
            raise ValueError("TLS certificate verification failed")
        return _response("OCR-42")

    logger = VisionInterceptionLogger(
        vision_model="primary-vision-model",
        fallback_vision_models=["fallback-vision-model"],
        target_models=["text-only"],
        completion_fn=completion_fn,
    )
    kwargs = {
        "model": "text-only",
        "messages": [
            {"role": "user", "content": [_image("https://example.com/a.png")]}
        ],
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, None)

    assert calls == ["primary-vision-model", "fallback-vision-model"]
    assert result["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "[Image transcription by fallback-vision-model]\nOCR-42",
        }
    ]


@pytest.mark.asyncio
async def test_deployment_hook_uses_fallback_when_primary_receives_no_image():
    calls = []

    async def completion_fn(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "primary-vision-model":
            return _response("No image provided.")
        return _response("OCR-42")

    logger = VisionInterceptionLogger(
        vision_model="primary-vision-model",
        fallback_vision_models=["fallback-vision-model"],
        target_models=["text-only"],
        completion_fn=completion_fn,
    )
    kwargs = {
        "model": "text-only",
        "messages": [
            {"role": "user", "content": [_image("https://example.com/a.png")]}
        ],
    }

    result = await logger.async_pre_call_deployment_hook(kwargs, None)

    assert calls == ["primary-vision-model", "fallback-vision-model"]
    assert result["messages"][0]["content"][0]["text"].endswith("OCR-42")


def test_initialize_from_proxy_config():
    logger = VisionInterceptionLogger.initialize_from_proxy_config(
        litellm_settings={
            "vision_interception_params": {
                "vision_model": "sophnet-claude-opus-4-7",
                "fallback_vision_models": ["azure-gpt-5.6-terra"],
                "target_models": ["deepseek-v4-flash", "sophnet-glm-5.2"],
            }
        },
        callback_specific_params={},
    )

    assert logger.vision_model == "sophnet-claude-opus-4-7"
    assert logger.fallback_vision_models == ["azure-gpt-5.6-terra"]
    assert logger.target_models == {"deepseek-v4-flash", "sophnet-glm-5.2"}
