import asyncio
import hashlib
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Dict, List, Optional, Set, Tuple

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes

CompletionFn = Callable[..., Awaitable[Any]]

_INTERNAL_REQUEST_KEY = "_vision_interception_internal"
_MAX_TRANSCRIPTION_CACHE = 256
_IMAGE_BLOCK_TYPES = {"image_url", "input_image"}
_TRANSCRIPTION_PROMPT = (
    "Transcribe all visible text in this image exactly. "
    "If there is no readable text, provide a concise factual description. "
    "If no image is available, return exactly [[NO_IMAGE]]. "
    "Return only the transcription or description."
)
_NO_IMAGE_RESPONSES = {"[[no_image]]", "no image provided", "no image was provided"}


class VisionInterceptionError(RuntimeError):
    status_code = 422


class VisionInterceptionLogger(CustomLogger):
    def __init__(
        self,
        vision_model: str,
        target_models: List[str],
        fallback_vision_models: Optional[List[str]] = None,
        completion_fn: Optional[CompletionFn] = None,
    ):
        super().__init__()
        if not vision_model:
            raise ValueError("vision_interception requires vision_model")
        self.vision_model = vision_model
        self.fallback_vision_models = [
            model
            for model in fallback_vision_models or []
            if model and model != vision_model
        ]
        self.target_models: Set[str] = set(target_models)
        self._completion_fn = completion_fn or self._router_acompletion
        self._transcription_cache: OrderedDict[str, Tuple[str, str]] = OrderedDict()
        self._inflight: Dict[str, asyncio.Task] = {}

    @classmethod
    def from_config_yaml(cls, config: Dict[str, Any]) -> "VisionInterceptionLogger":
        return cls(
            vision_model=str(config.get("vision_model", "")),
            fallback_vision_models=list(config.get("fallback_vision_models", [])),
            target_models=list(config.get("target_models", [])),
        )

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "VisionInterceptionLogger":
        config = litellm_settings.get("vision_interception_params")
        if config is None:
            config = callback_specific_params.get("vision_interception", {})
        return VisionInterceptionLogger.from_config_yaml(config)

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        if kwargs.pop(_INTERNAL_REQUEST_KEY, False):
            return kwargs

        if self._model_group(kwargs) not in self.target_models:
            return None

        if not self._has_image_payload(kwargs):
            return None

        modified_kwargs = kwargs.copy()
        if "messages" in modified_kwargs:
            modified_kwargs["messages"] = await self._replace_images_in_value(
                modified_kwargs["messages"]
            )
        if "input" in modified_kwargs:
            modified_kwargs["input"] = await self._replace_images_in_value(
                modified_kwargs["input"]
            )
        verbose_logger.info(
            "VisionInterception: replaced image inputs [target_model=%s vision_model=%s]",
            self._model_group(kwargs),
            self.vision_model,
        )
        return modified_kwargs

    @staticmethod
    def _model_group(kwargs: Dict[str, Any]) -> str:
        for metadata_key in ("metadata", "litellm_metadata"):
            metadata = kwargs.get(metadata_key)
            if isinstance(metadata, Mapping):
                model_group = metadata.get("model_group")
                if isinstance(model_group, str):
                    return model_group

        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, Mapping):
            for metadata_key in ("metadata", "litellm_metadata"):
                nested_metadata = litellm_params.get(metadata_key)
                if isinstance(nested_metadata, Mapping):
                    model_group = nested_metadata.get("model_group")
                    if isinstance(model_group, str):
                        return model_group

        model = kwargs.get("model")
        return model if isinstance(model, str) else ""

    @classmethod
    def _has_image_payload(cls, kwargs: Mapping[str, Any]) -> bool:
        return cls._contains_image(kwargs.get("messages")) or cls._contains_image(
            kwargs.get("input")
        )

    @classmethod
    def _contains_image(cls, value: Any) -> bool:
        if isinstance(value, list):
            return any(cls._contains_image(item) for item in value)
        if not isinstance(value, Mapping):
            return False
        if cls._is_image_block(value):
            return True
        content = value.get("content")
        if isinstance(content, list):
            return cls._contains_image(content)
        return False

    @staticmethod
    def _is_image_block(block: Any) -> bool:
        return isinstance(block, Mapping) and block.get("type") in _IMAGE_BLOCK_TYPES

    async def _replace_images_in_value(self, value: Any) -> Any:
        if isinstance(value, list):
            return list(
                await asyncio.gather(
                    *[self._replace_images_in_value(item) for item in value]
                )
            )
        if isinstance(value, Mapping):
            if self._is_image_block(value):
                vision_model, transcription = await self._transcribe_image(dict(value))
                return self._text_replacement(value, vision_model, transcription)
            content = value.get("content")
            if isinstance(content, list):
                updated = dict(value)
                updated["content"] = await self._replace_images_in_value(content)
                return updated
        return value

    @staticmethod
    def _text_replacement(
        block: Mapping[str, Any], vision_model: str, transcription: str
    ) -> Dict[str, Any]:
        text = f"[Image transcription by {vision_model}]\n{transcription}"
        if block.get("type") == "input_image":
            return {"type": "input_text", "text": text}
        return {"type": "text", "text": text}

    @staticmethod
    def _as_chat_image_block(block: Mapping[str, Any]) -> Dict[str, Any]:
        image_url = block.get("image_url")
        if image_url is None:
            image_url = block.get("url")
        if isinstance(image_url, Mapping):
            chat_image_url: Dict[str, Any] = {"url": image_url.get("url") or ""}
            detail = image_url.get("detail") or block.get("detail")
            if detail:
                chat_image_url["detail"] = detail
            return {"type": "image_url", "image_url": chat_image_url}
        chat_image_url = {"url": image_url if isinstance(image_url, str) else ""}
        if block.get("detail"):
            chat_image_url["detail"] = block["detail"]
        return {"type": "image_url", "image_url": chat_image_url}

    @classmethod
    def _image_cache_key(cls, image_block: Mapping[str, Any]) -> str:
        chat_image_block = cls._as_chat_image_block(image_block)
        image_url = chat_image_block.get("image_url")
        raw = ""
        if isinstance(image_url, Mapping):
            raw = str(image_url.get("url") or "")
        elif isinstance(image_url, str):
            raw = image_url
        return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()

    def _remember_transcription(self, key: str, result: Tuple[str, str]) -> None:
        self._transcription_cache[key] = result
        self._transcription_cache.move_to_end(key)
        while len(self._transcription_cache) > _MAX_TRANSCRIPTION_CACHE:
            self._transcription_cache.popitem(last=False)

    async def _transcribe_image(self, image_block: Dict[str, Any]) -> Tuple[str, str]:
        key = self._image_cache_key(image_block)
        cached = self._transcription_cache.get(key)
        if cached is not None:
            self._transcription_cache.move_to_end(key)
            return cached

        inflight = self._inflight.get(key)
        if inflight is None:
            inflight = asyncio.create_task(self._transcribe_image_uncached(image_block))
            self._inflight[key] = inflight
        try:
            result = await inflight
            self._remember_transcription(key, result)
            return result
        finally:
            if self._inflight.get(key) is inflight:
                self._inflight.pop(key, None)

    async def _transcribe_image_uncached(
        self, image_block: Dict[str, Any]
    ) -> Tuple[str, str]:
        failures = []
        chat_image_block = self._as_chat_image_block(image_block)
        for vision_model in [self.vision_model, *self.fallback_vision_models]:
            try:
                response = await self._completion_fn(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _TRANSCRIPTION_PROMPT},
                                chat_image_block,
                            ],
                        }
                    ],
                    max_tokens=512,
                    temperature=0,
                    timeout=30,
                    num_retries=0,
                    disable_fallbacks=True,
                    **{_INTERNAL_REQUEST_KEY: True},
                )
                content = self._response_content(response)
            except Exception as exc:
                failures.append(f"{vision_model}: {type(exc).__name__}")
                continue
            if content and not self._is_no_image_response(content):
                return vision_model, content
            failure = "no image received" if content else "empty response"
            failures.append(f"{vision_model}: {failure}")
        raise VisionInterceptionError(
            "Vision interception failed for all vision models: " + "; ".join(failures)
        )

    @staticmethod
    def _is_no_image_response(content: str) -> bool:
        return content.strip().lower().rstrip(".") in _NO_IMAGE_RESPONSES

    @staticmethod
    async def _router_acompletion(**kwargs: Any) -> Any:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            raise RuntimeError("LiteLLM proxy router is not initialized")
        return await llm_router.acompletion(**kwargs)

    @staticmethod
    def _response_content(response: Any) -> str:
        choices = (
            response.get("choices", [])
            if isinstance(response, Mapping)
            else getattr(response, "choices", [])
        )
        if not choices:
            return ""
        choice = choices[0]
        message = (
            choice.get("message", {}) if isinstance(choice, Mapping) else choice.message
        )
        content = (
            message.get("content") if isinstance(message, Mapping) else message.content
        )
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                item["text"].strip()
                for item in content
                if isinstance(item, Mapping)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ).strip()
        return ""
