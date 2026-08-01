import copy
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Dict, List, Optional, Set, Tuple

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes

CompletionFn = Callable[..., Awaitable[Any]]

_INTERNAL_REQUEST_KEY = "_vision_interception_internal"
_TRANSCRIPTION_PROMPT = (
    "Transcribe all visible text in this image exactly. "
    "If there is no readable text, provide a concise factual description. "
    "If no image is available, return exactly [[NO_IMAGE]]. "
    "Return only the transcription or description."
)
_NO_IMAGE_RESPONSES = {"[[no_image]]", "no image provided", "no image was provided"}


class VisionInterceptionError(RuntimeError):
    pass


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

        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not self._contains_image(messages):
            return None

        transformed_messages = await self._replace_images(messages)
        modified_kwargs = kwargs.copy()
        modified_kwargs["messages"] = transformed_messages
        verbose_logger.info(
            "VisionInterception: replaced image inputs [target_model=%s vision_model=%s]",
            self._model_group(kwargs),
            self.vision_model,
        )
        return modified_kwargs

    @staticmethod
    def _model_group(kwargs: Dict[str, Any]) -> str:
        metadata = kwargs.get("metadata")
        if isinstance(metadata, Mapping):
            model_group = metadata.get("model_group")
            if isinstance(model_group, str):
                return model_group

        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, Mapping):
            nested_metadata = litellm_params.get("metadata")
            if isinstance(nested_metadata, Mapping):
                model_group = nested_metadata.get("model_group")
                if isinstance(model_group, str):
                    return model_group

        model = kwargs.get("model")
        return model if isinstance(model, str) else ""

    @staticmethod
    def _contains_image(messages: List[Dict[str, Any]]) -> bool:
        return any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(block, Mapping) and block.get("type") == "image_url"
                for block in message["content"]
            )
            for message in messages
            if isinstance(message, Mapping)
        )

    async def _replace_images(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        transformed = copy.deepcopy(messages)
        for message in transformed:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            replacement = []
            for block in content:
                if not (
                    isinstance(block, Mapping) and block.get("type") == "image_url"
                ):
                    replacement.append(block)
                    continue
                vision_model, transcription = await self._transcribe_image(dict(block))
                replacement.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Image transcription by {vision_model}]\n"
                            f"{transcription}"
                        ),
                    }
                )
            message["content"] = replacement
        return transformed

    async def _transcribe_image(self, image_block: Dict[str, Any]) -> Tuple[str, str]:
        failures = []
        for vision_model in [self.vision_model, *self.fallback_vision_models]:
            try:
                response = await self._completion_fn(
                    model=vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _TRANSCRIPTION_PROMPT},
                                image_block,
                            ],
                        }
                    ],
                    max_tokens=512,
                    temperature=0,
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
