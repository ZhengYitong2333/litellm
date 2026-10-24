import hashlib
import json
from typing import Any, Dict, Mapping, Optional

from litellm.integrations.custom_logger import CustomLogger

_OPENCODE_API_BASE_PREFIX = "https://opencode.ai/inference/"
_USER_AGENT = "litellm-proxy/1.0"


class OpencodeSessionLogger(CustomLogger):
    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Any
    ) -> Optional[dict]:
        api_base = kwargs.get("api_base")
        if not isinstance(api_base, str) or not api_base.startswith(
            _OPENCODE_API_BASE_PREFIX
        ):
            return None

        headers = dict(kwargs.get("extra_headers") or {})
        headers["x-opencode-session"] = self._session_id(kwargs)
        headers.setdefault("User-Agent", _USER_AGENT)
        kwargs["extra_headers"] = headers
        return kwargs

    @classmethod
    def _session_id(cls, kwargs: Mapping[str, Any]) -> str:
        metadata = cls._metadata(kwargs)
        for key in ("session_id", "litellm_session_id", "trace_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                return value

        prompt_cache_key = kwargs.get("prompt_cache_key")
        if isinstance(prompt_cache_key, str) and prompt_cache_key:
            return prompt_cache_key

        user = kwargs.get("user") or metadata.get("user_api_key_alias") or ""
        digest = hashlib.sha256(
            json.dumps(
                [user, cls._first_user_message(kwargs)],
                default=str,
            ).encode()
        ).hexdigest()
        return f"litellm-{digest[:32]}"

    @staticmethod
    def _metadata(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
        for key in ("metadata", "litellm_metadata"):
            value = kwargs.get(key)
            if isinstance(value, Mapping):
                return value
            litellm_params = kwargs.get("litellm_params")
            if isinstance(litellm_params, Mapping):
                nested = litellm_params.get(key)
                if isinstance(nested, Mapping):
                    return nested
        return {}

    @classmethod
    def _first_user_message(cls, kwargs: Mapping[str, Any]) -> str:
        for collection in (kwargs.get("messages"), kwargs.get("input")):
            for item in cls._iter_items(collection):
                if cls._role(item, collection) == "user":
                    text = cls._text(item)
                    if text:
                        return text
        return ""

    @staticmethod
    def _iter_items(value: Any) -> list:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
        return []

    @staticmethod
    def _role(item: Any, collection: Any) -> Optional[str]:
        if not isinstance(item, Mapping):
            return None
        role = item.get("role")
        if isinstance(role, str):
            return role
        return "user" if isinstance(collection, str) else None

    @classmethod
    def _text(cls, item: Any) -> str:
        if isinstance(item, str):
            return item
        if not isinstance(item, Mapping):
            return ""
        content = item.get("content")
        if isinstance(content, str):
            return content
        for block in cls._iter_items(content):
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                return block["text"]
        return ""


opencode_session_logger = OpencodeSessionLogger()
