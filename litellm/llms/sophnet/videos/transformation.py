from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_PROJECT_BASE = "https://www.sophnet.com/api/open-apis/projects/easyllms"
CHAT_API_BASE_MARKERS = ("/open-apis/v1",)
HAILUO_DURATIONS = (6, 10)
STATUS_MAP = {
    "queued": "queued",
    "pending": "queued",
    "running": "in_progress",
    "processing": "in_progress",
    "succeeded": "completed",
    "success": "completed",
    "failed": "failed",
    "fail": "failed",
    "cancelled": "failed",
    "canceled": "failed",
}


def _project_base(api_base: Optional[str]) -> str:
    if not api_base:
        return DEFAULT_PROJECT_BASE
    base = api_base.rstrip("/")
    if any(marker in base for marker in CHAT_API_BASE_MARKERS):
        return DEFAULT_PROJECT_BASE
    for suffix in (
        "/videogenerator/generate",
        "/videogenerator",
        "/imagegenerator/task",
        "/imagegenerator",
    ):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _generate_url(api_base: Optional[str]) -> str:
    return f"{_project_base(api_base)}/videogenerator/generate"


def _task_url(api_base: Optional[str], video_id: str) -> str:
    original = extract_original_video_id(video_id)
    encoded = encode_url_path_segment(original, field_name="video_id")
    return f"{_generate_url(api_base)}/{encoded}"


def _is_hailuo(model: str) -> bool:
    return "hailuo" in model.lower()


def _created_at(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _map_duration(model: str, seconds: Optional[Any]) -> int:
    if seconds is None or seconds == "":
        return 6 if _is_hailuo(model) else 5
    try:
        duration = int(float(seconds))
    except (TypeError, ValueError):
        return 6 if _is_hailuo(model) else 5
    if _is_hailuo(model) and duration not in HAILUO_DURATIONS:
        return 6 if duration <= 6 else 10
    return duration


def _map_size(size: Optional[str]) -> Dict[str, str]:
    if not size:
        return {}
    mapped: Dict[str, str] = {}
    normalized = size.lower().replace("*", "x")
    size_map = {
        "1920x1080": ("1080P", "16:9"),
        "1280x720": ("720P", "16:9"),
        "1080x1920": ("1080P", "9:16"),
        "720x1280": ("720P", "9:16"),
        "1024x1024": ("720P", "1:1"),
        "1440x1440": ("1080P", "1:1"),
    }
    if normalized in size_map:
        resolution, ratio = size_map[normalized]
        mapped["resolution"] = resolution
        mapped["ratio"] = ratio
        return mapped
    if "x" in normalized:
        width_str, _, height_str = normalized.partition("x")
        try:
            width, height = int(width_str), int(height_str)
        except ValueError:
            return mapped
        if width == height:
            mapped["ratio"] = "1:1"
        elif width > height:
            mapped["ratio"] = "16:9"
        else:
            mapped["ratio"] = "9:16"
        mapped["resolution"] = "1080P" if max(width, height) >= 1080 else "720P"
    return mapped


class SophnetVideoConfig(BaseVideoConfig):
    def __init__(self):
        super().__init__()
        self._incoming_video_id: Optional[str] = None

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped: Dict[str, Any] = {}
        if "seconds" in video_create_optional_params:
            mapped["duration"] = _map_duration(
                model, video_create_optional_params.get("seconds")
            )
        if "size" in video_create_optional_params:
            mapped.update(_map_size(video_create_optional_params.get("size")))
        if "input_reference" in video_create_optional_params:
            mapped["input_reference"] = video_create_optional_params["input_reference"]
        supported = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key not in supported:
                mapped[key] = value
        return mapped

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        api_key = (
            api_key
            or get_secret_str("SOPHNET_API_KEY")
            or get_secret_str("SOPHNET_API_KEY_1")
        )
        if not api_key:
            raise ValueError("SOPHNET_API_KEY is not set")
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        return _project_base(api_base or get_secret_str("SOPHNET_API_BASE"))

    def _raise_if_error(self, raw_response: httpx.Response, data: Any) -> None:
        if not isinstance(data, dict):
            if raw_response.status_code >= 400:
                raise self.get_error_class(
                    error_message=raw_response.text,
                    status_code=raw_response.status_code,
                    headers=raw_response.headers,
                )
            return
        status = data.get("status")
        message = data.get("message")
        result = data.get("result")
        is_error_status = isinstance(status, int) and status not in (0, 200)
        if raw_response.status_code >= 400 or (is_error_status and not result):
            raise self.get_error_class(
                error_message=str(message or data),
                status_code=(
                    raw_response.status_code if raw_response.status_code >= 400 else 400
                ),
                headers=raw_response.headers,
            )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        input_reference = video_create_optional_request_params.get("input_reference")
        if input_reference:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": input_reference},
                }
            )

        parameters: Dict[str, Any] = {"watermark": False}
        duration = video_create_optional_request_params.get("duration")
        parameters["duration"] = _map_duration(model, duration)
        for key in ("resolution", "ratio", "fps", "seed"):
            if key in video_create_optional_request_params:
                parameters[key] = video_create_optional_request_params[key]

        request_data: Dict[str, Any] = {
            "model": model,
            "content": content,
            "parameters": parameters,
        }
        files_list: List[Tuple[str, Any]] = []
        return request_data, files_list, _generate_url(api_base)

    def _to_video_object(
        self,
        data: dict,
        model: Optional[str],
        custom_llm_provider: Optional[str],
        request_data: Optional[Dict] = None,
        fallback_status: str = "queued",
    ) -> VideoObject:
        result = data.get("result") or {}
        task_id = (
            result.get("task_id")
            or result.get("id")
            or data.get("task_id")
            or data.get("id")
            or ""
        )
        usage = result.get("usage") or {}
        content = result.get("content") or {}
        video_data: Dict[str, Any] = {
            "id": str(task_id),
            "object": "video",
            "status": STATUS_MAP.get(
                str(result.get("status") or fallback_status).lower(),
                fallback_status,
            ),
            "created_at": _created_at(result.get("created_at")),
            "model": result.get("model") or (request_data or {}).get("model") or model,
        }
        if usage.get("duration") is not None:
            video_data["seconds"] = str(usage["duration"])
        elif request_data and request_data.get("parameters", {}).get("duration"):
            video_data["seconds"] = str(request_data["parameters"]["duration"])
        if usage.get("size"):
            video_data["size"] = str(usage["size"]).replace("*", "x")
        if content.get("video_url"):
            video_data["usage"] = {
                "duration_seconds": float(usage.get("duration") or 0)
            }
        video_obj = VideoObject(**video_data)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )
        return video_obj

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        data = raw_response.json()
        self._raise_if_error(raw_response, data)
        return self._to_video_object(
            data=data,
            model=model,
            custom_llm_provider=custom_llm_provider,
            request_data=request_data,
            fallback_status="queued",
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        return _task_url(api_base, video_id), {}

    def _extract_video_url(self, response_data: Dict[str, Any]) -> str:
        result = response_data.get("result") or {}
        content = result.get("content") or {}
        video_url = content.get("video_url")
        if video_url:
            return video_url
        status = str(result.get("status") or "").lower()
        if status in ("queued", "pending", "running", "processing"):
            raise ValueError(
                f"Video is still processing (status: {status}). Please wait and try again."
            )
        if status in ("failed", "fail", "cancelled", "canceled"):
            raise ValueError(
                f"Video generation failed: {response_data.get('message') or status}"
            )
        raise ValueError("Video URL not found in response. Video may not be ready yet.")

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        data = raw_response.json()
        self._raise_if_error(raw_response, data)
        video_url = self._extract_video_url(data)
        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        data = raw_response.json()
        self._raise_if_error(raw_response, data)
        video_url = self._extract_video_url(data)
        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.SOPHNET,
        )
        video_response = await async_httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video remix is not supported by Sophnet")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by Sophnet")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video listing is not supported by Sophnet")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video listing is not supported by Sophnet")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        return _task_url(api_base, video_id), {}

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        data = raw_response.json()
        self._raise_if_error(raw_response, data)
        result = data.get("result") or {}
        return VideoObject(
            id=str(result.get("id") or result.get("task_id") or ""),
            object="video",
            status="cancelled",
            created_at=_created_at(result.get("created_at")),
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        self._incoming_video_id = video_id
        return _task_url(api_base, video_id), {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        data = raw_response.json()
        self._raise_if_error(raw_response, data)
        incoming = getattr(self, "_incoming_video_id", None)
        decoded = decode_video_id_with_provider(incoming) if incoming else {}
        video_obj = self._to_video_object(
            data=data,
            model=decoded.get("model_id") or None,
            custom_llm_provider=None,
            fallback_status="queued",
        )
        if incoming:
            video_obj.id = incoming
        elif custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, decoded.get("model_id")
            )
        return video_obj
