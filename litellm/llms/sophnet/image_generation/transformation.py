import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_PROJECT_BASE = "https://www.sophnet.com/api/open-apis/projects/easyllms"
DEFAULT_POLLING_TIMEOUT = 180
POLL_INTERVAL_SECONDS = 2
CHAT_API_BASE_MARKERS = ("/open-apis/v1",)

SUCCEEDED_STATUSES = {"SUCCEEDED", "SUCCESS"}
FAILED_STATUSES = {"FAILED", "FAIL", "ERROR", "CANCELED", "CANCELLED"}
RUNNING_STATUSES = {"PENDING", "RUNNING", "QUEUED", "PROCESSING", "CREATED"}


def _uses_async_task(model: str) -> bool:
    return "seedream" not in model.lower()


def _project_base(api_base: Optional[str]) -> str:
    if not api_base:
        return DEFAULT_PROJECT_BASE
    base = api_base.rstrip("/")
    if any(marker in base for marker in CHAT_API_BASE_MARKERS):
        return DEFAULT_PROJECT_BASE
    for suffix in ("/imagegenerator/task", "/imagegenerator"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _timeout_secs(litellm_params: dict, optional_params: dict) -> float:
    timeout = optional_params.get("timeout") or litellm_params.get("timeout")
    if timeout is None:
        return float(DEFAULT_POLLING_TIMEOUT)
    if isinstance(timeout, (int, float)):
        return float(timeout)
    read_timeout = getattr(timeout, "read", None)
    if read_timeout is not None:
        return float(read_timeout)
    return float(DEFAULT_POLLING_TIMEOUT)


def _map_size(model: str, size: Optional[str]) -> Optional[str]:
    if not size:
        if not _uses_async_task(model):
            return "2K"
        return None
    if _uses_async_task(model) and "x" in size.lower() and "*" not in size:
        return size.replace("x", "*").replace("X", "*")
    return size


class SophnetImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped = dict(optional_params)
        if "size" in non_default_params and "size" not in mapped:
            mapped_size = _map_size(model, non_default_params["size"])
            if mapped_size is not None:
                mapped["size"] = mapped_size
        if "n" in non_default_params and "n" not in mapped:
            mapped["n"] = non_default_params["n"]
        return mapped

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        project_base = _project_base(api_base or get_secret_str("SOPHNET_API_BASE"))
        if _uses_async_task(model):
            return f"{project_base}/imagegenerator/task"
        return f"{project_base}/imagegenerator"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        final_api_key = (
            api_key
            or get_secret_str("SOPHNET_API_KEY")
            or get_secret_str("SOPHNET_API_KEY_1")
        )
        if not final_api_key:
            raise ValueError("SOPHNET_API_KEY is not set")
        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        size = _map_size(model, optional_params.get("size"))
        n = optional_params.get("n")
        watermark = optional_params.get("watermark", False)

        if _uses_async_task(model):
            parameters: Dict[str, Any] = {"watermark": watermark}
            if n is not None:
                parameters["n"] = n
            if size is not None:
                parameters["size"] = size
            return {
                "model": model,
                "input": {"prompt": prompt},
                "parameters": parameters,
            }

        body: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "watermark": watermark,
        }
        if size is not None:
            body["size"] = size
        if n is not None:
            body["n"] = n
        return body

    def _raise_if_error(self, raw_response: httpx.Response, data: Any) -> None:
        if not isinstance(data, dict):
            if raw_response.status_code >= 400:
                raise self.get_error_class(
                    error_message=raw_response.text,
                    status_code=raw_response.status_code,
                    headers=raw_response.headers,
                )
            return

        message = data.get("message")
        error = data.get("error")
        output = data.get("output")
        openai_data = data.get("data")
        status = data.get("status")
        has_result = bool(output) or bool(openai_data)
        is_error_status = isinstance(status, int) and status not in (0, 200)

        if raw_response.status_code >= 400 or (
            not has_result and (error or is_error_status or message)
        ):
            raise self.get_error_class(
                error_message=str(message or error or data),
                status_code=(
                    raw_response.status_code if raw_response.status_code >= 400 else 400
                ),
                headers=raw_response.headers,
            )

    def _extract_task_id(self, data: dict) -> Optional[str]:
        output = data.get("output") or {}
        if isinstance(output, dict):
            task_id = output.get("taskId") or output.get("task_id") or output.get("id")
            if task_id:
                return str(task_id)
        return data.get("taskId") or data.get("task_id") or data.get("id")

    def _task_status(self, data: dict) -> str:
        output = data.get("output") or {}
        status = ""
        if isinstance(output, dict):
            status = output.get("taskStatus") or output.get("status") or ""
        if not status:
            status = data.get("taskStatus") or data.get("status") or ""
        return str(status).upper()

    def _extract_urls(self, data: dict) -> List[str]:
        urls: List[str] = []
        openai_data = data.get("data")
        if isinstance(openai_data, list):
            for item in openai_data:
                if isinstance(item, dict) and item.get("url"):
                    urls.append(item["url"])
                elif isinstance(item, str):
                    urls.append(item)

        output = data.get("output") or {}
        results: Any = []
        if isinstance(output, dict):
            results = output.get("results") or output.get("images") or []
            if isinstance(output.get("url"), str):
                urls.append(output["url"])
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    url = item.get("url") or item.get("image")
                    if url:
                        urls.append(url)
                elif isinstance(item, str):
                    urls.append(item)
        return urls

    def _check_timeout(self, start_time: float, timeout_secs: float) -> None:
        if time.time() - start_time > timeout_secs:
            raise TimeoutError(
                f"Sophnet image generation timed out after {timeout_secs} seconds"
            )

    def _poll_task_sync(
        self,
        task_id: str,
        project_base: str,
        headers: Dict[str, str],
        timeout_secs: float,
    ) -> dict:
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client = _get_httpx_client()
        start_time = time.time()
        task_url = f"{project_base}/imagegenerator/task/{task_id}"
        verbose_logger.debug("Polling Sophnet image task: %s", task_url)

        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)
            response = client.get(url=task_url, headers=headers)
            data = response.json()
            self._raise_if_error(response, data)
            status = self._task_status(data)
            if status in SUCCEEDED_STATUSES or self._extract_urls(data):
                return data
            if status in FAILED_STATUSES:
                output = data.get("output") if isinstance(data, dict) else {}
                reason = ""
                if isinstance(output, dict):
                    reason = output.get("message") or output.get("error") or ""
                raise ValueError(f"Sophnet image generation failed: {reason or data}")
            if status and status not in RUNNING_STATUSES:
                raise ValueError(f"Unknown Sophnet task status: {status}")
            time.sleep(POLL_INTERVAL_SECONDS)

    def _populate_response(
        self, data: dict, model_response: ImageResponse
    ) -> ImageResponse:
        if not model_response.data:
            model_response.data = []
        for url in self._extract_urls(data):
            model_response.data.append(ImageObject(url=url, b64_json=None))
        created = data.get("created")
        if created is not None:
            model_response.created = created
        return model_response

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse Sophnet image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        self._raise_if_error(raw_response, data)

        if not _uses_async_task(model):
            return self._populate_response(data, model_response)

        if self._extract_urls(data):
            return self._populate_response(data, model_response)

        task_id = self._extract_task_id(data) if isinstance(data, dict) else None
        if not task_id:
            raise self.get_error_class(
                error_message=f"Sophnet image generation response missing taskId: {data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        poll_headers = {
            "Authorization": raw_response.request.headers.get("Authorization", ""),
            "Content-Type": "application/json",
        }
        project_base = _project_base(str(raw_response.request.url))
        polled = self._poll_task_sync(
            task_id=task_id,
            project_base=project_base,
            headers=poll_headers,
            timeout_secs=_timeout_secs(litellm_params, optional_params),
        )
        return self._populate_response(polled, model_response)
