import os
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.sophnet.image_generation.transformation import (
    DEFAULT_PROJECT_BASE,
    SophnetImageGenerationConfig,
)
from litellm.types.utils import ImageResponse
from litellm.utils import ProviderConfigManager, get_llm_provider
from litellm.types.utils import LlmProviders


@pytest.mark.parametrize(
    "model_string, expected_model",
    [
        ("sophnet/Nano-Banana-Pro", "Nano-Banana-Pro"),
        ("sophnet/Doubao-Seedream-5.0-Pro", "Doubao-Seedream-5.0-Pro"),
    ],
)
def test_get_llm_provider_returns_sophnet(model_string: str, expected_model: str):
    model, provider, _, _ = get_llm_provider(model_string)
    assert provider == "sophnet"
    assert model == expected_model


@pytest.mark.parametrize(
    "model_string, custom_provider",
    [
        ("sophnet/Nano-Banana-Pro", "sophnet"),
        ("sophnet/Doubao-Seedream-5.0-Pro", "sophnet"),
    ],
)
def test_get_model_info_mode_is_image_generation(
    model_string: str, custom_provider: str
):
    prev_env = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    prev_model_cost = litellm.model_cost
    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        info = litellm.get_model_info(
            model=model_string, custom_llm_provider=custom_provider
        )
        assert info["mode"] == "image_generation"
    finally:
        if prev_env is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = prev_env
        litellm.model_cost = prev_model_cost


def test_provider_config_manager_returns_sophnet_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="Nano-Banana-Pro",
        provider=LlmProviders.SOPHNET,
    )
    assert isinstance(config, SophnetImageGenerationConfig)


class TestSophnetImageGenerationConfig:
    def setup_method(self):
        self.cfg = SophnetImageGenerationConfig()

    def test_get_complete_url_async_default(self):
        url = self.cfg.get_complete_url(None, None, "Nano-Banana-Pro", {}, {})
        assert url == f"{DEFAULT_PROJECT_BASE}/imagegenerator/task"

    def test_get_complete_url_seedream_default(self):
        url = self.cfg.get_complete_url(None, None, "Doubao-Seedream-5.0-Pro", {}, {})
        assert url == f"{DEFAULT_PROJECT_BASE}/imagegenerator"

    def test_get_complete_url_ignores_chat_v1_base(self):
        url = self.cfg.get_complete_url(
            "https://www.sophnet.com/api/open-apis/v1",
            None,
            "Nano-Banana-Pro",
            {},
            {},
        )
        assert url == f"{DEFAULT_PROJECT_BASE}/imagegenerator/task"

    def test_get_complete_url_custom_project(self):
        url = self.cfg.get_complete_url(
            "https://www.sophnet.com/api/open-apis/projects/myproj",
            None,
            "Doubao-Seedream-5.0-Pro",
            {},
            {},
        )
        assert (
            url
            == "https://www.sophnet.com/api/open-apis/projects/myproj/imagegenerator"
        )

    def test_validate_environment_sets_auth_header(self):
        headers = self.cfg.validate_environment(
            headers={},
            model="Nano-Banana-Pro",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="sk-test-key",
        )
        assert headers["Authorization"] == "Bearer sk-test-key"
        assert headers["Content-Type"] == "application/json"

    def test_validate_environment_raises_without_key(self):
        with patch(
            "litellm.llms.sophnet.image_generation.transformation.get_secret_str",
            return_value=None,
        ):
            with pytest.raises(ValueError, match="SOPHNET_API_KEY"):
                self.cfg.validate_environment(
                    headers={},
                    model="Nano-Banana-Pro",
                    messages=[],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                )

    def test_transform_request_nano_banana_pro(self):
        req = self.cfg.transform_image_generation_request(
            model="Nano-Banana-Pro",
            prompt="a red apple",
            optional_params={"size": "1024*1024", "n": 1},
            litellm_params={},
            headers={},
        )
        assert req["model"] == "Nano-Banana-Pro"
        assert req["input"]["prompt"] == "a red apple"
        assert req["parameters"]["watermark"] is False
        assert req["parameters"]["n"] == 1
        assert req["parameters"]["size"] == "1024*1024"

    def test_transform_request_seedream_sync(self):
        req = self.cfg.transform_image_generation_request(
            model="Doubao-Seedream-5.0-Pro",
            prompt="a red apple",
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert req == {
            "model": "Doubao-Seedream-5.0-Pro",
            "prompt": "a red apple",
            "watermark": False,
            "size": "2K",
        }

    def test_map_openai_params_size_for_banana(self):
        mapped = self.cfg.map_openai_params(
            non_default_params={"size": "1024x1024"},
            optional_params={},
            model="Nano-Banana-Pro",
            drop_params=False,
        )
        assert mapped["size"] == "1024*1024"

    def test_map_openai_params_size_for_seedream(self):
        mapped = self.cfg.map_openai_params(
            non_default_params={"size": "2048x2048"},
            optional_params={},
            model="Doubao-Seedream-5.0-Pro",
            drop_params=False,
        )
        assert mapped["size"] == "2048x2048"

    def _mock_response(self, body: dict, status_code: int = 200) -> httpx.Response:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = status_code
        mock_resp.headers = {}
        mock_resp.text = str(body)
        mock_resp.json.return_value = body
        mock_resp.request = MagicMock()
        mock_resp.request.headers = {"Authorization": "Bearer sk-test-key"}
        mock_resp.request.url = f"{DEFAULT_PROJECT_BASE}/imagegenerator/task"
        return mock_resp

    def test_transform_response_seedream_extracts_url(self):
        image_url = "https://example.volces.com/seedream.png"
        mock_resp = self._mock_response(
            {
                "model": "Doubao-Seedream-5.0-Pro",
                "created": 1788433244,
                "data": [{"url": image_url}],
            }
        )
        result = self.cfg.transform_image_generation_response(
            model="Doubao-Seedream-5.0-Pro",
            raw_response=mock_resp,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
        assert result.data is not None
        assert len(result.data) == 1
        assert result.data[0].url == image_url

    def test_transform_response_raises_on_api_error_body(self):
        mock_resp = self._mock_response(
            {
                "status": 20031,
                "message": "不支持的模型: gpt-image-1.5",
                "result": None,
                "error": {"code": "EASYLLM_IMAGEGENERATOR_CREATE_ERROR"},
            }
        )
        with pytest.raises(Exception, match="不支持的模型"):
            self.cfg.transform_image_generation_response(
                model="Doubao-Seedream-5.0-Pro",
                raw_response=mock_resp,
                model_response=ImageResponse(),
                logging_obj=MagicMock(),
                request_data={},
                optional_params={},
                litellm_params={},
                encoding=None,
            )

    def test_transform_response_polls_nano_banana_task(self):
        create_resp = self._mock_response(
            {"output": {"taskId": "task-123", "taskStatus": "PENDING"}}
        )
        polled = {
            "output": {
                "taskId": "task-123",
                "taskStatus": "SUCCEEDED",
                "results": [{"url": "https://example.com/banana.png"}],
            }
        }
        poll_resp = self._mock_response(polled)
        mock_client = MagicMock()
        mock_client.get.return_value = poll_resp

        with patch(
            "litellm.llms.custom_httpx.http_handler._get_httpx_client",
            return_value=mock_client,
        ):
            result = self.cfg.transform_image_generation_response(
                model="Nano-Banana-Pro",
                raw_response=create_resp,
                model_response=ImageResponse(),
                logging_obj=MagicMock(),
                request_data={},
                optional_params={},
                litellm_params={"timeout": 30},
                encoding=None,
            )

        assert result.data[0].url == "https://example.com/banana.png"
        mock_client.get.assert_called_once()
        called_url = mock_client.get.call_args.kwargs["url"]
        assert called_url.endswith("/imagegenerator/task/task-123")


def test_litellm_image_generation_seedream_end_to_end():
    mock_response_body = {
        "model": "Doubao-Seedream-5.0-Pro",
        "created": 1788433244,
        "data": [{"url": "https://example.volces.com/seedream.png"}],
    }

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
    ) as mock_post:
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_body
        mock_http_response.status_code = 200
        mock_http_response.headers = {}
        mock_http_response.text = ""
        mock_http_response.request = MagicMock()
        mock_http_response.request.headers = {"Authorization": "Bearer sk-test-key"}
        mock_http_response.request.url = f"{DEFAULT_PROJECT_BASE}/imagegenerator"
        mock_post.return_value = mock_http_response

        response = litellm.image_generation(
            model="sophnet/Doubao-Seedream-5.0-Pro",
            prompt="a red apple",
            api_key="sk-test-key",
            size="2K",
        )

        assert response.data[0].url == "https://example.volces.com/seedream.png"
        called_url = mock_post.call_args.kwargs.get("url") or mock_post.call_args[
            1
        ].get("url")
        assert called_url.endswith("/imagegenerator")
        body = mock_post.call_args.kwargs.get("json") or {}
        assert body["model"] == "Doubao-Seedream-5.0-Pro"
        assert body["prompt"] == "a red apple"
        assert body["size"] == "2K"
        assert "input" not in body


def test_litellm_image_generation_nano_banana_end_to_end():
    create_body = {"output": {"taskId": "task-abc", "taskStatus": "PENDING"}}
    poll_body = {
        "output": {
            "taskId": "task-abc",
            "taskStatus": "SUCCEEDED",
            "results": [{"url": "https://example.com/nano-banana-pro.png"}],
        }
    }

    with (
        patch(
            "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
        ) as mock_post,
        patch(
            "litellm.llms.custom_httpx.http_handler._get_httpx_client"
        ) as mock_get_client,
    ):
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = create_body
        mock_http_response.status_code = 200
        mock_http_response.headers = {}
        mock_http_response.text = ""
        mock_http_response.request = MagicMock()
        mock_http_response.request.headers = {"Authorization": "Bearer sk-test-key"}
        mock_http_response.request.url = f"{DEFAULT_PROJECT_BASE}/imagegenerator/task"
        mock_post.return_value = mock_http_response

        poll_resp = MagicMock()
        poll_resp.json.return_value = poll_body
        poll_resp.status_code = 200
        poll_resp.headers = {}
        poll_resp.text = ""
        mock_client = MagicMock()
        mock_client.get.return_value = poll_resp
        mock_get_client.return_value = mock_client

        response = litellm.image_generation(
            model="sophnet/Nano-Banana-Pro",
            prompt="a red apple",
            api_key="sk-test-key",
        )

        assert response.data[0].url == "https://example.com/nano-banana-pro.png"
        body = mock_post.call_args.kwargs.get("json") or {}
        assert body["model"] == "Nano-Banana-Pro"
        assert body["input"]["prompt"] == "a red apple"
        assert body["parameters"]["watermark"] is False
