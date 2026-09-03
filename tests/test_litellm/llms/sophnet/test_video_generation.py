import os
import sys
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.sophnet.videos.transformation import (
    DEFAULT_PROJECT_BASE,
    SophnetVideoConfig,
    _map_duration,
    _map_size,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.utils import ProviderConfigManager, get_llm_provider


@pytest.mark.parametrize(
    "model_string, expected_model",
    [
        ("sophnet/Wan3.0-Video-Prime", "Wan3.0-Video-Prime"),
        ("sophnet/Hailuo-2.3-T2V", "Hailuo-2.3-T2V"),
    ],
)
def test_get_llm_provider_returns_sophnet(model_string: str, expected_model: str):
    model, provider, _, _ = get_llm_provider(model_string)
    assert provider == "sophnet"
    assert model == expected_model


@pytest.mark.parametrize(
    "model_string",
    ["sophnet/Wan3.0-Video-Prime", "sophnet/Hailuo-2.3-T2V"],
)
def test_get_model_info_mode_is_video_generation(model_string: str):
    prev_env = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    prev_model_cost = litellm.model_cost
    try:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        info = litellm.get_model_info(model=model_string, custom_llm_provider="sophnet")
        assert info["mode"] == "video_generation"
    finally:
        if prev_env is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = prev_env
        litellm.model_cost = prev_model_cost


def test_provider_config_manager_returns_sophnet_video_config():
    config = ProviderConfigManager.get_provider_video_config(
        model="Wan3.0-Video-Prime",
        provider=LlmProviders.SOPHNET,
    )
    assert isinstance(config, SophnetVideoConfig)


@pytest.mark.parametrize(
    "model, seconds, expected",
    [
        ("Wan3.0-Video-Prime", None, 5),
        ("Wan3.0-Video-Prime", "5", 5),
        ("Hailuo-2.3-T2V", None, 6),
        ("Hailuo-2.3-T2V", "5", 6),
        ("Hailuo-2.3-T2V", "6", 6),
        ("Hailuo-2.3-T2V", "8", 10),
        ("Hailuo-2.3-T2V", "10", 10),
        ("Hailuo-2.3-T2V", "15", 10),
    ],
)
def test_map_duration(model: str, seconds, expected: int):
    assert _map_duration(model, seconds) == expected


def test_map_size_common_resolutions():
    assert _map_size("1920x1080") == {"resolution": "1080P", "ratio": "16:9"}
    assert _map_size("1080x1920") == {"resolution": "1080P", "ratio": "9:16"}
    assert _map_size("1280x720") == {"resolution": "720P", "ratio": "16:9"}
    assert _map_size(None) == {}


class TestSophnetVideoConfig:
    def setup_method(self):
        self.cfg = SophnetVideoConfig()
        self.mock_logging_obj = Mock()

    def test_create_request_wan_defaults(self):
        data, files, url = self.cfg.transform_video_create_request(
            model="Wan3.0-Video-Prime",
            prompt="a red apple rotating on a table",
            api_base=DEFAULT_PROJECT_BASE,
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url == f"{DEFAULT_PROJECT_BASE}/videogenerator/generate"
        assert files == []
        assert data["model"] == "Wan3.0-Video-Prime"
        assert data["content"] == [
            {"type": "text", "text": "a red apple rotating on a table"}
        ]
        assert data["parameters"]["duration"] == 5
        assert data["parameters"]["watermark"] is False

    def test_create_request_hailuo_maps_invalid_duration(self):
        mapped = self.cfg.map_openai_params(
            video_create_optional_params={"seconds": "5", "size": "1920x1080"},
            model="Hailuo-2.3-T2V",
            drop_params=False,
        )
        data, _, url = self.cfg.transform_video_create_request(
            model="Hailuo-2.3-T2V",
            prompt="a cat walking",
            api_base=DEFAULT_PROJECT_BASE,
            video_create_optional_request_params=mapped,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert url.endswith("/videogenerator/generate")
        assert data["parameters"]["duration"] == 6
        assert data["parameters"]["resolution"] == "1080P"
        assert data["parameters"]["ratio"] == "16:9"

    def test_create_request_adds_image_reference(self):
        data, _, _ = self.cfg.transform_video_create_request(
            model="Wan3.0-Video-Prime",
            prompt="animate this",
            api_base=DEFAULT_PROJECT_BASE,
            video_create_optional_request_params={
                "input_reference": "https://example.com/ref.png",
                "duration": 5,
            },
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

        assert data["content"][1] == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/ref.png"},
        }

    def test_create_request_rewrites_chat_api_base(self):
        _, _, url = self.cfg.transform_video_create_request(
            model="Wan3.0-Video-Prime",
            prompt="ok",
            api_base="https://www.sophnet.com/api/open-apis/v1",
            video_create_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{DEFAULT_PROJECT_BASE}/videogenerator/generate"

    def test_create_response_encodes_video_id(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {
            "status": 0,
            "result": {"task_id": "task-abc-123"},
        }

        result = self.cfg.transform_video_create_response(
            model="Wan3.0-Video-Prime",
            raw_response=mock_response,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="sophnet",
            request_data={"model": "Wan3.0-Video-Prime", "parameters": {"duration": 5}},
        )

        assert isinstance(result, VideoObject)
        assert result.status == "queued"
        assert result.model == "Wan3.0-Video-Prime"
        assert result.seconds == "5"
        assert result.id.startswith("video_")
        assert result.id == encode_video_id_with_provider(
            "task-abc-123", "sophnet", "Wan3.0-Video-Prime"
        )

    def test_status_retrieve_url_strips_encoded_id(self):
        video_id = encode_video_id_with_provider(
            "task-abc-123", "sophnet", "Wan3.0-Video-Prime"
        )
        url, params = self.cfg.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base=DEFAULT_PROJECT_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        assert url == f"{DEFAULT_PROJECT_BASE}/videogenerator/generate/task-abc-123"
        assert params == {}

    def test_status_retrieve_preserves_encoded_video_id(self):
        video_id = encode_video_id_with_provider(
            "task-abc-123", "sophnet", "Hailuo-2.3-T2V"
        )
        self.cfg.transform_video_status_retrieve_request(
            video_id=video_id,
            api_base=DEFAULT_PROJECT_BASE,
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )
        mock_running = Mock(spec=httpx.Response)
        mock_running.status_code = 200
        mock_running.headers = {}
        mock_running.json.return_value = {
            "status": 0,
            "result": {"task_id": "task-abc-123", "status": "running"},
        }
        running = self.cfg.transform_video_status_retrieve_response(
            raw_response=mock_running,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="sophnet",
        )
        assert running.status == "in_progress"
        assert running.id == video_id

    def test_status_retrieve_maps_running_and_succeeded(self):
        mock_running = Mock(spec=httpx.Response)
        mock_running.status_code = 200
        mock_running.headers = {}
        mock_running.json.return_value = {
            "status": 0,
            "result": {"task_id": "task-abc-123", "status": "running"},
        }
        running = self.cfg.transform_video_status_retrieve_response(
            raw_response=mock_running,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="sophnet",
        )
        assert running.status == "in_progress"

        mock_done = Mock(spec=httpx.Response)
        mock_done.status_code = 200
        mock_done.headers = {}
        mock_done.json.return_value = {
            "status": 0,
            "result": {
                "task_id": "task-abc-123",
                "status": "succeeded",
                "content": {"video_url": "https://example.com/out.mp4"},
                "usage": {"duration": 6, "size": "1920*1080"},
            },
        }
        done = self.cfg.transform_video_status_retrieve_response(
            raw_response=mock_done,
            logging_obj=self.mock_logging_obj,
            custom_llm_provider="sophnet",
        )
        assert done.status == "completed"
        assert done.seconds == "6"
        assert done.size == "1920x1080"
        assert done.usage == {"duration_seconds": 6.0}

    def test_content_extracts_url_and_rejects_in_progress(self):
        url = self.cfg._extract_video_url(
            {
                "result": {
                    "status": "succeeded",
                    "content": {"video_url": "https://example.com/out.mp4"},
                }
            }
        )
        assert url == "https://example.com/out.mp4"

        with pytest.raises(ValueError, match="still processing"):
            self.cfg._extract_video_url(
                {"result": {"status": "running", "content": {}}}
            )

    def test_remix_and_list_are_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.cfg.transform_video_remix_request(
                video_id="x",
                prompt="y",
                api_base=DEFAULT_PROJECT_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )
        with pytest.raises(NotImplementedError):
            self.cfg.transform_video_list_request(
                api_base=DEFAULT_PROJECT_BASE,
                litellm_params=GenericLiteLLMParams(),
                headers={},
            )

    def test_create_error_status_raises(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.headers = {}
        mock_response.text = "invalid duration"
        mock_response.json.return_value = {
            "status": 400,
            "message": "invalid duration 5, must be 6 or 10",
        }
        with pytest.raises(Exception, match="invalid duration"):
            self.cfg.transform_video_create_response(
                model="Hailuo-2.3-T2V",
                raw_response=mock_response,
                logging_obj=self.mock_logging_obj,
                custom_llm_provider="sophnet",
            )


def test_litellm_video_generation_end_to_end():
    mock_response_body = {
        "status": 0,
        "result": {"task_id": "task-live-1"},
    }

    with patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client"
    ) as mock_get_client:
        mock_http_response = MagicMock()
        mock_http_response.json.return_value = mock_response_body
        mock_http_response.status_code = 200
        mock_http_response.headers = {}
        mock_http_response.text = ""
        mock_http_response.request = MagicMock()
        mock_http_response.request.headers = {"Authorization": "Bearer sk-test-key"}
        mock_http_response.request.url = (
            f"{DEFAULT_PROJECT_BASE}/videogenerator/generate"
        )
        mock_client = MagicMock()
        mock_client.post.return_value = mock_http_response
        mock_get_client.return_value = mock_client

        response = litellm.video_generation(
            model="sophnet/Hailuo-2.3-T2V",
            prompt="a red apple rotating",
            api_key="sk-test-key",
            seconds="5",
        )

        assert response.status == "queued"
        assert response.id == encode_video_id_with_provider(
            "task-live-1", "sophnet", "Hailuo-2.3-T2V"
        )
        body = mock_client.post.call_args.kwargs.get("json") or {}
        assert body["model"] == "Hailuo-2.3-T2V"
        assert body["parameters"]["duration"] == 6
        called_url = mock_client.post.call_args.kwargs.get("url")
        assert called_url.endswith("/videogenerator/generate")
