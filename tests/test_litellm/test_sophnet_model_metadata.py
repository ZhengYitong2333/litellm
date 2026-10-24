"""
Metadata for the Sophnet gateway models that LiteLLM routes by bare name.

These entries drive capability flags, so a missing entry silently disables the
sanitizers that keep GLM/DeepSeek requests valid upstream instead of failing.
"""

import json
import os
import sys
from importlib.resources import files

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.utils import (
    get_model_info,
    requires_anthropic_request_sanitize,
    requires_json_keyword_for_json_object,
)


def _load_cost_maps() -> tuple[dict, dict]:
    backup = json.loads(
        files("litellm")
        .joinpath("model_prices_and_context_window_backup.json")
        .read_text(encoding="utf-8")
    )
    root_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "model_prices_and_context_window.json",
    )
    with open(root_path, encoding="utf-8") as f:
        return backup, json.load(f)


DEPLOY_MODELS = [
    "GLM-5.3",
    "GLM-5.3-Flash",
    "DeepSeek-V4-Flash-Vision-Exp",
    "Kimi-K3",
    "anthropic/MiniMax-M3",
]


@pytest.mark.parametrize("model_key", DEPLOY_MODELS)
def test_entry_present_in_both_cost_maps(model_key):
    backup, root = _load_cost_maps()
    assert model_key in backup
    assert model_key in root


@pytest.mark.parametrize(
    ("model", "expected_key"),
    [
        ("custom_openai/GLM-5.3", "GLM-5.3"),
        ("custom_openai/GLM-5.3-Flash", "GLM-5.3-Flash"),
        (
            "custom_openai/DeepSeek-V4-Flash-Vision-Exp",
            "DeepSeek-V4-Flash-Vision-Exp",
        ),
        ("custom_openai/Kimi-K3", "Kimi-K3"),
        ("anthropic/MiniMax-M3", "anthropic/MiniMax-M3"),
    ],
)
def test_get_model_info_resolves_deploy_model(model, expected_key):
    assert get_model_info(model=model)["key"] == expected_key


@pytest.mark.parametrize(
    "model", ["custom_openai/GLM-5.3", "custom_openai/GLM-5.3-Flash"]
)
def test_glm_53_flags_reach_the_sanitizers(model):
    assert requires_anthropic_request_sanitize(model=model) is True
    assert requires_json_keyword_for_json_object(model=model) is True


@pytest.mark.parametrize(
    "model",
    [
        "custom_openai/GLM-5.3",
        "custom_openai/DeepSeek-V4-Flash-Vision-Exp",
    ],
)
def test_optionally_json_models_require_json_keyword(model):
    assert requires_json_keyword_for_json_object(model=model) is True


def test_flags_are_set_only_where_the_gateway_needs_them():
    backup, _ = _load_cost_maps()
    assert backup["Kimi-K3"].get("requires_anthropic_request_sanitize") is None
    assert backup["Kimi-K3"].get("requires_json_keyword_for_json_object") is None
    assert (
        backup["anthropic/MiniMax-M3"].get("requires_anthropic_request_sanitize")
        is None
    )
