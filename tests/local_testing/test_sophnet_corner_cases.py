import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "deploy/sophnet-azure/test_corner_cases.py"
    )
    spec = importlib.util.spec_from_file_location("sophnet_corner_cases_unit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        {"status": 400, "error": "Invalid model name passed"},
        {"status": 401, "error": "Invalid credential"},
        {"status": 400, "error": "Input should be 'web_search'"},
    ],
)
def test_web_search_does_not_hide_client_errors(runner, strict, error):
    with (
        patch.object(runner, "STRICT_UPSTREAM", strict),
        patch.object(runner, "post", return_value={"ok": False, **error}),
    ):
        result = runner.case_messages_web_search_tool()
    assert result["ok"] is False
    assert not result.get("skip")


@pytest.mark.parametrize("strict", [False, True])
def test_web_search_rate_limit_respects_strict_mode(runner, strict):
    with (
        patch.object(runner, "STRICT_UPSTREAM", strict),
        patch.object(
            runner,
            "post",
            return_value={"ok": False, "status": 429, "error": "Too many requests"},
        ),
    ):
        result = runner.case_messages_web_search_tool()
    assert bool(result.get("skip")) is (not strict)
    assert result["ok"] is (not strict)


@pytest.mark.parametrize(
    "content,expected_ok", [(None, False), ("  ", False), ("OK", True)]
)
def test_chat_basic_requires_final_answer(runner, content, expected_ok):
    response = {
        "ok": True,
        "status": 200,
        "body": {
            "choices": [
                {
                    "message": {"content": content, "reasoning_content": "Thinking"},
                    "finish_reason": "length" if not expected_ok else "stop",
                }
            ]
        },
    }
    with patch.object(runner, "post", return_value=response):
        result = runner.case_chat_basic(runner.GLM)
    assert result["ok"] is expected_ok
    if not expected_ok:
        assert "final answer" in result["error"]
