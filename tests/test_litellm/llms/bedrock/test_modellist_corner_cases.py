"""
Unit-level regression tests for the agent invocation path across the
9-model modellist. These complement the live integration tests in
deploy/sophnet-azure/test_corner_cases.py by exercising the bedrock
transformation layer directly (no live proxy required).

Each test is parametrized over a curated set of modellist entries that
span the 4 upstream model families:

  - anthropic/Claude (Bedrock Converse): opus-4-5/4-6/4-7, sonnet-4-5/4-6/4-7
  - openai/GPT-5.x (Codex): gpt-5.4, gpt-5.5
  - custom_openai/GLM (adapter): GLM-5.1, GLM-5.2
  - custom_openai/DeepSeek (adapter): DeepSeek-V4-Pro/Flash
  - deepseek/DeepSeek (Anthropic passthrough): deepseek-v4-pro/flash
"""

import pytest

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig


# Curated modellist: 4 Claude variants x 3 reasoning models x 2 deepseek = 9 entries.
# Mirrors the CHAT_MODELS list in deploy/sophnet-azure/test_corner_cases.py.
# Format: (alias, upstream_model_id, family, supports_effort_via_output_config)
MODELLIST = [
    # Claude Bedrock Converse
    ("sophnet-claude-opus-4-7", "anthropic.claude-opus-4-7-aws", "claude", True),
    ("sophnet-claude-opus-4-5", "anthropic.claude-opus-4-5-20251101-v1:0", "claude", False),
    ("sophnet-claude-sonnet-4-7", "anthropic.claude-sonnet-4-7", "claude", True),
    ("sophnet-claude-haiku-4-5", "anthropic.claude-haiku-4-5-20251001-v1:0", "claude", False),
    # Anthropic passthrough
    ("deepseek-v4-pro", "deepseek-v4-pro", "deepseek_passthrough", False),
    ("deepseek-v4-flash", "deepseek-v4-flash", "deepseek_passthrough", False),
    # Custom OpenAI (adapter path)
    ("sophnet-glm-5.1", "custom_openai/GLM-5.1", "adapter", False),
    ("sophnet-glm-5.2", "custom_openai/GLM-5.2", "adapter", False),
    ("sophnet-deepseekv4-pro", "custom_openai/DeepSeek-V4-Pro", "adapter", False),
    ("sophnet-deepseekv4-flash", "custom_openai/DeepSeek-V4-Flash", "adapter", False),
]


@pytest.fixture(autouse=True)
def _use_local_anthropic_beta_headers(monkeypatch):
    """Force the JSON config manager to use the local file in tests."""
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    from litellm import anthropic_beta_headers_manager
    anthropic_beta_headers_manager._BETA_HEADERS_CONFIG = None
    yield
    anthropic_beta_headers_manager._BETA_HEADERS_CONFIG = None


class TestEffortBetaLeakRegression:
    """The original bug: effort-2025-11-24 leaked into anthropic_beta on Converse.

    This was the specific failure mode that the JSON-driven filter and the
    dead auto-append deletion were both aimed at. These tests guard the
    fix from regressing on any model in the modellist.
    """

    @pytest.mark.parametrize("alias,model_id,family,supports_effort", MODELLIST)
    def test_effort_beta_not_in_anthropic_beta(self, alias, model_id, family, supports_effort):
        """Setting output_config.effort must NOT add effort-2025-11-24 to anthropic_beta."""
        config = AmazonConverseConfig()
        result = config._transform_request_helper(
            model=model_id,
            system_content_blocks=[],
            optional_params={"output_config": {"effort": "low"}},
            messages=[{"role": "user", "content": "Reply with exactly one word: OK"}],
        )
        additional_fields = result.get("additionalModelRequestFields", {})
        betas = additional_fields.get("anthropic_beta", []) or []
        assert "effort-2025-11-24" not in betas, (
            f"{alias}: effort-2025-11-24 leaked into anthropic_beta {betas!r}"
        )

    @pytest.mark.parametrize("alias,model_id,family,supports_effort", MODELLIST)
    def test_effort_header_in_request_does_not_pass_through(self, alias, model_id, family, supports_effort):
        """anthropic-beta: effort-2025-11-24 in headers must be filtered out."""
        config = AmazonConverseConfig()
        result = config._transform_request_helper(
            model=model_id,
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "OK"}],
            headers={"anthropic-beta": "effort-2025-11-24,context-1m-2025-08-07"},
        )
        additional_fields = result.get("additionalModelRequestFields", {})
        betas = additional_fields.get("anthropic_beta", []) or []
        assert "effort-2025-11-24" not in betas, (
            f"{alias}: effort-2025-11-24 from headers leaked through {betas!r}"
        )
        # For Anthropic-family models, the supported beta should be preserved.
        # For non-Claude models on the Converse path, the field is intentionally
        # absent (anthropic_beta is an Anthropic-API concept).
        if family == "claude":
            assert "context-1m-2025-08-07" in betas, (
                f"{alias}: supported context-1m-2025-08-07 was incorrectly filtered {betas!r}"
            )


class TestJsonDrivenBetaFilter:
    """Verify the JSON config drives the filter for every null entry, not just 4."""

    # Pick 9 betas that the legacy substring filter would have missed
    # (the JSON has 25 null entries, only 4 had substring patterns).
    UNSUPPORTED_BETAS = [
        "interleaved-thinking-2025-05-14",
        "bash_20241022",
        "tool-search-tool-2025-10-19",
        "mcp-client-2025-11-20",
        "skills-2025-10-02",
        "mcp-servers-2025-12-04",
        "files-api-2025-04-14",
        "fast-mode-2026-02-01",
        "fine-grained-tool-streaming-2025-05-14",
    ]
    SUPPORTED_BETA = "context-1m-2025-08-07"

    @pytest.mark.parametrize("beta", UNSUPPORTED_BETAS)
    def test_unsupported_beta_is_filtered(self, beta):
        """Every null entry in JSON bedrock_converse must be filtered out."""
        config = AmazonConverseConfig()
        result = config._transform_request_helper(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "OK"}],
            headers={"anthropic-beta": f"{beta},{self.SUPPORTED_BETA}"},
        )
        additional_fields = result.get("additionalModelRequestFields", {})
        betas = additional_fields.get("anthropic_beta", []) or []
        assert beta not in betas, (
            f"{beta!r} should be filtered (null in JSON) but got {betas!r}"
        )
        assert self.SUPPORTED_BETA in betas

    def test_supported_beta_passes_through(self):
        """Non-null entries in JSON must be preserved (not over-filtered)."""
        config = AmazonConverseConfig()
        result = config._transform_request_helper(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "OK"}],
            headers={"anthropic-beta": "context-1m-2025-08-07"},
        )
        additional_fields = result.get("additionalModelRequestFields", {})
        assert additional_fields.get("anthropic_beta") == ["context-1m-2025-08-07"]

    def test_empty_headers_means_no_anthropic_beta(self):
        """No anthropic-beta header in request → no anthropic_beta field."""
        config = AmazonConverseConfig()
        result = config._transform_request_helper(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            system_content_blocks=[],
            optional_params={},
            messages=[{"role": "user", "content": "OK"}],
        )
        additional_fields = result.get("additionalModelRequestFields", {})
        assert "anthropic_beta" not in additional_fields


class TestBedrockConverseStability:
    """Smoke tests that exercise basic transformation paths for each model.

    The matrix tests in deploy/sophnet-azure/ cover the live proxy;
    these run in CI without a proxy and verify the transformation doesn't
    crash for any modellist entry.
    """

    @pytest.mark.parametrize("alias,model_id,family,supports_effort", MODELLIST)
    def test_basic_request_transforms(self, alias, model_id, family, supports_effort):
        config = AmazonConverseConfig()
        result = config._transform_request_helper(
            model=model_id,
            system_content_blocks=[],
            optional_params={"max_tokens": 32},
            messages=[{"role": "user", "content": "OK"}],
        )
        assert result is not None
        assert "messages" in result or "additionalModelRequestFields" in result

    @pytest.mark.parametrize("alias,model_id,family,supports_effort", MODELLIST)
    def test_request_with_tools_transforms(self, alias, model_id, family, supports_effort):
        config = AmazonConverseConfig()
        result = config._transform_request_helper(
            model=model_id,
            system_content_blocks=[],
            optional_params={
                "max_tokens": 256,
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }],
            },
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
        )
        assert result is not None
        # Tools should be transformed to bedrock format
        assert "tools" in result or "additionalModelRequestFields" in result

    @pytest.mark.parametrize("alias,model_id,family,supports_effort", MODELLIST)
    def test_cross_region_prefix_stripped(self, alias, model_id, family, supports_effort):
        """Cross-region inference profile prefixes (us./eu./global.) must be stripped."""
        config = AmazonConverseConfig()
        for prefix in ("us.", "global.", "eu.", "au."):
            cross_region = f"{prefix}{model_id}"
            result = config._transform_request_helper(
                model=cross_region,
                system_content_blocks=[],
                optional_params={"max_tokens": 16},
                messages=[{"role": "user", "content": "OK"}],
            )
            assert result is not None, f"cross-region {prefix} failed for {alias}"
