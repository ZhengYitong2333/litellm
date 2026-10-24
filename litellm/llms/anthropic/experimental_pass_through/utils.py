import os
from typing import Any, Mapping, Optional

import litellm
from litellm.types.llms.anthropic import UsageDelta
from litellm.types.utils import ModelInfo


def _usage_field(usage: Any, name: str) -> Any:
    return usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)


def _usage_tokens(usage: Any, name: str) -> int:
    value = _usage_field(usage, name)
    return value if isinstance(value, int) and value >= 0 else 0


def translate_openai_usage_to_anthropic(usage: Any) -> UsageDelta:
    details = _usage_field(usage, "prompt_tokens_details") or _usage_field(
        usage, "input_tokens_details"
    )
    cached = (
        _usage_tokens(details, "cached_tokens")
        or _usage_tokens(usage, "_cache_read_input_tokens")
        or _usage_tokens(usage, "cache_read_input_tokens")
    )
    created = (
        _usage_tokens(details, "cache_creation_tokens")
        or _usage_tokens(usage, "_cache_creation_input_tokens")
        or _usage_tokens(usage, "cache_creation_input_tokens")
    )
    total_input = _usage_tokens(usage, "prompt_tokens") or _usage_tokens(
        usage, "input_tokens"
    )
    return UsageDelta(
        input_tokens=max(0, total_input - cached - created),
        output_tokens=_usage_tokens(usage, "completion_tokens")
        or _usage_tokens(usage, "output_tokens"),
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=created,
    )


def uses_azure_openai_api_base(api_base: Optional[str]) -> bool:
    return isinstance(api_base, str) and (
        ".azure.com" in api_base or ".services.ai.azure.com" in api_base
    )


def is_reasoning_auto_summary_enabled() -> bool:
    """Check whether the default 'summary: detailed' injection is enabled (opt-in)."""
    return (
        litellm.reasoning_auto_summary
        or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"
    )


def normalize_reasoning_effort_value(
    effort: str,
    model: str,
    custom_llm_provider: Optional[str] = None,
) -> str:
    """
    Normalize a reasoning effort value based on model capabilities.

    Degradation chains:
    - "max"     → max / xhigh / high
    - "xhigh"   → xhigh / high
    - "minimal" → minimal / low
    - other values pass through unchanged
    """
    if effort not in ("max", "xhigh", "minimal"):
        return effort

    from litellm.utils import get_model_info

    model_info: Optional[ModelInfo] = None
    try:
        model_info = get_model_info(
            model=model, custom_llm_provider=custom_llm_provider
        )
    except Exception:
        model_info = None

    if effort == "max":
        if model_info and model_info.get("supports_max_reasoning_effort"):
            return "max"
        if model_info and model_info.get("supports_xhigh_reasoning_effort"):
            return "xhigh"
        return "high"
    elif effort == "xhigh":
        if model_info and model_info.get("supports_xhigh_reasoning_effort"):
            return "xhigh"
        return "high"
    elif effort == "minimal":
        if model_info and model_info.get("supports_minimal_reasoning_effort"):
            return "minimal"
        return "low"
    return "medium"
