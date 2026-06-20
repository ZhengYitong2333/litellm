"""
This file contains common utils for anthropic calls.
"""

import base64
import binascii
import copy
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_file_ids_from_messages,
)
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo, BaseTokenCounter
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.anthropic import (
    ANTHROPIC_HOSTED_TOOLS,
    ANTHROPIC_OAUTH_BETA_HEADER,
    ANTHROPIC_OAUTH_TOKEN_PREFIX,
    AllAnthropicToolsValues,
    AnthropicMcpServerTool,
)
from litellm.types.llms.openai import AllMessageValues


def is_anthropic_oauth_key(value: Optional[str]) -> bool:
    """Check if a value contains an Anthropic OAuth token (sk-ant-oat*)."""
    if value is None:
        return False
    # Handle both raw token and "Bearer <token>" format
    if value.startswith("Bearer "):
        value = value[7:]
    return value.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX)


def _merge_beta_headers(existing: Optional[str], new_beta: str) -> str:
    """Merge a new beta value into an existing comma-separated anthropic-beta header."""
    if not existing:
        return new_beta
    betas = {b.strip() for b in existing.split(",") if b.strip()}
    betas.add(new_beta)
    return ",".join(sorted(betas))


def optionally_handle_anthropic_oauth(
    headers: dict, api_key: Optional[str]
) -> tuple[dict, Optional[str]]:
    """
    Handle Anthropic OAuth token detection and header setup.

    If an OAuth token is detected in the Authorization header, extracts it
    and sets the required OAuth headers.

    Args:
        headers: Request headers dict
        api_key: Current API key (may be None)

    Returns:
        Tuple of (updated headers, api_key)
    """
    # Check Authorization header (passthrough / forwarded requests)
    auth_header = headers.get("authorization", "")
    if auth_header and auth_header.startswith(f"Bearer {ANTHROPIC_OAUTH_TOKEN_PREFIX}"):
        api_key = auth_header.replace("Bearer ", "")
        headers.pop("x-api-key", None)
        headers["anthropic-beta"] = _merge_beta_headers(
            headers.get("anthropic-beta"), ANTHROPIC_OAUTH_BETA_HEADER
        )
        headers["anthropic-dangerous-direct-browser-access"] = "true"
        return headers, api_key
    # Check api_key directly (standard chat/completion flow)
    if api_key and api_key.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX):
        headers.pop("x-api-key", None)
        headers["authorization"] = f"Bearer {api_key}"
        headers["anthropic-beta"] = _merge_beta_headers(
            headers.get("anthropic-beta"), ANTHROPIC_OAUTH_BETA_HEADER
        )
        headers["anthropic-dangerous-direct-browser-access"] = "true"
    return headers, api_key


class AnthropicError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message,
        headers: Optional[httpx.Headers] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class AnthropicModelInfo(BaseLLMModelInfo):
    def is_cache_control_set(self, messages: List[AllMessageValues]) -> bool:
        """
        Return if {"cache_control": ..} in message content block

        Used to check if anthropic prompt caching headers need to be set.
        """
        for message in messages:
            if message.get("cache_control", None) is not None:
                return True
            _message_content = message.get("content")
            if _message_content is not None and isinstance(_message_content, list):
                for content in _message_content:
                    if "cache_control" in content:
                        return True

        return False

    def is_file_id_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Return if {"source": {"type": "file", "file_id": ..}} in message content block
        """
        file_ids = get_file_ids_from_messages(messages)
        return len(file_ids) > 0

    def is_mcp_server_used(
        self, mcp_servers: Optional[List[AnthropicMcpServerTool]]
    ) -> bool:
        if mcp_servers is None:
            return False
        if mcp_servers:
            return True
        return False

    def is_computer_tool_used(
        self, tools: Optional[List[AllAnthropicToolsValues]]
    ) -> Optional[str]:
        """Returns the computer tool version if used, e.g. 'computer_20250124' or None"""
        if tools is None:
            return None
        for tool in tools:
            if "type" in tool and tool["type"].startswith("computer_"):
                return tool["type"]
        return None

    def is_web_search_tool_used(
        self, tools: Optional[List[AllAnthropicToolsValues]]
    ) -> bool:
        """Returns True if web_search tool is used"""
        if tools is None:
            return False
        for tool in tools:
            if "type" in tool and tool["type"].startswith(
                ANTHROPIC_HOSTED_TOOLS.WEB_SEARCH.value
            ):
                return True
        return False

    def is_pdf_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Set to true if media passed into messages.

        """
        for message in messages:
            if (
                "content" in message
                and message["content"] is not None
                and isinstance(message["content"], list)
            ):
                for content in message["content"]:
                    if "type" in content and content["type"] != "text":
                        return True
        return False

    def is_tool_search_used(self, tools: Optional[List]) -> bool:
        """
        Check if tool search tools are present in the tools list.
        """
        if not tools:
            return False

        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type in [
                "tool_search_tool_regex_20251119",
                "tool_search_tool_bm25_20251119",
            ]:
                return True
        return False

    def is_programmatic_tool_calling_used(self, tools: Optional[List]) -> bool:
        """
        Check if programmatic tool calling is being used (tools with allowed_callers field).

        Returns True if any tool has allowed_callers containing 'code_execution_20250825'.
        """
        if not tools:
            return False

        for tool in tools:
            # Check top-level allowed_callers
            allowed_callers = tool.get("allowed_callers", None)
            if allowed_callers and isinstance(allowed_callers, list):
                if "code_execution_20250825" in allowed_callers:
                    return True

            # Check function.allowed_callers for OpenAI format tools
            function = tool.get("function", {})
            if isinstance(function, dict):
                function_allowed_callers = function.get("allowed_callers", None)
                if function_allowed_callers and isinstance(
                    function_allowed_callers, list
                ):
                    if "code_execution_20250825" in function_allowed_callers:
                        return True

        return False

    def is_input_examples_used(self, tools: Optional[List]) -> bool:
        """
        Check if input_examples is being used in any tools.

        Returns True if any tool has input_examples field.
        """
        if not tools:
            return False

        for tool in tools:
            # Check top-level input_examples
            input_examples = tool.get("input_examples", None)
            if (
                input_examples
                and isinstance(input_examples, list)
                and len(input_examples) > 0
            ):
                return True

            # Check function.input_examples for OpenAI format tools
            function = tool.get("function", {})
            if isinstance(function, dict):
                function_input_examples = function.get("input_examples", None)
                if (
                    function_input_examples
                    and isinstance(function_input_examples, list)
                    and len(function_input_examples) > 0
                ):
                    return True

        return False

    @staticmethod
    def _is_claude_4_6_model(model: str) -> bool:
        """Check if the model is a Claude 4.6 model (Opus 4.6 or Sonnet 4.6)."""
        model_lower = model.lower()
        return any(
            v in model_lower
            for v in (
                "opus-4-6",
                "opus_4_6",
                "opus-4.6",
                "opus_4.6",
                "sonnet-4-6",
                "sonnet_4_6",
                "sonnet-4.6",
                "sonnet_4.6",
            )
        )

    @staticmethod
    def _is_claude_4_7_model(model: str) -> bool:
        """Check if the model is a Claude 4.7 model (Opus 4.7)."""
        model_lower = model.lower()
        return any(
            v in model_lower
            for v in (
                "opus-4-7",
                "opus_4_7",
                "opus-4.7",
                "opus_4.7",
            )
        )

    @staticmethod
    def _supports_model_capability(model: str, key: str) -> bool:
        """Check a boolean capability ``key`` in the model map.

        Strips bedrock/vertex prefixes so a provider-routed Claude still
        resolves to the Anthropic model-map entry.
        """
        from litellm.utils import _supports_factory

        try:
            if _supports_factory(
                model=model,
                custom_llm_provider="anthropic",
                key=key,
            ):
                return True
        except Exception:
            pass
        candidates = [model]
        for prefix in (
            "bedrock/converse/",
            "bedrock/invoke/",
            "bedrock/",
            "vertex_ai/",
        ):
            if model.startswith(prefix):
                candidates.append(model[len(prefix) :])
        try:
            from litellm.llms.bedrock.common_utils import BedrockModelInfo

            base = BedrockModelInfo.get_base_model(model)
            if base:
                candidates.append(base)
                candidates.append(f"bedrock/{base}")
        except Exception:
            pass
        try:
            for cand in candidates:
                if cand in litellm.model_cost and (
                    litellm.model_cost[cand].get(key) is True
                ):
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _is_adaptive_thinking_model(model: str) -> bool:
        """Claude 4.6+ models use adaptive thinking with ``output_config.effort``.

        Driven by the ``supports_adaptive_thinking`` flag in the model map; the
        4.6/4.7 name checks remain only as a fallback for provider-routed ids
        whose map entries predate the flag.
        """
        if AnthropicModelInfo._supports_model_capability(
            model, "supports_adaptive_thinking"
        ):
            return True
        return AnthropicModelInfo._is_claude_4_6_model(
            model
        ) or AnthropicModelInfo._is_claude_4_7_model(model)

    def is_effort_used(
        self, optional_params: Optional[dict], model: Optional[str] = None
    ) -> bool:
        """
        Check if effort parameter is being used and requires a beta header.

        Returns True if effort-related parameters are present and
        the model requires the effort beta header. Claude 4.6+ models
        use output_config as a stable API feature — no beta header needed.
        """
        if not optional_params:
            return False

        # Claude 4.6+ models use output_config as a stable API feature — no beta header needed
        if model and self._is_adaptive_thinking_model(model):
            return False

        # Check if reasoning_effort is provided for Claude Opus 4.5
        if model and ("opus-4-5" in model.lower() or "opus_4_5" in model.lower()):
            reasoning_effort = optional_params.get("reasoning_effort")
            if reasoning_effort and isinstance(reasoning_effort, str):
                return True

        # Check if output_config is directly provided (for non-4.6 models)
        output_config = optional_params.get("output_config")
        if output_config and isinstance(output_config, dict):
            effort = output_config.get("effort")
            if effort and isinstance(effort, str):
                return True

        return False

    def is_code_execution_tool_used(self, tools: Optional[List]) -> bool:
        """
        Check if code execution tool is being used.

        Returns True if any tool has type "code_execution_20250825".
        """
        if not tools:
            return False

        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type == "code_execution_20250825":
                return True
        return False

    def is_container_with_skills_used(self, optional_params: Optional[dict]) -> bool:
        """
        Check if container with skills is being used.

        Returns True if optional_params contains container with skills.
        """
        if not optional_params:
            return False

        container = optional_params.get("container")
        if container and isinstance(container, dict):
            skills = container.get("skills")
            if skills and isinstance(skills, list) and len(skills) > 0:
                return True
        return False

    def _get_user_anthropic_beta_headers(
        self, anthropic_beta_header: Optional[str]
    ) -> Optional[List[str]]:
        if anthropic_beta_header is None:
            return None
        return anthropic_beta_header.split(",")

    def get_computer_tool_beta_header(self, computer_tool_version: str) -> str:
        """
        Get the appropriate beta header for a given computer tool version.

        Args:
            computer_tool_version: The computer tool version (e.g., 'computer_20250124', 'computer_20241022')

        Returns:
            The corresponding beta header string
        """
        computer_tool_beta_mapping = {
            "computer_20250124": "computer-use-2025-01-24",
            "computer_20241022": "computer-use-2024-10-22",
        }
        return computer_tool_beta_mapping.get(
            computer_tool_version, "computer-use-2024-10-22"  # Default fallback
        )

    def get_anthropic_beta_list(
        self,
        model: str,
        optional_params: Optional[dict] = None,
        computer_tool_used: Optional[str] = None,
        prompt_caching_set: bool = False,
        file_id_used: bool = False,
        mcp_server_used: bool = False,
    ) -> List[str]:
        """
        Get list of common beta headers based on the features that are active.

        Returns:
            List of beta header strings
        """
        from litellm.types.llms.anthropic import ANTHROPIC_EFFORT_BETA_HEADER

        betas = []

        # Detect features
        effort_used = self.is_effort_used(optional_params, model)

        if effort_used:
            betas.append(ANTHROPIC_EFFORT_BETA_HEADER)  # effort-2025-11-24

        if computer_tool_used:
            beta_header = self.get_computer_tool_beta_header(computer_tool_used)
            betas.append(beta_header)

        # Anthropic no longer requires the prompt-caching beta header
        # Prompt caching now works automatically when cache_control is used in messages
        # Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

        if file_id_used:
            betas.append("files-api-2025-04-14")
            betas.append("code-execution-2025-05-22")

        if mcp_server_used:
            betas.append("mcp-client-2025-04-04")

        return list(set(betas))

    def get_anthropic_headers(
        self,
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,
        anthropic_version: Optional[str] = None,
        computer_tool_used: Optional[str] = None,
        prompt_caching_set: bool = False,
        pdf_used: bool = False,
        file_id_used: bool = False,
        mcp_server_used: bool = False,
        web_search_tool_used: bool = False,
        tool_search_used: bool = False,
        programmatic_tool_calling_used: bool = False,
        input_examples_used: bool = False,
        effort_used: bool = False,
        is_vertex_request: bool = False,
        user_anthropic_beta_headers: Optional[List[str]] = None,
        code_execution_tool_used: bool = False,
        container_with_skills_used: bool = False,
    ) -> dict:
        betas = set()
        # Anthropic no longer requires the prompt-caching beta header
        # Prompt caching now works automatically when cache_control is used in messages
        # Reference: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
        if computer_tool_used:
            beta_header = self.get_computer_tool_beta_header(computer_tool_used)
            betas.add(beta_header)
        # if pdf_used:
        #     betas.add("pdfs-2024-09-25")
        if file_id_used:
            betas.add("files-api-2025-04-14")
            betas.add("code-execution-2025-05-22")
        if mcp_server_used:
            betas.add("mcp-client-2025-04-04")
        # Tool search, programmatic tool calling, and input_examples all use the same beta header
        if tool_search_used or programmatic_tool_calling_used or input_examples_used:
            from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_BETA_HEADER

            betas.add(ANTHROPIC_TOOL_SEARCH_BETA_HEADER)

        # Effort parameter uses a separate beta header
        if effort_used:
            from litellm.types.llms.anthropic import ANTHROPIC_EFFORT_BETA_HEADER

            betas.add(ANTHROPIC_EFFORT_BETA_HEADER)

        # Code execution tool uses a separate beta header
        if code_execution_tool_used:
            betas.add("code-execution-2025-08-25")

        # Container with skills uses a separate beta header
        if container_with_skills_used:
            betas.add("skills-2025-10-02")

        _is_oauth = api_key and api_key.startswith(ANTHROPIC_OAUTH_TOKEN_PREFIX)
        headers = {
            "anthropic-version": anthropic_version or "2023-06-01",
            "accept": "application/json",
            "content-type": "application/json",
        }
        if _is_oauth:
            headers["authorization"] = f"Bearer {api_key}"
            headers["anthropic-dangerous-direct-browser-access"] = "true"
            betas.add(ANTHROPIC_OAUTH_BETA_HEADER)
        elif auth_token and not api_key:
            headers["authorization"] = f"Bearer {auth_token}"
        elif api_key:
            headers["x-api-key"] = api_key

        if user_anthropic_beta_headers is not None:
            betas.update(user_anthropic_beta_headers)

        # Don't send any beta headers to Vertex, except web search which is required
        if is_vertex_request is True:
            # Vertex AI requires web search beta header for web search to work
            if web_search_tool_used:
                from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES

                headers["anthropic-beta"] = (
                    ANTHROPIC_BETA_HEADER_VALUES.WEB_SEARCH_2025_03_05.value
                )
        elif len(betas) > 0:
            headers["anthropic-beta"] = ",".join(betas)

        return headers

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        # Check for Anthropic OAuth token in headers
        headers, api_key = optionally_handle_anthropic_oauth(
            headers=headers, api_key=api_key
        )
        api_key = AnthropicModelInfo.get_api_key(api_key)
        # Resolve auth_token from ANTHROPIC_AUTH_TOKEN if api_key is not set
        auth_token: Optional[str] = None
        if api_key is None:
            auth_token = AnthropicModelInfo.get_auth_token()
        if api_key is None and auth_token is None:
            raise litellm.AuthenticationError(
                message="Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params. Please set `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` in your environment vars",
                llm_provider="anthropic",
                model=model,
            )

        tools = optional_params.get("tools")
        prompt_caching_set = self.is_cache_control_set(messages=messages)
        computer_tool_used = self.is_computer_tool_used(tools=tools)
        mcp_server_used = self.is_mcp_server_used(
            mcp_servers=optional_params.get("mcp_servers")
        )
        pdf_used = self.is_pdf_used(messages=messages)
        file_id_used = self.is_file_id_used(messages=messages)
        web_search_tool_used = self.is_web_search_tool_used(tools=tools)
        tool_search_used = self.is_tool_search_used(tools=tools)
        programmatic_tool_calling_used = self.is_programmatic_tool_calling_used(
            tools=tools
        )
        input_examples_used = self.is_input_examples_used(tools=tools)
        effort_used = self.is_effort_used(optional_params=optional_params, model=model)
        code_execution_tool_used = self.is_code_execution_tool_used(tools=tools)
        container_with_skills_used = self.is_container_with_skills_used(
            optional_params=optional_params
        )
        user_anthropic_beta_headers = self._get_user_anthropic_beta_headers(
            anthropic_beta_header=headers.get("anthropic-beta")
        )
        anthropic_headers = self.get_anthropic_headers(
            computer_tool_used=computer_tool_used,
            prompt_caching_set=prompt_caching_set,
            pdf_used=pdf_used,
            api_key=api_key,
            auth_token=auth_token,
            file_id_used=file_id_used,
            web_search_tool_used=web_search_tool_used,
            is_vertex_request=optional_params.get("is_vertex_request", False),
            user_anthropic_beta_headers=user_anthropic_beta_headers,
            mcp_server_used=mcp_server_used,
            tool_search_used=tool_search_used,
            programmatic_tool_calling_used=programmatic_tool_calling_used,
            input_examples_used=input_examples_used,
            effort_used=effort_used,
            code_execution_tool_used=code_execution_tool_used,
            container_with_skills_used=container_with_skills_used,
        )

        headers = {**headers, **anthropic_headers}

        return headers

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        from litellm.secret_managers.main import get_secret_str

        return (
            api_base
            or get_secret_str("ANTHROPIC_API_BASE")
            or get_secret_str("ANTHROPIC_BASE_URL")
            or "https://api.anthropic.com"
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        from litellm.secret_managers.main import get_secret_str

        return api_key or get_secret_str("ANTHROPIC_API_KEY")

    @staticmethod
    def get_auth_token(auth_token: Optional[str] = None) -> Optional[str]:
        """Get auth token from ANTHROPIC_AUTH_TOKEN env var.

        Unlike api_key (which uses X-Api-Key header), auth_token uses
        Authorization: Bearer header, matching the official Anthropic SDK behavior.
        """
        from litellm.secret_managers.main import get_secret_str

        return auth_token or get_secret_str("ANTHROPIC_AUTH_TOKEN")

    @staticmethod
    def get_auth_header(api_key: Optional[str] = None) -> Optional[dict]:
        """Resolve Anthropic credentials and return the appropriate auth header dict.

        Checks ANTHROPIC_API_KEY first (-> x-api-key), then
        ANTHROPIC_AUTH_TOKEN (-> Authorization: Bearer).
        Returns None if neither is available.
        """
        resolved_key = AnthropicModelInfo.get_api_key(api_key)
        if resolved_key is not None:
            if is_anthropic_oauth_key(resolved_key):
                return {"authorization": f"Bearer {resolved_key}"}
            return {"x-api-key": resolved_key}
        auth_token = AnthropicModelInfo.get_auth_token()
        if auth_token is not None:
            return {"authorization": f"Bearer {auth_token}"}
        return None

    @staticmethod
    def get_base_model(model: Optional[str] = None) -> Optional[str]:
        return model.replace("anthropic/", "") if model else None

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        api_base = AnthropicModelInfo.get_api_base(api_base)
        auth_header = AnthropicModelInfo.get_auth_header(api_key)
        if api_base is None or auth_header is None:
            raise ValueError(
                "ANTHROPIC_API_BASE/ANTHROPIC_BASE_URL or ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN is not set. Please set the environment variable, to query Anthropic's `/models` endpoint."
            )
        headers = {"anthropic-version": "2023-06-01"}
        headers.update(auth_header)
        response = litellm.module_level_client.get(
            url=f"{api_base}/v1/models",
            headers=headers,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            raise Exception(
                f"Failed to fetch models from Anthropic. Status code: {response.status_code}, Response: {response.text}"
            )

        models = response.json()["data"]

        litellm_model_names = []
        for model in models:
            stripped_model_name = model["id"]
            litellm_model_name = "anthropic/" + stripped_model_name
            litellm_model_names.append(litellm_model_name)
        return litellm_model_names

    def get_token_counter(self) -> Optional[BaseTokenCounter]:
        """
        Factory method to create an Anthropic token counter.

        Returns:
            AnthropicTokenCounter instance for this provider.
        """
        from litellm.llms.anthropic.count_tokens.token_counter import (
            AnthropicTokenCounter,
        )

        return AnthropicTokenCounter()


def strip_advisor_blocks_from_messages(
    messages: List[Any], replace_with_text: bool = False
) -> List[Any]:
    """
    Remove (or replace) server_tool_use (name='advisor') and advisor_tool_result blocks
    from assistant message content.

    Prevents Anthropic 400 invalid_request_error: if advisor_tool_result blocks
    exist in history but the advisor tool is not in the tools array, the API rejects
    the request. This happens when the user has removed the advisor tool for cost
    control or on a follow-up turn.

    Args:
        messages: Conversation history to process (mutated in-place).
        replace_with_text: When True, replace the advisor exchange with an
            <advisor_feedback> text block so the executor retains the semantic
            context of what the advisor said.  When False (default), strip silently.
    """
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        # Collect advisor server_tool_use ids and their advice text (for replace mode).
        advisor_id_to_text: dict = {}
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "server_tool_use"
                and block.get("name") == "advisor"
            ):
                bid = block.get("id")
                if bid:
                    advisor_id_to_text[bid] = None  # text filled in below

        if not advisor_id_to_text:
            continue

        # If replacing, collect the advisor response text from advisor_tool_result blocks.
        if replace_with_text:
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "advisor_tool_result"
                    and block.get("tool_use_id") in advisor_id_to_text
                ):
                    raw = block.get("content") or ""
                    text = (
                        raw
                        if isinstance(raw, str)
                        else next(
                            (
                                b.get("text", "")
                                for b in raw
                                if isinstance(b, dict) and b.get("type") == "text"
                            ),
                            "",
                        )
                    )
                    advisor_id_to_text[block["tool_use_id"]] = text

        new_content = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            is_advisor_use = (
                block.get("type") == "server_tool_use"
                and block.get("name") == "advisor"
                and block.get("id") in advisor_id_to_text
            )
            is_advisor_result = (
                block.get("type") == "advisor_tool_result"
                and block.get("tool_use_id") in advisor_id_to_text
            )
            if is_advisor_use:
                if replace_with_text:
                    advice = advisor_id_to_text.get(block.get("id")) or ""
                    if advice:
                        new_content.append(
                            {
                                "type": "text",
                                "text": f"<advisor_feedback>\n{advice}\n</advisor_feedback>",
                            }
                        )
                # else: drop silently
            elif is_advisor_result:
                pass  # always drop — replaced above (or stripped)
            else:
                new_content.append(block)

        message["content"] = new_content
    return messages


def is_anthropic_invalid_redacted_thinking_data_error(error_text: str) -> bool:
    """Detect 400 when redacted_thinking ``data`` is rejected by the upstream API."""
    if not error_text:
        return False
    lower = error_text.lower()
    return "invalid" in lower and "redacted_thinking" in lower and "data" in lower


def is_anthropic_invalid_thinking_signature_error(error_text: str) -> bool:
    """
    Detect Anthropic 400 when encrypted thinking signatures in history do not match
    the current deployment (e.g. user rotated API key or switched model endpoint).

    Example API message:
    messages.N.content.M: Invalid `signature` in `thinking` block
    """
    if not error_text:
        return False
    lower = error_text.lower()
    return (
        "invalid" in lower
        and "signature" in lower
        and "thinking" in lower
        and "block" in lower
    )


def is_anthropic_compatible_gateway_opaque_error(error_text: str) -> bool:
    """
    Detect opaque 400 bodies from Anthropic-compatible gateways (e.g. Sophnet) that
    embed Go ``<nil>`` errors instead of a concrete validation message.

    Example (from LiteLLM proxy logs):
    {"error":{"type":"<nil>","message":" (request id: ...) (request id: ...)"}}
    """
    if not error_text:
        return False
    lower = error_text.lower()
    return "<nil>" in lower or "\\u003cnil\\u003e" in lower


def strip_redacted_thinking_blocks_from_anthropic_messages(
    messages: List[Any],
) -> List[Any]:
    """Remove all ``redacted_thinking`` blocks from message content lists."""
    out: List[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        filtered = [
            b
            for b in content
            if not (isinstance(b, dict) and b.get("type") == "redacted_thinking")
        ]
        if not filtered:
            continue
        if len(filtered) == len(content):
            out.append(m)
        else:
            out.append({**m, "content": filtered})
    return out


def strip_redacted_thinking_blocks_from_anthropic_messages_request_dict(
    data: Dict[str, Any],
) -> None:
    """Mutate an Anthropic Messages request dict: drop ``redacted_thinking`` blocks."""
    msgs = data.get("messages")
    if isinstance(msgs, list):
        data["messages"] = strip_redacted_thinking_blocks_from_anthropic_messages(msgs)


def _is_official_anthropic_api_base(api_base: Optional[str]) -> bool:
    if not api_base:
        return True
    lower = api_base.lower()
    return "anthropic.com" in lower


def infer_gateway_api_base_for_tool_sanitize(
    *,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve api_base for gateway tool sanitization when only the proxy alias is known.

    Claude Code calls /v1/messages with ``model=sophnet-gpt-5.5`` etc. without passing
    api_base; without this hint, sanitize would no-op (treat as official Anthropic).
    """
    if api_base:
        return api_base
    if isinstance(model, str):
        model_lower = model.lower()
        if "sophnet" in model_lower or "minimax" in model_lower:
            return "https://www.sophnet.com/api/open-apis/anthropic"
    return api_base


def _is_deepseek_api_base(api_base: Optional[str]) -> bool:
    """DeepSeek Anthropic Messages API requires thinking blocks in history round-trip."""
    if not api_base:
        return False
    return "deepseek.com" in api_base.lower()


def _anthropic_tool_effective_name(tool: dict) -> Optional[str]:
    """Return a non-empty tool name from Anthropic-native or OpenAI-wrapped tools."""
    name = tool.get("name")
    if isinstance(name, str):
        stripped = name.strip()
        if stripped:
            return stripped
    function = tool.get("function")
    if isinstance(function, dict):
        fn_name = function.get("name")
        if isinstance(fn_name, str):
            stripped = fn_name.strip()
            if stripped:
                return stripped
    return None


def _anthropic_tool_type_skips_name_and_schema_validation(
    tool_type: Optional[str],
) -> bool:
    if not isinstance(tool_type, str):
        return False
    return tool_type.startswith("web_search")


def _normalize_anthropic_tool_input_schema_for_gateway(tool: dict) -> dict:
    """
    Sophnet/MiniMax reject tools with empty parameters (error 2013).

    Coerce missing/empty schemas to a minimal object schema, and inject a
    benign placeholder property so zero-argument tools still satisfy the
    upstream "parameters must not be empty" validation.
    """
    tool_type = tool.get("type")
    if _anthropic_tool_type_skips_name_and_schema_validation(
        tool_type if isinstance(tool_type, str) else None
    ):
        return tool

    schema = tool.get("input_schema")
    if schema is None and isinstance(tool.get("function"), dict):
        schema = tool["function"].get("parameters")

    if not isinstance(schema, dict) or not schema:
        schema = {"type": "object", "properties": {}}
    else:
        schema = dict(schema)
        if schema.get("type") != "object":
            schema["type"] = "object"
        if "properties" not in schema:
            schema["properties"] = {}

    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        schema["properties"] = {
            "_unused": {
                "type": "string",
                "description": (
                    "Unused placeholder. This tool takes no arguments; "
                    "do not set this field."
                ),
            }
        }

    if "input_schema" in tool:
        return {**tool, "input_schema": schema}
    if isinstance(tool.get("function"), dict):
        return {
            **tool,
            "function": {**tool["function"], "parameters": schema},
        }
    return {**tool, "input_schema": schema}


def _should_drop_anthropic_tool_for_gateway(
    tool: Any,
    *,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> bool:
    """Drop tool definitions that third-party Anthropic gateways reject."""
    if _is_official_anthropic_api_base(api_base):
        return False
    if not isinstance(tool, dict):
        return True

    tool_type = tool.get("type")
    if isinstance(tool_type, str):
        if tool_type in {"shell", "namespace"}:
            return True

        if tool_type == "computer_use_preview" or tool_type.startswith("computer_"):
            return True

        # Anthropic-native server tools (``web_search_*``, etc.) have no
        # ``input_schema``; third-party gateways like Sophnet reject them with
        # "function name or parameters is empty (2013)". The gateway can't
        # execute the underlying server tool anyway, so drop instead of
        # forwarding a non-functional definition.
        if _anthropic_tool_type_skips_name_and_schema_validation(tool_type):
            return True

    if _anthropic_tool_effective_name(tool) is None:
        return True

    return False


def sanitize_anthropic_tools_for_upstream(
    tools: Optional[List[Any]],
    *,
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[List[Any]]:
    """Filter tool lists before forwarding to Anthropic-compatible gateways."""
    if not isinstance(tools, list):
        return tools

    filtered: List[Any] = []
    for tool in tools:
        if _should_drop_anthropic_tool_for_gateway(
            tool, api_base=api_base, model=model
        ):
            continue
        if isinstance(tool, dict) and not _is_official_anthropic_api_base(api_base):
            tool = _normalize_anthropic_tool_input_schema_for_gateway(tool)
        filtered.append(tool)
    return filtered


def normalize_anthropic_tool_use_blocks_in_messages(
    messages: List[Any],
) -> List[Any]:
    """
    Coerce assistant ``tool_use`` blocks for third-party Anthropic gateways.

    Sophnet/MiniMax reject empty function names or missing inputs (error 2013).
    Claude Code history can replay ``tool_use`` blocks with ``name: ""``.
    """
    out: List[Any] = []
    unnamed_index = 0
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue

        new_content: List[Any] = []
        message_changed = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                new_content.append(block)
                continue

            block_copy = dict(block)
            block_changed = False
            name = block_copy.get("name")
            if not isinstance(name, str) or not name.strip():
                block_copy["name"] = f"litellm_unnamed_tool_{unnamed_index}"
                unnamed_index += 1
                block_changed = True
            if block_copy.get("input") is None:
                block_copy["input"] = {}
                block_changed = True

            if block_changed:
                new_content.append(block_copy)
                message_changed = True
            else:
                new_content.append(block)

        if message_changed:
            out.append({**message, "content": new_content})
        else:
            out.append(message)
    return out


def sanitize_anthropic_messages_for_upstream(
    messages: List[Any],
    api_base: Optional[str] = None,
    model: Optional[str] = None,
) -> List[Any]:
    """
    Sanitize Anthropic Messages history before forwarding to an upstream API.

    - Drops empty text blocks and invalid ``redacted_thinking`` blocks
    - Drops all ``redacted_thinking`` for non-official Anthropic-compatible gateways
      (e.g. Sophnet) that cannot round-trip encrypted thinking payloads
    - Drops ``thinking`` history blocks for third-party gateways (e.g. Sophnet) where
      signatures are not portable; preserves them for DeepSeek's Anthropic Messages API
    - Inserts placeholder ``tool_result`` blocks for orphaned ``tool_use`` turns so
      Anthropic-compatible endpoints do not 400 on Claude Code history that drops them
    - Normalizes empty ``tool_use`` names/inputs for gateways such as Sophnet/MiniMax
    """
    effective_api_base = api_base or infer_gateway_api_base_for_tool_sanitize(
        model=model, api_base=api_base
    )
    messages = strip_empty_text_blocks_from_anthropic_messages(messages)
    messages = insert_missing_tool_result_blocks_for_anthropic_messages(messages)
    if _is_official_anthropic_api_base(effective_api_base):
        return messages
    messages = normalize_anthropic_tool_use_blocks_in_messages(messages)
    messages = strip_redacted_thinking_blocks_from_anthropic_messages(messages)
    if not _is_deepseek_api_base(effective_api_base):
        messages = strip_thinking_blocks_from_anthropic_messages(messages)
    return messages


def _collect_tool_use_ids(content: Any) -> List[str]:
    """Return ``tool_use`` block ids in an Anthropic message ``content`` list."""
    if not isinstance(content, list):
        return []
    return [
        block["id"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_use"
        and isinstance(block.get("id"), str)
        and block.get("id")
    ]


def _collect_tool_result_ids(content: Any) -> set:
    """Return ``tool_result`` block ``tool_use_id``s in a message ``content`` list."""
    if not isinstance(content, list):
        return set()
    return {
        block["tool_use_id"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("tool_use_id"), str)
    }


def insert_missing_tool_result_blocks_for_anthropic_messages(
    messages: List[Any],
) -> List[Any]:
    """
    Ensure every assistant ``tool_use`` block is answered by a ``tool_result`` block
    in the immediately following user turn.

    Claude Code history sometimes replays an assistant turn that made tool calls but
    omits the matching ``tool_result`` (the next user turn is plain text like
    "continue"). Anthropic-compatible endpoints (e.g. DeepSeek's ``/anthropic/v1/messages``)
    reject this with "tool_use ids were found without tool_result blocks immediately
    after". For any orphaned ``tool_use`` id we add a placeholder ``tool_result`` block.

    The chat/completions path already handles this via
    ``LiteLLMAnthropicMessagesAdapter._insert_missing_tool_result_placeholders``; this
    helper provides the equivalent guarantee for the native Anthropic Messages path.

    The caller's list and message dicts are never mutated; modified turns are returned
    as shallow copies with a fresh content list.
    """
    out: List[Any] = []
    index = 0
    total = len(messages)
    while index < total:
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            out.append(message)
            index += 1
            continue

        tool_use_ids = _collect_tool_use_ids(message.get("content"))
        if not tool_use_ids:
            out.append(message)
            index += 1
            continue

        out.append(message)
        next_message = messages[index + 1] if index + 1 < total else None
        next_is_user = (
            isinstance(next_message, dict) and next_message.get("role") == "user"
        )
        answered = (
            _collect_tool_result_ids(next_message.get("content"))
            if next_is_user and next_message is not None
            else set()
        )
        missing = [tid for tid in tool_use_ids if tid not in answered]

        if not missing:
            out.append(next_message)
            index += 2
            continue

        placeholders = [
            {"type": "tool_result", "tool_use_id": tid, "content": ""}
            for tid in missing
        ]
        if (
            next_is_user
            and next_message is not None
            and isinstance(next_message.get("content"), list)
        ):
            # Anthropic requires tool_result blocks at the start of the user turn.
            out.append(
                {**next_message, "content": placeholders + next_message["content"]}
            )
            index += 2
        else:
            out.append({"role": "user", "content": placeholders})
            index += 1
    return out


def strip_thinking_blocks_from_anthropic_messages(messages: List[Any]) -> List[Any]:
    """
    Return a new message list with thinking / redacted_thinking content blocks removed
    from each message. Used to recover from invalid thinking signatures on retry.

    Messages whose content is a list and becomes empty after stripping are omitted,
    since Anthropic rejects empty content arrays.
    """
    out: List[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        mm = copy.deepcopy(m)
        content = mm.get("content")
        if isinstance(content, list):
            filtered = [
                b
                for b in content
                if not (
                    isinstance(b, dict)
                    and b.get("type") in ("thinking", "redacted_thinking")
                )
            ]
            if not filtered:
                continue
            mm["content"] = filtered
        out.append(mm)
    return out


def strip_thinking_blocks_from_anthropic_messages_request_dict(
    data: Dict[str, Any],
) -> None:
    """
    Mutate an Anthropic Messages-style request dict: strip thinking blocks from
    ``messages`` and remove the top-level ``thinking`` extended-thinking param.
    """
    msgs = data.get("messages")
    if isinstance(msgs, list):
        data["messages"] = strip_thinking_blocks_from_anthropic_messages(msgs)
    data.pop("thinking", None)


def strip_empty_text_blocks_from_anthropic_messages(
    messages: List[Any],
) -> List[Any]:
    """
    Return a new message list with empty or whitespace-only ``{"type": "text"}``
    content blocks removed.

    Anthropic's API rejects requests containing such blocks with
    ``"messages: text content blocks must be non-empty"``, but assistant
    messages from Anthropic routinely arrive with ``{"type": "text", "text": ""}``
    alongside ``tool_use`` blocks (see anthropics/anthropic-sdk-python#461).
    Multi-turn tool-use clients (e.g. Claude Code) loop these prior responses
    back as conversation history, which then causes the next request to 400
    on the unified ``/v1/messages`` path.  ``/v1/chat/completions`` already
    handles this in ``anthropic_messages_pt``; this helper provides the
    equivalent guarantee for the native Anthropic Messages path.

    Also drops ``redacted_thinking`` blocks with missing/empty ``data``, which
    some Anthropic-compatible providers reject (``Invalid data in redacted_thinking``).

    Messages whose content is a list and becomes empty after stripping are
    omitted, matching :func:`strip_thinking_blocks_from_anthropic_messages`.
    The caller's list and its content blocks are never mutated; modified
    messages are returned as shallow copies with a fresh content list.
    """
    out: List[Any] = []
    for m in messages:
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            out.append(m)
            continue
        content = m["content"]
        filtered = [
            b
            for b in content
            if not _is_empty_text_block(b)
            and not _is_invalid_redacted_thinking_block(b)
        ]
        if len(filtered) == len(content):
            out.append(m)
        elif filtered:
            out.append({**m, "content": filtered})
    return out


def _is_empty_text_block(block: Any) -> bool:
    if not isinstance(block, dict) or block.get("type") != "text":
        return False
    text = block.get("text")
    return not isinstance(text, str) or not text.strip()


def _is_valid_redacted_thinking_data(data: Any) -> bool:
    if not isinstance(data, str) or not data.strip():
        return False
    try:
        base64.b64decode(data, validate=True)
        return True
    except (binascii.Error, ValueError):
        return False


def _is_invalid_redacted_thinking_block(block: Any) -> bool:
    """
    Anthropic-compatible providers reject redacted_thinking blocks with missing,
    empty, non-string, or non-base64 ``data`` (e.g. Sophnet 400 on Claude Code).
    """
    if not isinstance(block, dict) or block.get("type") != "redacted_thinking":
        return False
    return not _is_valid_redacted_thinking_data(block.get("data"))


def process_anthropic_headers(headers: Union[httpx.Headers, dict]) -> dict:
    openai_headers = {}
    if "anthropic-ratelimit-requests-limit" in headers:
        openai_headers["x-ratelimit-limit-requests"] = headers[
            "anthropic-ratelimit-requests-limit"
        ]
    if "anthropic-ratelimit-requests-remaining" in headers:
        openai_headers["x-ratelimit-remaining-requests"] = headers[
            "anthropic-ratelimit-requests-remaining"
        ]
    if "anthropic-ratelimit-tokens-limit" in headers:
        openai_headers["x-ratelimit-limit-tokens"] = headers[
            "anthropic-ratelimit-tokens-limit"
        ]
    if "anthropic-ratelimit-tokens-remaining" in headers:
        openai_headers["x-ratelimit-remaining-tokens"] = headers[
            "anthropic-ratelimit-tokens-remaining"
        ]

    llm_response_headers = {
        "{}-{}".format("llm_provider", k): v for k, v in headers.items()
    }

    additional_headers = {**llm_response_headers, **openai_headers}
    return additional_headers
