"""
Stream-state tracker for the /v1/responses (Codex) mid-stream fallback path.

Tracks text, tool-call, and reasoning state across stream events so that a
mid-stream failure can hand a richer continuation payload to the fallback
model — preserving what the client already saw and avoiding tool-call / reasoning
state loss.

Only depends on event-type enums — no imports from litellm.router or
litellm.exceptions to keep the import graph acyclic (router imports helper).
"""

from typing import Any, Dict, List, Optional

from litellm.types.llms.openai import ResponsesAPIStreamEvents as _StreamEvents


class _ResponsesStreamState:
    """
    Observes a /v1/responses streaming event sequence and accumulates:

    - visible_text  (from OUTPUT_TEXT_DELTA)
    - unresolved tool-call items by item_id
    - completed tool-call items by item_id
    - unresolved reasoning entries by (item_id, summary_index)
    - a "has_{text|tool|reasoning}_delta" aggregate for continuation-or-bare-retry
      decision.

    Usage:

        state = _ResponsesStreamState()
        async for event in source_iterator:
            state.observe(event)
            if state.is_visible(event):
                yielded_visible_output = True
            yield event

        # On failure:
        payload = state.to_error_payload()
        continuation = state.to_continuation_payload()
    """

    def __init__(self) -> None:
        # Textual content from output_text.delta
        self._visible_text: List[str] = []

        # In-flight items by item_id (added but not yet seen a delta or done)
        self._in_flight_items: Dict[str, Dict[str, Any]] = {}

        # Partial tool-call state by item_id
        # Keys match plan: {item_id, call_id, name, arguments_str, done}
        self._tool_calls: Dict[str, Dict[str, Any]] = {}

        # Partial reasoning state by (item_id, summary_index)
        # Keys match plan: {item_id, summary_index, summary_text, done}
        self._reasoning: Dict[str, Dict[str, Any]] = {}

        # Aggregate visibility flags (used for continuation-or-bare-retry decision)
        self.has_text: bool = False
        self.has_tool_delta: bool = False
        self.has_reasoning_delta: bool = False

    @staticmethod
    def _safe_str(v: Any) -> Optional[str]:
        """Return v if it is a non-empty string, else None.
        Guards against MagicMock auto-created attributes."""
        return v if isinstance(v, str) else None

    @staticmethod
    def _safe_int(v: Any, default: int = 0) -> int:
        """Return v if it is an int, else default."""
        return v if isinstance(v, int) else default

    def observe(self, event: Any) -> None:
        """
        Feed one stream event into the tracker.

        Handles: OUTPUT_ITEM_ADDED, FUNCTION_CALL_ARGUMENTS_DELTA/_DONE,
        REASONING_SUMMARY_TEXT_DELTA/_DONE, OUTPUT_TEXT_DELTA,
        OUTPUT_ITEM_DONE.
        """
        event_type = self._safe_str(getattr(event, "type", None))
        if event_type is None:
            return

        if event_type == _StreamEvents.OUTPUT_TEXT_DELTA:
            delta = self._safe_str(getattr(event, "delta", None))
            if delta is not None:
                self._visible_text.append(delta)
                self.has_text = True

        elif event_type == _StreamEvents.OUTPUT_ITEM_ADDED:
            item_id = self._resolve_item_id(event)
            if item_id is not None:
                # Determine item type and metadata from the nested `item` payload
                item_payload = getattr(event, "item", None)
                if item_payload is not None and not isinstance(item_payload, dict):
                    item_type = self._safe_str(getattr(item_payload, "type", None))
                    _name = self._safe_str(getattr(item_payload, "name", None))
                    _call_id = self._safe_str(getattr(item_payload, "call_id", None))
                elif isinstance(item_payload, dict):
                    item_type = self._safe_str(item_payload.get("type"))
                    _name = self._safe_str(item_payload.get("name"))
                    _call_id = self._safe_str(item_payload.get("call_id"))
                else:
                    item_type = None
                    _name = None
                    _call_id = None

                in_flight = {
                    "item_id": item_id,
                    "item_type": item_type,
                    "has_delta": False,
                }
                # Seed tool-call metadata for later delta tracking
                if item_type == "function_call":
                    if item_id not in self._tool_calls:
                        self._tool_calls[item_id] = {
                            "item_id": item_id,
                            "call_id": _call_id or "",
                            "name": _name or "",
                            "arguments_str": "",
                            "done": False,
                        }
                    else:
                        tc = self._tool_calls[item_id]
                        if _name:
                            tc["name"] = _name
                        if _call_id:
                            tc["call_id"] = _call_id
                self._in_flight_items[item_id] = in_flight

        elif event_type == _StreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA:
            item_id = self._safe_str(getattr(event, "item_id", None))
            delta = self._safe_str(getattr(event, "delta", None)) or ""
            if item_id is not None:
                self._ensure_tool_call(item_id)
                self._tool_calls[item_id]["arguments_str"] += delta
                self.has_tool_delta = True
                self._mark_item_has_delta(item_id)

        elif event_type == _StreamEvents.FUNCTION_CALL_ARGUMENTS_DONE:
            item_id = self._safe_str(getattr(event, "item_id", None))
            arguments = self._safe_str(getattr(event, "arguments", None)) or ""
            if item_id is not None:
                self._ensure_tool_call(item_id)
                # Override accumulated value with the complete payload
                self._tool_calls[item_id]["arguments_str"] = arguments
                self._tool_calls[item_id]["done"] = True
                self.has_tool_delta = True
                self._mark_item_has_delta(item_id)

        elif event_type == _StreamEvents.REASONING_SUMMARY_TEXT_DELTA:
            item_id = self._safe_str(getattr(event, "item_id", None))
            summary_index = self._safe_int(getattr(event, "summary_index", 0))
            delta = self._safe_str(getattr(event, "delta", None)) or ""
            if item_id is not None:
                key = self._reasoning_key(item_id, summary_index)
                if key not in self._reasoning:
                    self._reasoning[key] = {
                        "item_id": item_id,
                        "summary_index": summary_index,
                        "summary_text": "",
                        "done": False,
                    }
                self._reasoning[key]["summary_text"] += delta
                self.has_reasoning_delta = True
                self._mark_item_has_delta(item_id)

        elif event_type == _StreamEvents.REASONING_SUMMARY_TEXT_DONE:
            item_id = self._safe_str(getattr(event, "item_id", None))
            summary_index = self._safe_int(getattr(event, "summary_index", 0))
            text = self._safe_str(getattr(event, "text", None)) or ""
            if item_id is not None:
                key = self._reasoning_key(item_id, summary_index)
                # Override accumulated value with the complete payload
                self._reasoning[key] = {
                    "item_id": item_id,
                    "summary_index": summary_index,
                    "summary_text": text,
                    "done": True,
                }
                self.has_reasoning_delta = True
                self._mark_item_has_delta(item_id)

        elif event_type == _StreamEvents.OUTPUT_ITEM_DONE:
            item_id = self._resolve_item_id(event)
            if item_id is not None:
                self._in_flight_items.pop(item_id, None)
                # Saturate completed tool-call args from the event item payload
                item_payload = getattr(event, "item", None)
                if item_id in self._tool_calls:
                    tc = self._tool_calls[item_id]
                    tc["done"] = True
                    # If the event carries final arguments, overwrite
                    if item_payload is not None:
                        event_args = self._safe_str(
                            getattr(item_payload, "arguments", None)
                            if not isinstance(item_payload, dict)
                            else item_payload.get("arguments")
                        )
                        if event_args is not None:
                            tc["arguments_str"] = event_args
                            tc["done"] = True

    def is_visible(self, event: Any) -> bool:
        """
        True for events whose content was rendered to the client.

        Visible = has real content (text delta, tool arg delta, reasoning delta)
        or terminal event (done) that the client acted on.

        Lifecycle-only events (response.created, output_item.added with no
        subsequent delta) are never visible.
        """
        event_type = self._safe_str(getattr(event, "type", None))
        if event_type is None:
            return False
        # Terminal text events
        if event_type in (
            _StreamEvents.OUTPUT_TEXT_DELTA,
            _StreamEvents.OUTPUT_TEXT_DONE,
            _StreamEvents.CONTENT_PART_DONE,
            _StreamEvents.OUTPUT_ITEM_DONE,
        ):
            return True
        # Reasoning and tool-arg deltas carry content
        if event_type in (
            _StreamEvents.REASONING_SUMMARY_TEXT_DELTA,
            _StreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA,
            _StreamEvents.REASONING_SUMMARY_TEXT_DONE,
            _StreamEvents.FUNCTION_CALL_ARGUMENTS_DONE,
        ):
            return True
        return False

    def to_error_payload(self) -> Dict[str, Any]:
        """
        Build the extra keyword arguments for MidStreamFallbackError.
        """
        return {
            "partial_tool_calls": list(self._tool_calls.values()),
            "partial_reasoning": sorted(
                self._reasoning.values(),
                key=lambda r: (r["item_id"], r["summary_index"]),
            ),
            "had_in_flight_item": len(self._in_flight_items) > 0,
        }

    def to_continuation_payload(self) -> Dict[str, Any]:
        """
        Build the extra keyword arguments for _build_responses_continuation_input.

        Returns ALL tool calls (done and undone). The builder distinguishes them:
        done tool calls become paired function_call + function_call_output input
        items; undone tool calls are only described in the developer message.
        """
        all_tool_calls = sorted(
            self._tool_calls.values(),
            key=lambda tc: tc.get("item_id", ""),
        )
        partial_reasoning = sorted(
            self._reasoning.values(),
            key=lambda r: (r["item_id"], r["summary_index"]),
        )
        return {
            "partial_tool_calls": all_tool_calls,
            "partial_reasoning": partial_reasoning,
            "had_in_flight_item": len(self._in_flight_items) > 0,
        }

    def get_generated_content(self) -> str:
        """Collect visible text for the continuation prompt."""
        return "".join(self._visible_text)

    def has_recoverable_state(self) -> bool:
        """
        True when there is any state worth continuing into a fallback.

        Only pure OUTPUT_ITEM_ADDED (no subsequent delta) is NOT recoverable.
        """
        if self.has_text or self.has_tool_delta or self.has_reasoning_delta:
            return True
        if any(tc.get("done") for tc in self._tool_calls.values()):
            return True
        if any(inf.get("has_delta") for inf in self._in_flight_items.values()):
            return True
        return False

    # ---- internal helpers ----

    @staticmethod
    def _resolve_item_id(event: Any) -> Optional[str]:
        """Extract item_id from various event shapes, returning None when
        the value is not a real string (MagicMock auto-creates attributes)."""
        item_id = getattr(event, "item_id", None)
        if isinstance(item_id, str):
            return item_id
        # Some events carry the id inside the nested `item` payload
        item_payload = getattr(event, "item", None)
        if item_payload is not None:
            candidate = (
                getattr(item_payload, "item_id", None)
                or getattr(item_payload, "id", None)
                or (
                    item_payload.get("item_id")
                    if isinstance(item_payload, dict)
                    else None
                )
            )
            if isinstance(candidate, str):
                return candidate
        return None

    def _ensure_tool_call(self, item_id: str) -> None:
        """Initialize a partial tool-call tracking entry if missing."""
        if item_id not in self._tool_calls:
            in_flight = self._in_flight_items.get(item_id, {})
            self._tool_calls[item_id] = {
                "item_id": item_id,
                "call_id": "",
                "name": "",
                "arguments_str": "",
                "done": False,
            }

    def _mark_item_has_delta(self, item_id: str) -> None:
        """Mark in-flight item as having received content deltas."""
        if item_id in self._in_flight_items:
            self._in_flight_items[item_id]["has_delta"] = True

    @staticmethod
    def _reasoning_key(item_id: str, summary_index: int) -> str:
        return f"{item_id}:{summary_index}"
