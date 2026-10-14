"""Clear older tool results from requests; keep raw results for summaries and history."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from langchain.agents.middleware import ContextEditingMiddleware
from langchain_core.messages import ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

# Clear old tool results once the context reaches this share of the model's
# window — below the summarization trigger, so clearing is the first, cheaper
# response and the summary call only fires when clearing cannot keep up.
_CLEAR_TRIGGER_FRACTION = 0.6
# Without a discoverable window, a fixed threshold below the summarization
# fallback (100k) keeps the same ordering.
_FALLBACK_TRIGGER_TOKENS = 60_000
# Below this window, keep fewer recent tool results: on a small (local) window
# the kept working set must not itself fill the window.
_SMALL_WINDOW = 100_000
# Tools whose small, structured result the agent refers back to by value
# (e.g. an attempt id passed as a later parent_attempt_id). Clearing them frees
# almost nothing and could break that reference, so leave them intact.
_NEVER_CLEAR: tuple[str, ...] = ("record_attempt",)
SMALL_TOOL_RESULT_MAX_CHARS = 2048
CLEARED_RESULT_PREFIX_LINES = 3
CLEARED_RESULT_PREFIX_MAX_CHARS = 512


def clear_tool_uses_trigger_tokens(window: int | None) -> int:
    """The context size at which old tool results start being cleared."""
    return int(window * _CLEAR_TRIGGER_FRACTION) if window else _FALLBACK_TRIGGER_TOKENS


def keep_recent_tool_results(window: int | None) -> int:
    """How many recent tool results to keep verbatim, sized to the window.

    Clearing keeps the agent's working set, but that set must not itself fill a
    small window: a handful of medium results (each below the eviction
    threshold, so still inline) can otherwise exceed a local model's window.
    Keep fewer on a small window; large single results are evicted separately.
    """
    if window and window <= 32_768:
        return 1
    return 3 if (window and window < _SMALL_WINDOW) else 6


class FullRequestContextEditingMiddleware(ContextEditingMiddleware):
    """Include system text and tool schemas when deciding to clear results."""

    def edit_request(self, request, *, token_counter=None):
        def count(messages):
            if token_counter is not None:
                return token_counter(messages, request.system_message, request.tools)
            system = [request.system_message] if request.system_message else []
            return count_tokens_approximately([*system, *messages], tools=request.tools)

        messages = deepcopy(list(request.messages))
        original_results = {
            index: message
            for index, message in enumerate(messages)
            if isinstance(message, ToolMessage) and isinstance(message.content, str)
        }
        for edit in self.edits:
            edit.apply(messages, count_tokens=count)
        for index, original in original_results.items():
            reduced = messages[index]
            if not reduced.response_metadata.get("context_editing", {}).get("cleared"):
                continue
            if len(original.content) <= SMALL_TOOL_RESULT_MAX_CHARS:
                messages[index] = original
            else:
                prefix = "\n".join(original.content.splitlines()[:CLEARED_RESULT_PREFIX_LINES])
                messages[index] = reduced.model_copy(
                    update={
                        "content": "[cleared; first lines only]\n"
                        + prefix[:CLEARED_RESULT_PREFIX_MAX_CHARS]
                    }
                )
        return request.override(messages=messages)

    def wrap_model_call(self, request, handler):
        return handler(self.edit_request(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self.edit_request(request))


def _profile_window(model: Any | None) -> int | None:
    """Read a pre-built model's declared input window, if it has one."""
    profile = getattr(model, "profile", None)
    if not isinstance(profile, dict):
        return None
    value = profile.get("max_input_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def filesystem_tool_result_limit(window: int | None) -> int:
    return max(256, window // 32) if window and window <= 32_768 else 20_000


def build_context_editing_middleware(
    model_id: str | None = None,
    *,
    model: Any | None = None,
) -> Any:
    """Middleware that clears older tool results when the context grows large.

    The trigger is sized to the model's window so it fires before summarization;
    the most recent results are kept so the active working set is untouched, with
    fewer kept on a small window so that set can't fill it.
    """
    from langchain.agents.middleware import ClearToolUsesEdit

    # Eval bridge models carry the context budget in their profile.
    window = None
    if model_id is not None:
        from jutul_agent.models import context_window

        window = context_window(model_id)
    if window is None:
        window = _profile_window(model)
    return FullRequestContextEditingMiddleware(
        edits=[
            ClearToolUsesEdit(
                trigger=clear_tool_uses_trigger_tokens(window),
                keep=keep_recent_tool_results(window),
                exclude_tools=_NEVER_CLEAR,
            )
        ],
    )
