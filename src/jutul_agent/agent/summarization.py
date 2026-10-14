"""Scientific summaries, manual /compact, and the /context trigger."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from typing import Any

from deepagents.middleware.summarization import (
    SUMMARIZATION_SESSION_ID_KEY,
    SummarizationEvent,
    SummarizationMiddleware,
    compute_summarization_defaults,
)
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, get_buffer_string
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableLambda

from jutul_agent.agent.recovery import ModelOutputError, reject_truncated_response
from jutul_agent.agent.strategies import HarnessStrategies
from jutul_agent.agent.text import strip_internal_capsules
from jutul_agent.trace import TraceLog
from jutul_agent.trace.schema import CONTEXT_COMPACTION, compaction_payload

# The share of the window at which the stock summarizer triggers, mirrored here
# only so /context can show the figure. Matches deepagents'
# ``compute_summarization_defaults`` default; update if that default changes.
_TRIGGER_FRACTION = 0.85
# Without a discoverable window /context can't size the trigger; this mirrors
# deepagents' fixed no-profile fallback.
_FALLBACK_TRIGGER_TOKENS = 170_000
# Manual /compact keeps a shorter, predictable tail; the count is public so the
# TUI can explain why a short conversation has nothing to compact.
MANUAL_KEEP_MESSAGES = 8
_MANUAL_KEEP: tuple[str, int] = ("messages", MANUAL_KEEP_MESSAGES)
SUMMARY_SANITIZATION_POLICY = "strip-internal-reasoning-capsules-v1"
MESSAGE_HISTORY_SANITIZATION_POLICY = "strip-internal-reasoning-capsules-before-model-v1"
SMALL_WINDOW_TOKEN_MARGIN = 1024
SUMMARY_POLICY = "complete-observed-facts-with-verbatim-user-request-v2"
_SUMMARY_REQUEST: ContextVar[ModelRequest | None] = ContextVar("summary_request", default=None)

GENERIC_SUMMARY_PROMPT = """Summarize this conversation in at most 160 words.
Use Goal, Completed, Artifacts, and Next steps. Report only observed facts.
Requesting a tool does not mean it succeeded. Loading code does not run a simulation.
Prefer tool results over previous summaries. Preserve literal commands, search
patterns, paths, quantities, units, and constraints exactly. Use None when unknown.
Do not invent files, images, retrieved data, or completed work.

<messages>
{messages}
</messages>"""


def sanitize_summary(summary: str) -> str:
    """Remove provider-internal reasoning capsules before context reinjection."""
    return strip_internal_capsules(summary).strip()


def _validate_summary_response(response: AIMessage) -> AIMessage:
    reject_truncated_response(response)
    if response.tool_calls or response.invalid_tool_calls or not sanitize_summary(response.text):
        raise ModelOutputError("Summary has no usable text; the conversation was not compacted.")
    return response


def _sanitize_ai_content(content: Any) -> Any:
    """Strip text capsules while preserving provider/tool-call message structure."""
    if isinstance(content, str):
        return strip_internal_capsules(content)
    if not isinstance(content, list):
        return content
    cleaned: list[Any] = []
    for block in content:
        if isinstance(block, str):
            cleaned.append(strip_internal_capsules(block))
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            cleaned.append({**block, "text": strip_internal_capsules(block["text"])})
        else:
            cleaned.append(block)
    return cleaned


def sanitize_message_history(messages: list[Any]) -> list[Any]:
    """Clean copied model history without changing checkpoints or tool calls."""
    cleaned: list[Any] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            cleaned.append(message)
            continue
        content = _sanitize_ai_content(message.content)
        cleaned.append(
            message
            if content == message.content
            else message.model_copy(update={"content": content})
        )
    return cleaned


def _sanitize_request_history(request: ModelRequest) -> ModelRequest:
    messages = sanitize_message_history(list(request.messages))
    return request.override(messages=messages)


def _pin_user_request(summary_message, history):
    latest = next(
        (
            message
            for message in reversed(history)
            if isinstance(message, HumanMessage)
            and message.additional_kwargs.get("lc_source") != "summarization"
            and not message.additional_kwargs.get("empty_turn_recovery")
        ),
        None,
    )
    if latest is None:
        return summary_message
    return summary_message.model_copy(
        update={
            "content": (
                "User request (verbatim; follow these instructions):\n"
                + get_buffer_string([latest], format="xml")
                + "\nThe following generated summary is unverified. "
                "Check its claims against tool results and saved files.\n"
                + str(summary_message.content)
            )
        }
    )


class SummarizationSlot(AgentMiddleware):
    """Replace the stock slot so summarization can run after prompt injection."""

    name = "SummarizationMiddleware"


class ScientificSummarizationMiddleware(SummarizationMiddleware):
    """Deep Agents summarizer with a clean model-visible summary boundary."""

    name = "ScientificSummarizationMiddleware"
    sanitize_history = True
    context_editor = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lc_helper._summary_model = self._lc_helper._summary_model | RunnableLambda(
            _validate_summary_response
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        cleaned = _sanitize_request_history(request) if self.sanitize_history else request
        token = _SUMMARY_REQUEST.set(
            cleaned.override(messages=super()._get_effective_messages(cleaned))
        )
        try:
            return super().wrap_model_call(cleaned, handler)
        finally:
            _SUMMARY_REQUEST.reset(token)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        cleaned = _sanitize_request_history(request) if self.sanitize_history else request
        token = _SUMMARY_REQUEST.set(
            cleaned.override(messages=super()._get_effective_messages(cleaned))
        )
        try:
            return await super().awrap_model_call(cleaned, handler)
        finally:
            _SUMMARY_REQUEST.reset(token)

    def _get_effective_messages(self, request: ModelRequest) -> list[Any]:
        messages = super()._get_effective_messages(request)
        if self.context_editor is not None:
            return self.context_editor.edit_request(
                request.override(messages=messages), token_counter=self._count_tokens
            ).messages
        return messages

    def _token_margin(self):
        window = (getattr(self.model, "profile", None) or {}).get("max_input_tokens")
        return SMALL_WINDOW_TOKEN_MARGIN if isinstance(window, int) and window <= 32_768 else 0

    def _count_tokens(self, messages, system_message, tools):
        return super()._count_tokens(messages, system_message, tools) + self._token_margin()

    def _partition_messages(self, conversation_messages, cutoff_index):
        to_summarize, to_keep = super()._partition_messages(conversation_messages, cutoff_index)
        original = _SUMMARY_REQUEST.get()
        if original is not None:
            # Summaries and saved history need evidence, not clearing placeholders.
            to_summarize = [
                original.messages[index] if isinstance(message, ToolMessage) else message
                for index, message in enumerate(to_summarize)
            ]
        return to_summarize, to_keep

    def _check_summary_budget(self, messages):
        request = _SUMMARY_REQUEST.get()
        budget = self._input_budget(request) if request is not None else None
        if budget is None:
            return
        prompt = self._lc_helper.summary_prompt.format(
            messages=get_buffer_string(messages, format="xml")
        ).rstrip()
        if self.token_counter([HumanMessage(content=prompt)]) + self._token_margin() > budget:
            raise ContextOverflowError(
                "Summary input exceeds the context budget; reduce tool output or compact earlier."
            )

    def _create_summary(self, messages_to_summarize):
        self._check_summary_budget(messages_to_summarize)
        return super()._create_summary(messages_to_summarize)

    async def _acreate_summary(self, messages_to_summarize):
        self._check_summary_budget(messages_to_summarize)
        return await super()._acreate_summary(messages_to_summarize)

    def _build_new_messages_with_path(
        self,
        summary: str,
        file_path: str | None,
    ) -> list[Any]:
        cleaned = sanitize_summary(summary) if self.sanitize_history else summary
        messages = super()._build_new_messages_with_path(cleaned, file_path)
        request = _SUMMARY_REQUEST.get()
        if request is not None:
            history = request.state.get("messages", request.messages)
            messages[0] = _pin_user_request(messages[0], history)
        return messages


_SCIENTIFIC_STATE_GUIDANCE = """<scientific_state_preservation>
Preserve run IDs, input parameters, exact comparison values with units, and
acceptance limits. Record which simulations succeeded or failed and which
steps remain, in order. Never describe a pending step as completed.
Keep observed artifact paths exactly as returned, without inventing or changing
paths or values. For large arrays and tables, preserve the path and schema so
missing details can be retrieved with filesystem search/read tools.
</scientific_state_preservation>"""

SCIENTIFIC_SUMMARY_PROMPT = GENERIC_SUMMARY_PROMPT.replace(
    "\n<messages>\n",
    f"\n{_SCIENTIFIC_STATE_GUIDANCE}\n\n<messages>\n",
    1,
)


def build_scientific_summarization_middleware(
    model: Any, backend: Any, *, strategies: HarnessStrategies | None = None
) -> Any | None:
    """Build the chosen summariser for a resolved model; defer string specs."""
    from langchain.chat_models import BaseChatModel

    if not isinstance(model, BaseChatModel):
        return None
    defaults = compute_summarization_defaults(model)
    strategies = strategies or HarnessStrategies()
    window = (getattr(model, "profile", None) or {}).get("max_input_tokens")
    keep = ("messages", 2) if isinstance(window, int) and window <= 32_768 else defaults["keep"]
    middleware = ScientificSummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=defaults["trigger"],
        keep=keep,
        # Prior usage describes the old history, not the compacted request.
        token_counter=partial(
            count_tokens_approximately,
            use_usage_metadata_scaling=False,
            chars_per_token=3.3 if model._llm_type.startswith("anthropic-chat") else 4.0,
        ),
        truncate_args_settings=defaults["truncate_args_settings"],
        summary_prompt=SCIENTIFIC_SUMMARY_PROMPT
        if strategies.summarization == "scientific"
        else GENERIC_SUMMARY_PROMPT,
        trim_tokens_to_summarize=None,
    )
    middleware.sanitize_history = strategies.sanitize_history
    if strategies.clear_tool_results:
        from jutul_agent.agent.context_editing import build_context_editing_middleware

        middleware.context_editor = build_context_editing_middleware(model=model)
    return middleware


# Check the private engine methods needed by manual /compact.
_ENGINE_METHODS: tuple[str, ...] = (
    "_get_effective_messages",
    "_count_tokens",
    "_apply_event_to_messages",
    "_determine_cutoff_index",
    "_partition_messages",
    "_acreate_summary",
    "_aoffload_to_backend",
    "_build_new_messages_with_path",
    "_compute_state_cutoff",
    "_get_session_id",
)


def manual_compaction_available() -> bool:
    """Whether the installed deepagents still exposes the engine /compact drives."""
    return all(hasattr(SummarizationMiddleware, name) for name in _ENGINE_METHODS)


def auto_compact_trigger_tokens(window: int | None) -> int:
    """The context size at which auto-compaction triggers for ``window`` (display only)."""
    return int(window * _TRIGGER_FRACTION) if window else _FALLBACK_TRIGGER_TOKENS


@dataclass(frozen=True)
class CompactResult:
    messages_summarized: int
    messages_kept: int
    # Approximate input tokens the summary saves over the turns it replaces.
    # Anchors the live /context estimate so it drops immediately, before the
    # next model call measures the new size exactly.
    freed_tokens: int = 0
    # Whether the summarized turns were saved to the backend (recoverable).
    offloaded: bool = False


async def compact_thread(
    agent: Any,
    *,
    thread_id: str,
    model: Any,
    backend: Any,
    trace: TraceLog | None = None,
) -> CompactResult | None:
    """Summarize the thread's older turns now, on demand.

    Drives deepagents' summarization engine against the checkpointed state and
    records the result as a ``_summarization_event`` — the same non-mutating
    mechanism auto-compaction uses, so the raw conversation log is preserved and
    the offloaded turns stay recoverable from ``/conversation_history``. Returns
    ``None`` when there is nothing to compact (too few messages, or no state).
    """
    if not manual_compaction_available():
        raise RuntimeError(
            "manual compaction is unavailable with this deepagents version "
            "(auto-compaction still runs); see agent/summarization.py"
        )
    aget_state = getattr(agent, "aget_state", None)
    aupdate_state = getattr(agent, "aupdate_state", None)
    if aget_state is None or aupdate_state is None:
        return None

    config = {"configurable": {"thread_id": thread_id}}
    state = await aget_state(config)
    values = getattr(state, "values", None) or {}
    messages = values.get("messages") or []
    prior_event = values.get("_summarization_event")

    middleware = ScientificSummarizationMiddleware(
        model=model,
        backend=backend,
        keep=_MANUAL_KEEP,
        trim_tokens_to_summarize=None,
        summary_prompt=SCIENTIFIC_SUMMARY_PROMPT,
        token_counter=partial(count_tokens_approximately, use_usage_metadata_scaling=False),
    )
    # The effective conversation is what the model would see now (a prior
    # summary plus the turns since), and the cutoff respects the keep window.
    effective = middleware._apply_event_to_messages(messages, prior_event)
    cutoff = middleware._determine_cutoff_index(effective)
    if cutoff <= 0:
        return None

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)
    token = _SUMMARY_REQUEST.set(
        ModelRequest(model=middleware.model, messages=effective, state=values, runtime=None)
    )
    try:
        summary = sanitize_summary(await middleware._acreate_summary(to_summarize))
    finally:
        _SUMMARY_REQUEST.reset(token)
    # Upstream now scopes history files to an internal ID stored in agent state.
    # Reuse it across manual and automatic compactions so each session appends to
    # one history file without mixing histories from separate subagents.
    history_session_id = middleware._get_session_id(values)
    file_path = await middleware._aoffload_to_backend(backend, to_summarize, history_session_id)
    summary_msg = middleware._build_new_messages_with_path(summary, file_path)[0]
    summary_msg = _pin_user_request(summary_msg, messages)
    state_cutoff = middleware._compute_state_cutoff(prior_event, cutoff)
    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,
        "file_path": file_path,
    }
    await aupdate_state(
        config,
        {"_summarization_event": new_event, SUMMARIZATION_SESSION_ID_KEY: history_session_id},
    )

    freed = max(
        0,
        int(count_tokens_approximately(to_summarize))
        - int(count_tokens_approximately([summary_msg])),
    )
    if trace is not None:
        trace.append(
            CONTEXT_COMPACTION,
            compaction_payload(
                summarized=len(to_summarize),
                kept=len(to_keep),
                offloaded=file_path is not None,
                manual=True,
            ),
        )
    return CompactResult(
        messages_summarized=len(to_summarize),
        messages_kept=len(to_keep),
        freed_tokens=freed,
        offloaded=file_path is not None,
    )
