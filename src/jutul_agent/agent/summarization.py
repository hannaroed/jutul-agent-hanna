"""Scientific summaries, manual /compact, and the /context trigger."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from deepagents.middleware.summarization import (
    DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
    SUMMARIZATION_SESSION_ID_KEY,
    SummarizationEvent,
    SummarizationMiddleware,
    compute_summarization_defaults,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately

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


def sanitize_summary(summary: str) -> str:
    """Remove provider-internal reasoning capsules before context reinjection."""
    return strip_internal_capsules(summary).strip()


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


class ScientificSummarizationMiddleware(SummarizationMiddleware):
    """Deep Agents summarizer with a clean model-visible summary boundary."""

    # The shared name replaces Deep Agents' stock summariser.
    name = "SummarizationMiddleware"
    sanitize_history = True

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        cleaned = _sanitize_request_history(request) if self.sanitize_history else request
        return super().wrap_model_call(cleaned, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        cleaned = _sanitize_request_history(request) if self.sanitize_history else request
        return await super().awrap_model_call(cleaned, handler)

    def _build_new_messages_with_path(
        self,
        summary: str,
        file_path: str | None,
    ) -> list[Any]:
        cleaned = sanitize_summary(summary) if self.sanitize_history else summary
        return super()._build_new_messages_with_path(cleaned, file_path)


_SCIENTIFIC_STATE_GUIDANCE = """<scientific_state_preservation>
This conversation belongs to a scientific simulation and research agent.
When summarizing, preserve the experiment state needed for later decisions:

- exact run, case, and attempt identifiers;
- input parameters and every reported comparison metric, including units;
- acceptance constraints, thresholds, and which runs satisfy them;
- artifact paths and a precise note of what each artifact contains;
- commands already executed and whether they succeeded, so work is not repeated.
- every explicit verification gate or pending action, preserving its identifier,
  order, tool name, exact path or search pattern, and completion status verbatim.

Do not copy bulky arrays or tables into the summary. Preserve their path and
schema instead. Record workspace artifacts only with the exact workspace-relative
path that appeared in a user message or tool result. Never construct an absolute
artifact path from the working directory, change a filename or extension, or
invent a ledger/artifact that was not explicitly observed. If a path was not
observed, label it unknown. Never invent a missing value. If a future decision
requires a detail that remains only in an artifact, explicitly say to retrieve it
with the filesystem search/read tools and retain the exact observed path or path
pattern needed. A task is not complete while a verification-gate action is still
pending: never replace a required retrieval with a remembered, rounded, example,
or guessed value, and never summarize a pending action as if it had happened.
</scientific_state_preservation>"""

SCIENTIFIC_SUMMARY_PROMPT = DEEPAGENTS_DEFAULT_SUMMARY_PROMPT.replace(
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
    middleware = ScientificSummarizationMiddleware(
        model=model,
        backend=backend,
        trigger=defaults["trigger"],
        keep=defaults["keep"],
        truncate_args_settings=defaults["truncate_args_settings"],
        summary_prompt=SCIENTIFIC_SUMMARY_PROMPT
        if strategies.summarization == "scientific"
        else DEEPAGENTS_DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize=None,
    )
    middleware.sanitize_history = strategies.sanitize_history
    return middleware


# Check the private engine methods needed by manual /compact.
_ENGINE_METHODS: tuple[str, ...] = (
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

    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        keep=_MANUAL_KEEP,
        trim_tokens_to_summarize=None,
    )
    # The effective conversation is what the model would see now (a prior
    # summary plus the turns since), and the cutoff respects the keep window.
    effective = middleware._apply_event_to_messages(messages, prior_event)
    cutoff = middleware._determine_cutoff_index(effective)
    if cutoff <= 0:
        return None

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)
    summary = sanitize_summary(await middleware._acreate_summary(to_summarize))
    # Upstream now scopes history files to an internal ID stored in agent state.
    # Reuse it across manual and automatic compactions so each session appends to
    # one history file without mixing histories from separate subagents.
    history_session_id = middleware._get_session_id(values)
    file_path = await middleware._aoffload_to_backend(backend, to_summarize, history_session_id)
    summary_msg = middleware._build_new_messages_with_path(summary, file_path)[0]
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
