"""Request one tool call at a time from providers that support it."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

TOOL_SCHEDULING_POLICY = "request-sequential-provider-tools-v2"


def supports_sequential_tool_calls(model: Any) -> bool:
    """Whether ``model.bind_tools`` has an explicit parallel-call control."""
    try:
        parameters = inspect.signature(model.bind_tools).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return "parallel_tool_calls" in parameters


def _sequential(request: ModelRequest) -> ModelRequest:
    if not supports_sequential_tool_calls(request.model):
        return request
    settings = dict(request.model_settings or {})
    settings["parallel_tool_calls"] = False
    return request.override(model_settings=settings)


class SequentialToolCallsMiddleware(AgentMiddleware):
    """Ask capable providers to emit at most one tool call per model turn."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse | AIMessage],
    ) -> ModelResponse | AIMessage:
        return handler(_sequential(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse | AIMessage]],
    ) -> ModelResponse | AIMessage:
        return await handler(_sequential(request))
