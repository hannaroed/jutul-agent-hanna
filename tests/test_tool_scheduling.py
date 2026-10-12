"""Tests for provider-enforced sequential tool scheduling."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage

from jutul_agent.agent.tool_scheduling import (
    SequentialToolCallsMiddleware,
    supports_sequential_tool_calls,
)


class _CapableModel:
    def bind_tools(self, tools, *, parallel_tool_calls=None):
        return self


class _UnsupportedModel:
    def bind_tools(self, tools, **kwargs):
        return self


def _request(model, settings=None) -> ModelRequest:
    return ModelRequest(
        model=model,
        messages=[],
        system_message=None,
        runtime=MagicMock(),
        state={"messages": []},
        model_settings=settings,
    )


def test_detects_explicit_provider_control() -> None:
    assert supports_sequential_tool_calls(_CapableModel()) is True
    assert supports_sequential_tool_calls(_UnsupportedModel()) is False


async def test_enforces_one_tool_call_without_losing_existing_settings() -> None:
    captured = {}

    async def handler(request):
        captured.update(request.model_settings or {})
        return AIMessage(content="ok")

    middleware = SequentialToolCallsMiddleware()
    result = await middleware.awrap_model_call(
        _request(_CapableModel(), {"temperature": 0}), handler
    )

    assert result.content == "ok"
    assert captured == {"temperature": 0, "parallel_tool_calls": False}


async def test_leaves_unsupported_provider_unchanged() -> None:
    captured = None

    async def handler(request):
        nonlocal captured
        captured = request.model_settings
        return AIMessage(content="ok")

    middleware = SequentialToolCallsMiddleware()
    await middleware.awrap_model_call(_request(_UnsupportedModel()), handler)

    assert captured == {}
