"""Inspect model provider for NTNU's hosted LLM API: ``--model ntnu/<model id>``.

NTNU serves open-weight models behind an OpenAI-compatible (LiteLLM) gateway,
so this is Inspect's generic OpenAI-compatible API with two differences. The
base URL defaults to NTNU's, and the key is read from ``NTNU_API_KEY``. And
consecutive system messages are merged into one: the agent's system prompt
reaches Inspect as several blocks through the agent bridge, and chat templates
such as Qwen's reject any system message that is not the first message.

Importing this module registers the provider (the sim_qa task does).
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import GenerateConfig, modelapi

NTNU_BASE_URL = "https://llm.hpc.ntnu.no/v1"


@modelapi(name="ntnu")
def ntnu() -> type:
    from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

    class NTNUAPI(OpenAICompatibleAPI):
        def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig | None = None,
            **model_args: Any,
        ) -> None:
            super().__init__(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                config=config or GenerateConfig(),
                service="NTNU",
                service_base_url=NTNU_BASE_URL,
                **model_args,
            )

        def collapse_system_messages(self) -> bool:
            return True

    return NTNUAPI
