"""Configure and check NTNU's hosted LLM API without changing other providers.

Run with the existing project environment:
    uv run --no-sync python research/idun_connection.py setup
    uv run --no-sync python research/idun_connection.py models
    uv run --no-sync python research/idun_connection.py check --model EXACT_MODEL_ID
    uv run --no-sync python research/idun_connection.py agent --model EXACT_MODEL_ID tui

The check command tests the connection and a tiny tool round trip. The agent
command launches the existing agent in a separate local workspace. NTNU
network/VPN access is required. No new packages needed.
Official API guide:
https://www.hpc.ntnu.no/idun/documentation/ai-coding-assistant-and-large-language-models-llms-on-idun/
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

REPO = Path(__file__).resolve().parents[1]
KEY_FILE = REPO / ".jutul-agent" / "idun" / "api-key"
BASE_URL = "https://llm.hpc.ntnu.no/v1"
TIMEOUT = httpx.Timeout(90.0, connect=8.0)


def save_key() -> None:
    if not sys.stdin.isatty():
        raise ValueError("Run setup yourself in a terminal; key entry is hidden.")
    key = getpass.getpass("Paste your NTNU API key (hidden), then press Enter: ").strip()
    if not key or any(char.isspace() for char in key) or not key.isascii():
        raise ValueError("Expected a non-empty API key without whitespace.")
    KEY_FILE.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=KEY_FILE.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(key + "\n")
        Path(temporary).replace(KEY_FILE)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print("Key saved in the Git-ignored .jutul-agent/idun/api-key (owner access only).")
    print("Next: connect to NTNU VPN, then run this helper with 'models'.")


def read_key() -> str:
    if not KEY_FILE.is_file():
        raise ValueError("No local NTNU key yet. Run this helper with 'setup' in a terminal.")
    if KEY_FILE.stat().st_mode & 0o077:
        raise ValueError(
            "Key file permissions must be private: chmod 600 .jutul-agent/idun/api-key"
        )
    key = KEY_FILE.read_text().strip()
    if not key or any(char.isspace() for char in key) or not key.isascii():
        raise ValueError("Invalid local key file. Run 'setup' again.")
    return key


def list_models(key: str) -> None:
    with OpenAI(api_key=key, base_url=BASE_URL, timeout=TIMEOUT, max_retries=0) as client:
        models = sorted(item.id for item in client.models.list())
    print("NTNU authentication succeeded. Available model IDs:")
    for model in models:
        print(f"  {model}")
    print("\nChoose a chat model, then run 'check --model EXACT_MODEL_ID'.")


def check_model(key: str, model: str) -> None:
    from langchain_core.messages import HumanMessage, ToolMessage
    from langchain_openai import ChatOpenAI

    # Explicit endpoint and key avoid using any personal OpenAI credentials.
    llm = ChatOpenAI(
        model=model,
        api_key=key,
        base_url=BASE_URL,
        use_responses_api=False,
        timeout=TIMEOUT,
        max_retries=0,
        max_tokens=1024,
    )
    tool = {
        "type": "function",
        "function": {
            "name": "idun_connection_check",
            "description": "Return a local test value. This tool has no side effects.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }
    request = HumanMessage(
        content="Call idun_connection_check with value 7. Do not answer directly."
    )
    reply = llm.bind_tools([tool]).invoke([request])
    calls = reply.tool_calls
    if (
        len(calls) != 1
        or calls[0]["name"] != "idun_connection_check"
        or calls[0]["args"] != {"value": 7}
    ):
        raise ValueError(
            "API responded, but this small tool-call check did not pass. "
            "No scientific experiment was started."
        )
    result = ToolMessage(content=json.dumps({"value": 7}), tool_call_id=calls[0]["id"])
    final = llm.invoke(
        [
            request,
            reply,
            result,
            HumanMessage(content="Reply with exactly IDUN_OK_7 if the tool returned value 7."),
        ]
    )
    if final.text.strip() != "IDUN_OK_7":
        raise ValueError("Tool call worked, but the final response check did not pass.")
    print(f"Passed: NTNU model {model} completed a tool call and read its local result.")
    print("This verifies API/tool compatibility only; Julia and full experiments are separate.")


def launch_agent(key: str, model: str, interface: str, arguments: list[str]) -> None:
    """Scope the NTNU endpoint to this process; preserve other provider settings."""
    environment = dict(os.environ)
    environment.update(
        OPENAI_API_KEY=key,
        OPENAI_BASE_URL=BASE_URL,
        OPENAI_API_BASE=BASE_URL,
    )
    workspace = KEY_FILE.parent / "workspace"
    workspace.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "jutul_agent",
        interface,
        *arguments,
        "--model",
        f"openai:{model}",
        "--workspace",
        str(workspace),
        "--state-home",
        str(KEY_FILE.parent / "state"),
    ]
    os.execve(sys.executable, command, environment)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="Save your API key privately using hidden terminal input")
    commands.add_parser("models", help="Check authentication and list available model IDs")
    check = commands.add_parser("check", help="Run a tiny two-request tool compatibility check")
    check.add_argument("--model", required=True, help="Exact model ID from the models command")
    agent = commands.add_parser(
        "agent", help="Launch jutul-agent with NTNU in a separate local workspace"
    )
    agent.add_argument("--model", required=True, help="Exact model ID, without the openai: prefix")
    agent.add_argument("interface", choices=["tui", "web", "run"])
    agent.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="Remaining jutul-agent flags/prompt"
    )
    args = parser.parse_args()
    try:
        if args.command == "setup":
            save_key()
        elif args.command == "models":
            list_models(read_key())
        elif args.command == "check":
            check_model(read_key(), args.model)
        else:
            launch_agent(read_key(), args.model, args.interface, args.arguments)
    except APITimeoutError:
        print(
            "NTNU API timed out. Check the VPN or retry later if the service is busy.",
            file=sys.stderr,
        )
        return 1
    except APIConnectionError:
        print("Cannot reach NTNU's API. Connect to NTNU VPN or campus network.", file=sys.stderr)
        return 1
    except APIStatusError as exc:
        print(f"NTNU API returned HTTP {exc.status_code}.", file=sys.stderr)
        print(
            "401/403: check the key/access; 404: check the model ID; 429: retry later.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
