"""Explicit harness controls for controlled scientific comparisons."""

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class HarnessStrategies:
    summarization: Literal["generic", "scientific"] = "scientific"
    clear_tool_results: bool = True
    sequential_tools: bool = True
    sanitize_history: bool = True

    def __post_init__(self) -> None:
        if self.summarization not in ("generic", "scientific"):
            raise ValueError("summarization must be generic or scientific")
        for name in ("clear_tool_results", "sequential_tools", "sanitize_history"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)
