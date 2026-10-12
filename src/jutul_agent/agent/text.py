"""Text cleanup shared by compaction and evaluation output."""

import re

_INTERNAL_CAPSULE = re.compile(
    r"<think\b[^>]*>.*?</think>|<content-internal>.*?</content-internal>",
    re.DOTALL,
)


def strip_internal_capsules(text: str) -> str:
    """Remove serialized reasoning/bridge capsules without changing whitespace."""
    return _INTERNAL_CAPSULE.sub("", text)
