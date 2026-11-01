# -*- coding: utf-8 -*-
"""interject — mid-stream context injection for autoregressive decoders."""

from .runtime import (
    SUPPLEMENT_STYLES,
    SYSTEM_PROMPT,
    format_supplement,
    generate_chat_turn,
)

__version__ = "0.1.0"
__all__ = [
    "SUPPLEMENT_STYLES",
    "SYSTEM_PROMPT",
    "format_supplement",
    "generate_chat_turn",
    "__version__",
]
