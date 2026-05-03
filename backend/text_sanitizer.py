"""Utilities for sanitizing text before persistence."""


def sanitize_text(value: str | None) -> str:
    """Remove NUL and unsafe control characters while preserving common whitespace."""
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    cleaned_chars = []
    for ch in value:
        code = ord(ch)
        if ch in ("\n", "\r", "\t"):
            cleaned_chars.append(ch)
            continue
        if code == 0:
            continue
        if 0 <= code < 32 or 127 <= code < 160:
            continue
        cleaned_chars.append(ch)
    return "".join(cleaned_chars)
