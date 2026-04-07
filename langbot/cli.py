"""CLI entry point for the bot project."""

import openai

RETRIABLE_EXCEPTIONS = (
    openai.RateLimitError,       # 429
    openai.APITimeoutError,      # request timed out
    openai.APIConnectionError,   # DNS/socket/TLS/connectivity issues
    openai.InternalServerError,  # 5xx from provider
)

def main() -> int:
    return 0
