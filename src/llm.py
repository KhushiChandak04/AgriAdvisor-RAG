"""Groq LLM wrapper with clear network/SSL error messages."""
# Use the Windows certificate store (fixes antivirus / corporate SSL interception)
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import os

import httpx
from groq import (APIConnectionError, APIStatusError, AuthenticationError,
                  Groq, RateLimitError)
from langsmith import traceable

from . import config


def _client() -> Groq:
    kwargs = dict(api_key=config.GROQ_API_KEY, timeout=60.0, max_retries=2)
    # Last resort if your network still breaks SSL: set GROQ_INSECURE_SSL=1 in .env
    if os.getenv("GROQ_INSECURE_SSL", "0") == "1":
        kwargs["http_client"] = httpx.Client(verify=False, timeout=60.0)
    return Groq(**kwargs)


@traceable(name="groq_chat", run_type="llm", project_name=config.LANGSMITH_PROJECT)
def chat(messages, temperature: float = 0.2, max_tokens: int = 2500) -> str:
    if not config.GROQ_API_KEY or config.GROQ_API_KEY.startswith("your_"):
        raise RuntimeError("GROQ_API_KEY is missing. Add it to the .env file and restart the app.")

    params = dict(model=config.GROQ_MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if "gpt-oss" in config.GROQ_MODEL.lower():
        params["extra_body"] = {"reasoning_effort": "low"}

    try:
        resp = _client().chat.completions.create(**params)
    except APIConnectionError as e:
        cause = repr(getattr(e, "__cause__", None) or e)
        raise RuntimeError(
            "Could not reach Groq (api.groq.com). Check: internet / VPN / firewall / antivirus "
            "HTTPS scanning. If it is an SSL problem, add GROQ_INSECURE_SSL=1 to .env and restart. "
            f"Details: {cause}")
    except AuthenticationError:
        raise RuntimeError("Groq rejected the API key. Check GROQ_API_KEY in .env.")
    except RateLimitError:
        raise RuntimeError("Groq rate limit reached. Wait a minute and try again.")
    except APIStatusError as e:
        raise RuntimeError(f"Groq error {e.status_code}: {e.message}")

    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError("The model returned an empty answer. Please try again.")
    return text