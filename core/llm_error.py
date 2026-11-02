"""Plain-language text from an LLM client exception.

OpenAI-compatible clients attach the API's JSON body and then stringify the
whole object. Settings and probes want the message field, not that dump.
"""


def clean_llm_error(e) -> str:
    """Pull the useful message out of an LLM/HTTP error (esp. OpenAI-client 400s)."""
    body = getattr(e, 'body', None)
    if isinstance(body, dict):
        inner = body.get('error') if isinstance(body.get('error'), dict) else body
        msg = (inner or {}).get('message')
        if msg:
            return str(msg)[:250]
    return str(e)[:250]
