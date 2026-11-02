# llm_providers/gemini.py
"""
Google Gemini provider via OpenAI-compatible endpoint.

Strips unsupported params (penalties, stop).
Gemini 2.5 thinking uses reasoning_effort.
Gemini 3 thinking uses extra_body.google.thinking_config instead.
Google rejects a request that contains both, and it rejects a top-level
`google` field. The OpenAI SDK merges its extra_body kwarg into the JSON
root, so the google block is nested once more here.

Gemini 3 function calls also return extra_content.google.thought_signature.
That block is captured on the tool call and sent back by
OpenAICompatProvider._sanitize_messages. Dropping it makes the tool
result request 400.
"""

import logging
from typing import Dict, Any

from .openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)

GEMINI_THINKING_PREFIXES = ('gemini-2.5', 'gemini-3')
# Gemini 3 accepts these thinking_level values. Anything else falls back
# to medium so an unknown effort string cannot 400 the request.
GEMINI_3_LEVELS = ('minimal', 'low', 'medium', 'high')


class GeminiProvider(OpenAICompatProvider):
    """Gemini via OpenAI-compatible API with thinking support."""

    @property
    def supports_images(self) -> bool:
        return True

    def __init__(self, llm_config: Dict[str, Any], request_timeout: float = 240.0):
        super().__init__(llm_config, request_timeout)
        self._thinking_enabled = llm_config.get('thinking_enabled', True)
        self._reasoning_effort = llm_config.get('reasoning_effort', 'medium')

    def _transform_params_for_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Strip unsupported params, add thinking config for 2.5+ models."""
        # Penalties/top_k/stop before the base lift (else top_k would ride
        # extra_body into an endpoint that doesn't take it).
        params = {k: v for k, v in (params or {}).items()
                  if k not in ('stop', 'frequency_penalty', 'presence_penalty',
                               'repeat_penalty', 'top_k')}
        result = super()._transform_params_for_model(params)

        # Gemini 3 rejects reasoning_effort together with thinking_config
        # ("found both"). Depth and thought summaries go in thinking_config
        # only. Gemini 2.5 has no thinking_config on this endpoint, so it
        # keeps reasoning_effort.
        model_lower = (self.model or '').lower()
        if self._thinking_enabled and any(model_lower.startswith(p) for p in GEMINI_THINKING_PREFIXES):
            if model_lower.startswith('gemini-3'):
                level = self._reasoning_effort if self._reasoning_effort in GEMINI_3_LEVELS else 'medium'
                result['extra_body'] = {
                    "extra_body": {
                        "google": {
                            "thinking_config": {
                                "thinking_level": level,
                                "include_thoughts": True,
                            }
                        }
                    }
                }
            else:
                result['reasoning_effort'] = self._reasoning_effort
            logger.info(f"[GEMINI] Thinking enabled (effort={self._reasoning_effort}) for {self.model}")

        return result
