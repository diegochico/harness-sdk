"""Shared prompt-caching translation for OpenAI model providers.

OpenAI caches prompt prefixes automatically server-side and routes reads on a caller-supplied
``prompt_cache_key``. Cache-point *placement* (``prompt_cache_breakpoint``, or Anthropic-compatible
``cache_control`` under ``strategy="anthropic"``) is handled by ``OpenAIModel`` system-message
formatting, not here; this module maps only the remaining top-level ``CacheConfig`` fields,
``cache_key`` and, when it already names a valid retention literal, ``ttl``, onto the request.
"""

import warnings
from typing import Any

from ._validation import warn_on_cache_config_not_supported
from .model import CacheConfig

# OpenAI's prompt_cache_retention accepts only these literals. ttl maps through only on an exact
# match - the SDK never guesses a conversion from an arbitrary duration string.
#
# prompt_cache_retention is deprecated in openai 2.54.0 in favor of prompt_cache_options.ttl, whose only
# accepted value today is "30m" - so "24h"/"in_memory" remain expressible only through this field.
_RETENTION_LITERALS = frozenset({"in_memory", "24h"})


def apply_cache_config(request: dict[str, Any], cache_config: CacheConfig | None) -> None:
    """Map a ``CacheConfig`` onto an OpenAI request in place.

    An explicit value already present in ``request`` (carried in from the user's ``params``) always
    wins; this fills in only what ``params`` did not set. ``strategy`` is read by system-message
    formatting instead of this function, so it is accepted here without a warning even though this
    function itself does nothing with it. ``system_prompt_ttl`` is a genuine no-op (OpenAI has no
    auto-injection path), and a ``ttl`` that is not an OpenAI retention literal is ignored; each
    such no-op is surfaced through ``warnings.warn`` (deduped per call site by the standard
    library's default filter), matching the config-validation warnings in ``_validation.py``.

    Args:
        request: The request dict being assembled; mutated in place.
        cache_config: The provider's configured cache settings, if any.
    """
    if cache_config is None:
        return

    if cache_config.cache_key is not None and "prompt_cache_key" not in request:
        request["prompt_cache_key"] = cache_config.cache_key

    if cache_config.ttl is not None and "prompt_cache_retention" not in request:
        if cache_config.ttl in _RETENTION_LITERALS:
            request["prompt_cache_retention"] = cache_config.ttl
        else:
            warnings.warn(
                f"cache_config.ttl={cache_config.ttl!r} is not an openai retention value "
                "('in_memory' or '24h') and will be ignored",
                stacklevel=4,
            )

    warn_on_cache_config_not_supported(cache_config, "OpenAI", supported={"cache_key", "ttl", "strategy"})
