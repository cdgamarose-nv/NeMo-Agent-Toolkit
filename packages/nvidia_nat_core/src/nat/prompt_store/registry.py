# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global prompt-store registry.

A module-level dict is populated at startup by NAT register functions and
queried at runtime by agents and optimizers.

Example::

    from nat.prompt_store.registry import get_prompt_store

    store = get_prompt_store("production_prompts")
    record = await store.get("my_agent/system", "stable")
"""

from __future__ import annotations

import logging

from .base import PromptStore

logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
_stores: dict[str, PromptStore] = {}


# ── Registration ──────────────────────────────────────────────────────────────

def register_store(name: str, store: PromptStore) -> None:
    """Register a :class:`PromptStore` instance under *name*.

    Called by the NAT register function during startup; usually not invoked
    directly by application code.

    Args:
        name: Unique store name (matches ``store_name`` in YAML config).
        store: Concrete :class:`PromptStore` instance.
    """
    if name in _stores:
        logger.debug("Replacing existing prompt store registered as %r", name)
    _stores[name] = store


# ── Lookup ────────────────────────────────────────────────────────────────────

def get_prompt_store(name: str) -> PromptStore:
    """Return the registered store with *name*.

    Args:
        name: Store name as declared in the YAML config.

    Returns:
        The registered :class:`PromptStore` instance.

    Raises:
        KeyError: If no store has been registered under *name*.
    """
    if name not in _stores:
        raise KeyError(
            f"Prompt store '{name}' is not registered. "
            f"Check that a prompt_store function with store_name={name!r} "
            "appears in your NAT config."
        )
    return _stores[name]


def get_prompt_store_or_none(name: str) -> PromptStore | None:
    """Return the registered store with *name*, or ``None`` if not found."""
    return _stores.get(name)


def list_registered_stores() -> list[str]:
    """Return the names of all registered stores."""
    return sorted(_stores)


# ── Prompt fetching helper ────────────────────────────────────────────────────

async def fetch_prompts(
    store_name: str | None,
    prompt_map: dict[str, str],
    versions: dict[str, str],
) -> dict[str, str]:
    """Fetch a set of prompts from a registered store.

    This is the single shared implementation used by all agent register
    functions.  It handles the store-not-configured case, the store-not-
    registered warning, per-prompt errors, and INFO logging.

    Args:
        store_name: Value of ``config.prompt_store`` (``None`` = disabled).
        prompt_map: Maps short prompt name → store key, e.g.
            ``{"system": "my_agent/system"}``.
        versions: Value of ``config.prompt_versions``; missing keys default
            to ``"latest"``.

    Returns:
        ``{short_name: content}`` for every prompt that was fetched
        successfully.  Returns an empty dict when *store_name* is ``None``
        or the store is not registered.  Individual fetch failures are
        logged as warnings and omitted from the result, causing the agent
        to fall back to its local ``.j2`` file for that prompt.
    """
    if not store_name:
        return {}

    store = get_prompt_store_or_none(store_name)
    if store is None:
        logger.warning(
            "Prompt store '%s' is not registered; falling back to local .j2 files.",
            store_name,
        )
        return {}

    result: dict[str, str] = {}
    for short_name, store_key in prompt_map.items():
        version = versions.get(short_name, "latest")
        try:
            record = await store.get(store_key, version)
            result[short_name] = record.content
            logger.info(
                "Loaded prompt '%s' version '%s' from store '%s'",
                store_key,
                version,
                store_name,
            )
        except Exception:
            logger.warning(
                "Failed to load prompt '%s' (version '%s') from store '%s'; "
                "falling back to local .j2 file.",
                store_key,
                version,
                store_name,
                exc_info=True,
            )

    return result


# ── Test helpers ──────────────────────────────────────────────────────────────

def reset_registry() -> None:
    """Clear all registered stores (for testing)."""
    _stores.clear()
