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

"""NAT integration: PromptStoreRef, PromptStoreBaseConfig, PromptStoreMixin,
and @register_prompt_store.

This module also serves as the entry point for NAT component discovery;
importing it triggers registration of all built-in backends.

Usage – defining a backend::

    from nat.prompt_store.register import register_prompt_store, PromptStoreBaseConfig

    class MyStoreConfig(PromptStoreBaseConfig, name="my_prompt_store"):
        url: str

    @register_prompt_store(config_type=MyStoreConfig)
    async def my_prompt_store(config: MyStoreConfig, builder):
        store = MyPromptStore(url=config.url)
        register_store(config.store_name, store)
        async def _noop(query: str) -> str:
            return "prompt store"
        yield FunctionInfo.from_fn(_noop, description=f"Prompt store '{config.store_name}'")

Usage – adding prompt-store fields to an agent config::

    class MyAgentConfig(PromptStoreMixin, FunctionBaseConfig, name="my_agent"):
        llm: LLMRef = Field(...)

    # In the register function:
    prompts = await fetch_prompts(
        config.prompt_store,
        {"system": "my_agent/system"},
        config.prompt_versions,
    )
    system_prompt = prompts.get("system")
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Reference to a registered prompt store by name (mirrors ``LLMRef``).
#: Use as a field type in agent configs::
#:
#:     prompt_store: PromptStoreRef | None = Field(default=None)
PromptStoreRef = str


# ---------------------------------------------------------------------------
# Agent config mixin
# ---------------------------------------------------------------------------

class PromptStoreMixin(BaseModel):
    """Mixin that adds prompt-store fields to any agent config.

    Mix this in *before* :class:`~nat.data_models.function.FunctionBaseConfig`
    so the field definitions are visible to Pydantic::

        class MyConfig(PromptStoreMixin, FunctionBaseConfig, name="my_agent"):
            llm: LLMRef = Field(...)
    """

    prompt_store: PromptStoreRef | None = Field(
        default=None,
        description="Name of a registered prompt store to load prompts from. "
                    "If unset, prompts are loaded from local .j2 files (backwards-compatible).",
    )
    prompt_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Prompt name → version overrides, e.g. {\"orchestrator\": \"stable\"}. "
                    "Defaults to 'latest' for any prompt not listed here.",
    )


# ---------------------------------------------------------------------------
# Base config (for store backends)
# ---------------------------------------------------------------------------

class PromptStoreBaseConfig(FunctionBaseConfig):
    """Base configuration class for all prompt store backends.

    All concrete backend config classes should inherit from this.
    The ``store_name`` field is the key used to register the store in the
    global registry and the value agents put in ``prompt_store`` fields.

    Example YAML::

        functions:
          local_prompts:
            _type: filesystem_prompt_store
            root_dir: ./prompts
            store_name: local

          prod_prompts:
            _type: postgres_prompt_store
            dsn: postgresql://user:pass@host/db
            store_name: production
    """

    store_name: str = "default"


# ---------------------------------------------------------------------------
# Registration decorator
# ---------------------------------------------------------------------------

def register_prompt_store(
    config_type: type[PromptStoreBaseConfig],
) -> Callable[[Any], Any]:
    """Decorator for registering prompt store backends with NAT.

    Wraps :func:`nat.cli.register_workflow.register_function` and enforces
    that the config class is a :class:`PromptStoreBaseConfig` subclass.

    Args:
        config_type: The Pydantic config class for this backend.

    Returns:
        A decorator that registers the async generator function as a NAT
        function.

    Example::

        @register_prompt_store(config_type=FilesystemPromptStoreConfig)
        async def filesystem_prompt_store(config, builder):
            ...
            yield FunctionInfo.from_fn(_noop, description="...")
    """
    if not (isinstance(config_type, type) and issubclass(config_type, PromptStoreBaseConfig)):
        raise TypeError(
            f"register_prompt_store expects a PromptStoreBaseConfig subclass, got {config_type!r}"
        )

    def decorator(fn: Callable[..., AsyncGenerator]) -> Callable[..., AsyncGenerator]:
        return register_function(config_type=config_type)(fn)

    return decorator


# ---------------------------------------------------------------------------
# Import built-in backends (triggers registration at module load time)
# ---------------------------------------------------------------------------

# flake8: noqa
# isort:skip_file
from nat.prompt_store.backends import filesystem  # noqa: F401,E402
