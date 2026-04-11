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

"""Versioned prompt store for NAT agents.

Provides immutable, versioned storage of prompt templates with mutable
aliases (e.g. ``latest``, ``stable``), pluggable backends, and a
global registry that agents can query at runtime.

Quickstart::

    from nat.prompt_store import get_prompt_store

    store = get_prompt_store("my_store")
    record = await store.get("deep_researcher/orchestrator", version="stable")
    rendered = await store.render("deep_researcher/orchestrator", version="stable",
                                   current_datetime="2026-01-01")
"""

from .base import AliasNotFoundError
from .base import PromptNotFoundError
from .base import PromptStore
from .base import PromptStoreError
from .base import PromptVersionConflictError
from .base import PromptVersionProtectedError
from .model import PromptRecord
from .register import PromptStoreBaseConfig
from .register import PromptStoreMixin
from .register import PromptStoreRef
from .register import register_prompt_store
from .registry import fetch_prompts
from .registry import get_prompt_store
from .registry import get_prompt_store_or_none
from .registry import list_registered_stores
from .registry import register_store
from .registry import reset_registry

__all__ = [
    "AliasNotFoundError",
    "PromptNotFoundError",
    "PromptRecord",
    "PromptStore",
    "PromptStoreBaseConfig",
    "PromptStoreError",
    "PromptStoreMixin",
    "PromptStoreRef",
    "PromptVersionConflictError",
    "PromptVersionProtectedError",
    "fetch_prompts",
    "get_prompt_store",
    "get_prompt_store_or_none",
    "list_registered_stores",
    "register_prompt_store",
    "register_store",
    "reset_registry",
]
