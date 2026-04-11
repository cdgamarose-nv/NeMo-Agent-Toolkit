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

"""Abstract PromptStore interface and exception hierarchy."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from .model import PromptRecord


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PromptStoreError(Exception):
    """Base class for all prompt-store errors."""


class PromptNotFoundError(PromptStoreError):
    """Raised when a prompt name or version does not exist in the store."""


class PromptVersionConflictError(PromptStoreError):
    """Raised when trying to put() a version that already exists (write-once)."""


class PromptVersionProtectedError(PromptStoreError):
    """Raised when trying to delete() a version that is referenced by an alias."""


class AliasNotFoundError(PromptStoreError):
    """Raised when resolving an alias that has not been set."""


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class PromptStore(ABC):
    """Async interface for a versioned, immutable prompt store.

    All methods are coroutines.  Concrete backends (filesystem, Postgres, …)
    inherit from this class and implement every abstract method.

    The :meth:`render` method is provided as a concrete convenience on top of
    :meth:`get`; backends only need to implement the storage primitives.
    """

    # ------------------------------------------------------------------
    # Storage primitives (must be implemented by backends)
    # ------------------------------------------------------------------

    @abstractmethod
    async def get(self, name: str, version: str = "latest") -> PromptRecord:
        """Retrieve a prompt record.

        If *version* is an alias (e.g. ``"latest"`` or ``"stable"``) it is
        resolved to the canonical version string first.

        Args:
            name: Logical prompt name (e.g. ``"deep_researcher/orchestrator"``).
            version: Exact version string or alias. Defaults to ``"latest"``.

        Returns:
            The matching :class:`~nat.prompt_store.model.PromptRecord`.

        Raises:
            PromptNotFoundError: If the name or resolved version does not exist.
            AliasNotFoundError: If *version* is an alias that has not been set.
        """

    @abstractmethod
    async def put(self, record: PromptRecord) -> None:
        """Store a new prompt version (write-once).

        After a successful ``put`` the ``latest`` alias is updated to point to
        the new version.

        Args:
            record: The :class:`PromptRecord` to store.

        Raises:
            PromptVersionConflictError: If the version already exists.
        """

    @abstractmethod
    async def list_versions(self, name: str) -> list[str]:
        """Return all version strings for a prompt, in creation order.

        Args:
            name: Logical prompt name.

        Returns:
            List of version strings (oldest first).

        Raises:
            PromptNotFoundError: If no versions exist for *name*.
        """

    @abstractmethod
    async def list_names(self, tag: str | None = None) -> list[str]:
        """Return all prompt names registered in the store.

        Args:
            tag: If given, only return prompts that carry this tag.

        Returns:
            Sorted list of logical prompt names.
        """

    @abstractmethod
    async def set_alias(self, name: str, alias: str, version: str) -> None:
        """Point a mutable alias to a specific version.

        Args:
            name: Logical prompt name.
            alias: Alias label (e.g. ``"stable"``).
            version: Exact version string the alias should point to.

        Raises:
            PromptNotFoundError: If *name* / *version* do not exist.
        """

    @abstractmethod
    async def resolve_alias(self, name: str, alias: str) -> str:
        """Resolve an alias to its current version string.

        Args:
            name: Logical prompt name.
            alias: Alias label.

        Returns:
            The version string the alias currently points to.

        Raises:
            PromptNotFoundError: If *name* does not exist.
            AliasNotFoundError: If *alias* has not been set.
        """

    @abstractmethod
    async def delete(self, name: str, version: str) -> None:
        """Delete a specific version (leaves aliases unchanged).

        Aliases that previously pointed to the deleted version will
        *dangle*; callers must update them if desired.

        Args:
            name: Logical prompt name.
            version: Exact version string to delete.

        Raises:
            PromptNotFoundError: If *name* / *version* do not exist.
            PromptVersionProtectedError: If backend policy prevents deletion.
        """

    # ------------------------------------------------------------------
    # Convenience (provided; backends may override for efficiency)
    # ------------------------------------------------------------------

    async def render(self, name: str, /, version: str = "latest", **kwargs: Any) -> str:
        """Fetch a prompt and render its template.

        Uses Jinja2 for ``jinja2`` format records, Python ``str.format``
        for ``f-string`` records, and returns the raw content unchanged for
        all other formats.

        *name* is positional-only so template variables whose key is ``"name"``
        do not conflict with this parameter.

        Args:
            name: Logical prompt name (positional-only).
            version: Exact version string or alias. Defaults to ``"latest"``.
            **kwargs: Variables passed to the template engine.

        Returns:
            The rendered prompt string.
        """
        record = await self.get(name, version)
        return _render_content(record.content, record.format, **kwargs)


def _render_content(content: str, fmt: str, **kwargs: Any) -> str:
    """Render *content* using the appropriate template engine."""
    if fmt == "jinja2":
        import jinja2
        template = jinja2.Template(content, undefined=jinja2.StrictUndefined)
        return template.render(**kwargs)
    if fmt == "f-string":
        return content.format(**kwargs)
    # mustache or unknown – return as-is; callers can do their own rendering
    return content
