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

"""Filesystem-backed prompt store (development / single-node use).

Directory layout::

    root_dir/
      my_agent/
        system/
          aliases.yml          ← mutable; the only file that changes
          1.0.0/
            prompt.j2          ← raw template (human-readable)
            meta.yml           ← PromptRecord metadata
          1.1.0/
            prompt.j2
            meta.yml
          1.1.0-opt.3/
            prompt.j2
            meta.yml

``aliases.yml`` maps alias labels to version strings, e.g.::

    latest: "1.1.0"
    stable: "1.0.0"

``meta.yml`` stores everything in :class:`~nat.prompt_store.model.PromptRecord`
except ``name``, ``version``, and ``content``.

Concurrency
-----------
Per-name :class:`asyncio.Lock` instances prevent concurrent writes to the
same prompt entry.  Reads are lock-free.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import yaml
from pydantic import Field

from nat.prompt_store.base import AliasNotFoundError
from nat.prompt_store.base import PromptNotFoundError
from nat.prompt_store.base import PromptStore
from nat.prompt_store.base import PromptVersionConflictError
from nat.prompt_store.model import PromptRecord
from nat.prompt_store.register import PromptStoreBaseConfig
from nat.prompt_store.register import register_prompt_store
from nat.prompt_store.registry import register_store

logger = logging.getLogger(__name__)

# Name of the aliases file inside each prompt directory
_ALIASES_FILE = "aliases.yml"
# Name of the template file inside each version directory
_PROMPT_FILE = "prompt.j2"
# Name of the metadata file inside each version directory
_META_FILE = "meta.yml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class FilesystemPromptStoreConfig(PromptStoreBaseConfig, name="filesystem_prompt_store"):
    """Configuration for the filesystem-backed prompt store.

    Example YAML::

        functions:
          local_prompts:
            _type: filesystem_prompt_store
            root_dir: ./prompts
            store_name: local
    """

    root_dir: str = Field(default="./prompts", description="Root directory for the prompt store")


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class FilesystemPromptStore(PromptStore):
    """Filesystem-backed :class:`~nat.prompt_store.base.PromptStore`.

    Args:
        root_dir: Root directory.  Created on first write if absent.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prompt_dir(self, name: str) -> Path:
        """Return the directory for a given prompt name.

        ``"/"`` in *name* becomes a directory separator, so
        ``"my_agent/system"`` maps to ``{root}/my_agent/system/``.
        """
        return self._root.joinpath(*name.split("/"))

    def _version_dir(self, name: str, version: str) -> Path:
        return self._prompt_dir(name) / version

    def _aliases_path(self, name: str) -> Path:
        return self._prompt_dir(name) / _ALIASES_FILE

    def _lock(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    def _require_prompt_dir(self, name: str) -> Path:
        """Return the on-disk directory for *name*, raising if it is absent."""
        pdir = self._prompt_dir(name)
        if not pdir.exists():
            raise PromptNotFoundError(f"Prompt '{name}' not found in store")
        return pdir

    def _sorted_version_dirs(self, pdir: Path) -> list[Path]:
        """Subdirectories of *pdir* that contain a prompt file, sorted by folder name."""
        return sorted(
            (d for d in pdir.iterdir() if d.is_dir() and (d / _PROMPT_FILE).exists()),
            key=lambda d: d.name,
        )

    async def _read_tags_from_version_dir(self, vdir: Path) -> tuple[str, ...]:
        """Load ``tags`` from the version's ``meta.yml`` only (no prompt body I/O)."""
        meta_path = vdir / _META_FILE
        if not meta_path.exists():
            return ()
        async with aiofiles.open(meta_path) as fh:
            raw = await fh.read()
        meta = yaml.safe_load(raw) or {}
        raw_tags = meta.get("tags", [])
        if isinstance(raw_tags, list):
            return tuple(str(t) for t in raw_tags)
        return ()

    # ------------------------------------------------------------------
    # Alias helpers
    # ------------------------------------------------------------------

    async def _read_aliases(self, name: str) -> dict[str, str]:
        path = self._aliases_path(name)
        if not path.exists():
            return {}
        async with aiofiles.open(path) as fh:
            raw = await fh.read()
        data = yaml.safe_load(raw) or {}
        return {str(k): str(v) for k, v in data.items()}

    async def _write_aliases(self, name: str, aliases: dict[str, str]) -> None:
        path = self._aliases_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w") as fh:
            await fh.write(yaml.safe_dump(aliases, default_flow_style=False))

    # ------------------------------------------------------------------
    # Version resolution
    # ------------------------------------------------------------------

    async def _resolve_version(self, name: str, version: str) -> str:
        """Return the canonical version string, resolving aliases if needed."""
        vdir = self._version_dir(name, version)
        if vdir.exists():
            return version
        # May be an alias
        aliases = await self._read_aliases(name)
        if version in aliases:
            return aliases[version]
        raise AliasNotFoundError(
            f"'{version}' is neither a version nor an alias for prompt '{name}'"
        )

    # ------------------------------------------------------------------
    # PromptStore interface
    # ------------------------------------------------------------------

    async def get(self, name: str, version: str = "latest") -> PromptRecord:
        self._require_prompt_dir(name)

        canonical = await self._resolve_version(name, version)
        vdir = self._version_dir(name, canonical)
        if not vdir.exists():
            raise PromptNotFoundError(
                f"Prompt '{name}' version '{canonical}' not found"
            )

        prompt_path = vdir / _PROMPT_FILE
        meta_path = vdir / _META_FILE

        async with aiofiles.open(prompt_path) as fh:
            content = await fh.read()

        meta: dict[str, Any] = {}
        if meta_path.exists():
            async with aiofiles.open(meta_path) as fh:
                raw = await fh.read()
            meta = yaml.safe_load(raw) or {}

        # Parse created_at – stored as ISO string
        created_at_raw = meta.get("created_at")
        if isinstance(created_at_raw, str):
            created_at = datetime.fromisoformat(created_at_raw)
        elif isinstance(created_at_raw, datetime):
            created_at = created_at_raw
        else:
            created_at = datetime.now(UTC)

        return PromptRecord(
            name=name,
            version=canonical,
            content=content,
            format=meta.get("format", "jinja2"),
            input_schema=meta.get("input_schema", {}),
            created_at=created_at,
            author=meta.get("author") or None,
            tags=tuple(meta.get("tags", [])),
            parent_version=meta.get("parent_version"),
            eval_scores=meta.get("eval_scores", {}),
            metadata=meta.get("metadata", {}),
        )

    async def put(self, record: PromptRecord) -> None:
        async with self._lock(record.name):
            vdir = self._version_dir(record.name, record.version)
            if vdir.exists():
                raise PromptVersionConflictError(
                    f"Prompt '{record.name}' version '{record.version}' already exists"
                )

            vdir.mkdir(parents=True, exist_ok=True)

            # Write template
            async with aiofiles.open(vdir / _PROMPT_FILE, "w") as fh:
                await fh.write(record.content)

            # Write metadata (everything except name, version, content)
            meta: dict[str, Any] = {
                "format": record.format,
                "input_schema": record.input_schema,
                "created_at": record.created_at.isoformat(),
                "tags": list(record.tags),
                "parent_version": record.parent_version,
                "eval_scores": record.eval_scores,
                "metadata": record.metadata,
            }
            if record.author is not None:
                meta["author"] = record.author
            async with aiofiles.open(vdir / _META_FILE, "w") as fh:
                await fh.write(yaml.safe_dump(meta, default_flow_style=False))

            # Update the 'latest' alias
            aliases = await self._read_aliases(record.name)
            aliases["latest"] = record.version
            await self._write_aliases(record.name, aliases)

        logger.debug("Stored prompt '%s' version '%s'", record.name, record.version)

    async def list_versions(self, name: str) -> list[str]:
        pdir = self._require_prompt_dir(name)
        vdirs = self._sorted_version_dirs(pdir)
        if not vdirs:
            raise PromptNotFoundError(f"No versions found for prompt '{name}'")
        return [d.name for d in vdirs]

    async def list_names(self, tag: str | None = None) -> list[str]:
        if not self._root.exists():
            return []

        names: list[str] = []
        for pdir in sorted(self._root.rglob(_ALIASES_FILE)):
            # Reconstruct logical name from relative path
            rel = pdir.parent.relative_to(self._root)
            name = "/".join(rel.parts)
            if tag is None:
                names.append(name)
            else:
                # Same behavior as before: tags from the first sorted version dir only.
                try:
                    pdir = self._require_prompt_dir(name)
                    vdirs = self._sorted_version_dirs(pdir)
                    if not vdirs:
                        continue
                    tags = await self._read_tags_from_version_dir(vdirs[0])
                    if tag in tags:
                        names.append(name)
                except PromptNotFoundError:
                    pass
        return names

    async def set_alias(self, name: str, alias: str, version: str) -> None:
        self._require_prompt_dir(name)

        vdir = self._version_dir(name, version)
        if not vdir.exists():
            raise PromptNotFoundError(
                f"Prompt '{name}' version '{version}' not found"
            )

        async with self._lock(name):
            aliases = await self._read_aliases(name)
            aliases[alias] = version
            await self._write_aliases(name, aliases)

    async def resolve_alias(self, name: str, alias: str) -> str:
        self._require_prompt_dir(name)

        aliases = await self._read_aliases(name)
        if alias not in aliases:
            raise AliasNotFoundError(
                f"Alias '{alias}' not set for prompt '{name}'"
            )
        return aliases[alias]

    async def delete(self, name: str, version: str) -> None:
        vdir = self._version_dir(name, version)
        if not vdir.exists():
            raise PromptNotFoundError(
                f"Prompt '{name}' version '{version}' not found"
            )

        async with self._lock(name):
            await asyncio.to_thread(shutil.rmtree, vdir)

        logger.debug("Deleted prompt '%s' version '%s'", name, version)


# ---------------------------------------------------------------------------
# NAT register function
# ---------------------------------------------------------------------------

@register_prompt_store(config_type=FilesystemPromptStoreConfig)
async def filesystem_prompt_store(config: FilesystemPromptStoreConfig, builder: Any):
    """Register a filesystem-backed prompt store with NAT.

    Populates the global prompt-store registry so agents can call
    ``get_prompt_store(config.store_name)`` at runtime.
    """
    from nat.builder.function_info import FunctionInfo

    store = FilesystemPromptStore(root_dir=config.root_dir)
    register_store(config.store_name, store)

    logger.info(
        "Filesystem prompt store '%s' registered at '%s'",
        config.store_name,
        config.root_dir,
    )

    async def _noop(query: str) -> str:
        """Prompt store (config-only, not a tool)."""
        return "This is a config-only function."

    yield FunctionInfo.from_fn(
        _noop,
        description=f"Filesystem prompt store '{config.store_name}' at '{config.root_dir}'",
    )
