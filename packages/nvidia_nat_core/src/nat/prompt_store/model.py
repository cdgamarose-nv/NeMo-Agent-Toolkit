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

"""PromptRecord – the core versioned prompt data model."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class PromptRecord:
    """An immutable, versioned snapshot of a prompt template.

    Once written to a :class:`PromptStore` a record is never modified.
    Mutable aliases (e.g. ``latest``, ``stable``) point to a specific
    version string and can be updated independently.

    Attributes:
        name: Logical identifier, may contain ``/`` for namespacing
              (e.g. ``my_agent/system``).
        version: Semantic version string (e.g. ``1.2.0``, ``1.2.0-opt.3``).
        content: Raw template string (Jinja2, f-string, or plain text).
        format: Template engine – ``jinja2`` (default), ``f-string``, or
                ``mustache``.
        input_schema: JSON Schema dict describing expected template variables.
        created_at: UTC timestamp when this record was created.
        author: Optional free-form author identifier (user, process, optimizer run).
        tags: Searchable tags (e.g. ``draft``, ``stable``, ``optimized``).
        parent_version: Version this record was derived from (lineage).
        eval_scores: Benchmark results keyed by evaluator name
                     (e.g. ``{"freshqa": 0.83}``).
        metadata: Arbitrary key/value pairs (optimizer run IDs, hyperparams…).
    """

    name: str
    version: str
    content: str
    format: Literal["jinja2", "f-string", "mustache"] = "jinja2"
    input_schema: dict = field(default_factory=dict, compare=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC), compare=False)
    author: str | None = field(default=None, compare=False)
    tags: tuple[str, ...] = field(default_factory=tuple, compare=False)
    parent_version: str | None = field(default=None, compare=False)
    eval_scores: dict[str, float] = field(default_factory=dict, compare=False)
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PromptRecord.name must not be empty")
        if not self.version:
            raise ValueError("PromptRecord.version must not be empty")
        # Normalise tags to tuple so the frozen dataclass stays hashable
        if isinstance(self.tags, list):
            object.__setattr__(self, "tags", tuple(self.tags))
