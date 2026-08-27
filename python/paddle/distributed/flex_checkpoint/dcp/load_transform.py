# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from paddle import Tensor, dtype

    from .metadata import LocalTensorMetadata


class LoadTransform(Protocol):
    """Format-independent extension point for checkpoint load transforms.

    A transform exposes virtual logical tensors to AOA, lists the physical
    checkpoint tensors required to materialize each logical tensor, and runs
    only after those physical tensors have been fully assembled. Implementers
    may additionally provide read_plan() and read_plan_for() to request local
    physical component shards; the three methods below form the required
    contract.
    """

    def logical_metadata(self) -> dict[str, LocalTensorMetadata]: ...

    def source_keys(self, logical_key: str) -> list[str]: ...

    def apply(
        self,
        logical_key: str,
        source_tensors: dict[str, Tensor],
        output_dtype: dtype,
    ) -> Tensor: ...
