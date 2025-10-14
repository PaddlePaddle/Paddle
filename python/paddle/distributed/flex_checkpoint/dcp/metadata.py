# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass


@dataclass(frozen=True)
class LocalTensorMetadata:
    """
    The location of a local tensor in the global tensor.
    """

    global_offset: tuple[int]
    local_shape: tuple[int]
    dtype: str
    global_shape: tuple[int] | None = None
    is_flattened: bool = False
    flattened_range: tuple[int] | None = None


@dataclass(frozen=True)
class LocalTensorIndex:
    """
    The identifier of a local tensor.
    """

    tensor_key: str
    global_offset: tuple[int]
    is_flattened: bool = False
    flattened_range: tuple[int] | None = None


@dataclass
class Metadata:
    state_dict_metadata: dict[str, list[LocalTensorMetadata]] = None
    storage_metadata: dict[LocalTensorIndex, str] = None
    flat_mapping: dict[str, tuple[str]] = None
