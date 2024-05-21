# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from paddle import (
        CPUPlace,
        CUDAPinnedPlace,
        CUDAPlace,
        CustomPlace,
        IPUPlace,
        XPUPlace,
    )

PlaceLike: TypeAlias = Union[
    "CPUPlace",
    "CUDAPlace",
    "CUDAPinnedPlace",
    "IPUPlace",
    "CustomPlace",
    "XPUPlace",
    str,  # some string like "cpu", "gpu:0", etc.
]
