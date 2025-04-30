# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = [
    'DimVar',
    'DTypeVar',
    'Tensor',
]


# Usage:
#   N = paddle.incubate.cc.typing.DimVar("N")
#   M = paddle.incubate.cc.typing.DimVar(4096)
class DimVar:
    def __init__(
        self,
        name_or_value: str | int,
        min: int | None = None,
        max: int | None = None,
    ):
        self.name_or_value = name_or_value
        self.min = min
        self.max = max


# Usage:
#   T = paddle.incubate.cc.typing.DTypeVar("T", "bfloat16", "float32")
class DTypeVar:
    def __init__(self, name: str, *candidates):
        assert len(candidates) > 0
        assert len(candidates) == len(set(candidates))
        for candidate in candidates:
            assert isinstance(candidate, str)
        self.name = str
        self.candidates = candidates


# Usage:
#
# import paddle.incubate.cc.typing as pct
# N = pct.DimVar("N")
# M = pct.DimVar("M")
# DType = pct.DTypeVar("T")
# def foo(x: paddle.cc.typing.Tensor([N, M], DType)):
#   ...
#
class Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
