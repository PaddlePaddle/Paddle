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


class DimVar:
    r"""
    Dimension expression used in PCC.

    Parameters:
        name_or_value (str|int): the name or value of the dimension.
        min (int, optional): the min value of name_or_value. It is only used when name_or_value is a str.
        max (int, optional): the max value of name_or_value. It is only used when name_or_value is a str.

    Examples:
        .. code-block:: python

            >>> import paddle.incubate.cc.typing as pct
            >>> M = pct.DimVar(128)
            >>> N = pct.DimVar('N', min=32)
    """

    def __init__(
        self,
        name_or_value: str | int,
        min: int | None = None,
        max: int | None = None,
    ):
        self.name_or_value = name_or_value
        self.min = min
        self.max = max


class DTypeVar:
    r"""
    Data type expression used in PCC.

    Parameters:
        name (str): the name of the data type.

    Examples:
        .. code-block:: python

            >>> import paddle.incubate.cc.typing as pct
            >>> T = pct.DTypeVar("T", "float32")
    """

    def __init__(self, name: str, *candidates):
        assert len(candidates) > 0
        assert len(candidates) == len(set(candidates))
        for candidate in candidates:
            assert isinstance(candidate, str)
        self.name = str
        self.candidates = candidates


class Tensor:
    r"""
    Tensor expression used in PCC.

    Parameters:
        shape (DimVar): dimension expression of the tensor.
        dtype (DTypeVar): data type of the tensor.

    Examples:
        .. code-block:: python

            >>> import paddle.incubate.cc.typing as pct

            >>> M = pct.DimVar("M")
            >>> N = pct.DimVar("N")
            >>> DType = pct.DTypeVar("T")

            >>> def foo(x: pct.Tensor([N, M], DType), y: pct.Tensor([N, M], DType)):
            >>>     return x + y
    """

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
