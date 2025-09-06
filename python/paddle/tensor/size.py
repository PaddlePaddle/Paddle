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
import functools
from collections.abc import Iterable, Sequence


class Size(tuple):
    """The result type of a call to ``paddle.Tensor.size()``.
    It describes the size of all dimensions of the original tensor. As a subclass of tuple,
    it supports all common sequence operations like indexing, slicing, concatenation, etc.

    Args:
        *args: Either a sequence of integers or multiple integer arguments representing dimensions.

    Returns:
        Size: A special tuple subclass representing tensor dimensions.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> size = paddle.Size([2, 3, 4])
            >>> print(size)
            paddle.Size([2, 3, 4])
    """

    def __new__(cls, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Sequence):
            seq = args[0]
        else:
            seq = args

        if len(seq) == 1 and hasattr(seq[0], 'ndim') and seq[0].ndim == 1:
            seq = seq[0].tolist()

        converted = []
        for item in seq:
            if hasattr(item, '__index__'):
                converted.append(int(item.__index__()))
            else:
                raise TypeError(
                    f"paddle.Size() takes an iterable of 'int' (got {type(item).__name__})"
                )

        return super().__new__(cls, converted)

    def __repr__(self):
        if not self:
            return "paddle.Size([])"
        return f"paddle.Size([{', '.join(map(str, self))}])"

    def __add__(self, other: Iterable):
        if isinstance(other, (tuple)):
            return Size(super().__add__(tuple(other)))
        raise TypeError(
            f"can only concatenate tuple (not {type(other).__name__}) to Size"
        )

    def __radd__(self, other: Iterable):
        if isinstance(other, (tuple)):
            return Size(tuple(other).__add__(self))
        raise TypeError(
            f"can only concatenate tuple (not {type(other).__name__}) to Size"
        )

    def __mul__(self, other: Iterable):
        if isinstance(other, int):
            return Size(super().__mul__(other))
        return NotImplemented

    __rmul__ = __mul__

    def numel(self):
        return functools.reduce(lambda x, y: x * y, self, 1)

    def __reduce__(self):
        return (Size, (tuple(self),))

    def __concat__(self, other: Iterable):
        if not isinstance(other, (tuple, Size)):
            raise TypeError(
                f"can only concatenate tuple (not {type(other).__name__}) to paddle.Size"
            )
        return self + other

    def __getitem__(self, key):
        from builtins import slice

        result = super().__getitem__(key)
        if isinstance(key, slice):
            return Size(result)
        return result
