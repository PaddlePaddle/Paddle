#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import Tensor
from paddle.framework import (
    in_dynamic_mode,
)


class _CompatClassMeta(type):
    """Keep compat classes recognizable as native classes across dispatch."""

    def __new__(mcls, name, bases, namespace, *, native_cls=None, **kwargs):
        if native_cls is None:
            module_parts = namespace.get('__module__', '').split('.')
            if (
                module_parts[:2] == ['paddle', 'compat']
                and len(module_parts) > 2
            ):
                native_module = getattr(paddle, module_parts[2], None)
                native_cls = getattr(native_module, name, None)
        if isinstance(native_cls, type) and not any(
            issubclass(base, native_cls) for base in bases
        ):
            bases = tuple(
                native_cls if issubclass(native_cls, base) else base
                for base in bases
            )
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __instancecheck__(cls, instance: object) -> bool:
        native_cls = cls.__dict__.get('__native_cls__')
        if native_cls is not None:
            return isinstance(instance, native_cls)
        return super().__instancecheck__(instance)

    def __subclasscheck__(cls, subclass: type) -> bool:
        native_cls = cls.__dict__.get('__native_cls__')
        if native_cls is not None:
            return issubclass(subclass, native_cls)
        return super().__subclasscheck__(subclass)


def _check_out_status(
    out: Tensor | tuple[Tensor, Tensor] | list[Tensor],
    expect_multiple: bool = False,
):
    if out is None:
        return
    if not in_dynamic_mode():
        raise RuntimeError(
            "Using `out` static graph CINN backend is currently not supported. Directly return the tensor tuple instead.\n"
        )
    if expect_multiple:
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise TypeError(
                f"Expected a list or tuple of two tensors, got {type(out)} instead."
            )
        if not (
            isinstance(out[0], paddle.Tensor)
            and isinstance(out[1], paddle.Tensor)
        ):
            raise TypeError(
                f"Expected Tensor type in the tuple/list, got ({type(out[0])}, {type(out[1])}) instead."
            )
    else:
        if not isinstance(out, paddle.Tensor):
            raise TypeError(f"Expected a Tensor, got {type(out)} instead.")
