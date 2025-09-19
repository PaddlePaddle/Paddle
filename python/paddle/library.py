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

#  #The file has been adapted from pytorch project
#  #Licensed under  BSD-style license -
#  https://github.com/pytorch/pytorch/blob/main/LICENSE

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Union, overload

from typing_extensions import TypeAlias

from ._ops import PYTHON_OP_REGISTRY

_DeviceTypes: TypeAlias = Union[str, Sequence[str], None]


def warn_about_unimplemented_torch_features(feature: str, fn_name: str) -> None:
    warnings.warn(
        f"The feature '{feature}' in function '{fn_name}' is not implemented in PaddlePaddle's custom operator interface.",
        UserWarning,
        stacklevel=2,
    )


class Tag: ...


class CustomOpDef:
    def __init__(
        self,
        namespace: str,
        name: str,
        schema: str,
        fn: Callable,
        tags: Sequence[Tag] | None = None,
    ) -> None:
        self._namespace = namespace
        self._name = name
        self._schema = schema
        self._fn = fn
        self._tags = tags if tags is not None else []

    @property
    def _qualname(self) -> str:
        return f"{self._namespace}::{self._name}"

    def __repr__(self) -> str:
        return f"<CustomOpDef({self._qualname})>"

    def register_fake(
        self, fn: Callable[..., object], /
    ) -> Callable[..., object]:
        warn_about_unimplemented_torch_features(
            "register_fake", "torch.library.CustomOpDef"
        )
        return fn


@overload
def custom_op(
    name: str,
    fn: None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: _DeviceTypes = None,
    schema: str | None = None,
    tags: Sequence[Tag] | None = None,
) -> Callable[[Callable[..., object]], CustomOpDef]: ...


@overload
def custom_op(
    name: str,
    fn: Callable[..., object],
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: _DeviceTypes = None,
    schema: str | None = None,
    tags: Sequence[Tag] | None = None,
) -> CustomOpDef: ...


def custom_op(
    name: str,
    fn: Callable[..., object] | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: _DeviceTypes = None,
    schema: str | None = None,
    tags: Sequence[Tag] | None = None,
) -> Callable[[Callable[..., object]], CustomOpDef] | CustomOpDef:
    if device_types:
        warn_about_unimplemented_torch_features(
            "device_types", "torch.library.custom_op"
        )
    if schema:
        warn_about_unimplemented_torch_features(
            "schema", "torch.library.custom_op"
        )
    if tags:
        warn_about_unimplemented_torch_features(
            "tags", "torch.library.custom_op"
        )

    assert "::" in name, (
        "The custom operator name should be qualified with a namespace, "
        "like 'my_namespace::my_op'."
    )
    namespace, op_name = name.split("::", 1)

    def inner(fn: Callable[..., object]) -> CustomOpDef:
        PYTHON_OP_REGISTRY.register(name, fn)
        return CustomOpDef(
            namespace=namespace,
            name=op_name,
            schema=schema if schema is not None else "",
            fn=fn,
            tags=tags,
        )

    if fn is None:
        return inner
    return inner(fn)


def register_fake(
    op: str | CustomOpDef,
    func: Callable[..., object] | None = None,
    /,
    *,
    lib: None = None,
    _stacklevel: int = 1,
    allow_override: bool = False,
):
    warn_about_unimplemented_torch_features(
        "register_fake", "torch.library.register_fake"
    )

    def register(func):
        return func

    if func is None:
        return register
    else:
        return register(func)
