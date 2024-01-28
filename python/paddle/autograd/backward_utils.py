# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,tes
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import collections
import warnings
from collections.abc import Sequence
from typing import Any

from paddle import pir
from paddle.base import core
from paddle.base.wrapped_decorator import signature_safe_contextmanager


class ValueWrapper:
    def __init__(self, value) -> None:
        if isinstance(value, ValueWrapper):
            assert isinstance(value._value, (type(None), pir.Value))
        else:
            assert isinstance(value, (type(None), pir.Value))
        self._value = value._value if isinstance(value, ValueWrapper) else value

    def __hash__(self) -> int:
        if isinstance(self._value, pir.Value):
            return self._value.hash()
        else:
            return hash(self._value)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ValueWrapper):
            warnings.warn(
                f'In ValueWrapper.__eq__ expected type of `other` is ValueWrapper but received {other.__class__}.'
            )
            return False

        if self._value is None or other._value is None:
            return self._value is None and other._value is None
        return self._value.is_same(other._value)


class ValueDict:
    def __init__(
        self,
        iter=None,
        *,
        default_factory=None,
    ):
        self._items: dict[ValueWrapper] = {}
        self._default_factory = default_factory
        if iter is not None:
            for key, val in iter.items():
                self[key] = val

    def copy(self):
        ret = ValueDict()
        ret._items = self._items.copy()
        ret._default_factory = self._default_factory
        return ret

    def update(self, other_dict):
        for key, val in other_dict.items():
            self[key] = val

    def keys(self):
        for key in self._items.keys():
            yield key._value

    def values(self):
        return self._items.values()

    def items(self):
        for key, val in self._items.items():
            yield key._value, val

    def pop(self, key):
        if not self.__contains__(key):
            raise KeyError(f'{key} is not in ValueDict')
        return self._items.pop(ValueWrapper(key))

    def __setitem__(self, key, val: Any):
        self._items[ValueWrapper(key)] = val

    def __getitem__(self, key):
        if not self.__contains__(key):
            if self._default_factory is not None:
                self[key] = self._default_factory()
            else:
                raise KeyError(f'{key} is not in ValueDict')
        return self._items[ValueWrapper(key)]

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        return ValueWrapper(key) in self._items

    def __repr__(self) -> str:
        items_str = ", ".join(f"{key}: {val}" for key, val in self.items())
        return f'ValueDict({items_str})'


class ValueSet:
    def __init__(
        self, iter: Sequence[ValueWrapper] | set[ValueWrapper] | None = None
    ):
        self._set: set[ValueWrapper] = set()
        if iter is not None:
            for val in iter:
                self.add(val)

    def copy(self):
        ret = ValueSet()
        ret._set = self._set.copy()
        return ret

    def add(self, val):
        if not self.__contains__(val):
            self._set.add(ValueWrapper(val))

    def update(self, other: set):
        for val in other:
            self.add(val)

    def __and__(self, other: ValueSet):
        return ValueSet(self._set & other._set)

    def __or__(self, other: ValueSet):
        return ValueSet(self._set | other._set)

    def __bool__(self):
        return bool(self._set)

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        for val in self._set:
            yield val._value

    def __contains__(self, val):
        return ValueWrapper(val) in self._set

    def __repr__(self) -> str:
        items_str = ", ".join(repr(item) for item in self)
        return f'ValueSet({items_str})'


class State:
    """
    record relationship of forward op/value and backward op/value
    one state must be bining with a block, if block has parent block,
    state will include parent block info.

    """

    def __init__(self, block):
        self.block = block
        # value -> list(list(value))
        self.value_to_valuegrad = ValueDict(default_factory=list)
        self.value_to_sumvaluegrad = ValueDict(default_factory=list)
        # operation -> list(operation)
        self.op_to_opgrad = collections.defaultdict(list)

        # value -> list(value)
        self.valuegrad_to_value = ValueDict(default_factory=list)
        self.sumvaluegrad_to_value = ValueDict(default_factory=list)
        # operation -> list(operation)
        self.opgrad_to_op = collections.defaultdict(list)
        # only for controlflow
        # inside_value is sub block value, which will yield to parent block,
        # parant block value is outside_value
        self.inside_value_to_outside_value_map = ValueDict()

    def turn_map(self) -> None:
        self.valuegrad_to_value = ValueDict(default_factory=list)
        self.sumvaluegrad_to_value = ValueDict(default_factory=list)
        self.opgrad_to_op = collections.defaultdict(list)

        for k, v in self.value_to_valuegrad.items():
            if v != []:
                for value in v[0]:
                    self.valuegrad_to_value[value] = [k]
        for k, v in self.value_to_sumvaluegrad.items():
            if v != []:
                for value in v[0]:
                    self.sumvaluegrad_to_value[value] = [k]
        for k, v in self.op_to_opgrad.items():
            if v != []:
                self.opgrad_to_op[v[0]] = [k]

    def copy(self, new_block):
        state = State(new_block)
        state.value_to_valuegrad = self.value_to_valuegrad.copy()
        state.value_to_sumvaluegrad = self.value_to_sumvaluegrad.copy()

        # operation -> list(operation)
        state.op_to_opgrad = self.op_to_opgrad.copy()

        # value -> list(value)
        state.valuegrad_to_value = self.valuegrad_to_value.copy()
        state.sumvaluegrad_to_value = self.sumvaluegrad_to_value.copy()
        # operation -> list(operation)
        state.opgrad_to_op = self.opgrad_to_op.copy()

        # only for controlflow
        state.inside_value_to_outside_value_map = (
            self.inside_value_to_outside_value_map.copy()
        )

        return state


def _check_vjp_dynamic_shape(op, inputs):
    for items in inputs:
        for item in items:
            shape = item.shape
            if -1 in shape:
                warnings.warn(
                    f"[Prim] Decomp op does not support dynamic shape -1, but got shape {item.shape} in inputs of op {op.name()} . Prim will skip its vjp op."
                )
                return True


# Prim currently does not support dynamic shape, when dynamic shape exits in shape of op inputs, prim will be skipped its vjp op.
@signature_safe_contextmanager
def dynamic_shape_prim_vjp_guard(op, inputs):
    skip_prim = (
        core._is_bwd_prim_enabled()
        and core._enable_prim_dynamic_shape()
        and _check_vjp_dynamic_shape(op, inputs)
    )
    try:
        if skip_prim:
            core._set_prim_backward_enabled(False)
        yield
    finally:
        if skip_prim:
            core._set_prim_backward_enabled(True)
