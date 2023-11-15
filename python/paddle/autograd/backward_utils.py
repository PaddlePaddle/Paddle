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
from collections.abc import Sequence
from typing import Any


class ValueWrapper:
    def __init__(self, value) -> None:
        if isinstance(value, ValueWrapper):
            value = value.value
        self.value = value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other) -> bool:
        if isinstance(other, ValueWrapper):
            other = other.value
        if self.value is None:
            return other is None
        return self.value.is_same(other)


class ValueDict:
    def __init__(
        self,
        iter: dict[ValueWrapper, Any] | None = None,
        *,
        default_factory=None,
    ):
        self._items: dict[ValueWrapper, Any] = {}
        self._default_factory = default_factory
        if iter is not None:
            for key, val in iter.items():
                self[key] = val

    def update(self, other_dict):
        for key, val in other_dict:
            self[ValueWrapper(key)] = val

    def keys(self):
        for key in self._items.keys():
            yield key.value

    def values(self):
        return self._items.values()

    def items(self):
        for key, val in self._items.items():
            yield key.value, val

    def __setitem__(self, key, val: Any):
        if not isinstance(key, ValueWrapper):
            key = ValueWrapper(key)
        self._items[key] = val

    def __getitem__(self, key):
        if not self.__contains__(key):
            if self._default_factory is not None:
                self[key] = self._default_factory()
            else:
                raise KeyError(f'{key} not in ValueDict')
        return self._items[key]

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self.keys()

    def __contains__(self, key):
        if isinstance(key, ValueWrapper):
            return key in self._items
        return ValueWrapper(key) in self._items


class ValueSet:
    def __init__(
        self, iter: Sequence[ValueWrapper] | set[ValueWrapper] | None = None
    ):
        self._values: set[ValueWrapper] = set()
        if iter is not None:
            for val in iter:
                self.add(val)

    def add(self, val):
        val = ValueWrapper(val)
        if not self.__contains__(val):
            self._values.add(val)

    def update(self, other_set: set):
        for val in other_set:
            self.add(ValueWrapper(val))

    def __and__(self, other_set: ValueSet):
        ret = ValueSet()
        for val in self._values:
            if val in other_set:
                ret.add(val)
        return ret

    def __or__(self, other_set: ValueSet):
        return ValueSet(self._values | other_set._values)

    def __bool__(self):
        return bool(self._values)

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        for val in self._values:
            yield val.value

    def __contains__(self, val):
        if isinstance(val, ValueWrapper):
            return val in self._values
        return ValueWrapper(val) in self._values


class State:
    """
    record relationship of forward op/value and backward op/value
    one state must be bining with a program

    """

    def __init__(self, program):
        self.program = program
        # opresult -> list(list(opresult))
        self.value_to_valuegrad = ValueDict(default_factory=list)
        self.value_to_sumvaluegrad = ValueDict(default_factory=list)
        # operation -> list(operation)
        self.op_to_opgrad = collections.defaultdict(list)

        # opresult -> list(opresult)
        self.valuegrad_to_value = ValueDict(default_factory=list)
        self.sumvaluegrad_to_value = ValueDict(default_factory=list)
        # operation -> list(operation)
        self.opgrad_to_op = collections.defaultdict(list)

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
