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

    def __setitem__(self, other_key, other_val: Any):
        if not isinstance(other_key, ValueWrapper):
            other_key = ValueWrapper(other_key)
        self._items[other_key] = other_val

    def __getitem__(self, other_key):
        if not self.__contains__(other_key):
            if self._default_factory is not None:
                self[other_key] = self._default_factory()
            else:
                self[other_key] = None
        return self._items[other_key]

    def __and__(self, other_dict: ValueDict):
        ret = ValueDict()
        for key, val in self._items.items():
            if key in other_dict:
                ret[key] = val
        return ret

    def __or__(self, other_dict: ValueDict):
        return ValueDict(self._items | other_dict._items)

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self.keys()

    def __contains__(self, other_key):
        return other_key in self._items


class ValueSet:
    def __init__(
        self, iter: Sequence[ValueWrapper] | set[ValueWrapper] | None = None
    ):
        self._values: set[ValueWrapper] = set()
        if iter is not None:
            for val in iter:
                self.add(val)

    def add(self, other_val):
        other_val = ValueWrapper(other_val)
        if not self.__contains__(other_val):
            self._values.add(other_val)

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

    def __contains__(self, other_val):
        return other_val in self._values


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
