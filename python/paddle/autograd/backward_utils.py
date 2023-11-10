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

from typing import Any


class ValueDict:
    def __init__(self, iter=None, *, default_factory=None):
        self._items: list[tuple[Any, Any]] = []
        self._default_factory = default_factory
        if iter is not None:
            for key, val in iter:
                self[key] = val

    def update(self, other_dict):
        for key, val in other_dict:
            self[key] = val

    def keys(self):
        for key, _ in self._items:
            yield key

    def values(self):
        for _, val in self._items:
            yield val

    def items(self):
        yield from self._items

    def __setitem__(self, other_key, other_val: Any):
        if self.__contains__(other_key):
            for i, (key, val) in enumerate(self._items):
                if hash(key) == hash(other_key) and key.is_same(other_key):
                    self._items[i] = (other_key, other_val)
                    break
        else:
            self._items.append((other_key, other_val))

    def __getitem__(self, other_key):
        for key, val in self._items:
            if hash(key) == hash(other_key) and key.is_same(other_key):
                return val

        if self._default_factory is not None:
            val = self._default_factory()
            self._items.append((other_key, val))
            return val
        else:
            return None

    def __and__(self, other_dict: "ValueDict"):
        ret = ValueDict()
        for key, val in self._items:
            if key in other_dict:
                ret[key] = val
        return ret

    def __or__(self, other_dict: "ValueDict"):
        return ValueDict(self._items + other_dict._items)

    def __bool__(self):
        return bool(self._items)

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self.keys()

    def __contains__(self, other_key):
        for key, _ in self._items:
            if hash(key) == hash(other_key) and key.is_same(other_key):
                return True
        return False


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
        self.op_to_opgrad = ValueDict(default_factory=list)

        # opresult -> list(opresult)
        self.valuegrad_to_value = ValueDict(default_factory=list)
        self.sumvaluegrad_to_value = ValueDict(default_factory=list)
        # operation -> list(operation)
        self.opgrad_to_op = ValueDict(default_factory=list)

    def turn_map(self) -> None:
        self.valuegrad_to_value = ValueDict(default_factory=list)
        self.sumvaluegrad_to_value = ValueDict(default_factory=list)
        self.opgrad_to_op = ValueDict(default_factory=list)

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
