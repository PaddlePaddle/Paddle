# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any, Callable, Generic, TypeVar

from typing_extensions import Concatenate, ParamSpec, TypeAlias

P = ParamSpec("P")
R = TypeVar("R")

MutableDataT = TypeVar("MutableDataT", bound="MutableData")
DataGetter: TypeAlias = Callable[[MutableDataT, Any], Any]

InnerMutableDataT = TypeVar(
    "InnerMutableDataT", bound="dict[str, Any] | list[Any]"
)


class Mutation:
    ABBR: str


class MutationSet(Mutation):
    """
    Setting a value.
    This mutation is used for MutableDictLikeData and MutableListLikeData.
    """

    ABBR = "S"

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __repr__(self):
        return f"MutationSet({self.key}, {self.value})"


class MutationDel(Mutation):
    """
    Deleting a value.
    This mutation is used for MutableDictLikeData and MutableListLikeData.
    """

    ABBR = "D"

    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f"MutationDel({self.key})"


class MutationNew(Mutation):
    """
    Adding a new value.
    This mutation is only used for MutableDictLikeData.
    """

    ABBR = "N"

    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __repr__(self):
        return f"MutationNew({self.key}, {self.value})"


class MutationInsert(Mutation):
    """
    Inserting a value.
    This mutation is only used for MutableListLikeData.
    """

    ABBR = "I"

    def __init__(self, index, value):
        self.index = index
        self.value = value

    def __repr__(self):
        return f"MutationInsert({self.index}, {self.value})"


class MutationPermutate(Mutation):
    """
    Permutating all the values.
    This mutation is only used for MutableListLikeData.
    """

    ABBR = "P"

    def __init__(self, permutation):
        self.permutation = permutation

    def __repr__(self):
        return f"MutationPermutate({self.permutation})"


def record_mutation(
    mutation_fn: Callable[Concatenate[MutableDataT, P], Mutation]
) -> Callable[Concatenate[MutableDataT, P], None]:
    def wrapper(self, *args: P.args, **kwargs: P.kwargs):
        mutation = mutation_fn(self, *args, **kwargs)
        self.records.append(mutation)

    return wrapper


class MutableData(Generic[InnerMutableDataT]):
    """
    An intermediate data structure between data and variable, it records all the mutations.
    """

    read_cache: InnerMutableDataT

    class Empty:
        def __repr__(self):
            return "Empty()"

    def __init__(self, data: Any, getter: DataGetter):
        self.original_data = data
        self.getter = getter
        self.records: list[Mutation] = []

    def is_empty(self, value):
        return isinstance(value, MutableData.Empty)

    @property
    def version(self):
        return len(self.records)

    @property
    def has_changed(self):
        return self.version != 0

    def rollback(self, version: int):
        assert version <= self.version
        self.records[:] = self.records[:version]

    def get(self, key):
        raise NotImplementedError()

    def set(self, key, value):
        raise NotImplementedError()

    def apply(self, mutation: Mutation, write_cache: InnerMutableDataT):
        raise NotImplementedError()

    def reproduce(self, version: int | None = None) -> InnerMutableDataT:
        if version is None:
            version = self.version
        write_cache = self.read_cache.copy()
        for mutation in self.records[:version]:
            self.apply(mutation, write_cache)
        return write_cache

    def __repr__(self) -> str:
        records_abbrs = "".join([mutation.ABBR for mutation in self.records])
        return f"{self.__class__.__name__}({records_abbrs})"


class MutableDictLikeData(MutableData["dict[str, Any]"]):
    def __init__(self, data: Any, getter: DataGetter):
        super().__init__(data, getter)
        self.read_cache = {}

    def clear_read_cache(self):
        self.read_cache.clear()

    def get(self, key: Any):
        # TODO(SigureMo): Optimize performance of this.
        write_cache = self.reproduce(self.version)
        if key not in write_cache:
            self.read_cache[key] = self.getter(self, key)
        return self.reproduce(self.version)[key]

    def get_all(self):
        original_keys = list(self.original_data.keys())
        for mutation in self.records:
            if isinstance(mutation, MutationNew):
                original_keys.append(mutation.key)
            elif isinstance(mutation, MutationDel):
                original_keys.remove(mutation.key)
        return {key: self.get(key) for key in original_keys}

    @record_mutation
    def set(self, key: Any, value: Any) -> Mutation:
        is_new = False
        if self.is_empty(self.get(key)):
            is_new = True
        return (
            MutationSet(key, value) if not is_new else MutationNew(key, value)
        )

    @record_mutation
    def delete(self, key):
        return MutationDel(key)

    def apply(self, mutation: Mutation, write_cache: dict[str, Any]):
        if isinstance(mutation, MutationNew):
            write_cache[mutation.key] = mutation.value
        elif isinstance(mutation, MutationSet):
            write_cache[mutation.key] = mutation.value
        elif isinstance(mutation, MutationDel):
            write_cache[mutation.key] = MutableData.Empty()
        else:
            raise ValueError(f"Unknown mutation type {mutation}")

    def reproduce(self, version: int | None = None):
        if version is None:
            version = self.version
        write_cache = self.read_cache.copy()
        for mutation in self.records[:version]:
            self.apply(mutation, write_cache)
        return write_cache


class MutableListLikeData(MutableData["list[Any]"]):
    def __init__(self, data: Any, getter: DataGetter):
        super().__init__(data, getter)
        self.read_cache = [
            self.getter(self, idx) for idx in range(len(self.original_data))
        ]

    def clear_read_cache(self):
        self.read_cache[:] = []

    @property
    def length(self):
        return len(self.reproduce())

    def get(self, key):
        write_cache = self.reproduce(self.version)
        return write_cache[key]

    def get_all(self) -> list[Any]:
        items = self.reproduce(self.version)
        return items

    @record_mutation
    def set(self, key: int, value: Any):
        return MutationSet(self._regularize_index(key), value)

    @record_mutation
    def delete(self, key: int):
        return MutationDel(self._regularize_index(key))

    @record_mutation
    def insert(self, index: int, value: Any):
        return MutationInsert(self._regularize_index(index), value)

    @record_mutation
    def permutate(self, permutation: list[int]):
        return MutationPermutate(permutation)

    def _regularize_index(self, index: int):
        if index < 0:
            index += self.length
        return index

    def apply(self, mutation: Mutation, write_cache: list[Any]):
        if isinstance(mutation, MutationSet):
            write_cache[mutation.key] = mutation.value
        elif isinstance(mutation, MutationDel):
            write_cache[:] = (
                write_cache[: mutation.key] + write_cache[mutation.key + 1 :]
            )
        elif isinstance(mutation, MutationInsert):
            write_cache.insert(mutation.index, mutation.value)
        elif isinstance(mutation, MutationPermutate):
            write_cache[:] = [write_cache[i] for i in mutation.permutation]
        else:
            raise ValueError(f"Unknown mutation type {mutation}")
