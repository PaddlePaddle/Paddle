# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Generic, Iterable, Iterator, TypeVar

T = TypeVar("T")


class OrderedSet(Generic[T]):
    """
    A set that preserves the order of insertion.
    """

    _data: dict[T, None]

    def __init__(self, items: Iterable[T] | None = None):
        """
        Examples:
            >>> s = OrderedSet([1, 2, 3])
            >>> s
            OrderedSet(1, 2, 3)
            >>> s = OrderedSet()
            >>> s
            OrderedSet()
        """
        self._data = dict.fromkeys(items) if items is not None else {}

    def __iter__(self) -> Iterator[T]:
        """
        Examples:
            >>> s = OrderedSet([1, 2, 3])
            >>> for item in s:
            ...     print(item)
            1
            2
            3
        """
        return iter(self._data)

    def __or__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        """
        Union two sets.

        Args:
            other: Another set to be unioned.

        Returns:
            The union of two sets.

        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([2, 3, 4])
            >>> s1 | s2
            OrderedSet(1, 2, 3, 4)
        """
        return OrderedSet(list(self) + list(other))

    def __ior__(self, other: OrderedSet[T]):
        """
        Union two sets in place.

        Args:
            other: Another set to be unioned.

        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([2, 3, 4])
            >>> s1 |= s2
            >>> s1
            OrderedSet(1, 2, 3, 4)
        """
        self._data.update(dict.fromkeys(other))
        return self

    def __and__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        """
        Intersect two sets.

        Args:
            other: Another set to be intersected.

        Returns:
            The intersection of two sets.

        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([2, 3, 4])
            >>> s1 & s2
            OrderedSet(2, 3)
        """
        return OrderedSet([item for item in self if item in other])

    def __iand__(self, other: OrderedSet[T]):
        """
        Intersect two sets in place.

        Args:
            other: Another set to be intersected.

        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([2, 3, 4])
            >>> s1 &= s2
            >>> s1
            OrderedSet(2, 3)
        """
        self._data = {item: None for item in self if item in other}
        return self

    def __sub__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        """
        Subtract two sets.

        Args:
            other: Another set to be subtracted.

        Returns:
            The subtraction of two sets.

        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([2, 3, 4])
            >>> s1 - s2
            OrderedSet(1)
        """
        return OrderedSet([item for item in self if item not in other])

    def __isub__(self, other: OrderedSet[T]):
        """
        Subtract two sets in place.

        Args:
            other: Another set to be subtracted.

        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([2, 3, 4])
            >>> s1 -= s2
            >>> s1
            OrderedSet(1)
        """
        self._data = {item: None for item in self if item not in other}
        return self

    def __xor__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        """
        Symmetric difference of two sets.

        Args:
            other: Another set to be xor'ed.

        Returns:
            The symmetric difference of two sets.

        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([2, 3, 4])
            >>> s1 ^ s2
            OrderedSet(1, 4)
        """
        return OrderedSet(
            [item for item in self if item not in other]
        ) | OrderedSet([item for item in other if item not in self])

    def __ixor__(self, other: OrderedSet[T]):
        """
        Symmetric difference of two sets in place.

        Args:
            other: Another set to be xor'ed.

        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([2, 3, 4])
            >>> s1 ^= s2
            >>> s1
            OrderedSet(1, 4)
        """
        # TODO(Python3.8-cleanup): Use dict union syntax when Python 3.9 is
        # minimum supported version.
        # self._data = {item: None for item in self if item not in other} | {
        #     item: None for item in other if item not in self
        # }
        self._data = {
            **{item: None for item in self if item not in other},
            **{item: None for item in other if item not in self},
        }
        return self

    def add(self, item: T):
        """
        Add an item to the set.

        Args:
            item: The item to be added.

        Examples:
            >>> s = OrderedSet([1, 2, 3])
            >>> s.add(4)
            >>> s
            OrderedSet(1, 2, 3, 4)
        """
        self._data.setdefault(item)

    def remove(self, item: T):
        """
        Remove an item from the set.

        Args:
            item: The item to be removed.

        Examples:
            >>> s = OrderedSet([1, 2, 3])
            >>> s.remove(2)
            >>> s
            OrderedSet(1, 3)
        """
        del self._data[item]

    def __contains__(self, item: T) -> bool:
        """
        Examples:
            >>> s = OrderedSet([1, 2, 3])
            >>> 1 in s
            True
            >>> 4 in s
            False
        """
        return item in self._data

    def __len__(self) -> int:
        """
        Examples:
            >>> s = OrderedSet([1, 2, 3])
            >>> len(s)
            3
        """
        return len(self._data)

    def __bool__(self) -> bool:
        """
        Examples:
            >>> s = OrderedSet([1, 2, 3])
            >>> bool(s)
            True
            >>> s = OrderedSet()
            >>> bool(s)
            False
        """
        return bool(self._data)

    def __eq__(self, other: object) -> bool:
        """
        Examples:
            >>> s1 = OrderedSet([1, 2, 3])
            >>> s2 = OrderedSet([1, 2, 3])
            >>> s1 == s2
            True
            >>> s3 = OrderedSet([3, 2, 1])
            >>> s1 == s3
            False
        """
        if not isinstance(other, OrderedSet):
            return NotImplemented
        return list(self) == list(other)

    def __repr__(self) -> str:
        data_repr = ", ".join(map(repr, self._data))
        return f"OrderedSet({data_repr})"
