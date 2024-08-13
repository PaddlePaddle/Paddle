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

from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, overload

if TYPE_CHECKING:
    ValidateValueFunc = Callable[[Any], None]


StackDataT = TypeVar("StackDataT")


class VariableStack(Generic[StackDataT]):
    """
    A stack class for storing variables.

    Examples:
        >>> var1, var2, var3, var4 = range(1, 5)
        >>> stack = VariableStack()
        >>> stack.push(var1)
        >>> stack.push(var3)
        >>> stack.insert(1, var2)
        >>> stack
        [1, 2, 3]
        >>> stack.pop()
        3
        >>> stack.pop_n(2)
        [1, 2]
        >>> stack.push(var1)
        >>> stack.push(var2)
        >>> stack.push(var3)
        >>> stack
        [1, 2, 3]
        >>> stack.top
        3
        >>> stack.peek[1]
        3
        >>> stack.peek[:1]
        [3]
        >>> stack.peek[:2]
        [2, 3]
        >>> stack.peek[1] = var4
        >>> stack
        [1, 2, 4]

    """

    class VariablePeeker:
        @overload
        def __getitem__(self, index: int) -> StackDataT: ...

        @overload
        def __getitem__(self, index: slice) -> list[StackDataT]: ...

        @overload
        def __call__(self, index: int = 1) -> StackDataT: ...

        @overload
        def __call__(self, index: slice) -> list[StackDataT]: ...

        def __init__(
            self, data: list[StackDataT], validate_value_func: ValidateValueFunc
        ):
            self._data = data
            self.validate_value_func = validate_value_func

        def __getitem__(
            self, index: int | slice
        ) -> StackDataT | list[StackDataT]:
            if isinstance(index, int):
                assert 0 < index <= len(self._data)
                return self._data[-index]
            if isinstance(index, slice):
                assert (
                    index.start is None and index.step is None
                ), "slice which has start or step not supported"
                assert 0 < index.stop <= len(self._data)
                return self._data[-index.stop :]
            raise NotImplementedError(f"index type {type(index)} not supported")

        def __setitem__(self, index: int, value: Any):
            assert isinstance(
                index, int
            ), f"index type {type(index)} not supported"
            assert (
                0 < index <= len(self._data)
            ), f"index should be in [1, {len(self._data)}], but get {index}"
            self.validate_value_func(value)
            self._data[-index] = value

        def __call__(
            self, index: int | slice = 1
        ) -> StackDataT | list[StackDataT]:
            return self[index]

    def __init__(
        self,
        data: list[StackDataT] | None = None,
        *,
        validate_value_func: ValidateValueFunc | None = None,
    ):
        if data is None:
            data = []
        else:
            data = data.copy()
        self.validate_value_func = (
            (lambda _: None)
            if validate_value_func is None
            else validate_value_func
        )
        self._data = data
        self._peeker = VariableStack.VariablePeeker(
            self._data, self.validate_value_func
        )

    def copy(self):
        return VariableStack(
            self._data, validate_value_func=self.validate_value_func
        )

    def push(self, val: StackDataT):
        """
        Pushes a variable onto the stack.

        Args:
            val: The variable to be pushed.

        """
        self.validate_value_func(val)
        self._data.append(val)

    def insert(self, index: int, val: StackDataT):
        """
        Inserts a variable onto the stack.

        Args:
            index: The index at which the variable is to be inserted, the top of the stack is at index 0.
            val: The variable to be inserted.

        """
        assert (
            0 <= index <= len(self)
        ), f"index should be in [0, {len(self)}], but get {index}"
        self.validate_value_func(val)
        self._data.insert(len(self) - index, val)

    def pop(self) -> StackDataT:
        """
        Pops the top value from the stack.

        Returns:
            The popped value.

        """
        assert len(self) > 0, "stack is empty"
        return self._data.pop()

    def pop_n(self, n: int) -> list[StackDataT]:
        """
        Pops the top n values from the stack.

        Args:
            n: The number of values to pop.

        Returns:
            A list of the popped values.

        """
        assert (
            len(self) >= n >= 0
        ), f"n should be in [0, {len(self)}], but get {n}"
        if n == 0:
            return []
        retval = self._data[-n:]
        self._data[-n:] = []
        return retval

    @property
    def peek(self) -> VariablePeeker:
        return self._peeker

    @property
    def top(self) -> StackDataT:
        assert len(self) > 0, "stack is empty"
        return self.peek[1]

    @top.setter
    def top(self, value):
        assert len(self) > 0, "stack is empty"
        self.peek[1] = value

    def __contains__(self, value):
        return value in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return str(self._data)
