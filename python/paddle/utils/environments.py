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

import os
from typing import Generic, TypeVar

T = TypeVar("T")


class EnvironmentVariable(Generic[T]):
    name: str
    default: T

    def __init__(self, name: str, default: T):
        self.name = name
        self.default = default

    def get(self) -> T:
        raise NotImplementedError()

    def set(self, value: T) -> None:
        raise NotImplementedError()

    def delete(self) -> None:
        del os.environ[self.name]

    def __repr__(self) -> str:
        return f"Env({self.name}={self.get()!r})"


class StringEnvironmentVariable(EnvironmentVariable[str]):
    def __init__(self, name: str, default: str):
        super().__init__(name, default)
        assert isinstance(default, str), "default must be a string"

    def get(self) -> str:
        return os.getenv(self.name, self.default)

    def set(self, value: str) -> None:
        assert isinstance(value, str), "value must be a string"
        os.environ[self.name] = value


class BooleanEnvironmentVariable(EnvironmentVariable[bool]):
    BOOLEAN_IS_SET = ("y", "yes", "t", "true", "on", "1")

    def __init__(self, name: str, default: bool):
        super().__init__(name, default)
        assert isinstance(default, bool), "default must be a boolean"

    def get(self) -> bool:
        default = str(self.default).lower()
        env_str = os.getenv(self.name, default).lower()
        return env_str in BooleanEnvironmentVariable.BOOLEAN_IS_SET

    def set(self, value: bool) -> None:
        assert isinstance(value, bool), "value must be a boolean"
        os.environ[self.name] = str(value).lower()


class IntegerEnvironmentVariable(EnvironmentVariable[int]):
    def __init__(self, name: str, default: int):
        super().__init__(name, default)
        assert isinstance(default, int) and not isinstance(
            default, bool
        ), "default must be an integer"

    def get(self) -> int:
        try:
            return int(os.getenv(self.name, str(self.default)))
        except ValueError:
            return self.default

    def set(self, value: int) -> None:
        assert isinstance(value, int) and not isinstance(
            value, bool
        ), "value must be an integer"
        os.environ[self.name] = str(value)


class EnvironmentVariableGuard(Generic[T]):
    variable: EnvironmentVariable[T]
    original_value: T

    def __init__(self, variable: EnvironmentVariable[T], value: T):
        self.variable = variable
        self.original_value = variable.get()
        self.variable.set(value)

    def __enter__(self) -> EnvironmentVariableGuard:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.variable.set(self.original_value)
