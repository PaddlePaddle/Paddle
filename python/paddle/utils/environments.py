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
from typing import Generic, List, TypeVar

from typing_extensions import Self

T = TypeVar("T")


def strtobool(val):
    val = val.lower()
    if val in ['y', 'yes', 't', 'true', 'on', '1']:
        return True
    elif val in ['n', 'no', 'f', 'false', 'off', '0']:
        return False
    else:
        raise ValueError(f"Invalid truth value {val!r}")


class EnvironmentVariable(Generic[T]):
    name: str
    default: T

    def __init__(self, name: str, default: T):
        self.name = name
        self.default = default
        self._last_env_value: str | None = None
        self._cached_value: T | None = None

    def get(self) -> T:
        _current_env_value = os.getenv(self.name)
        if (
            self._cached_value is None
            or self._last_env_value != _current_env_value
        ):
            self._cached_value = self.parse_from_string()
            self._last_env_value = _current_env_value
        return self._cached_value

    def set(self, value: T) -> None:
        os.environ[self.name] = self.convert_to_string(value)
        self._cached_value = value

    def parse_from_string(self) -> T:
        raise NotImplementedError

    def convert_to_string(self, value: T) -> str:
        raise NotImplementedError

    def delete(self) -> None:
        del os.environ[self.name]

    def __repr__(self) -> str:
        return f"Env({self.name}={self.get()!r})"


class StringEnvironmentVariable(EnvironmentVariable[str]):
    def __init__(self, name: str, default: str):
        super().__init__(name, default)
        assert isinstance(default, str), "default must be a string"

    def parse_from_string(self) -> str:
        return os.getenv(self.name, self.default)

    def convert_to_string(self, value: str) -> str:
        assert isinstance(value, str), "value must be a string"
        return value


class BooleanEnvironmentVariable(EnvironmentVariable[bool]):
    def __init__(self, name: str, default: bool):
        super().__init__(name, default)
        assert isinstance(default, bool), "default must be a boolean"

    def parse_from_string(self) -> bool:
        default = str(self.default)
        env_str = os.getenv(self.name, default)
        return strtobool(env_str)

    def convert_to_string(self, value: bool) -> str:
        assert isinstance(value, bool), "value must be a boolean"
        return str(value).lower()

    def __bool__(self) -> bool:
        raise ValueError(
            "BooleanEnvironmentVariable does not support bool(), "
            "please use get() instead."
        )


class IntegerEnvironmentVariable(EnvironmentVariable[int]):
    def __init__(self, name: str, default: int):
        super().__init__(name, default)
        assert isinstance(default, int) and not isinstance(
            default, bool
        ), "default must be an integer"

    def parse_from_string(self) -> int:
        try:
            return int(os.getenv(self.name, str(self.default)))
        except ValueError:
            return self.default

    def convert_to_string(self, value: int) -> str:
        assert isinstance(value, int) and not isinstance(
            value, bool
        ), "value must be an integer"
        return str(value)


class StringListEnvironmentVariable(EnvironmentVariable[List[str]]):
    def __init__(self, name: str, default: list[str]):
        super().__init__(name, default)
        assert isinstance(default, list), "default must be a list"

    def parse_from_string(self) -> list[str]:
        return os.getenv(self.name, ",".join(self.default)).split(",")

    def convert_to_string(self, value: list[str]) -> str:
        assert isinstance(value, list), "value must be a list"
        assert all(
            isinstance(x, str) for x in value
        ), "value must be a list of strings"
        return ",".join(value)


class EnvironmentVariableGuard(Generic[T]):
    variable: EnvironmentVariable[T]
    original_value: T

    def __init__(self, variable: EnvironmentVariable[T], value: T):
        self.variable = variable
        self.original_value = variable.get()
        self.variable.set(value)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.variable.set(self.original_value)
