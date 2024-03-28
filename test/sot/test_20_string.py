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

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import assert_true, check_no_breakgraph


def string_format(x: paddle.Tensor):
    whilespace = 123
    hello_world = f"Hello {whilespace} World"
    z = assert_true(hello_world == "Hello 123 World")
    hello_world2 = f"Hello {whilespace}{whilespace} World"
    z = assert_true(hello_world2 == "Hello 123123 World")
    hello_world_lower = "Hello World".lower()
    z = assert_true(hello_world_lower == "hello world")
    return x + 1


def string_lower(x: paddle.Tensor):
    hello_world_lower = "Hello World".lower()
    z = assert_true(hello_world_lower == "hello world")
    return x + 1


@check_no_breakgraph
def str_startswith():
    s = "Hello World"
    a1 = s.startswith("Hello")
    a2 = s.startswith("World")
    a3 = s.startswith("Hello World")
    a4 = s.startswith("Hello World!")
    a5 = s.startswith("Hello", 5)
    a6 = s.startswith("Hello", 1, 4)
    a7 = s.startswith("Hello", 0, 11)
    return (a1, a2, a3, a4, a5, a6, a7)


@check_no_breakgraph
def str_endswith():
    s = "Hello World"
    a1 = s.endswith("Hello")
    a2 = s.endswith("World")
    a3 = s.endswith("Hello World")
    a4 = s.endswith("Hello World!")
    a5 = s.endswith("Hello", 5)
    a6 = s.endswith("Hello", 0, 4)
    a7 = s.endswith("Hello", 1, 11)
    return (a1, a2, a3, a4, a5, a6, a7)


class TestString(TestCaseBase):
    def test_string_format(self):
        self.assert_results(string_format, paddle.to_tensor(1))

    def test_string_lower(self):
        self.assert_results(string_lower, paddle.to_tensor(1))

    def test_str_startswith(self):
        self.assert_results(str_startswith)

    def test_str_endswith(self):
        self.assert_results(str_endswith)


if __name__ == "__main__":
    unittest.main()
