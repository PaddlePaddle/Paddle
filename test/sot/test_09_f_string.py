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

# FORMAT_VALUE (new)
# BUILD_STRING (new)
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import assert_true


def foo(x: paddle.Tensor):
    whilespace = 123
    hello_world = f"Hello {whilespace} World"
    z = assert_true(hello_world == "Hello 123 World")
    x = x + 1
    return x


def fstring_with_convert_test(x: paddle.Tensor):
    obj = 42
    formatted_string = f"{obj!r}"
    z = assert_true(formatted_string == "42")
    x = x + 1
    return x


def fstring_with_spec(x: paddle.Tensor):
    int_num = 42
    float_num = 3.1415
    formatted_string = f"{int_num} and {float_num:.2f}"
    z = assert_true(formatted_string == "42 and 3.14")
    x = x + 1
    return x


class TestFString(TestCaseBase):
    def test_f_string(self):
        self.assert_results(foo, paddle.to_tensor(1))

    def test_f_string_with_spec(self):
        self.assert_results(fstring_with_spec, paddle.to_tensor(1))

    def test_f_string_with_convert_test(self):
        self.assert_results(fstring_with_convert_test, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
