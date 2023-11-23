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

import sys
import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit import sot
from paddle.jit.sot.utils import strict_mode_guard


def foo(x, y):
    if x.dtype == paddle.float32:
        out = x + y
    else:
        out = x - y
    return out


def dtype_in_guard(x, y):
    sot.psdb.fallback()
    with paddle.amp.auto_cast(level='O2'):
        for i in range(10):
            z = foo(x, y)
            x = z
        return x


def bar(x, y):
    if x == paddle.float32:
        return y + 1
    else:
        return y - 1


def dtype_as_input(x, y):
    sot.psdb.fallback()
    with paddle.amp.auto_cast(level='O2'):
        for i in range(10):
            z = bar(x, y)
            y = z
        return y


class TestDtypeInGuard(TestCaseBase):
    @strict_mode_guard(False)
    def test_dtype_in_guard(self):
        with test_instruction_translator_cache_context() as ctx:
            x = paddle.to_tensor([2], dtype="float32")
            y = paddle.to_tensor([3], dtype="float32")
            self.assert_results(dtype_in_guard, x, y)
            if sys.version_info >= (3, 11):
                # skipped with co_exceptiontable flag
                self.assertEqual(ctx.translate_count, 1)
            else:
                self.assertEqual(ctx.translate_count, 2)

    @strict_mode_guard(False)
    def test_input_dtype_in_guard(self):
        with test_instruction_translator_cache_context() as ctx:
            x = paddle.float32
            y = paddle.to_tensor([3], dtype="float32")
            self.assert_results(dtype_as_input, x, y)
            if sys.version_info >= (3, 11):
                # skipped with co_exceptiontable flag
                self.assertEqual(ctx.translate_count, 1)
            else:
                self.assertEqual(ctx.translate_count, 2)


if __name__ == "__main__":
    unittest.main()
