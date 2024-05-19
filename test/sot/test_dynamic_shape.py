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

import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.utils import with_allow_dynamic_shape_guard


def foo(x):
    s = x.shape[0]
    return x + s


def dynamic_int_input_func(x, n):
    x = paddle.reshape(x, [n, -1])
    return (x + n) * 2 - 1, (n + 1) * 2 - 1


class TestOpcodeExecutorDynamicShapeCache(TestCaseBase):
    def test_dynamic_int_input_cache_hit(self):
        with with_allow_dynamic_shape_guard(
            True
        ), test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                dynamic_int_input_func, paddle.randn([3, 4, 5]), 1
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                dynamic_int_input_func, paddle.randn([3, 4, 5]), 2
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                dynamic_int_input_func, paddle.randn([3, 4, 5]), 3
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                dynamic_int_input_func, paddle.randn([3, 4, 5]), 4
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                dynamic_int_input_func, paddle.randn([3, 4, 5]), 5
            )
            self.assertEqual(ctx.translate_count, 2)


if __name__ == '__main__':
    unittest.main()
