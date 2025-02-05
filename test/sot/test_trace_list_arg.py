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

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.utils.envs import allow_dynamic_shape_guard


def foo(x: list[paddle.Tensor], y: list[paddle.Tensor]):
    return x[0] + y[0]


def bar(x: list[paddle.Tensor], y: int, z: int):
    return x[y + z] + 1


class TestTraceListArg(TestCaseBase):
    def test_foo(self):
        a = paddle.to_tensor(1)
        b = paddle.to_tensor(2)
        c = paddle.to_tensor([3, 4])

        with test_instruction_translator_cache_context() as cache:
            self.assert_results(foo, [a], [b])
            self.assertEqual(cache.translate_count, 1)
            self.assert_results(foo, [b], [a])  # Cache hit
            self.assertEqual(cache.translate_count, 1)
            self.assert_results(foo, [a], [c])  # Cache miss
            self.assertEqual(cache.translate_count, 2)

    @allow_dynamic_shape_guard(False)
    def test_bar_static_shape(self):
        a = [paddle.to_tensor(1), paddle.to_tensor(2), paddle.to_tensor(3)]
        b = [paddle.to_tensor([2, 3]), paddle.to_tensor(4), paddle.to_tensor(5)]

        with test_instruction_translator_cache_context() as cache:
            self.assert_results(bar, a, 1, 1)
            self.assertEqual(cache.translate_count, 1)
            self.assert_results(bar, a, 2, 0)  # Cache miss
            self.assertEqual(cache.translate_count, 2)
            self.assert_results(bar, b, 1, 1)  # Cache hit
            self.assertEqual(cache.translate_count, 2)

    @allow_dynamic_shape_guard(True)
    def test_bar_dynamic_shape(self):
        # TODO(zrr1999): mv to dynamic shape test
        a = [paddle.to_tensor(1), paddle.to_tensor(2), paddle.to_tensor(3)]
        b = [paddle.to_tensor([2, 3]), paddle.to_tensor(4), paddle.to_tensor(5)]
        with test_instruction_translator_cache_context() as cache:
            self.assert_results(bar, a, 1, 1)
            self.assertEqual(cache.translate_count, 1)
            self.assert_results(bar, a, 2, 0)  # Cache miss
            self.assertEqual(cache.translate_count, 2)
            self.assert_results(bar, b, 2, 0)  # Cache miss
            self.assertEqual(cache.translate_count, 3)
            self.assert_results(bar, b, 2, 0)  # Cache hit
            self.assertEqual(cache.translate_count, 3)
            self.assert_results(bar, b, 1, 1)  # Cache hit
            self.assertEqual(cache.translate_count, 3)
            self.assert_results(bar, b, 0, 2)  # Cache miss
            self.assertEqual(cache.translate_count, 4)


if __name__ == "__main__":
    unittest.main()
