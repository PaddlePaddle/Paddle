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
    test_with_faster_guard,
)

import paddle


def non_operator_related_fn(x: int, y: int):
    return x + y


def partial_non_operator_related_fn(x: paddle.Tensor, y: paddle.Tensor, z: int):
    a = x + y
    return [a, z + z]


def guard_inputs(x: int, y: int, z: int):
    return x + y + z


class TestGuardOutputs(TestCaseBase):
    @test_with_faster_guard
    def test_non_operator_related_fn(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(non_operator_related_fn, 1, 2)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(non_operator_related_fn, 3, 4)
            self.assertEqual(ctx.translate_count, 2)

    @test_with_faster_guard
    def test_partial_non_operator_related_fn(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                partial_non_operator_related_fn,
                paddle.to_tensor(1),
                paddle.to_tensor(2),
                3,
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                partial_non_operator_related_fn,
                paddle.to_tensor(4),
                paddle.to_tensor(5),
                6,
            )
            self.assertEqual(ctx.translate_count, 2)

    @test_with_faster_guard
    def test_guard_inputs(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(guard_inputs, 1, 2, 3)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(guard_inputs, 0, 2, 3)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(guard_inputs, 1, 0, 3)
            self.assertEqual(ctx.translate_count, 3)
            self.assert_results(guard_inputs, 1, 2, 0)
            self.assertEqual(ctx.translate_count, 4)


if __name__ == "__main__":
    unittest.main()
