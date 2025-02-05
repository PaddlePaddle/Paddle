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


def test_guard_fn(fn, inp):
    if fn is None:
        return 0
    else:
        return fn(inp)


class TestGuardOutputs(TestCaseBase):
    @test_with_faster_guard
    def test_non_operator_related_fn(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                test_guard_fn,
                paddle.nn.functional.relu,
                paddle.to_tensor([1.0, -1.0]),
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                test_guard_fn,
                paddle.nn.functional.gelu,
                paddle.to_tensor([1.0, -1.0]),
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                test_guard_fn,
                paddle.nn.functional.relu,
                paddle.to_tensor([-1.0, -1.0]),
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                test_guard_fn, None, paddle.to_tensor([-1.0, -1.0])
            )
            self.assertEqual(ctx.translate_count, 3)

        deleted_cnt = 0

        class Callable:
            def __call__(self, var):
                return paddle.nn.functional.relu(var)

            def __del__(self):
                nonlocal deleted_cnt
                deleted_cnt += 1

        fn1 = Callable()
        fn2 = Callable()
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(
                test_guard_fn, fn1, paddle.to_tensor([1.0, -1.0])
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                test_guard_fn, fn2, paddle.to_tensor([1.0, -1.0])
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                test_guard_fn, fn2, paddle.to_tensor([1.0, -1.0])
            )
            self.assertEqual(ctx.translate_count, 2)


if __name__ == "__main__":
    unittest.main()
