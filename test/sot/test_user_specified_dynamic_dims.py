# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.jit.sot.psdb import check_no_breakgraph


@check_no_breakgraph
def tensor_with_marked_dynamic_dims(tensor):
    return tensor + 1


@check_no_breakgraph
def multiple_tensors_with_marked_dynamic_dims(tensor1, tensor2):
    return tensor1.sum() + tensor2.sum()


class TestUserSpecifiedDynamicDims(TestCaseBase):
    def test_auto_inferred_dynamic_dims(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x1 = paddle.rand([5, 6], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x1)
            self.assertEqual(ctx.translate_count, 1)
            x2 = paddle.rand([4, 6], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x2)
            self.assertEqual(ctx.translate_count, 2)
            x3 = paddle.rand([3, 6], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x3)
            self.assertEqual(ctx.translate_count, 2)

    def test_user_specified_dynamic_dims(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x1 = paddle.rand([5, 6], dtype='float32')
            paddle.jit.marker.dynamic_dims(x1, [0])
            self.assert_results(tensor_with_marked_dynamic_dims, x1)
            self.assertEqual(ctx.translate_count, 1)
            x2 = paddle.rand([4, 6], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x2)
            self.assertEqual(ctx.translate_count, 1)
            x3 = paddle.rand([3, 6], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x3)
            self.assertEqual(ctx.translate_count, 1)

    def test_user_specified_multiple_dynamic_dims(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x1 = paddle.rand([5, 6], dtype='float32')
            paddle.jit.marker.dynamic_dims(x1, [0, 1])
            self.assert_results(tensor_with_marked_dynamic_dims, x1)
            self.assertEqual(ctx.translate_count, 1)
            x2 = paddle.rand([4, 6], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x2)
            self.assertEqual(ctx.translate_count, 1)
            x3 = paddle.rand([3, 7], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x3)
            self.assertEqual(ctx.translate_count, 1)

    def test_multiple_tensors_with_marked_dynamic_dims(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x1 = paddle.rand([5, 6], dtype='float32')
            x2 = paddle.rand([5, 6], dtype='float32')
            paddle.jit.marker.dynamic_dims(x1, [0])
            paddle.jit.marker.dynamic_dims(x2, [0])
            self.assert_results(
                multiple_tensors_with_marked_dynamic_dims, x1, x2
            )
            self.assertEqual(ctx.translate_count, 1)
            x3 = paddle.rand([4, 6], dtype='float32')
            x4 = paddle.rand([5, 6], dtype='float32')
            self.assert_results(
                multiple_tensors_with_marked_dynamic_dims, x3, x4
            )
            self.assertEqual(ctx.translate_count, 1)
            x5 = paddle.rand([4, 6], dtype='float32')
            x6 = paddle.rand([4, 6], dtype='float32')
            self.assert_results(
                multiple_tensors_with_marked_dynamic_dims, x5, x6
            )
            self.assertEqual(ctx.translate_count, 1)

    def test_mix_auto_inferred_and_specified_dynamic_dims(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            x1 = paddle.rand([5, 6], dtype='float32')
            paddle.jit.marker.dynamic_dims(x1, [0])
            self.assert_results(tensor_with_marked_dynamic_dims, x1)
            self.assertEqual(ctx.translate_count, 1)
            x2 = paddle.rand([4, 6], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x2)
            self.assertEqual(ctx.translate_count, 1)
            x3 = paddle.rand([3, 7], dtype='float32')
            self.assert_results(tensor_with_marked_dynamic_dims, x3)
            self.assertEqual(ctx.translate_count, 2)


if __name__ == '__main__':
    unittest.main()
