#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
)

import paddle


class TestSetDynamicShape(Dy2StTestBase):
    @test_ast_only
    def test_start(self):
        def dygraph_func(loop_number):
            mask = paddle.randn([2, 2])
            paddle.jit.api.set_dynamic_shape(mask, [-1, 2])
            n = paddle.randn([1, 2])
            for i in range(loop_number):
                mask = paddle.concat([mask, n], axis=0)
                if mask.shape[0] == 5:
                    break
            return mask

        loop_num = paddle.to_tensor(10)
        expected_shape = dygraph_func(loop_num).shape
        actual_shape = paddle.jit.to_static(dygraph_func)(loop_num).shape
        self.assertEqual(expected_shape, actual_shape)

    @test_ast_only
    def test_pir_ast(self):
        def dygraph_func():
            mask = paddle.randn([2, 2])
            paddle.jit.api.set_dynamic_shape(mask, [-1, 2])
            return mask.shape[0]

        out = paddle.jit.to_static(dygraph_func)()
        self.assertTrue(isinstance(out, paddle.Tensor))
        self.assertEqual(out.numpy(), 2)


if __name__ == '__main__':
    unittest.main()
