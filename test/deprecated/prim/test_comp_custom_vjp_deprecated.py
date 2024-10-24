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
import unittest

import paddle
from paddle.base import core


class TestCustomVJP(unittest.TestCase):
    def setUp(self):
        def func():
            x = paddle.rand((1,))
            x.stop_gradient = False
            return paddle.nn.functional.dropout(x)

        self.f = func
        self.ops_fwd_enable_bwd_disable = (
            'uniform_random',
            'uniform_random',
            'fill_constant',
            'greater_equal',
            'cast',
            'elementwise_mul',
            'scale',
            'cast',
            'fill_any_like',
            'scale',
            'elementwise_mul_grad',
        )
        self.ops_fwd_disable_bwd_enable = (
            'uniform_random',
            'dropout',
            'fill_any_like',
            'fill_any_like',
            'cast',
            'elementwise_mul',
            'scale',
        )
        self.ops_all_enable = (
            'uniform_random',
            'uniform_random',
            'fill_constant',
            'greater_equal',
            'cast',
            'elementwise_mul',
            'scale',
            'cast',
            'fill_constant',
            'fill_constant',
            'cast',
            'elementwise_mul',
            'scale',
        )

    def test_enable_prim_fwd(self):
        core._set_prim_forward_enabled(True)
        core._set_prim_backward_enabled(False)
        self.assertEqual(
            self.ops_fwd_enable_bwd_disable,
            tuple(
                op.type
                for op in paddle.jit.to_static(full_graph=True)(self.f)
                .get_concrete_program()[1]
                ._train_program.block(0)
                .ops
            ),
        )
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)

    def test_enable_prim_bwd(self):
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(True)
        self.assertEqual(
            self.ops_fwd_disable_bwd_enable,
            tuple(
                op.type
                for op in paddle.jit.to_static(full_graph=True)(self.f)
                .get_concrete_program()[1]
                ._train_program.block(0)
                .ops
            ),
        )
        core._set_prim_forward_enabled(False)
        core._set_prim_backward_enabled(False)

    def test_enable_prim_all(self):
        core._set_prim_all_enabled(True)
        self.assertEqual(
            self.ops_all_enable,
            tuple(
                op.type
                for op in paddle.jit.to_static(full_graph=True)(self.f)
                .get_concrete_program()[1]
                ._train_program.block(0)
                .ops
            ),
        )
        core._set_prim_all_enabled(False)


if __name__ == '__main__':
    unittest.main()
