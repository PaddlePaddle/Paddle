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

import numpy as np

import paddle
from paddle import _ir_ops, nn
from paddle.autograd.ir_backward import grad
from paddle.decomposition import decompose
from paddle.framework import core

paddle.enable_static()


class SimpNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, scale, bias, ep, begin_axis):
        out = _ir_ops.layer_norm(x, scale, bias, ep, begin_axis)
        return out


class TestPrimMode(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 1024, 1024]
        self.norm_shape = [1048576]
        self.ep = 1e-5
        self.begin_axis = 1

        self.x = np.random.random(self.shape_x).astype("float32")
        self.scale = np.random.random(self.norm_shape).astype("float32")
        self.bias = np.random.random(self.norm_shape).astype("float32")

    def base_net(self, flag=None):
        if flag == "backward":
            core._set_prim_backward_enabled(True)
        main_program = paddle.ir.core.Program()
        start_program = paddle.ir.core.Program()
        with paddle.static.program_guard(main_program, start_program):
            net = SimpNet()
            x = paddle.static.data('x', shape=self.shape_x, dtype='float32')
            x.stop_gradient = False
            scale = paddle.static.data(
                'scale', shape=self.norm_shape, dtype='float32'
            )
            scale.stop_gradient = True
            bias = paddle.static.data(
                'bias', shape=self.norm_shape, dtype='float32'
            )
            bias.stop_gradient = True
            output_grad = paddle.tensor.fill_constant(
                self.shape_x, dtype='float32', value=1.0
            )

            res = net(x, scale, bias, self.ep, self.begin_axis)
            whole_ops_before = [
                op.name() for op in main_program.global_block().ops
            ]
            [res2] = decompose(
                main_program,
                [res],
            )
            gradients = grad(res2, x, output_grad)

            if flag == "backward":
                assert (
                    "pd_op.layer_norm" in whole_ops_before
                    and "pd_op.layer_norm_grad" not in whole_ops_before
                )
                core._set_prim_forward_enabled(True)
                [res2] = decompose(
                    main_program, [res2], whitelist={"pd_op.layer_norm"}
                )
                whole_ops_after = [
                    op.name() for op in main_program.global_block().ops
                ]
                assert "pd_op.layer_norm" not in whole_ops_after
                core._set_prim_forward_enabled(False)

            exe = paddle.static.Executor()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'scale': self.scale,
                    'bias': self.bias,
                },
                fetch_list=[res2, gradients[0]],
            )

        if flag == "backward":
            core._set_prim_backward_enabled(False)
        return outs

    def test_prim_custom_vjp(self):
        res_ref = self.base_net()
        res = self.base_net("backward")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
