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
from paddle.decomposition import decomp
from paddle.framework import core

paddle.enable_static()


class LlamaRMSNorm(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        # 1. variance = hidden_states.pow(2).mean(-1, keepdim=True)

        axis_rst = 2
        # 1.1 decomp pow -> elementwise_pow
        pow_tensor = paddle.full([1], axis_rst, hidden_states.dtype)
        pow_rst = paddle.pow(hidden_states, pow_tensor)

        # 1.2 decomp mean -> sum & div
        sum_rst = paddle.sum(pow_rst, [axis_rst], keepdim=True)
        shape_rst = paddle.shape(sum_rst)
        div_by = paddle.full(shape_rst, hidden_states.shape[axis_rst])
        variance = paddle.divide(sum_rst, div_by)

        # 2. paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        # 2.1 decomp variance + self.variance_epsilon -> full + scale
        scale_tensor = paddle.full([1], 1.0)
        scale_rst = paddle.scale(variance, scale_tensor, self.variance_epsilon)

        # 2.2 decomp rsqrt -> pow(-0.5)
        rsqrt_tensor = paddle.full([1], -0.5)
        rsqrt_rst = paddle.pow(scale_rst, rsqrt_tensor)

        hidden_states = rsqrt_rst * hidden_states

        # hidden_states = (
        #     paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        # )

        return hidden_states * self.weight


class TestPrimMode(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 1024, 1024]
        self.shape_y = [2, 1024, 1]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")

    def base_net(self, flag=None):
        if flag == "all":
            core._set_prim_all_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data('x', [-1, 1024, 1024], dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            # x1 = paddle.nn.functional.relu(x)
            # x1 = paddle.prod(x[:-1])
            # y.stop_gradient = False
            print(x.shape, y.shape)
            z = paddle.maximum(x, y)
            # res = paddle.nn.functional.gelu(z)

            # pm = paddle.base.libpaddle.pir.PassManager()
            # paddle.base.libpaddle.pir.infer_symbolic_shape_pass(pm, main_program)
            # pm.run(main_program)

            [res2] = decomp.decompose(main_program, [z])
            # paddle.base.libpaddle.pir.infer_symbolic_shape_pass(pm, main_program)
            # pm.run(main_program)
            # gradients = grad(res2, (x, y))
            exe = paddle.static.Executor()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                },
                fetch_list=[res2],
            )

        whole_ops = [op.name() for op in main_program.global_block().ops]
        if flag == "all":
            core._set_prim_all_enabled(False)
            # assert (
            #     'pd_op.gelu' not in whole_ops
            #     and 'pd_op.divide_grad' in whole_ops
            # )
        return outs

    def test_prim_all_dynamic(self):
        res_ref = self.base_net()
        res = self.base_net("all")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
