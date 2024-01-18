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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^Res2Net^Res2Net50_14w_8s
# api||paddle.nn.functional.pooling.avg_pool2d,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR87(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1070,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1071,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1072,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1073,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1074,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1075,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1076,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1084,  # (shape: [22, 28, 56, 56], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1086 = paddle.nn.functional.pooling.avg_pool2d(
            var_1084,
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=False,
            exclusive=True,
            divisor_override=None,
            data_format='NCHW',
            name=None,
        )
        var_1087 = paddle.tensor.manipulation.concat(
            [
                var_1070,
                var_1071,
                var_1072,
                var_1073,
                var_1074,
                var_1075,
                var_1076,
                var_1086,
            ],
            axis=1,
        )
        return var_1087, var_1086


class TestSIR87(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 56, 56], dtype=paddle.float32),
        )
        self.net = SIR87()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
