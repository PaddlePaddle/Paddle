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

# repo: diffusers_sub_graph
# model: stable_diffusion
# api:paddle.nn.functional.activation.silu||api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.conv.conv2d||method:__add__||method:__truediv__
import unittest

import numpy as np

import paddle

paddle.seed(2024)


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1280, 1280, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1280],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 1280, 1, 1], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 1280, 1, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.activation.silu(var_0, None)
        var_3 = paddle.nn.functional.common.dropout(
            var_2,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_4 = paddle.nn.functional.conv.conv2d(
            var_3, self.parameter_0, self.parameter_1, [1, 1], 1, [1, 1], 1
        )
        var_5 = var_1 + var_4
        var_6 = var_5 / 1.0
        return var_6


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 1280, 1, 1], dtype=paddle.float32),
        paddle.rand(shape=[1, 1280, 1, 1], dtype=paddle.float32),
    )
    return inputs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = create_paddle_inputs()
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
