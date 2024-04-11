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

# repo: diffusers_sub_grpah
# model: stable_diffusion
# api:paddle.nn.functional.activation.silu||api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.conv.conv2d||api:paddle.nn.functional.conv.conv2d||method:__add__||method:__truediv__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 256, 4, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        return paddle.nn.functional.batch_norm(
            var_0,
            self.parameter_0,
            self.parameter_1,
            self.parameter_2,
            self.parameter_3,
            training=True,
        )


def create_paddle_inputs():
    inputs = (paddle.rand(shape=[16, 32, 12, 12], dtype=paddle.float32),)
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
            np.testing.assert_allclose(
                st.numpy(), cinn.numpy(), atol=1e-6, rtol=1e-5
            )


if __name__ == '__main__':
    unittest.main()
