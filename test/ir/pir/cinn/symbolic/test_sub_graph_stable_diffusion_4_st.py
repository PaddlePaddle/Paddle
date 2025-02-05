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
# api:paddle.randn||method:__mul__||method:__add__||method:__mul__||api:paddle.randn||api:paddle.randint||method:cast||method:__getitem__||method:__pow__||method:flatten||method:unsqueeze||method:unsqueeze||method:unsqueeze||method:__getitem__||method:__rsub__||method:__pow__||method:flatten||method:unsqueeze||method:unsqueeze||method:unsqueeze||method:__mul__||method:__mul__||method:__add__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 4, 1, 1], dtype: paddle.float32, stop_gradient: True)
        var_1,  # (shape: [1, 4, 1, 1], dtype: paddle.float32, stop_gradient: True)
        var_2,  # (shape: [1000], dtype: paddle.float32, stop_gradient: True)
    ):
        var_3 = paddle.randn([1, 4, 1, 1], dtype='float32')
        var_4 = var_1 * var_3
        var_5 = var_0 + var_4
        var_6 = var_5 * 0.18215
        var_7 = paddle.randn([1, 4, 1, 1])
        var_8 = paddle.randint(0, 1000, (1,))
        var_9 = var_8.cast('int64')
        var_10 = var_2[var_9]
        var_11 = var_10**0.5
        var_12 = var_11.flatten()
        var_13 = var_12.unsqueeze(-1)
        var_14 = var_13.unsqueeze(-1)
        var_15 = var_14.unsqueeze(-1)
        var_16 = var_2[var_9]
        var_17 = 1 - var_16
        var_18 = var_17**0.5
        var_19 = var_18.flatten()
        var_20 = var_19.unsqueeze(-1)
        var_21 = var_20.unsqueeze(-1)
        var_22 = var_21.unsqueeze(-1)
        var_23 = var_15 * var_6
        var_24 = var_22 * var_7
        var_25 = var_23 + var_24
        return var_25, var_9, var_6, var_7


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 4, 1, 1], dtype=paddle.float32),
        paddle.rand(shape=[1, 4, 1, 1], dtype=paddle.float32),
        paddle.rand(shape=[1000], dtype=paddle.float32),
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
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
