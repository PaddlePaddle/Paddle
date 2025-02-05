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
# api:paddle.tensor.creation.arange||method:__rmul__||method:__truediv__||api:paddle.tensor.ops.exp||method:__getitem__||method:cast||method:__getitem__||method:__mul__||method:__rmul__||api:paddle.tensor.ops.sin||api:paddle.tensor.ops.cos||api:paddle.tensor.manipulation.concat||method:__getitem__||method:__getitem__||api:paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1], dtype: paddle.int64, stop_gradient: True)
    ):
        var_1 = paddle.tensor.creation.arange(start=0, end=160, dtype='float32')
        var_2 = -9.210340371976184 * var_1
        var_3 = var_2 / 160
        var_4 = paddle.tensor.ops.exp(var_3)
        var_5 = var_0[
            (
                slice(None, None, None),
                None,
            )
        ]
        var_6 = var_5.cast('float32')
        var_7 = var_4[
            (
                None,
                slice(None, None, None),
            )
        ]
        var_8 = var_6 * var_7
        var_9 = 1 * var_8
        var_10 = paddle.tensor.ops.sin(var_9)
        var_11 = paddle.tensor.ops.cos(var_9)
        var_12 = paddle.tensor.manipulation.concat([var_10, var_11], axis=-1)
        var_13 = var_12[
            (
                slice(None, None, None),
                slice(160, None, None),
            )
        ]
        var_14 = var_12[
            (
                slice(None, None, None),
                slice(None, 160, None),
            )
        ]
        var_15 = paddle.tensor.manipulation.concat([var_13, var_14], axis=-1)
        return var_15


def create_paddle_inputs():
    inputs = (paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int64),)
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
