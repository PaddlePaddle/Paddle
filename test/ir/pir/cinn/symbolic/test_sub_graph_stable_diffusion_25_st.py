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
# method:cast||method:cast||api:paddle.nn.functional.loss.mse_loss||method:mean||method:mean
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 4, 1, 1], dtype: paddle.float32, stop_gradient: True)
        var_1,  # (shape: [1, 4, 1, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = var_1.cast('float32')
        var_3 = var_0.cast('float32')
        var_4 = paddle.nn.functional.loss.mse_loss(
            var_2, var_3, reduction='none'
        )
        var_5 = var_4.mean([1, 2, 3])
        var_6 = var_5.mean()
        return var_6


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 4, 1, 1], dtype=paddle.float32),
        paddle.rand(shape=[1, 4, 1, 1], dtype=paddle.float32),
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
