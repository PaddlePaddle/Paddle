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
# method:transpose||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:reshape||method:transpose||method:reshape||method:transpose||api:paddle.tensor.linalg.matmul||method:__mul__||method:cast||api:paddle.nn.functional.activation.softmax||method:cast||api:paddle.tensor.linalg.matmul||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout||method:transpose||method:reshape||method:__add__||method:__truediv__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[512, 512],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[512, 512],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[512, 512],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[512, 512],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 512, 1], dtype: paddle.float32, stop_gradient: True)
        var_1,  # (shape: [1, 512, 1, 1], dtype: paddle.float32, stop_gradient: True)
    ):
        var_2 = var_0.transpose([0, 2, 1])
        var_3 = paddle.nn.functional.common.linear(
            x=var_2, weight=self.parameter_0, bias=self.parameter_6, name=None
        )
        var_4 = paddle.nn.functional.common.linear(
            x=var_2, weight=self.parameter_2, bias=self.parameter_1, name=None
        )
        var_5 = paddle.nn.functional.common.linear(
            x=var_2, weight=self.parameter_5, bias=self.parameter_4, name=None
        )
        var_6 = var_3.reshape([0, 0, 1, 512])
        var_7 = var_6.transpose([0, 2, 1, 3])
        var_8 = var_4.reshape([0, 0, 1, 512])
        var_9 = var_8.transpose([0, 2, 1, 3])
        var_10 = var_5.reshape([0, 0, 1, 512])
        var_11 = var_10.transpose([0, 2, 1, 3])
        var_12 = paddle.tensor.linalg.matmul(var_7, var_9, transpose_y=True)
        var_13 = var_12 * 0.04419417382415922
        var_14 = var_13.cast('float32')
        var_15 = paddle.nn.functional.activation.softmax(var_14, axis=-1)
        var_16 = var_15.cast('float32')
        var_17 = paddle.tensor.linalg.matmul(var_16, var_11)
        var_18 = var_17.transpose([0, 2, 1, 3])
        var_19 = var_18.reshape([0, 0, 512])
        var_20 = paddle.nn.functional.common.linear(
            x=var_19, weight=self.parameter_3, bias=self.parameter_7, name=None
        )
        var_21 = paddle.nn.functional.common.dropout(
            var_20,
            p=0.0,
            axis=None,
            training=False,
            mode='upscale_in_train',
            name=None,
        )
        var_22 = var_21.transpose([0, 2, 1])
        var_23 = var_22.reshape([1, 512, 1, 1])
        var_24 = var_23 + var_1
        var_25 = var_24 / 1
        return var_25


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1, 512, 1], dtype=paddle.float32),
        paddle.rand(shape=[1, 512, 1, 1], dtype=paddle.float32),
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
