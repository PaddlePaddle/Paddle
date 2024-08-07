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

# repo: llm_sub_graphs
# model: chatglm2
# method:reshape||method:reshape||method:transpose||method:transpose||api:paddle.tensor.linalg.bmm||method:__mul__||method:reshape||method:astype||method:__mul__||method:__add__||method:astype||api:paddle.nn.functional.activation.softmax||method:astype||api:paddle.nn.functional.common.dropout||method:reshape||method:reshape||method:transpose||api:paddle.tensor.linalg.bmm||method:reshape||method:transpose||method:reshape
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1024, 4, 4, 8], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1024, 4, 4, 8], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1024, 4, 4, 8], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [4, 1, 1024, 1024], dtype: paddle.float32, stop_gradient: True)
    ):
        var_4 = var_0.reshape([1024, 16, -1])
        var_5 = var_1.reshape([1024, 16, -1])
        var_6 = var_4.transpose([1, 0, 2])
        var_7 = var_5.transpose([1, 2, 0])
        var_8 = paddle.tensor.linalg.bmm(var_6, var_7)
        var_9 = var_8 * 0.027196414661021056
        var_10 = var_9.reshape((4, 4, 1024, 1024))
        var_11 = var_10.astype('float32')
        var_12 = var_11 * 13
        var_13 = var_12 + var_3
        var_14 = var_13.astype('float32')
        var_15 = paddle.nn.functional.activation.softmax(var_14, axis=-1)
        var_16 = var_15.astype('float32')
        var_17 = paddle.nn.functional.common.dropout(
            var_16,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_18 = var_2.reshape([1024, 16, -1])
        var_19 = var_17.reshape([16, 1024, -1])
        var_20 = var_18.transpose([1, 0, 2])
        var_21 = paddle.tensor.linalg.bmm(var_19, var_20)
        var_22 = var_21.reshape((4, 4, 1024, 8))
        var_23 = var_22.transpose([2, 0, 1, 3])
        var_24 = var_23.reshape([1024, 4, 32])
        return var_24


def create_paddle_inputs():
    inputs = (
        paddle.rand(shape=[1024, 4, 4, 8], dtype=paddle.float32),
        paddle.rand(shape=[1024, 4, 4, 8], dtype=paddle.float32),
        paddle.rand(shape=[1024, 4, 4, 8], dtype=paddle.float32),
        paddle.rand(shape=[4, 1, 1024, 1024], dtype=paddle.float32),
    )
    return inputs


def create_numpy_inputs():
    inputs = (
        np.random.random(size=[1024, 4, 4, 8]).astype('float32'),
        np.random.random(size=[1024, 4, 4, 8]).astype('float32'),
        np.random.random(size=[1024, 4, 4, 8]).astype('float32'),
        np.random.random(size=[4, 1, 1024, 1024]).astype('float32'),
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
