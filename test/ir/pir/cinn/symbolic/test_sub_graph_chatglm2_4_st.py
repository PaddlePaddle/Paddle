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
# api:paddle.nn.functional.input.embedding||method:transpose||api:paddle.tensor.creation.ones||api:paddle.tensor.creation.tril||method:astype||api:paddle.tensor.creation.ones||method:astype||method:__and__||api:paddle.tensor.creation.arange||method:__truediv__||method:__rpow__||method:__rtruediv__||api:paddle.tensor.creation.arange||api:paddle.tensor.math.outer||method:astype||api:paddle.tensor.ops.cos||api:paddle.tensor.ops.sin||api:paddle.tensor.manipulation.stack||method:__getitem__||method:transpose
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[64896, 32],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [4, 1024], dtype: paddle.int64, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.input.embedding(
            var_0,
            weight=self.parameter_0,
            padding_idx=None,
            sparse=False,
            name=None,
        )
        var_2 = var_1.transpose([1, 0, 2])
        var_3 = paddle.tensor.creation.ones([4, 1, 1024, 1024])
        var_4 = paddle.tensor.creation.tril(var_3)
        var_5 = var_4.astype('bool')
        var_6 = paddle.tensor.creation.ones(
            (4, 1, 1024, 1024),
            dtype='bool',
        )
        var_7 = var_6.astype('bool')
        var_8 = var_5 and var_7
        var_9 = paddle.tensor.creation.arange(0, 4, 2, dtype='float32')
        var_10 = var_9 / 4
        var_11 = 10000**var_10
        var_12 = 1.0 / var_11
        var_13 = paddle.tensor.creation.arange(0, 1024, dtype='float32')
        var_14 = paddle.tensor.math.outer(var_13, var_12)
        var_15 = var_14.astype('float32')
        var_16 = paddle.tensor.ops.cos(var_15)
        var_17 = paddle.tensor.ops.sin(var_15)
        var_18 = paddle.tensor.manipulation.stack([var_16, var_17], axis=-1)
        var_19 = var_18[(None, slice(None, 1024, None))]
        var_20 = var_19.transpose([1, 0, 2, 3])
        return var_2, var_8, var_20


def create_paddle_inputs():
    inputs = (
        paddle.randint(low=0, high=10, shape=[4, 1024], dtype=paddle.int64),
    )
    return inputs


def create_numpy_inputs():
    inputs = (np.random.randint(low=0, high=10, size=[4, 1024], dtype='int64'),)
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
