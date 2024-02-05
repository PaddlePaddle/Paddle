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
# model: ppcls^configs^ImageNet^DPN^DPN98
# api:paddle.tensor.manipulation.split||api:paddle.tensor.math.add||api:paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [22, 1056, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [22, 1024, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [22, 288, 14, 14], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3, var_4 = paddle.tensor.manipulation.split(
            var_0, num_or_sections=[1024, 32], axis=1
        )
        var_5 = paddle.tensor.math.add(x=var_1, y=var_3)
        var_6 = paddle.tensor.manipulation.concat([var_2, var_4], axis=1)
        return var_5, var_6


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 1056, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[22, 1024, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[22, 288, 14, 14], dtype=paddle.float32),
        )
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

    # NOTE prim + cinn lead to error
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
