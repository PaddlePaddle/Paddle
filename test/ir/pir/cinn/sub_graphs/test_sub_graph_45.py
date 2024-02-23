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

# repo: PaddleDetection
# model: configs^sparse_rcnn^sparse_rcnn_r50_fpn_3x_pro100_coco_single_dy2st_train
# method:clone||method:unbind||method:__rmul__||method:__sub__||method:__rmul__||method:__sub__||method:__rmul__||method:__add__||method:__rmul__||method:__add__||api:paddle.tensor.manipulation.stack||method:unsqueeze||method:unsqueeze||method:__mul__||method:unsqueeze||method:tile||method:clone
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[100, 256],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[100, 4],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = self.parameter_1.clone()
        var_2, var_3, var_4, var_5 = var_1.unbind(-1)
        var_6 = 0.5 * var_4
        var_7 = var_2 - var_6
        var_8 = 0.5 * var_5
        var_9 = var_3 - var_8
        var_10 = 0.5 * var_4
        var_11 = var_2 + var_10
        var_12 = 0.5 * var_5
        var_13 = var_3 + var_12
        var_14 = paddle.tensor.manipulation.stack(
            [var_7, var_9, var_11, var_13], axis=-1
        )
        var_15 = var_14.unsqueeze(0)
        var_16 = var_0.unsqueeze(-2)
        var_17 = var_15 * var_16
        var_18 = self.parameter_0.unsqueeze(0)
        var_19 = var_18.tile([1, 1, 1])
        var_20 = var_19.clone()
        return var_20, var_17


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[1, 4], dtype=paddle.float32),)
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
