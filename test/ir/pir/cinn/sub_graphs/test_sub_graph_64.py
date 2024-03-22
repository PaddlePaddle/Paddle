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
# model: configs^cascade_rcnn^cascade_rcnn_r50_fpn_1x_coco_single_dy2st_train
# api:paddle.tensor.manipulation.flatten||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.activation.relu
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[12544, 1024],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[1024, 1024],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [512, 256, 7, 7], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.tensor.manipulation.flatten(
            var_0, start_axis=1, stop_axis=-1
        )
        var_2 = paddle.nn.functional.common.linear(
            x=var_1, weight=self.parameter_0, bias=self.parameter_3, name=None
        )
        var_3 = paddle.nn.functional.activation.relu(var_2)
        var_4 = paddle.nn.functional.common.linear(
            x=var_3, weight=self.parameter_2, bias=self.parameter_1, name=None
        )
        var_5 = paddle.nn.functional.activation.relu(var_4)
        return var_5


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[512, 256, 7, 7], dtype=paddle.float32),
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

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
