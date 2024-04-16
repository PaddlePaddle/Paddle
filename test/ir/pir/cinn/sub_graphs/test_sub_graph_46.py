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
# model: configs^yolox^yolox_l_300e_coco_single_dy2st_train
# api:paddle.tensor.ops.sigmoid||method:flatten||method:transpose||api:paddle.tensor.manipulation.split||method:flatten||method:transpose||api:paddle.tensor.ops.sigmoid||method:flatten||method:transpose
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 5, 50, 50], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 80, 50, 50], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.tensor.ops.sigmoid(var_1)
        var_3 = var_2.flatten(2)
        var_4 = var_3.transpose([0, 2, 1])
        var_5, var_6 = paddle.tensor.manipulation.split(var_0, [4, 1], axis=1)
        var_7 = var_5.flatten(2)
        var_8 = var_7.transpose([0, 2, 1])
        var_9 = paddle.tensor.ops.sigmoid(var_6)
        var_10 = var_9.flatten(2)
        var_11 = var_10.transpose([0, 2, 1])
        return var_2, var_8, var_6, var_9, var_4, var_11


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 5, 50, 50], dtype=paddle.float32),
            paddle.rand(shape=[1, 80, 50, 50], dtype=paddle.float32),
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
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
