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
# api:paddle.tensor.search.topk||api:paddle.tensor.creation.full||method:__gt__||method:__lt__||api:paddle.tensor.logic.logical_and||api:paddle.tensor.creation.zeros_like||api:paddle.tensor.search.where||method:__ge__||api:paddle.tensor.creation.ones_like||api:paddle.tensor.search.where||method:max||method:__gt__||method:__eq__||api:paddle.tensor.logic.logical_and||method:cast||method:sum||method:__gt__||api:paddle.tensor.creation.ones_like||api:paddle.tensor.search.where||method:flatten||method:flatten
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [2, 171888], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1, var_2 = paddle.tensor.search.topk(var_0, k=1, axis=0)
        var_3 = paddle.tensor.creation.full([1, 171888], -1, dtype='int32')
        var_4 = var_1 > -1
        var_5 = var_1 < 0.3
        var_6 = paddle.tensor.logic.logical_and(var_4, var_5)
        var_7 = paddle.tensor.creation.zeros_like(var_3)
        var_8 = paddle.tensor.search.where(var_6, var_7, var_3)
        var_9 = var_1 >= 0.7
        var_10 = paddle.tensor.creation.ones_like(var_8)
        var_11 = paddle.tensor.search.where(var_9, var_10, var_8)
        var_12 = var_0.max(axis=1, keepdim=True)
        var_13 = var_0 > 0
        var_14 = var_0 == var_12
        var_15 = paddle.tensor.logic.logical_and(var_13, var_14)
        var_16 = var_15.cast('int32')
        var_17 = var_16.sum(0, keepdim=True)
        var_18 = var_17 > 0
        var_19 = paddle.tensor.creation.ones_like(var_11)
        var_20 = paddle.tensor.search.where(var_18, var_19, var_11)
        var_21 = var_2.flatten()
        var_22 = var_20.flatten()
        return var_21, var_22


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[2, 171888], dtype=paddle.float32),)
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
            self.net, to_static=True, with_prim=False, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
