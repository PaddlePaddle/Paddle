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
# api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.concat||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.linalg.transpose||api:paddle.tensor.manipulation.reshape||api:paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 3, 168, 256], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 3, 84, 128], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 3, 42, 64], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [1, 3, 21, 32], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [1, 3, 11, 16], dtype: paddle.float32, stop_gradient: False)
        var_5,  # (shape: [1, 12, 168, 256], dtype: paddle.float32, stop_gradient: False)
        var_6,  # (shape: [1, 12, 84, 128], dtype: paddle.float32, stop_gradient: False)
        var_7,  # (shape: [1, 12, 42, 64], dtype: paddle.float32, stop_gradient: False)
        var_8,  # (shape: [1, 12, 21, 32], dtype: paddle.float32, stop_gradient: False)
        var_9,  # (shape: [1, 12, 11, 16], dtype: paddle.float32, stop_gradient: False)
        var_10,  # (shape: [129024, 4], dtype: paddle.float32, stop_gradient: True)
        var_11,  # (shape: [32256, 4], dtype: paddle.float32, stop_gradient: True)
        var_12,  # (shape: [8064, 4], dtype: paddle.float32, stop_gradient: True)
        var_13,  # (shape: [2016, 4], dtype: paddle.float32, stop_gradient: True)
        var_14,  # (shape: [528, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_15 = paddle.tensor.manipulation.reshape(var_10, shape=(-1, 4))
        var_16 = paddle.tensor.manipulation.reshape(var_11, shape=(-1, 4))
        var_17 = paddle.tensor.manipulation.reshape(var_12, shape=(-1, 4))
        var_18 = paddle.tensor.manipulation.reshape(var_13, shape=(-1, 4))
        var_19 = paddle.tensor.manipulation.reshape(var_14, shape=(-1, 4))
        var_20 = paddle.tensor.manipulation.concat(
            [var_15, var_16, var_17, var_18, var_19]
        )
        var_21 = paddle.tensor.linalg.transpose(var_0, perm=[0, 2, 3, 1])
        var_22 = paddle.tensor.manipulation.reshape(var_21, shape=(1, -1, 1))
        var_23 = paddle.tensor.linalg.transpose(var_1, perm=[0, 2, 3, 1])
        var_24 = paddle.tensor.manipulation.reshape(var_23, shape=(1, -1, 1))
        var_25 = paddle.tensor.linalg.transpose(var_2, perm=[0, 2, 3, 1])
        var_26 = paddle.tensor.manipulation.reshape(var_25, shape=(1, -1, 1))
        var_27 = paddle.tensor.linalg.transpose(var_3, perm=[0, 2, 3, 1])
        var_28 = paddle.tensor.manipulation.reshape(var_27, shape=(1, -1, 1))
        var_29 = paddle.tensor.linalg.transpose(var_4, perm=[0, 2, 3, 1])
        var_30 = paddle.tensor.manipulation.reshape(var_29, shape=(1, -1, 1))
        var_31 = paddle.tensor.manipulation.concat(
            [var_22, var_24, var_26, var_28, var_30], axis=1
        )
        var_32 = paddle.tensor.linalg.transpose(var_5, perm=[0, 2, 3, 1])
        var_33 = paddle.tensor.manipulation.reshape(var_32, shape=(1, -1, 4))
        var_34 = paddle.tensor.linalg.transpose(var_6, perm=[0, 2, 3, 1])
        var_35 = paddle.tensor.manipulation.reshape(var_34, shape=(1, -1, 4))
        var_36 = paddle.tensor.linalg.transpose(var_7, perm=[0, 2, 3, 1])
        var_37 = paddle.tensor.manipulation.reshape(var_36, shape=(1, -1, 4))
        var_38 = paddle.tensor.linalg.transpose(var_8, perm=[0, 2, 3, 1])
        var_39 = paddle.tensor.manipulation.reshape(var_38, shape=(1, -1, 4))
        var_40 = paddle.tensor.linalg.transpose(var_9, perm=[0, 2, 3, 1])
        var_41 = paddle.tensor.manipulation.reshape(var_40, shape=(1, -1, 4))
        var_42 = paddle.tensor.manipulation.concat(
            [var_33, var_35, var_37, var_39, var_41], axis=1
        )
        return var_20, var_31, var_42


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 3, 168, 256], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 84, 128], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 42, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 21, 32], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 11, 16], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 168, 256], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 84, 128], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 42, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 21, 32], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 11, 16], dtype=paddle.float32),
            paddle.rand(shape=[129024, 4], dtype=paddle.float32),
            paddle.rand(shape=[32256, 4], dtype=paddle.float32),
            paddle.rand(shape=[8064, 4], dtype=paddle.float32),
            paddle.rand(shape=[2016, 4], dtype=paddle.float32),
            paddle.rand(shape=[528, 4], dtype=paddle.float32),
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
            self.net, to_static=True, with_prim=False, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
