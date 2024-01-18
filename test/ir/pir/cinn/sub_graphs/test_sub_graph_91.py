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
# api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.concat,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.concat,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR42(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_570,  # (shape: [1, 3, 192, 288], dtype: paddle.float32, stop_gradient: False)
        var_571,  # (shape: [1, 3, 96, 144], dtype: paddle.float32, stop_gradient: False)
        var_572,  # (shape: [1, 3, 48, 72], dtype: paddle.float32, stop_gradient: False)
        var_573,  # (shape: [1, 3, 24, 36], dtype: paddle.float32, stop_gradient: False)
        var_574,  # (shape: [1, 3, 12, 18], dtype: paddle.float32, stop_gradient: False)
        var_575,  # (shape: [1, 12, 192, 288], dtype: paddle.float32, stop_gradient: False)
        var_576,  # (shape: [1, 12, 96, 144], dtype: paddle.float32, stop_gradient: False)
        var_577,  # (shape: [1, 12, 48, 72], dtype: paddle.float32, stop_gradient: False)
        var_578,  # (shape: [1, 12, 24, 36], dtype: paddle.float32, stop_gradient: False)
        var_579,  # (shape: [1, 12, 12, 18], dtype: paddle.float32, stop_gradient: False)
        var_580,  # (shape: [165888, 4], dtype: paddle.float32, stop_gradient: True)
        var_581,  # (shape: [41472, 4], dtype: paddle.float32, stop_gradient: True)
        var_582,  # (shape: [10368, 4], dtype: paddle.float32, stop_gradient: True)
        var_583,  # (shape: [2592, 4], dtype: paddle.float32, stop_gradient: True)
        var_584,  # (shape: [648, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_585 = paddle.tensor.manipulation.reshape(var_580, shape=(-1, 4))
        var_586 = paddle.tensor.manipulation.reshape(var_581, shape=(-1, 4))
        var_587 = paddle.tensor.manipulation.reshape(var_582, shape=(-1, 4))
        var_588 = paddle.tensor.manipulation.reshape(var_583, shape=(-1, 4))
        var_589 = paddle.tensor.manipulation.reshape(var_584, shape=(-1, 4))
        var_590 = paddle.tensor.manipulation.concat(
            [var_585, var_586, var_587, var_588, var_589]
        )
        var_591 = paddle.tensor.linalg.transpose(var_570, perm=[0, 2, 3, 1])
        var_592 = paddle.tensor.manipulation.reshape(var_591, shape=(1, -1, 1))
        var_593 = paddle.tensor.linalg.transpose(var_571, perm=[0, 2, 3, 1])
        var_594 = paddle.tensor.manipulation.reshape(var_593, shape=(1, -1, 1))
        var_595 = paddle.tensor.linalg.transpose(var_572, perm=[0, 2, 3, 1])
        var_596 = paddle.tensor.manipulation.reshape(var_595, shape=(1, -1, 1))
        var_597 = paddle.tensor.linalg.transpose(var_573, perm=[0, 2, 3, 1])
        var_598 = paddle.tensor.manipulation.reshape(var_597, shape=(1, -1, 1))
        var_599 = paddle.tensor.linalg.transpose(var_574, perm=[0, 2, 3, 1])
        var_600 = paddle.tensor.manipulation.reshape(var_599, shape=(1, -1, 1))
        var_601 = paddle.tensor.manipulation.concat(
            [var_592, var_594, var_596, var_598, var_600], axis=1
        )
        var_602 = paddle.tensor.linalg.transpose(var_575, perm=[0, 2, 3, 1])
        var_603 = paddle.tensor.manipulation.reshape(var_602, shape=(1, -1, 4))
        var_604 = paddle.tensor.linalg.transpose(var_576, perm=[0, 2, 3, 1])
        var_605 = paddle.tensor.manipulation.reshape(var_604, shape=(1, -1, 4))
        var_606 = paddle.tensor.linalg.transpose(var_577, perm=[0, 2, 3, 1])
        var_607 = paddle.tensor.manipulation.reshape(var_606, shape=(1, -1, 4))
        var_608 = paddle.tensor.linalg.transpose(var_578, perm=[0, 2, 3, 1])
        var_609 = paddle.tensor.manipulation.reshape(var_608, shape=(1, -1, 4))
        var_610 = paddle.tensor.linalg.transpose(var_579, perm=[0, 2, 3, 1])
        var_611 = paddle.tensor.manipulation.reshape(var_610, shape=(1, -1, 4))
        var_612 = paddle.tensor.manipulation.concat(
            [var_603, var_605, var_607, var_609, var_611], axis=1
        )
        return var_590, var_601, var_612


class TestSIR42(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 3, 192, 288], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 96, 144], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 48, 72], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 24, 36], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 12, 18], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 192, 288], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 96, 144], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 48, 72], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 24, 36], dtype=paddle.float32),
            paddle.rand(shape=[1, 12, 12, 18], dtype=paddle.float32),
            paddle.rand(shape=[165888, 4], dtype=paddle.float32),
            paddle.rand(shape=[41472, 4], dtype=paddle.float32),
            paddle.rand(shape=[10368, 4], dtype=paddle.float32),
            paddle.rand(shape=[2592, 4], dtype=paddle.float32),
            paddle.rand(shape=[648, 4], dtype=paddle.float32),
        )
        self.net = SIR42()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
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
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
