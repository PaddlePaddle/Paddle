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
# model: configs^mask_rcnn^mask_rcnn_r101_vd_fpn_1x_coco_single_dy2st_train
# api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR106(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1557,  # (shape: [2, 1], dtype: paddle.int64, stop_gradient: True)
        var_1558,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
        var_1559,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
        var_1560,  # (shape: [2, 28, 28], dtype: paddle.int32, stop_gradient: True)
        var_1561,  # (shape: [2], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1563 = paddle.tensor.manipulation.concat([var_1557])
        var_1564 = paddle.tensor.manipulation.concat([var_1558])
        var_1565 = paddle.tensor.manipulation.concat([var_1559], axis=0)
        var_1566 = paddle.tensor.manipulation.concat([var_1560], axis=0)
        var_1567 = paddle.tensor.manipulation.concat([var_1561], axis=0)
        return var_1564, var_1565, var_1566, var_1563, var_1567


class TestSIR106(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.randint(low=0, high=10, shape=[2, 1], dtype=paddle.int64),
            paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
            paddle.randint(
                low=0, high=10, shape=[2, 28, 28], dtype=paddle.int32
            ),
            paddle.rand(shape=[2], dtype=paddle.float32),
        )
        self.net = SIR106()

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
