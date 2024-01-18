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
# model: configs^picodet^legacy_model^picodet_s_320_coco_single_dy2st_train
# method||cast,method||__add__,method||cast,method||__sub__,method||cast,method||__sub__,api||paddle.nn.functional.loss.cross_entropy,method||__mul__,api||paddle.nn.functional.loss.cross_entropy,method||__mul__,method||__add__,method||__rmul__,method||__mul__,method||sum,method||__truediv__
import unittest

import numpy as np

import paddle


class SIR286(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1638,  # (shape: [4, 8], dtype: paddle.float32, stop_gradient: False)
        var_1639,  # (shape: [4], dtype: paddle.float32, stop_gradient: True)
        var_1640,  # (shape: [4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1641 = var_1639.cast('int64')
        var_1642 = var_1641.__add__(1)
        var_1643 = var_1642.cast('float32')
        var_1644 = var_1643.__sub__(var_1639)
        var_1645 = var_1641.cast('float32')
        var_1646 = var_1639.__sub__(var_1645)
        var_1647 = paddle.nn.functional.loss.cross_entropy(
            var_1638, var_1641, reduction='none'
        )
        var_1648 = var_1647.__mul__(var_1644)
        var_1649 = paddle.nn.functional.loss.cross_entropy(
            var_1638, var_1642, reduction='none'
        )
        var_1650 = var_1649.__mul__(var_1646)
        var_1651 = var_1648.__add__(var_1650)
        var_1652 = var_1651.__rmul__(0.25)
        var_1653 = var_1652.__mul__(var_1640)
        var_1654 = var_1653.sum()
        var_1655 = var_1654.__truediv__(4.0)
        return var_1655


class TestSIR286(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[4, 8], dtype=paddle.float32),
            paddle.rand(shape=[4], dtype=paddle.float32),
            paddle.rand(shape=[4], dtype=paddle.float32),
        )
        self.net = SIR286()

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
