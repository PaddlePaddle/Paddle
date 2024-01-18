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
# model: configs^ppyolo^ppyolov2_r50vd_dcn_365e_coco_single_dy2st_train
# api||paddle.tensor.math.maximum,api||paddle.tensor.math.maximum,api||paddle.tensor.math.minimum,api||paddle.tensor.math.minimum,method||__sub__,method||clip,method||__sub__,method||clip,method||__mul__,method||__sub__,method||__sub__,method||__mul__,method||clip,method||__sub__,method||__sub__,method||__mul__,method||clip,method||__add__,method||__sub__,method||__add__,method||__truediv__,method||__mul__,method||__rsub__,method||__mul__
import unittest

import numpy as np

import paddle


class SIR81(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_624,  # (shape: [1, 3, 26, 26, 1], dtype: paddle.float32, stop_gradient: False)
        var_625,  # (shape: [1, 3, 26, 26, 1], dtype: paddle.float32, stop_gradient: False)
        var_626,  # (shape: [1, 3, 26, 26, 1], dtype: paddle.float32, stop_gradient: False)
        var_627,  # (shape: [1, 3, 26, 26, 1], dtype: paddle.float32, stop_gradient: False)
        var_628,  # (shape: [1, 3, 26, 26, 1], dtype: paddle.float32, stop_gradient: True)
        var_629,  # (shape: [1, 3, 26, 26, 1], dtype: paddle.float32, stop_gradient: True)
        var_630,  # (shape: [1, 3, 26, 26, 1], dtype: paddle.float32, stop_gradient: True)
        var_631,  # (shape: [1, 3, 26, 26, 1], dtype: paddle.float32, stop_gradient: True)
    ):
        var_632 = paddle.tensor.math.maximum(var_624, var_628)
        var_633 = paddle.tensor.math.maximum(var_625, var_629)
        var_634 = paddle.tensor.math.minimum(var_626, var_630)
        var_635 = paddle.tensor.math.minimum(var_627, var_631)
        var_636 = var_634.__sub__(var_632)
        var_637 = var_636.clip(0)
        var_638 = var_635.__sub__(var_633)
        var_639 = var_638.clip(0)
        var_640 = var_637.__mul__(var_639)
        var_641 = var_626.__sub__(var_624)
        var_642 = var_627.__sub__(var_625)
        var_643 = var_641.__mul__(var_642)
        var_644 = var_643.clip(0)
        var_645 = var_630.__sub__(var_628)
        var_646 = var_631.__sub__(var_629)
        var_647 = var_645.__mul__(var_646)
        var_648 = var_647.clip(0)
        var_649 = var_644.__add__(var_648)
        var_650 = var_649.__sub__(var_640)
        var_651 = var_650.__add__(1e-09)
        var_652 = var_640.__truediv__(var_651)
        var_653 = var_652.__mul__(var_652)
        var_654 = var_653.__rsub__(1)
        var_655 = var_654.__mul__(2.5)
        return var_655


class TestSIR81(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 3, 26, 26, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 26, 26, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 26, 26, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 26, 26, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 26, 26, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 26, 26, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 26, 26, 1], dtype=paddle.float32),
            paddle.rand(shape=[1, 3, 26, 26, 1], dtype=paddle.float32),
        )
        self.net = SIR81()

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
