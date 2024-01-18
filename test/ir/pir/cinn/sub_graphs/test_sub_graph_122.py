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
# model: configs^gfl^gfl_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.tensor.ops.sigmoid,api||paddle.tensor.creation.zeros,api||paddle.nn.functional.loss.binary_cross_entropy_with_logits,method||pow,method||__mul__,method||__ge__,method||__lt__,api||paddle.tensor.logic.logical_and,method||nonzero,method||squeeze
import unittest

import numpy as np

import paddle


class SIR149(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1000,  # (shape: [3800], dtype: paddle.int64, stop_gradient: True)
        var_999,  # (shape: [3800, 80], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1002 = paddle.tensor.ops.sigmoid(var_999)
        var_1003 = paddle.tensor.creation.zeros([3800, 80], dtype='float32')
        var_1004 = paddle.nn.functional.loss.binary_cross_entropy_with_logits(
            var_999, var_1003, reduction='none'
        )
        var_1005 = var_1002.pow(2.0)
        var_1006 = var_1004.__mul__(var_1005)
        var_1007 = var_1000.__ge__(0)
        var_1008 = var_1000.__lt__(80)
        var_1009 = paddle.tensor.logic.logical_and(var_1007, var_1008)
        var_1010 = var_1009.nonzero()
        var_1011 = var_1010.squeeze(1)
        return var_1011, var_1002, var_1006


class TestSIR149(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.randint(low=0, high=10, shape=[3800], dtype=paddle.int64),
            paddle.rand(shape=[3800, 80], dtype=paddle.float32),
        )
        self.net = SIR149()

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
