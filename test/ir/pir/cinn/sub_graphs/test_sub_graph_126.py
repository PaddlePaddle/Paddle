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
# model: configs^gfl^gflv2_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.tensor.creation.zeros,api||paddle.nn.functional.loss.binary_cross_entropy,method||pow,method||__mul__,method||__ge__,method||__lt__,api||paddle.tensor.logic.logical_and,method||nonzero,method||squeeze
import unittest

import numpy as np

import paddle


class SIR167(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1110,  # (shape: [247, 81], dtype: paddle.float32, stop_gradient: False)
        var_1111,  # (shape: [247], dtype: paddle.int64, stop_gradient: True)
    ):
        var_1113 = paddle.tensor.creation.zeros([247, 81], dtype='float32')
        var_1114 = paddle.nn.functional.loss.binary_cross_entropy(
            var_1110, var_1113, reduction='none'
        )
        var_1115 = var_1110.pow(2.0)
        var_1116 = var_1114.__mul__(var_1115)
        var_1117 = var_1111.__ge__(0)
        var_1118 = var_1111.__lt__(81)
        var_1119 = paddle.tensor.logic.logical_and(var_1117, var_1118)
        var_1120 = var_1119.nonzero()
        var_1121 = var_1120.squeeze(1)
        return var_1121, var_1116


class TestSIR167(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[247, 81], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[247], dtype=paddle.int64),
        )
        self.net = SIR167()

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
