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
# model: configs^rotate^fcosr^fcosr_x50_3x_dota_single_dy2st_train
# method||__sub__,method||pow,api||paddle.nn.functional.loss.binary_cross_entropy,method||__truediv__,method||__rmul__,method||__rmul__,method||__add__
import unittest

import numpy as np

import paddle


class SIR92(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1226,  # (shape: [1, 21824, 15], dtype: paddle.float32, stop_gradient: False)
        var_1227,  # (shape: [1, 21824, 15], dtype: paddle.float32, stop_gradient: True)
        var_1228,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1229 = var_1226.__sub__(var_1227)
        var_1230 = var_1229.pow(2.0)
        var_1231 = paddle.nn.functional.loss.binary_cross_entropy(
            var_1226, var_1227, weight=var_1230, reduction='sum'
        )
        var_1232 = var_1231.__truediv__(12)
        var_1233 = var_1232.__rmul__(1.0)
        var_1234 = var_1228.__rmul__(1.0)
        var_1235 = var_1233.__add__(var_1234)
        return var_1235, var_1232


class TestSIR92(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 21824, 15], dtype=paddle.float32),
            paddle.rand(shape=[1, 21824, 15], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
        )
        self.net = SIR92()

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
