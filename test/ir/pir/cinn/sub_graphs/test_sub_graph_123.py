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
# method||__rmul__,method||__mul__,method||sum,method||__truediv__
import unittest

import numpy as np

import paddle


class SIR160(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1088,  # (shape: [247], dtype: paddle.float32, stop_gradient: False)
        var_1089,  # (shape: [247], dtype: paddle.float32, stop_gradient: True)
        var_1090,  # (shape: [], dtype: paddle.int64, stop_gradient: True)
    ):
        var_1091 = var_1088.__rmul__(1.0)
        var_1092 = var_1091.__mul__(var_1089)
        var_1093 = var_1092.sum()
        var_1094 = var_1093.__truediv__(var_1090)
        return var_1094


class TestSIR160(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[247], dtype=paddle.float32),
            paddle.rand(shape=[247], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int64),
        )
        self.net = SIR160()

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
