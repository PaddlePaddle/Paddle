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
# api||paddle.tensor.math.add_n
import unittest

import numpy as np

import paddle


class SIR106(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_2056,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2057,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2058,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2059,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2060,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2061,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2062,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_2063,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2064 = paddle.tensor.math.add_n(
            [
                var_2062,
                var_2063,
                var_2056,
                var_2057,
                var_2058,
                var_2059,
                var_2060,
                var_2061,
            ]
        )
        return var_2064


class TestSIR106(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
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
