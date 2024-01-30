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
# method:cast||method:__add__||method:cast||method:__sub__||method:cast||method:__sub__||api:paddle.nn.functional.loss.cross_entropy||method:__mul__||api:paddle.nn.functional.loss.cross_entropy||method:__mul__||method:__add__||method:__rmul__||method:__mul__||method:sum||method:__truediv__
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [24, 8], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [24], dtype: paddle.float32, stop_gradient: True)
        var_2,  # (shape: [24], dtype: paddle.float32, stop_gradient: True)
    ):
        var_3 = var_1.cast('int64')
        var_4 = var_3 + 1
        var_5 = var_4.cast('float32')
        var_6 = var_5 - var_1
        var_7 = var_3.cast('float32')
        var_8 = var_1 - var_7
        var_9 = paddle.nn.functional.loss.cross_entropy(
            var_0, var_3, reduction='none'
        )
        var_10 = var_9 * var_6
        var_11 = paddle.nn.functional.loss.cross_entropy(
            var_0, var_4, reduction='none'
        )
        var_12 = var_11 * var_8
        var_13 = var_10 + var_12
        var_14 = 0.25 * var_13
        var_15 = var_14 * var_2
        var_16 = var_15.sum()
        var_17 = var_16 / 4.0
        return var_17


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[24, 8], dtype=paddle.float32),
            paddle.rand(shape=[24], dtype=paddle.float32),
            paddle.rand(shape=[24], dtype=paddle.float32),
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
            self.net, to_static=True, with_prim=True, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
