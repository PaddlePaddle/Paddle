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
# model: configs^rotate^ppyoloe_r^ppyoloe_r_crn_s_3x_dota_single_dy2st_train
# api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 192, 32, 32], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.pooling.max_pool2d(
            var_0,
            kernel_size=5,
            stride=1,
            padding=2,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_2 = paddle.nn.functional.pooling.max_pool2d(
            var_0,
            kernel_size=9,
            stride=1,
            padding=4,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_3 = paddle.nn.functional.pooling.max_pool2d(
            var_0,
            kernel_size=13,
            stride=1,
            padding=6,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_4 = paddle.tensor.manipulation.concat(
            [var_0, var_1, var_2, var_3], axis=1
        )
        return var_4


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 192, 32, 32], dtype=paddle.float32),
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
        # NOTE(Aurelius84): cinn_op.pool2d only support pool_type='avg' under adaptive=True
        paddle.set_flags({"FLAGS_deny_cinn_ops": "pool2d"})
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
