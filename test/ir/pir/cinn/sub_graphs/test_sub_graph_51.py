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
# api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||method:flatten||method:transpose||method:flatten||method:transpose||method:flatten||method:transpose
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[91, 192, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[91],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 192, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 15, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 4, 64, 64], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_0,
            bias=self.parameter_1,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_4 = paddle.tensor.ops.sigmoid(var_1)
        var_5 = var_4.flatten(2)
        var_6 = var_5.transpose([0, 2, 1])
        var_7 = var_2.flatten(2)
        var_8 = var_7.transpose([0, 2, 1])
        var_9 = var_3.flatten(2)
        var_10 = var_9.transpose([0, 2, 1])
        return var_3, var_4, var_6, var_8, var_10


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 192, 64, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 15, 64, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 4, 64, 64], dtype=paddle.float32),
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
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
