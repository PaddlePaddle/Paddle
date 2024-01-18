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
# api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.hardsigmoid,api||paddle.tensor.math.multiply
import unittest

import numpy as np

import paddle


class SIR70(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_275 = self.create_parameter(
            shape=[24],
            dtype=paddle.float32,
        )
        self.var_278 = self.create_parameter(
            shape=[96, 24, 1, 1],
            dtype=paddle.float32,
        )
        self.var_279 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )
        self.var_274 = self.create_parameter(
            shape=[24, 96, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_269,  # (shape: [1, 48, 20, 20], dtype: paddle.float32, stop_gradient: False)
        var_270,  # (shape: [1, 48, 20, 20], dtype: paddle.float32, stop_gradient: False)
    ):
        var_272 = paddle.tensor.manipulation.concat([var_270, var_269], axis=1)
        var_273 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_272, output_size=1, data_format='NCHW', name=None
        )
        var_276 = paddle.nn.functional.conv._conv_nd(
            var_273,
            self.var_274,
            bias=self.var_275,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_277 = paddle.nn.functional.activation.relu(var_276)
        var_280 = paddle.nn.functional.conv._conv_nd(
            var_277,
            self.var_278,
            bias=self.var_279,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_281 = paddle.nn.functional.activation.hardsigmoid(var_280)
        var_282 = paddle.tensor.math.multiply(x=var_272, y=var_281)
        return var_282


class TestSIR70(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 48, 20, 20], dtype=paddle.float32),
            paddle.rand(shape=[1, 48, 20, 20], dtype=paddle.float32),
        )
        self.net = SIR70()

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
