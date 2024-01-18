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
# model: configs^ssd^ssd_vgg16_300_240e_voc_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape
import unittest

import numpy as np

import paddle


class SIR3(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_522 = self.create_parameter(
            shape=[16, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_567 = self.create_parameter(
            shape=[84, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_543 = self.create_parameter(
            shape=[24],
            dtype=paddle.float32,
        )
        self.var_538 = self.create_parameter(
            shape=[126],
            dtype=paddle.float32,
        )
        self.var_553 = self.create_parameter(
            shape=[24],
            dtype=paddle.float32,
        )
        self.var_563 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.var_527 = self.create_parameter(
            shape=[84, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_523 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.var_542 = self.create_parameter(
            shape=[24, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_572 = self.create_parameter(
            shape=[16, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_548 = self.create_parameter(
            shape=[126],
            dtype=paddle.float32,
        )
        self.var_577 = self.create_parameter(
            shape=[84, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_528 = self.create_parameter(
            shape=[84],
            dtype=paddle.float32,
        )
        self.var_578 = self.create_parameter(
            shape=[84],
            dtype=paddle.float32,
        )
        self.var_537 = self.create_parameter(
            shape=[126, 1024, 3, 3],
            dtype=paddle.float32,
        )
        self.var_573 = self.create_parameter(
            shape=[16],
            dtype=paddle.float32,
        )
        self.var_532 = self.create_parameter(
            shape=[24, 1024, 3, 3],
            dtype=paddle.float32,
        )
        self.var_552 = self.create_parameter(
            shape=[24, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_558 = self.create_parameter(
            shape=[126],
            dtype=paddle.float32,
        )
        self.var_533 = self.create_parameter(
            shape=[24],
            dtype=paddle.float32,
        )
        self.var_568 = self.create_parameter(
            shape=[84],
            dtype=paddle.float32,
        )
        self.var_557 = self.create_parameter(
            shape=[126, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_547 = self.create_parameter(
            shape=[126, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_562 = self.create_parameter(
            shape=[16, 256, 3, 3],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_513,  # (shape: [1, 512, 38, 38], dtype: paddle.float32, stop_gradient: False)
        var_514,  # (shape: [1, 1024, 19, 19], dtype: paddle.float32, stop_gradient: False)
        var_515,  # (shape: [1, 512, 10, 10], dtype: paddle.float32, stop_gradient: False)
        var_516,  # (shape: [1, 256, 5, 5], dtype: paddle.float32, stop_gradient: False)
        var_517,  # (shape: [1, 256, 3, 3], dtype: paddle.float32, stop_gradient: False)
        var_518,  # (shape: [1, 256, 1, 1], dtype: paddle.float32, stop_gradient: False)
    ):
        var_524 = paddle.nn.functional.conv._conv_nd(
            var_513,
            self.var_522,
            bias=self.var_523,
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
        var_525 = paddle.tensor.linalg.transpose(var_524, [0, 2, 3, 1])
        var_526 = paddle.tensor.manipulation.reshape(var_525, [0, -1, 4])
        var_529 = paddle.nn.functional.conv._conv_nd(
            var_513,
            self.var_527,
            bias=self.var_528,
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
        var_530 = paddle.tensor.linalg.transpose(var_529, [0, 2, 3, 1])
        var_531 = paddle.tensor.manipulation.reshape(var_530, [0, -1, 21])
        var_534 = paddle.nn.functional.conv._conv_nd(
            var_514,
            self.var_532,
            bias=self.var_533,
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
        var_535 = paddle.tensor.linalg.transpose(var_534, [0, 2, 3, 1])
        var_536 = paddle.tensor.manipulation.reshape(var_535, [0, -1, 4])
        var_539 = paddle.nn.functional.conv._conv_nd(
            var_514,
            self.var_537,
            bias=self.var_538,
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
        var_540 = paddle.tensor.linalg.transpose(var_539, [0, 2, 3, 1])
        var_541 = paddle.tensor.manipulation.reshape(var_540, [0, -1, 21])
        var_544 = paddle.nn.functional.conv._conv_nd(
            var_515,
            self.var_542,
            bias=self.var_543,
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
        var_545 = paddle.tensor.linalg.transpose(var_544, [0, 2, 3, 1])
        var_546 = paddle.tensor.manipulation.reshape(var_545, [0, -1, 4])
        var_549 = paddle.nn.functional.conv._conv_nd(
            var_515,
            self.var_547,
            bias=self.var_548,
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
        var_550 = paddle.tensor.linalg.transpose(var_549, [0, 2, 3, 1])
        var_551 = paddle.tensor.manipulation.reshape(var_550, [0, -1, 21])
        var_554 = paddle.nn.functional.conv._conv_nd(
            var_516,
            self.var_552,
            bias=self.var_553,
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
        var_555 = paddle.tensor.linalg.transpose(var_554, [0, 2, 3, 1])
        var_556 = paddle.tensor.manipulation.reshape(var_555, [0, -1, 4])
        var_559 = paddle.nn.functional.conv._conv_nd(
            var_516,
            self.var_557,
            bias=self.var_558,
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
        var_560 = paddle.tensor.linalg.transpose(var_559, [0, 2, 3, 1])
        var_561 = paddle.tensor.manipulation.reshape(var_560, [0, -1, 21])
        var_564 = paddle.nn.functional.conv._conv_nd(
            var_517,
            self.var_562,
            bias=self.var_563,
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
        var_565 = paddle.tensor.linalg.transpose(var_564, [0, 2, 3, 1])
        var_566 = paddle.tensor.manipulation.reshape(var_565, [0, -1, 4])
        var_569 = paddle.nn.functional.conv._conv_nd(
            var_517,
            self.var_567,
            bias=self.var_568,
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
        var_570 = paddle.tensor.linalg.transpose(var_569, [0, 2, 3, 1])
        var_571 = paddle.tensor.manipulation.reshape(var_570, [0, -1, 21])
        var_574 = paddle.nn.functional.conv._conv_nd(
            var_518,
            self.var_572,
            bias=self.var_573,
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
        var_575 = paddle.tensor.linalg.transpose(var_574, [0, 2, 3, 1])
        var_576 = paddle.tensor.manipulation.reshape(var_575, [0, -1, 4])
        var_579 = paddle.nn.functional.conv._conv_nd(
            var_518,
            self.var_577,
            bias=self.var_578,
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
        var_580 = paddle.tensor.linalg.transpose(var_579, [0, 2, 3, 1])
        var_581 = paddle.tensor.manipulation.reshape(var_580, [0, -1, 21])
        return (
            var_526,
            var_536,
            var_546,
            var_556,
            var_566,
            var_576,
            var_531,
            var_541,
            var_551,
            var_561,
            var_571,
            var_581,
        )


class TestSIR3(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 512, 38, 38], dtype=paddle.float32),
            paddle.rand(shape=[1, 1024, 19, 19], dtype=paddle.float32),
            paddle.rand(shape=[1, 512, 10, 10], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 5, 5], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 3, 3], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 1, 1], dtype=paddle.float32),
        )
        self.net = SIR3()

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
