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
# method||reshape,api||paddle.nn.functional.activation.softmax,method||topk,method||mean,api||paddle.tensor.manipulation.concat,method||reshape,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.ops.sigmoid
import unittest

import numpy as np

import paddle


class SIR112(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_557 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_552 = self.create_parameter(
            shape=[64, 20, 1, 1],
            dtype=paddle.float32,
        )
        self.var_553 = self.create_parameter(
            shape=[64],
            dtype=paddle.float32,
        )
        self.var_556 = self.create_parameter(
            shape=[1, 64, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_544,  # (shape: [1, 68, 25, 38], dtype: paddle.float32, stop_gradient: False)
    ):
        var_545 = var_544.reshape([1, 4, -1, 25, 38])
        var_546 = paddle.nn.functional.activation.softmax(var_545, axis=2)
        out = var_546.topk(4, axis=2)
        var_547 = out[0]
        var_548 = out[1]
        var_549 = var_547.mean(axis=2, keepdim=True)
        var_550 = paddle.tensor.manipulation.concat([var_547, var_549], axis=2)
        var_551 = var_550.reshape([1, 20, 25, 38])
        var_554 = paddle.nn.functional.conv._conv_nd(
            var_551,
            self.var_552,
            bias=self.var_553,
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
        var_555 = paddle.nn.functional.activation.relu(var_554)
        var_558 = paddle.nn.functional.conv._conv_nd(
            var_555,
            self.var_556,
            bias=self.var_557,
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
        var_559 = paddle.tensor.ops.sigmoid(var_558)
        return var_559


class TestSIR112(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 68, 25, 38], dtype=paddle.float32),
        )
        self.net = SIR112()

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
