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

import unittest

import numpy as np

import paddle
from paddle.jit import to_static
from paddle.static import InputSpec


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn

    return paddle.jit.to_static(
        net,
        build_strategy=build_strategy,
        full_graph=True,
        input_spec=input_spec,
    )


class SubGraph1(paddle.nn.Layer):
    # subgraph from fcos_r50_fpn_1x_coco
    def __init__(self):
        super().__init__()

    @to_static
    def forward(self, x):
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
        x = paddle.cast(x, dtype='float32')
        shape = paddle.shape(x)
        new_shape = [shape[0] * 32, -1]
        x = paddle.reshape(x, new_shape)
        sum_1 = paddle.sum(x, axis=1, dtype='float32', keepdim=True)
        x = paddle.multiply(x, x)
        sum_2 = paddle.sum(x, axis=1, keepdim=True)
        return sum_1, sum_2


class SubGraph2(paddle.nn.Layer):
    # subgraph from faster_rcnn_r50_1x_coco
    def __init__(self):
        super().__init__()

    @to_static
    def forward(self, x, y):
        x = paddle.add(x, y)
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
        zero_tensor = paddle.full(shape=[], fill_value=0, dtype='float16')
        shape_tensor = paddle.shape(x)
        expanded_tensor = paddle.expand(zero_tensor, shape=shape_tensor)
        max_tensor = paddle.maximum(x, expanded_tensor)
        transposed_back = paddle.transpose(max_tensor, perm=[0, 2, 3, 1])
        return max_tensor, transposed_back


class SubGraph3(paddle.nn.Layer):
    # subgraph from msvsr
    def __init__(self):
        super().__init__()

    @to_static
    def forward(self, x):
        x = paddle.transpose(x, perm=[0, 3, 1, 2])
        return x


class TestSubGraph(unittest.TestCase):
    def eval(self, net_class, use_cinn, *args, input_spec=None):
        net = net_class()
        if input_spec:
            net = apply_to_static(net, use_cinn, input_spec)
        else:
            net = apply_to_static(net, use_cinn)
        net.eval()
        with paddle.no_grad():
            outputs = net(*args)
        return outputs

    def test_eval_subgraph1(self):
        inputs1 = paddle.uniform(
            [1, 16, 256, 256], dtype="float16", min=-0.5, max=0.5
        )
        inputs1.stop_gradient = False
        dy_out = self.eval(
            SubGraph1,
            False,
            inputs1,
            input_spec=[
                InputSpec(shape=[None, None, None, 256], dtype='float16')
            ],
        )
        cinn_out = self.eval(
            SubGraph1,
            True,
            inputs1,
            input_spec=[
                InputSpec(shape=[None, None, None, 256], dtype='float16')
            ],
        )
        np.testing.assert_allclose(
            dy_out[0].numpy(), cinn_out[0].numpy(), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            dy_out[1].numpy(), cinn_out[1].numpy(), atol=1e-5, rtol=1e-5
        )

    def test_eval_subgraph2(self):
        inputs1 = paddle.uniform(
            [1, 7, 7, 2048], dtype="float16", min=-0.5, max=0.5
        )
        inputs2 = paddle.uniform(
            [1, 7, 7, 2048], dtype="float16", min=-0.5, max=0.5
        )
        inputs1.stop_gradient = False
        inputs2.stop_gradient = False
        dy_out = self.eval(
            SubGraph2,
            False,
            inputs1,
            inputs2,
            input_spec=[
                InputSpec(shape=[None, None, None, 2048], dtype='float16'),
                InputSpec(shape=[None, None, None, 2048], dtype='float16'),
            ],
        )
        cinn_out = self.eval(
            SubGraph2,
            True,
            inputs1,
            inputs2,
            input_spec=[
                InputSpec(shape=[None, None, None, 2048], dtype='float16'),
                InputSpec(shape=[None, None, None, 2048], dtype='float16'),
            ],
        )
        np.testing.assert_allclose(
            dy_out[0].numpy(), cinn_out[0].numpy(), atol=1e-5, rtol=1e-5
        )
        np.testing.assert_allclose(
            dy_out[1].numpy(), cinn_out[1].numpy(), atol=1e-5, rtol=1e-5
        )

    def test_eval_subgraph3(self):
        inputs3 = paddle.uniform(
            [1, 360, 640, 128], dtype="float16", min=-0.5, max=0.5
        )
        inputs3.stop_gradient = False
        dy_out = self.eval(SubGraph3, False, inputs3)
        cinn_out = self.eval(SubGraph3, True, inputs3)
        np.testing.assert_allclose(
            dy_out.numpy(), cinn_out.numpy(), atol=1e-5, rtol=1e-5
        )


if __name__ == '__main__':
    unittest.main()
