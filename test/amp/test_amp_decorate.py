# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn.functional as F


class ConvBNLayer(paddle.nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        stride=1,
        groups=1,
        act=None,
    ):
        super().__init__()

        self._conv = paddle.nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=None,
        )

        self._batch_norm = paddle.nn.BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)

        return y


class Model(paddle.nn.Layer):
    def __init__(self, input_channel, hidden_size, fp32_layers=False):
        super().__init__()
        self.conv = ConvBNLayer(input_channel, 8, 3)
        self.linear = paddle.nn.Linear(8, hidden_size)
        self.layernorm = paddle.nn.Sequential(
            paddle.nn.LayerNorm(hidden_size),
            paddle.nn.LayerNorm(hidden_size),
        )
        self.fp32_layers = fp32_layers

    def forward(self, inputs):
        with paddle.amp.auto_cast(enable=self.fp32_layers):
            x = self.conv(inputs)
        x = self.linear(x)
        x = F.relu(x)
        x = self.layernorm(x)
        return x


class TestAMPDecorate(unittest.TestCase):
    def check_results(self, model, dtype=paddle.float32):
        self.assertEqual(model.conv._conv.weight.dtype, dtype)
        self.assertEqual(model.conv._conv.bias.dtype, dtype)
        self.assertEqual(model.conv._batch_norm.weight.dtype, paddle.float32)
        self.assertEqual(model.conv._batch_norm.weight.dtype, paddle.float32)
        self.assertEqual(model.linear.weight.dtype, paddle.float16)
        self.assertEqual(model.linear.bias.dtype, paddle.float16)
        self.assertEqual(model.layernorm[0].weight.dtype, paddle.float32)
        self.assertEqual(model.layernorm[0].bias.dtype, paddle.float32)
        self.assertEqual(model.layernorm[1].weight.dtype, paddle.float32)
        self.assertEqual(model.layernorm[1].bias.dtype, paddle.float32)

    def test_excluded_layers(self):
        if not paddle.amp.is_float16_supported:
            return
        model = Model(4, 8, fp32_layers=True)
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            dtype='float16',
            excluded_layers=model.conv,
        )

        with paddle.amp.auto_cast(level='O2'):
            out = model(paddle.rand(shape=[2, 4, 8, 8], dtype='float32'))

        self.check_results(model, dtype=paddle.float32)

    def test_excluded_layers_attr_list(self):
        if not paddle.amp.is_float16_supported:
            return
        model = Model(4, 8, fp32_layers=True)
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            dtype='float16',
            excluded_layers=[model.conv],
        )

        with paddle.amp.auto_cast(level='O2'):
            out = model(paddle.rand(shape=[2, 4, 8, 8], dtype='float32'))

        self.check_results(model)

    def test_excluded_layers_attr_none(self):
        if not paddle.amp.is_float16_supported:
            return
        model = Model(4, 8, fp32_layers=False)
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            dtype='float16',
            excluded_layers=None,
        )

        with paddle.amp.auto_cast(level='O2'):
            out = model(paddle.rand(shape=[2, 4, 8, 8], dtype='float16'))

        self.check_results(model, dtype=paddle.float16)


if __name__ == '__main__':
    unittest.main()
