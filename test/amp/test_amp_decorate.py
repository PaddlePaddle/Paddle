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
from paddle.base import core


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
    def __init__(
        self, input_channel, hidden_size, fp16_conv=True, fp16_linear=True
    ):
        super().__init__()
        self.conv = ConvBNLayer(input_channel, 8, 3)
        self.linear = paddle.nn.Linear(8, hidden_size)
        self.layernorm = paddle.nn.Sequential(
            paddle.nn.LayerNorm(hidden_size),
            paddle.nn.LayerNorm(hidden_size),
        )
        self.fp16_conv = fp16_conv
        self.fp16_linear = fp16_linear

    def forward(self, inputs):
        with paddle.amp.auto_cast(enable=self.fp16_conv):
            if not self.fp16_conv:
                inputs = inputs.astype('float32')
            x = self.conv(inputs)
        with paddle.amp.auto_cast(enable=self.fp16_linear):
            if not self.fp16_linear:
                x = x.astype('float32')
            x = self.linear(x)
        x = F.relu(x)
        x = self.layernorm(x)
        return x


class LayerNorm2D(paddle.nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(args, *kwargs)

    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])
        x = super().forward(x)
        return x.transpose([0, 3, 1, 2])


class CustomLayer(paddle.nn.Layer):
    def __init__(
        self, input_channel, hidden_size, fp16_conv=True, fp16_linear=True
    ):
        super().__init__()
        self.conv = ConvBNLayer(input_channel, 8, 3)
        self.linear = paddle.nn.Linear(8, hidden_size)
        self.layernorm = paddle.nn.Sequential(
            LayerNorm2D(hidden_size),
            LayerNorm2D(hidden_size),
        )
        self.fp16_conv = fp16_conv
        self.fp16_linear = fp16_linear

    def forward(self, inputs):
        with paddle.amp.auto_cast(enable=self.fp16_conv):
            if not self.fp16_conv:
                inputs = inputs.astype('float32')
            x = self.conv(inputs)
        with paddle.amp.auto_cast(enable=self.fp16_linear):
            if not self.fp16_linear:
                x = x.astype('float32')
            x = self.linear(x)
        x = F.relu(x)
        x = self.layernorm(x)
        return x


@unittest.skipIf(
    not core.is_compiled_with_cuda() and not core.is_compiled_with_xpu(),
    "Require compiled with CUDA or XPU.",
)
@unittest.skipIf(
    core.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] < 7.0,
    "run test when gpu's compute capability is at least 7.0.",
)
@unittest.skipIf(
    core.is_compiled_with_xpu()
    and core.get_xpu_device_version(0) < core.XPUVersion.XPU3,
    "run test when xpu's compute capability >= xpu3.",
)
class TestAMPDecorate(unittest.TestCase):
    def check_results(self, fp32_layers=[], fp16_layers=[]):
        for idx in range(len(fp32_layers)):
            for layer in fp32_layers[idx].sublayers(include_self=False):
                self.assertTrue(
                    layer.weight.dtype
                    in (paddle.float32, core.DataType.FLOAT32)
                )
                self.assertTrue(
                    layer.bias.dtype in (paddle.float32, core.DataType.FLOAT32)
                )

        for idx in range(len(fp16_layers)):
            for layer in fp16_layers[idx].sublayers(include_self=False):
                self.assertTrue(
                    layer.weight.dtype
                    in (paddle.float16, core.DataType.FLOAT16)
                )
                self.assertTrue(
                    layer.bias.dtype in (paddle.float16, core.DataType.FLOAT16)
                )

    def test_excluded_layers(self):
        model = Model(4, 8, fp16_conv=False)
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            dtype='float16',
            excluded_layers=model.conv,
        )
        with paddle.amp.auto_cast(level='O2'):
            out = model(paddle.rand(shape=[2, 4, 8, 8], dtype='float32'))
        self.check_results(
            fp32_layers=[model.conv, model.layernorm],
            fp16_layers=[model.linear],
        )

    def test_excluded_layers_attr_list(self):
        model = Model(4, 8, fp16_conv=False, fp16_linear=False)
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            dtype='float16',
            excluded_layers=[model.conv, model.linear],
        )

        with paddle.amp.auto_cast(level='O2'):
            out = model(paddle.rand(shape=[2, 4, 8, 8], dtype='float32'))

        self.check_results(
            fp32_layers=[model.conv, model.linear, model.layernorm]
        )

    def test_excluded_layers_attr_types(self):
        model = Model(4, 8)
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            dtype='float16',
            excluded_layers=[paddle.nn.Conv2D, model.linear],
        )

        with paddle.amp.auto_cast(level='O2'):
            out = model(paddle.rand(shape=[2, 4, 8, 8], dtype='float16'))

        self.check_results(
            fp32_layers=[model.conv, model.linear, model.layernorm]
        )

    def test_excluded_layers_attr_none(self):
        model = Model(4, 8)
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            dtype='float16',
            excluded_layers=None,
        )

        with paddle.amp.auto_cast(level='O2'):
            out = model(paddle.rand(shape=[2, 4, 8, 8], dtype='float16'))

        self.check_results(
            fp32_layers=[model.layernorm, model.conv._batch_norm],
            fp16_layers=[model.conv._conv, model.linear],
        )

    def test_excluded_layers_custom_layer(self):
        model = CustomLayer(4, 8)
        model = paddle.amp.decorate(
            models=model,
            level='O2',
            dtype='bfloat16',
            excluded_layers=[paddle.nn.LayerNorm, paddle.nn.BatchNorm],
        )
        with paddle.amp.auto_cast(level='O2'):
            out = model(paddle.rand(shape=[2, 4, 8, 8], dtype='float32'))
        self.check_results(
            fp32_layers=[model.layernorm, model.conv._batch_norm],
        )

    def test_pir(self):
        with paddle.pir_utils.IrGuard():
            with paddle.static.program_guard(
                paddle.static.Program(), paddle.static.Program()
            ):
                self.test_excluded_layers()
                self.test_excluded_layers_attr_list()
                self.test_excluded_layers_attr_types()
                self.test_excluded_layers_attr_none()
                self.test_excluded_layers_custom_layer()


if __name__ == '__main__':
    unittest.main()
