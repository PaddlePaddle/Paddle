# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.base.dygraph as dg
from paddle import base, nn


def _reverse_repeat_list(t, n):
    return [x for x in reversed(t) for _ in range(n)]


class Conv2DTestCase(unittest.TestCase):
    def __init__(
        self,
        methodName='runTest',
        batch_size=4,
        spartial_shape=(16, 16),
        num_channels=6,
        num_filters=8,
        filter_size=3,
        padding=0,
        padding_mode='zeros',
        stride=1,
        dilation=1,
        groups=1,
        no_bias=False,
        data_format="NCHW",
        dtype="float32",
    ):
        super().__init__(methodName)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.spartial_shape = spartial_shape
        self.filter_size = filter_size

        self.padding = padding
        if padding_mode in {'reflect', 'replicate', 'circular'}:
            _paired_padding = paddle.utils.convert_to_list(
                padding, 2, 'padding'
            )
            self._reversed_padding_repeated_twice = _reverse_repeat_list(
                _paired_padding, 2
            )
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.no_bias = no_bias
        self.data_format = data_format
        self.dtype = dtype

    def setUp(self):
        self.channel_last = self.data_format == "NHWC"
        if self.channel_last:
            input_shape = (
                self.batch_size,
                *self.spartial_shape,
                self.num_channels,
            )
        else:
            input_shape = (
                self.batch_size,
                self.num_channels,
                *self.spartial_shape,
            )
        self.input = np.random.randn(*input_shape).astype(self.dtype)

        if isinstance(self.filter_size, int):
            filter_size = [self.filter_size] * 2
        else:
            filter_size = self.filter_size
        self.weight_shape = weight_shape = (
            self.num_filters,
            self.num_channels // self.groups,
            *filter_size,
        )
        self.weight = np.random.uniform(-1, 1, size=weight_shape).astype(
            self.dtype
        )
        if not self.no_bias:
            self.bias = np.random.uniform(
                -1, 1, size=(self.num_filters,)
            ).astype(self.dtype)
        else:
            self.bias = None

    def paddle_nn_layer(self):
        x_var = paddle.to_tensor(self.input)
        x_var.stop_gradient = False
        conv = nn.Conv2D(
            self.num_channels,
            self.num_filters,
            self.filter_size,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            data_format=self.data_format,
        )
        conv.weight.set_value(self.weight)
        if not self.no_bias:
            conv.bias.set_value(self.bias)
        y_var = conv(x_var)
        y_var.backward()
        y_np = y_var.numpy()
        t1 = x_var.gradient()
        return y_np, t1

    def run_Conv2D_static(self, place):
        paddle.seed(2023)
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                x_var = paddle.static.data(
                    "input", self.input.shape, dtype=self.dtype
                )
                conv = nn.Conv2D(
                    self.num_channels,
                    self.num_filters,
                    self.filter_size,
                    padding=self.padding,
                    padding_mode=self.padding_mode,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    data_format=self.data_format,
                )
                y_var = conv(x_var)
        feed_dict = {"input": self.input}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def run_Conv2D_dygraph(self):
        paddle.seed(2023)
        x_var = paddle.to_tensor(self.input)
        x_var.stop_gradient = False
        conv = nn.Conv2D(
            self.num_channels,
            self.num_filters,
            self.filter_size,
            padding=self.padding,
            padding_mode=self.padding_mode,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            data_format=self.data_format,
        )
        y_var = conv(x_var)
        y_np = y_var.numpy()
        return y_np

    def _test_equivalence_in_pir(self, place):
        with paddle.pir_utils.IrGuard():
            result1 = self.run_Conv2D_static(place)
            with dg.guard(place):
                result2 = self.run_Conv2D_dygraph()
            np.testing.assert_array_almost_equal(result1, result2)

    def runTest(self):
        place = base.CPUPlace()
        self._test_equivalence_in_pir(place)

        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self._test_equivalence_in_pir(place)


class Conv2DErrorTestCase(Conv2DTestCase):
    def runTest(self):
        place = base.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_nn_layer()


def add_cases(suite):
    suite.addTest(Conv2DTestCase(methodName='runTest'))
    suite.addTest(
        Conv2DTestCase(methodName='runTest', stride=[1, 2], dilation=2)
    )
    suite.addTest(
        Conv2DTestCase(methodName='runTest', stride=2, dilation=(2, 1))
    )
    suite.addTest(
        Conv2DTestCase(methodName='runTest', padding="same", no_bias=True)
    )
    suite.addTest(
        Conv2DTestCase(
            methodName='runTest', filter_size=(3, 3), padding='valid'
        )
    )
    suite.addTest(Conv2DTestCase(methodName='runTest', padding=(2, 3)))
    suite.addTest(Conv2DTestCase(methodName='runTest', padding=[1, 2, 2, 1]))
    suite.addTest(
        Conv2DTestCase(
            methodName='runTest', padding=[[0, 0], [0, 0], [1, 2], [2, 1]]
        )
    )
    suite.addTest(Conv2DTestCase(methodName='runTest', data_format="NHWC"))
    suite.addTest(
        Conv2DTestCase(
            methodName='runTest',
            data_format="NHWC",
            padding=[[0, 0], [1, 1], [2, 2], [0, 0]],
        )
    )
    suite.addTest(
        Conv2DTestCase(methodName='runTest', groups=2, padding="valid")
    )
    suite.addTest(
        Conv2DTestCase(
            methodName='runTest',
            num_filters=6,
            num_channels=3,
            groups=3,
            padding="valid",
        )
    )
    suite.addTest(
        Conv2DTestCase(
            methodName='runTest',
            filter_size=(3, 3),
            padding=1,
            padding_mode='reflect',
        )
    )
    suite.addTest(
        Conv2DTestCase(
            methodName='runTest',
            filter_size=(3, 3),
            padding=1,
            padding_mode='replicate',
        )
    )
    suite.addTest(
        Conv2DTestCase(
            methodName='runTest',
            filter_size=(3, 3),
            padding=1,
            padding_mode='circular',
        )
    )


def add_error_cases(suite):
    suite.addTest(
        Conv2DErrorTestCase(methodName='runTest', num_channels=5, groups=2)
    )
    suite.addTest(
        Conv2DErrorTestCase(
            methodName='runTest', num_channels=5, groups=2, stride=0
        )
    )
    suite.addTest(
        Conv2DErrorTestCase(
            methodName='runTest', num_channels=5, groups=2, padding=[-1, -1]
        )
    )


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
