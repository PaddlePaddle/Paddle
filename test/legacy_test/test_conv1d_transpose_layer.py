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
import paddle.nn.functional as F
from paddle import base, nn


class Conv1DTransposeTestCase(unittest.TestCase):
    def __init__(
        self,
        methodName='runTest',
        batch_size=4,
        spartial_shape=16,
        in_channels=6,
        out_channels=8,
        filter_size=3,
        output_size=None,
        padding=0,
        output_padding=0,
        stride=1,
        dilation=1,
        groups=1,
        no_bias=False,
        data_format="NCL",
        dtype="float32",
    ):
        super().__init__(methodName)
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spartial_shape = spartial_shape
        self.filter_size = filter_size
        self.output_size = output_size

        self.padding = padding
        self.output_padding = output_padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.no_bias = no_bias
        self.data_format = data_format
        self.dtype = dtype

    def setUp(self):
        self.channel_last = False if self.data_format == "NCL" else True
        input_shape = (
            (self.batch_size, self.in_channels, self.spartial_shape)
            if not self.channel_last
            else (
                self.batch_size,
                self.spartial_shape,
                self.in_channels,
            )
        )
        self.input = np.random.randn(*input_shape).astype(self.dtype)

        if isinstance(self.filter_size, int):
            filter_size = [self.filter_size]
        else:
            filter_size = self.filter_size
        self.weight_shape = weight_shape = (
            self.in_channels,
            self.out_channels // self.groups,
            *filter_size,
        )
        self.weight = np.random.uniform(-1, 1, size=weight_shape).astype(
            self.dtype
        )
        if not self.no_bias:
            self.bias = np.random.uniform(
                -1, 1, size=(self.out_channels,)
            ).astype(self.dtype)
        else:
            self.bias = None

    def functional(self, place):
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                input_shape = (
                    (-1, self.in_channels, -1)
                    if not self.channel_last
                    else (-1, -1, self.in_channels)
                )
                x_var = paddle.static.data(
                    "input", input_shape, dtype=self.dtype
                )
                w_var = paddle.static.data(
                    "weight", self.weight_shape, dtype=self.dtype
                )
                if not self.no_bias:
                    b_var = paddle.static.data(
                        "bias", (self.out_channels,), dtype=self.dtype
                    )
                else:
                    b_var = None
                y_var = F.conv1d_transpose(
                    x_var,
                    w_var,
                    None if self.no_bias else b_var,
                    output_size=self.output_size,
                    padding=self.padding,
                    output_padding=self.output_padding,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    data_format=self.data_format,
                )
        feed_dict = {"input": self.input, "weight": self.weight}
        if self.bias is not None:
            feed_dict["bias"] = self.bias
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def paddle_nn_layer(self):
        x_var = paddle.to_tensor(self.input)
        conv = nn.Conv1DTranspose(
            self.in_channels,
            self.out_channels,
            self.filter_size,
            padding=self.padding,
            output_padding=self.output_padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            data_format=self.data_format,
        )
        conv.weight.set_value(self.weight)
        if not self.no_bias:
            conv.bias.set_value(self.bias)
        y_var = conv(x_var, output_size=self.output_size)
        y_np = y_var.numpy()
        return y_np

    def _test_pir_equivalence(self, place):
        with paddle.pir_utils.IrGuard():
            result1 = self.functional(place)
        with dg.guard(place):
            result2 = self.paddle_nn_layer()
        np.testing.assert_array_almost_equal(result1, result2)

    def runTest(self):
        place = base.CPUPlace()
        self._test_pir_equivalence(place)

        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self._test_pir_equivalence(place)


class Conv1DTransposeErrorTestCase(Conv1DTransposeTestCase):
    def runTest(self):
        place = base.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_nn_layer()


def add_cases(suite):
    suite.addTest(Conv1DTransposeTestCase(methodName='runTest'))
    suite.addTest(
        Conv1DTransposeTestCase(
            methodName='runTest', stride=[2], no_bias=True, dilation=2
        )
    )
    suite.addTest(
        Conv1DTransposeTestCase(
            methodName='runTest',
            filter_size=(3),
            output_size=[36],
            stride=[2],
            dilation=2,
        )
    )
    suite.addTest(
        Conv1DTransposeTestCase(methodName='runTest', stride=2, dilation=(2))
    )
    suite.addTest(
        Conv1DTransposeTestCase(methodName='runTest', padding="valid")
    )
    suite.addTest(
        Conv1DTransposeTestCase(methodName='runTest', padding='valid')
    )
    suite.addTest(
        Conv1DTransposeTestCase(methodName='runTest', filter_size=1, padding=3)
    )
    suite.addTest(Conv1DTransposeTestCase(methodName='runTest', padding=[2]))
    suite.addTest(
        Conv1DTransposeTestCase(methodName='runTest', data_format="NLC")
    )
    suite.addTest(
        Conv1DTransposeTestCase(methodName='runTest', groups=2, padding="valid")
    )
    suite.addTest(
        Conv1DTransposeTestCase(
            methodName='runTest',
            out_channels=6,
            in_channels=3,
            groups=3,
            padding="valid",
        )
    )
    suite.addTest(
        Conv1DTransposeTestCase(
            methodName='runTest',
            data_format="NLC",
            spartial_shape=16,
            output_size=18,
        )
    )
    suite.addTest(
        Conv1DTransposeTestCase(
            methodName='runTest', data_format="NLC", stride=3, output_padding=2
        )
    )
    suite.addTest(Conv1DTransposeTestCase(methodName='runTest', padding=[1, 2]))


def add_error_cases(suite):
    suite.addTest(
        Conv1DTransposeErrorTestCase(
            methodName='runTest', data_format="not_valid"
        )
    )
    suite.addTest(
        Conv1DTransposeErrorTestCase(
            methodName='runTest', in_channels=5, groups=2
        )
    )
    suite.addTest(
        Conv1DTransposeErrorTestCase(
            methodName='runTest', stride=2, output_padding=3
        )
    )
    suite.addTest(
        Conv1DTransposeErrorTestCase(
            methodName='runTest', output_size="not_valid"
        )
    )


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()
