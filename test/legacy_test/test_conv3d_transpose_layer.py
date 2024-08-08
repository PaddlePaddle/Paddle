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
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base, nn


class Conv3DTransposeTestCase(unittest.TestCase):
    def __init__(
        self,
        methodName='runTest',
        batch_size=2,
        spatial_shape=(8, 8, 8),
        num_channels=6,
        num_filters=8,
        filter_size=3,
        output_size=None,
        padding=0,
        stride=1,
        dilation=1,
        groups=1,
        no_bias=False,
        data_format="NCDHW",
        dtype="float32",
    ):
        super().__init__(methodName)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.spatial_shape = spatial_shape
        self.filter_size = filter_size
        self.output_size = output_size

        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.no_bias = no_bias
        self.data_format = data_format
        self.dtype = dtype

    def setUp(self):
        self.channel_last = self.data_format == "NDHWC"
        if self.channel_last:
            input_shape = (
                self.batch_size,
                *self.spatial_shape,
                self.num_channels,
            )
        else:
            input_shape = (
                self.batch_size,
                self.num_channels,
                *self.spatial_shape,
            )
        self.input = np.random.randn(*input_shape).astype(self.dtype)

        if isinstance(self.filter_size, int):
            filter_size = [self.filter_size] * 3
        else:
            filter_size = self.filter_size
        self.weight_shape = weight_shape = (
            self.num_channels,
            self.num_filters // self.groups,
            *filter_size,
        )
        self.weight = np.random.uniform(-1, 1, size=weight_shape).astype(
            self.dtype
        )
        if self.no_bias:
            self.bias = None
        else:
            self.bias = np.random.uniform(
                -1, 1, size=(self.num_filters,)
            ).astype(self.dtype)

    def base_layer(self, place):
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                input_shape = (
                    (-1, -1, -1, -1, self.num_channels)
                    if self.channel_last
                    else (-1, self.num_channels, -1, -1, -1)
                )
                x_var = paddle.static.data(
                    "input", input_shape, dtype=self.dtype
                )
                weight_attr = paddle.nn.initializer.Assign(self.weight)
                if self.bias is None:
                    bias_attr = False
                else:
                    bias_attr = paddle.nn.initializer.Assign(self.bias)
                y_var = paddle.nn.Conv3DTranspose(
                    in_channels=self.num_channels,
                    out_channels=self.num_filters,
                    kernel_size=self.filter_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr,
                    data_format=self.data_format,
                )(x_var, self.output_size)
        feed_dict = {"input": self.input}
        exe = base.Executor(place)
        exe.run(start)
        (y_np,) = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def functional(self, place):
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                input_shape = (
                    (-1, -1, -1, -1, self.num_channels)
                    if self.channel_last
                    else (-1, self.num_channels, -1, -1, -1)
                )
                x_var = paddle.static.data(
                    "input", input_shape, dtype=self.dtype
                )
                w_var = paddle.static.data(
                    "weight", self.weight_shape, dtype=self.dtype
                )
                if not self.no_bias:
                    b_var = paddle.static.data(
                        "bias", (self.num_filters,), dtype=self.dtype
                    )
                else:
                    b_var = None
                y_var = F.conv3d_transpose(
                    x_var,
                    w_var,
                    None if self.no_bias else b_var,
                    output_size=self.output_size,
                    padding=self.padding,
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
        conv = nn.Conv3DTranspose(
            self.num_channels,
            self.num_filters,
            self.filter_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            data_format=self.data_format,
        )
        conv.weight.set_value(self.weight)
        if not self.no_bias:
            conv.bias.set_value(self.bias)
        y_var = conv(x_var, self.output_size)
        y_np = y_var.numpy()
        return y_np

    def _test_pir_equivalence(self, place):
        place = base.CPUPlace()
        with paddle.pir_utils.IrGuard():
            result1 = self.base_layer(place)
            result2 = self.functional(place)
        with dg.guard(place):
            result3 = self.paddle_nn_layer()
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def _test_equivalence(self, place):
        place = base.CPUPlace()
        with paddle.pir_utils.OldIrGuard():
            result1 = self.base_layer(place)
            result2 = self.functional(place)
        with dg.guard(place):
            result3 = self.paddle_nn_layer()
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        place = base.CPUPlace()
        self._test_equivalence(place)
        self._test_pir_equivalence(place)

        if base.core.is_compiled_with_cuda():
            place = base.CUDAPlace(0)
            self._test_equivalence(place)
            self._test_pir_equivalence(place)


class Conv3DTransposeErrorTestCase(Conv3DTransposeTestCase):
    def runTest(self):
        place = base.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_nn_layer()


def add_cases(suite):
    suite.addTest(Conv3DTransposeTestCase(methodName='runTest'))
    suite.addTest(
        Conv3DTransposeTestCase(
            methodName='runTest', stride=[1, 2, 1], dilation=2, no_bias=True
        )
    )
    suite.addTest(
        Conv3DTransposeTestCase(
            methodName='runTest',
            output_size=[12, 19, 12],
            stride=[1, 2, 1],
            dilation=2,
        )
    )
    suite.addTest(
        Conv3DTransposeTestCase(
            methodName='runTest', stride=2, dilation=(2, 1, 2)
        )
    )
    suite.addTest(
        Conv3DTransposeTestCase(methodName='runTest', padding="valid")
    )
    suite.addTest(
        Conv3DTransposeTestCase(methodName='runTest', padding='valid')
    )
    suite.addTest(
        Conv3DTransposeTestCase(
            methodName='runTest', filter_size=1, padding=(2, 3, 1)
        )
    )
    suite.addTest(
        Conv3DTransposeTestCase(
            methodName='runTest', padding=[1, 2, 2, 3, 2, 1]
        )
    )
    suite.addTest(
        Conv3DTransposeTestCase(
            methodName='runTest',
            padding=[[0, 0], [0, 0], [2, 3], [1, 2], [2, 1]],
        )
    )
    suite.addTest(
        Conv3DTransposeTestCase(methodName='runTest', data_format="NDHWC")
    )
    suite.addTest(
        Conv3DTransposeTestCase(
            methodName='runTest',
            data_format="NDHWC",
            padding=[[0, 0], [1, 1], [2, 2], [3, 3], [0, 0]],
        )
    )
    suite.addTest(
        Conv3DTransposeTestCase(methodName='runTest', groups=2, padding="valid")
    )
    suite.addTest(
        Conv3DTransposeTestCase(
            methodName='runTest',
            num_filters=6,
            num_channels=3,
            groups=3,
            padding="valid",
        )
    )


def add_error_cases(suite):
    suite.addTest(
        Conv3DTransposeErrorTestCase(
            methodName='runTest', num_channels=5, groups=2
        )
    )
    suite.addTest(
        Conv3DTransposeErrorTestCase(
            methodName='runTest', output_size="not_valid"
        )
    )
    suite.addTest(
        Conv3DTransposeErrorTestCase(
            methodName='runTest', num_channels=5, groups=2, padding=[-1, 1, 3]
        )
    )


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()
