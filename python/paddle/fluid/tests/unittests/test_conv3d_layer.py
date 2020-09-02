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

import numpy as np
from paddle import fluid, nn
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F
import paddle.fluid.initializer as I
import unittest


class Conv3DTestCase(unittest.TestCase):
    def __init__(self,
                 methodName='runTest',
                 batch_size=4,
                 spartial_shape=(8, 8, 8),
                 num_channels=6,
                 num_filters=8,
                 filter_size=3,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 no_bias=False,
                 data_format="NCDHW",
                 dtype="float32"):
        super(Conv3DTestCase, self).__init__(methodName)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.spartial_shape = spartial_shape
        self.filter_size = filter_size

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
            input_shape = (self.batch_size, ) + self.spartial_shape + (
                self.num_channels, )
        else:
            input_shape = (self.batch_size, self.num_channels
                           ) + self.spartial_shape
        self.input = np.random.randn(*input_shape).astype(self.dtype)

        if isinstance(self.filter_size, int):
            filter_size = [self.filter_size] * 3
        else:
            filter_size = self.filter_size
        self.weight_shape = weight_shape = (self.num_filters, self.num_channels
                                            // self.groups) + tuple(filter_size)
        self.weight = np.random.uniform(
            -1, 1, size=weight_shape).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(
                -1, 1, size=(self.num_filters, )).astype(self.dtype)
        else:
            self.bias = None

    def fluid_layer(self, place):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                input_shape = (-1, -1, -1, -1, self.num_channels) \
                    if self.channel_last else (-1, self.num_channels, -1, -1, -1)
                x_var = fluid.data("input", input_shape, dtype=self.dtype)
                weight_attr = I.NumpyArrayInitializer(self.weight)
                if self.bias is None:
                    bias_attr = False
                else:
                    bias_attr = I.NumpyArrayInitializer(self.bias)
                y_var = fluid.layers.conv3d(
                    x_var,
                    self.num_filters,
                    self.filter_size,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    param_attr=weight_attr,
                    bias_attr=bias_attr,
                    data_format=self.data_format)
        feed_dict = {"input": self.input}
        exe = fluid.Executor(place)
        exe.run(start)
        y_np, = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def functional(self, place):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                input_shape = (-1, -1, -1, -1, self.num_channels) \
                    if self.channel_last else (-1, self.num_channels, -1, -1, -1)
                x_var = fluid.data("input", input_shape, dtype=self.dtype)
                w_var = fluid.data(
                    "weight", self.weight_shape, dtype=self.dtype)
                b_var = fluid.data(
                    "bias", (self.num_filters, ), dtype=self.dtype)
                y_var = F.conv3d(
                    x_var,
                    w_var,
                    None if self.no_bias else b_var,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    data_format=self.data_format)
        feed_dict = {"input": self.input, "weight": self.weight}
        if self.bias is not None:
            feed_dict["bias"] = self.bias
        exe = fluid.Executor(place)
        exe.run(start)
        y_np, = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def paddle_nn_layer(self):
        x_var = dg.to_variable(self.input)
        conv = nn.Conv3d(
            self.num_channels,
            self.num_filters,
            self.filter_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            data_format=self.data_format)
        conv.weight.set_value(self.weight)
        if not self.no_bias:
            conv.bias.set_value(self.bias)
        y_var = conv(x_var)
        y_np = y_var.numpy()
        return y_np

    def _test_equivalence(self, place):
        place = fluid.CPUPlace()
        result1 = self.fluid_layer(place)
        result2 = self.functional(place)
        with dg.guard(place):
            result3 = self.paddle_nn_layer()
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        place = fluid.CPUPlace()
        self._test_equivalence(place)

        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self._test_equivalence(place)


class Conv3DErrorTestCase(Conv3DTestCase):
    def runTest(self):
        place = fluid.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_nn_layer()


def add_cases(suite):
    suite.addTest(Conv3DTestCase(methodName='runTest'))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest', stride=[1, 2, 1], dilation=2))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest', stride=2, dilation=(2, 1, 2)))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest', padding="same", no_bias=True))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest', filter_size=(3, 2, 3), padding='valid'))
    suite.addTest(Conv3DTestCase(methodName='runTest', padding=(2, 3, 1)))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest', padding=[1, 2, 2, 1, 2, 3]))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest',
            padding=[[0, 0], [0, 0], [1, 2], [2, 1], [2, 2]]))
    suite.addTest(Conv3DTestCase(methodName='runTest', data_format="NDHWC"))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest',
            data_format="NDHWC",
            padding=[[0, 0], [1, 1], [3, 3], [2, 2], [0, 0]]))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest', groups=2, padding="valid"))
    suite.addTest(
        Conv3DTestCase(
            methodName='runTest',
            num_filters=6,
            num_channels=3,
            groups=3,
            padding="valid"))


def add_error_cases(suite):
    suite.addTest(
        Conv3DErrorTestCase(
            methodName='runTest', num_channels=5, groups=2))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()
