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
import paddle
from paddle import fluid, nn
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F
import paddle.fluid.initializer as I
import unittest


class Conv1dTestCase(unittest.TestCase):
    def __init__(self,
                 methodName='runTest',
                 batch_size=4,
                 spartial_shape=(16, ),
                 num_channels=6,
                 num_filters=8,
                 filter_size=3,
                 padding=0,
                 padding_mode="zeros",
                 stride=1,
                 dilation=1,
                 groups=1,
                 no_bias=False,
                 dtype="float32",
                 data_format="NCL"):
        super(Conv1dTestCase, self).__init__(methodName)
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.spartial_shape = spartial_shape
        self.filter_size = filter_size
        self.data_format = data_format
        self.channel_last = (self.data_format == "NLC")

        self.padding = padding
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.no_bias = no_bias
        self.dtype = dtype

    def setUp(self):
        input_shape = (self.batch_size, self.num_channels
                       ) + self.spartial_shape if not self.channel_last else (
                           self.batch_size, ) + self.spartial_shape + (
                               self.num_channels, )
        self.input = np.random.randn(*input_shape).astype(self.dtype)

        if isinstance(self.filter_size, int):
            filter_size = [self.filter_size]
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

    def functional(self, place):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                input_shape = (-1, self.num_channels,
                               -1) if not self.channel_last else (
                                   -1, -1, self.num_channels)
                x_var = fluid.data("input", input_shape, dtype=self.dtype)
                w_var = fluid.data(
                    "weight", self.weight_shape, dtype=self.dtype)
                b_var = fluid.data(
                    "bias", (self.num_filters, ), dtype=self.dtype)
                y_var = F.conv1d(
                    x_var,
                    w_var,
                    b_var if not self.no_bias else None,
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
        x_var = paddle.to_tensor(self.input)
        conv = nn.Conv1d(
            self.num_channels,
            self.num_filters,
            self.filter_size,
            padding=self.padding,
            padding_mode=self.padding_mode,
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
        result1 = self.functional(place)
        with dg.guard(place):
            result2 = self.paddle_nn_layer()
        np.testing.assert_array_almost_equal(result1, result2)

    def runTest(self):
        place = fluid.CPUPlace()
        self._test_equivalence(place)

        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self._test_equivalence(place)


class Conv1dErrorTestCase(Conv1dTestCase):
    def runTest(self):
        place = fluid.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_nn_layer()


class Conv1dTypeErrorTestCase(Conv1dTestCase):
    def runTest(self):
        place = fluid.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(TypeError):
                self.paddle_nn_layer()


def add_cases(suite):
    suite.addTest(Conv1dTestCase(methodName='runTest'))
    suite.addTest(Conv1dTestCase(methodName='runTest', stride=[1], dilation=2))
    suite.addTest(Conv1dTestCase(methodName='runTest', stride=2, dilation=(1)))
    suite.addTest(
        Conv1dTestCase(
            methodName='runTest', padding="same", no_bias=True))
    suite.addTest(
        Conv1dTestCase(
            methodName='runTest', filter_size=3, padding='valid'))
    suite.addTest(
        Conv1dTestCase(
            methodName='runTest', padding=2, data_format='NLC'))
    suite.addTest(Conv1dTestCase(methodName='runTest', padding=[1]))
    suite.addTest(Conv1dTestCase(methodName='runTest', padding=[1, 2]))
    suite.addTest(Conv1dTestCase(methodName='runTest', padding=2))
    suite.addTest(Conv1dTestCase(methodName='runTest'))
    suite.addTest(
        Conv1dTestCase(
            methodName='runTest', groups=2, padding="valid"))
    suite.addTest(
        Conv1dTestCase(
            methodName='runTest',
            num_filters=6,
            num_channels=3,
            groups=3,
            padding="valid",
            data_format='NLC'))


def add_error_cases(suite):
    suite.addTest(
        Conv1dTypeErrorTestCase(
            methodName='runTest', padding_mode="reflect", padding="valid"))
    suite.addTest(
        Conv1dErrorTestCase(
            methodName='runTest', data_format="VALID"))
    suite.addTest(
        Conv1dErrorTestCase(
            methodName='runTest', padding_mode="VALID"))
    suite.addTest(
        Conv1dErrorTestCase(
            methodName='runTest', num_channels=5, groups=2))
    suite.addTest(
        Conv1dErrorTestCase(
            methodName='runTest', num_filters=8, num_channels=15, groups=3))
    suite.addTest(
        Conv1dErrorTestCase(
            methodName='runTest', padding=[1, 2, 3, 4, 5]))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()
