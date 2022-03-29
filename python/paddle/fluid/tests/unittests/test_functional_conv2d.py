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

import paddle
import paddle.nn.functional as F
from paddle import fluid
import paddle.fluid.dygraph as dg
import paddle.fluid.initializer as I
import numpy as np
import unittest
from unittest import TestCase


class TestFunctionalConv2D(TestCase):
    batch_size = 4
    spatial_shape = (16, 16)
    dtype = "float32"

    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NHWC"

    def prepare(self):
        if isinstance(self.filter_shape, int):
            filter_shape = (self.filter_shape, ) * 2
        else:
            filter_shape = tuple(self.filter_shape)

        self.weight = np.random.uniform(
            -1, 1, (self.out_channels, self.in_channels // self.groups
                    ) + filter_shape).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(-1, 1, (
                self.out_channels, )).astype(self.dtype)

        self.channel_last = (self.data_format == "NHWC")
        if self.channel_last:
            self.input_shape = (self.batch_size, ) + self.spatial_shape + (
                self.in_channels, )
        else:
            self.input_shape = (self.batch_size, self.in_channels
                                ) + self.spatial_shape

        self.input = np.random.uniform(-1, 1,
                                       self.input_shape).astype(self.dtype)

    def static_graph_case_1(self):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                if self.channel_last:
                    x = fluid.data(
                        "input", (-1, -1, -1, self.in_channels),
                        dtype=self.dtype)
                else:
                    x = fluid.data(
                        "input", (-1, self.in_channels, -1, -1),
                        dtype=self.dtype)
                y = fluid.layers.conv2d(
                    x,
                    self.out_channels,
                    self.filter_shape,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    param_attr=I.NumpyArrayInitializer(self.weight),
                    bias_attr=False
                    if self.no_bias else I.NumpyArrayInitializer(self.bias),
                    act=self.act,
                    data_format=self.data_format)
        exe = fluid.Executor(self.place)
        exe.run(start)
        out, = exe.run(main, feed={"input": self.input}, fetch_list=[y])
        return out

    def static_graph_case_2(self):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                if self.channel_last:
                    x = x = fluid.data(
                        "input", (-1, -1, -1, self.in_channels),
                        dtype=self.dtype)
                else:
                    x = fluid.data(
                        "input", (-1, self.in_channels, -1, -1),
                        dtype=self.dtype)
                weight = fluid.data(
                    "weight", self.weight.shape, dtype=self.dtype)
                if not self.no_bias:
                    bias = fluid.data("bias", self.bias.shape, dtype=self.dtype)
                y = F.conv2d(
                    x,
                    weight,
                    None if self.no_bias else bias,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    data_format=self.data_format)

                if self.act == 'sigmoid':
                    y = F.sigmoid(y)

        exe = fluid.Executor(self.place)
        exe.run(start)
        feed_dict = {"input": self.input, "weight": self.weight}
        if not self.no_bias:
            feed_dict["bias"] = self.bias
        out, = exe.run(main, feed=feed_dict, fetch_list=[y])
        return out

    def dygraph_case(self):
        with dg.guard(self.place):
            x = dg.to_variable(self.input)
            weight = dg.to_variable(self.weight)
            bias = None if self.no_bias else dg.to_variable(self.bias)
            y = F.conv2d(
                x,
                weight,
                bias,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                data_format=self.data_format)

            if self.act == 'sigmoid':
                y = F.sigmoid(y)

            out = y.numpy()
        return out

    def _test_identity(self):
        self.prepare()
        out1 = self.static_graph_case_1()
        out2 = self.static_graph_case_2()
        out3 = self.dygraph_case()
        np.testing.assert_array_almost_equal(out1, out2)
        np.testing.assert_array_almost_equal(out2, out3)

    def test_identity_cpu(self):
        self.place = fluid.CPUPlace()
        self._test_identity()

    @unittest.skipIf(not fluid.core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    def test_identity_gpu(self):
        self.place = fluid.CUDAPlace(0)
        self._test_identity()


class TestFunctionalConv2DError(TestCase):
    batch_size = 4
    spatial_shape = (16, 16)
    dtype = "float32"

    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = "not_valid"
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NHWC"

    def test_exception(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.static_graph_case()

    def prepare(self):
        if isinstance(self.filter_shape, int):
            filter_shape = (self.filter_shape, ) * 2
        else:
            filter_shape = tuple(self.filter_shape)
        self.weight_shape = (self.out_channels, self.in_channels // self.groups
                             ) + filter_shape
        self.bias_shape = (self.out_channels, )

    def static_graph_case(self):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                self.channel_last = self.data_format == "NHWC"
                if self.channel_last:
                    x = x = fluid.data(
                        "input", (-1, -1, -1, self.in_channels),
                        dtype=self.dtype)
                else:
                    x = fluid.data(
                        "input", (-1, self.in_channels, -1, -1),
                        dtype=self.dtype)
                weight = fluid.data(
                    "weight", self.weight_shape, dtype=self.dtype)
                if not self.no_bias:
                    bias = fluid.data("bias", self.bias_shape, dtype=self.dtype)
                y = F.conv2d(
                    x,
                    weight,
                    None if self.no_bias else bias,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    data_format=self.data_format)


class TestFunctionalConv2DCase2(TestFunctionalConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = True
        self.data_format = "NHWC"


class TestFunctionalConv2DCase3(TestFunctionalConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2, 3, 1]
        self.stride = 2
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = True
        self.data_format = "NHWC"


class TestFunctionalConv2DCase4(TestFunctionalConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 1, 2, 2]
        self.stride = 1
        self.dilation = 2
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = True
        self.data_format = "NHWC"


class TestFunctionalConv2DCase5(TestFunctionalConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [1, 1], [2, 2], [0, 0]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = True
        self.data_format = "NHWC"


class TestFunctionalConv2DCase6(TestFunctionalConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [0, 0], [1, 1], [2, 2]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = True
        self.data_format = "NCHW"


class TestFunctionalConv2DCase7(TestFunctionalConv2D):
    def setUp(self):
        self.in_channels = 6
        self.out_channels = 8
        self.filter_shape = 3
        self.padding = "same"
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = True
        self.data_format = "NCHW"


class TestFunctionalConv2DCase8(TestFunctionalConv2D):
    def setUp(self):
        self.in_channels = 6
        self.out_channels = 12
        self.filter_shape = 3
        self.padding = "valid"
        self.stride = 1
        self.dilation = 1
        self.groups = 6
        self.no_bias = True
        self.act = None
        self.use_cudnn = False
        self.data_format = "NCHW"


class TestFunctionalConv2DErrorCase2(TestFunctionalConv2DError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [1, 2], [3, 4], [5, 6]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = False
        self.data_format = "NCHW"


class TestFunctionalConv2DErrorCase3(TestFunctionalConv2DError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 4
        self.filter_shape = 3
        self.padding = "same"
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = False
        self.data_format = "not_valid"


class TestFunctionalConv2DErrorCase4(TestFunctionalConv2DError):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 3
        self.filter_shape = 3
        self.padding = "same"
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = False
        self.data_format = "NCHW"


class TestFunctionalConv2DErrorCase7(TestFunctionalConv2DError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = "same"
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = True
        self.data_format = "not_valid"


class TestFunctionalConv2DErrorCase8(TestFunctionalConv2DError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2, 1, 2, 1]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = True
        self.data_format = "NCHW"


class TestFunctionalConv2DErrorCase9(TestFunctionalConv2DError):
    def setUp(self):
        self.in_channels = -5
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [0, 0], [3, 2], [1, 2]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = False
        self.data_format = "NCHW"


class TestFunctionalConv2DErrorCase10(TestFunctionalConv2DError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 4
        self.filter_shape = 3
        self.padding = "same"
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = False
        self.data_format = "NHWC"


class TestFunctionalConv2DErrorCase11(TestFunctionalConv2DError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.use_cudnn = False
        self.data_format = "NHCW"


class TestFunctionalConv2DErrorCase12(TestCase):
    def setUp(self):
        self.input = np.array([])
        self.filter = np.array([])
        self.num_filters = 0
        self.filter_size = 0
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.data_format = "NCHW"

    def static_graph_case(self):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                x = fluid.data("input", self.input.shape, dtype=paddle.float32)
                y = fluid.layers.conv2d(
                    x,
                    self.num_filters,
                    self.filter_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    param_attr=I.NumpyArrayInitializer(self.filter),
                    bias_attr=False if self.bias is None else
                    I.NumpyArrayInitializer(self.bias),
                    act=None,
                    data_format=self.data_format)
        exe = fluid.Executor()
        exe.run(start)
        out, = exe.run(main, feed={"input": self.input}, fetch_list=[y])
        return out

    def dygraph_case(self):
        with dg.guard():
            x = dg.to_variable(self.input, dtype=paddle.float32)
            w = dg.to_variable(self.filter, dtype=paddle.float32)
            b = None if self.bias is None else dg.to_variable(
                self.bias, dtype=paddle.float32)
            y = F.conv2d(
                x,
                w,
                b,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                data_format=self.data_format)

    def test_dygraph_exception(self):
        with self.assertRaises(ValueError):
            self.dygraph_case()

    def test_static_exception(self):
        with self.assertRaises(ValueError):
            self.static_graph_case()


class TestFunctionalConv2DErrorCase13(TestFunctionalConv2DErrorCase12):
    def setUp(self):
        self.input = np.random.randn(1, 3, 3, 3)
        self.filter = np.random.randn(3, 3, 1, 1)
        self.num_filters = 3
        self.filter_size = 1
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 0
        self.data_format = "NCHW"


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
