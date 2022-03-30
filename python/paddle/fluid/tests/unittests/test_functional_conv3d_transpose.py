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
import numpy as np
import paddle.fluid.dygraph as dg
import paddle.fluid.initializer as I
import paddle.nn.functional as F
import unittest
from paddle import fluid
from paddle.fluid.framework import _test_eager_guard
from unittest import TestCase


class TestFunctionalConv3DTranspose(TestCase):
    batch_size = 4
    spatial_shape = (8, 8, 8)
    dtype = "float32"
    output_size = None

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
        self.data_format = "NDHWC"

    def prepare(self):
        if isinstance(self.filter_shape, int):
            filter_shape = (self.filter_shape, ) * 3
        else:
            filter_shape = tuple(self.filter_shape)

        self.weight = np.random.uniform(
            -1, 1, (self.in_channels, self.out_channels // self.groups
                    ) + filter_shape).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(-1, 1, (
                self.out_channels, )).astype(self.dtype)

        self.channel_last = (self.data_format == "NDHWC")
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
                        "input", (-1, -1, -1, -1, self.in_channels),
                        dtype=self.dtype)
                else:
                    x = fluid.data(
                        "input", (-1, self.in_channels, -1, -1, -1),
                        dtype=self.dtype)
                y = fluid.layers.conv3d_transpose(
                    x,
                    self.out_channels,
                    output_size=self.output_size,
                    filter_size=self.filter_shape,
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
                        "input", (-1, -1, -1, -1, self.in_channels),
                        dtype=self.dtype)
                else:
                    x = fluid.data(
                        "input", (-1, self.in_channels, -1, -1, -1),
                        dtype=self.dtype)
                weight = fluid.data(
                    "weight", self.weight.shape, dtype=self.dtype)
                if not self.no_bias:
                    bias = fluid.data("bias", self.bias.shape, dtype=self.dtype)
                y = F.conv3d_transpose(
                    x,
                    weight,
                    None if self.no_bias else bias,
                    output_size=self.output_size,
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
            y = F.conv3d_transpose(
                x,
                weight,
                bias,
                output_size=self.output_size,
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

    def test_identity_cpu_check_eager(self):
        with _test_eager_guard():
            self.test_identity_cpu()

    @unittest.skipIf(not fluid.core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    def test_identity_gpu(self):
        self.place = fluid.CUDAPlace(0)
        self._test_identity()

    @unittest.skipIf(not fluid.core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    def test_identity_gpu_check_eager(self):
        with _test_eager_guard():
            self.test_identity_gpu()


class TestFunctionalConv3DTransposeError(TestCase):
    batch_size = 4
    spatial_shape = (8, 8, 8)
    dtype = "float32"
    output_size = None

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
        self.data_format = "NDHWC"

    def test_exception(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.static_graph_case()

    def prepare(self):
        if isinstance(self.filter_shape, int):
            filter_shape = (self.filter_shape, ) * 3
        else:
            filter_shape = tuple(self.filter_shape)
        self.weight_shape = (self.in_channels, self.out_channels // self.groups
                             ) + filter_shape
        self.bias_shape = (self.out_channels, )

    def static_graph_case(self):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                self.channel_last = self.data_format == "NDHWC"
                if self.channel_last:
                    x = x = fluid.data(
                        "input", (-1, -1, -1, -1, self.in_channels),
                        dtype=self.dtype)
                else:
                    x = fluid.data(
                        "input", (-1, self.in_channels, -1, -1, -1),
                        dtype=self.dtype)
                weight = fluid.data(
                    "weight", self.weight_shape, dtype=self.dtype)
                if not self.no_bias:
                    bias = fluid.data("bias", self.bias_shape, dtype=self.dtype)
                y = F.conv3d_transpose(
                    x,
                    weight,
                    None if self.no_bias else bias,
                    output_size=self.output_size,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    data_format=self.data_format)
                if self.act == 'sigmoid':
                    y = F.sigmoid(y)


class TestFunctionalConv3DTransposeCase2(TestFunctionalConv3DTranspose):
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
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeCase3(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 6
        self.filter_shape = 3
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DTransposeCase4(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 6
        self.filter_shape = 3
        self.padding = "same"
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = True
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DTransposeCase5(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 6
        self.filter_shape = 3
        self.padding = "valid"
        self.stride = (1, 2, 1)
        self.dilation = (2, 1, 1)
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DTransposeCase6(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 4
        self.filter_shape = 3
        self.padding = "valid"
        self.stride = (1, 2, 1)
        self.dilation = 1
        self.groups = 4
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DTransposeCase7(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 4
        self.filter_shape = 3
        self.padding = "valid"
        self.output_size = (10, 17, 10)
        self.stride = (1, 2, 1)
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeCase8(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 6
        self.filter_shape = 3
        self.padding = [[0, 0], [1, 2], [1, 2], [2, 1], [0, 0]]
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DTransposeCase9(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 6
        self.filter_shape = 3
        self.padding = [[0, 0], [0, 0], [1, 1], [1, 1], [2, 2]]
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeCase10(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 6
        self.filter_shape = 3
        self.padding = [1, 1, 2, 2, 1, 1]
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeCase11(TestFunctionalConv3DTranspose):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 6
        self.filter_shape = 3
        self.padding = [1, 2, 1]
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeErrorCase2(
        TestFunctionalConv3DTransposeError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2, 2, 1, 3]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DTransposeErrorCase3(
        TestFunctionalConv3DTransposeError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [0, 0], [1, 1], [1, 2], [2, 1]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DTransposeErrorCase4(
        TestFunctionalConv3DTransposeError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [1, 2], [1, 1], [0, 0], [2, 1]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeErrorCase5(
        TestFunctionalConv3DTransposeError):
    def setUp(self):
        self.in_channels = -2
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeErrorCase7(
        TestFunctionalConv3DTransposeError):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = 0
        self.output_size = "not_valid"
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeErrorCase8(
        TestFunctionalConv3DTransposeError):
    def setUp(self):
        self.in_channels = 4
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "not_valid"


class TestFunctionalConv3DTransposeErrorCase9(
        TestFunctionalConv3DTransposeError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 4
        self.filter_shape = 3
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 2
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DTransposeErrorCase10(TestCase):
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
        self.data_format = "NCDHW"

    def static_graph_case(self):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                x = fluid.data("input", self.input.shape, dtype=paddle.float32)
                y = fluid.layers.conv3d_transpose(
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
            y = F.conv3d_transpose(
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

    def test_dygraph_exception_check_eager(self):
        with _test_eager_guard():
            self.test_dygraph_exception()

    def test_static_exception(self):
        with self.assertRaises(ValueError):
            self.static_graph_case()


class TestFunctionalConv3DTransposeErrorCase11(
        TestFunctionalConv3DTransposeErrorCase10):
    def setUp(self):
        self.input = np.random.randn(1, 3, 3, 3, 3)
        self.filter = np.random.randn(3, 3, 1, 1, 1)
        self.num_filters = 3
        self.filter_size = 1
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 0
        self.data_format = "NCDHW"


if __name__ == "__main__":
    unittest.main()
