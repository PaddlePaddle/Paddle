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
from unittest import TestCase

import numpy as np

import paddle
import paddle.base.dygraph as dg
import paddle.nn.functional as F
from paddle import base

paddle.enable_static()


class TestFunctionalConv3D(TestCase):
    batch_size = 4
    spatial_shape = (8, 8, 8)
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
        self.data_format = "NDHWC"

    def prepare(self):
        if isinstance(self.filter_shape, int):
            filter_shape = (self.filter_shape,) * 3
        else:
            filter_shape = tuple(self.filter_shape)

        self.weight = np.random.uniform(
            -1,
            1,
            (
                self.out_channels,
                self.in_channels // self.groups,
                *filter_shape,
            ),
        ).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(-1, 1, (self.out_channels,)).astype(
                self.dtype
            )

        self.channel_last = self.data_format == "NDHWC"
        if self.channel_last:
            self.input_shape = (
                self.batch_size,
                *self.spatial_shape,
                self.in_channels,
            )
        else:
            self.input_shape = (
                self.batch_size,
                self.in_channels,
                *self.spatial_shape,
            )

        self.input = np.random.uniform(-1, 1, self.input_shape).astype(
            self.dtype
        )

    def static_graph_case_1(self):
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                if self.channel_last:
                    x = paddle.static.data(
                        "input",
                        (-1, -1, -1, -1, self.in_channels),
                        dtype=self.dtype,
                    )
                else:
                    x = paddle.static.data(
                        "input",
                        (-1, self.in_channels, -1, -1, -1),
                        dtype=self.dtype,
                    )
                y = paddle.static.nn.conv3d(
                    x,
                    self.out_channels,
                    self.filter_shape,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    param_attr=paddle.nn.initializer.Assign(self.weight),
                    bias_attr=(
                        False
                        if self.no_bias
                        else paddle.nn.initializer.Assign(self.bias)
                    ),
                    act=self.act,
                    data_format=self.data_format,
                )
        exe = base.Executor(self.place)
        exe.run(start)
        (out,) = exe.run(main, feed={"input": self.input}, fetch_list=[y])
        return out

    def static_graph_case_2(self):
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                if self.channel_last:
                    x = x = paddle.static.data(
                        "input",
                        (-1, -1, -1, -1, self.in_channels),
                        dtype=self.dtype,
                    )
                else:
                    x = paddle.static.data(
                        "input",
                        (-1, self.in_channels, -1, -1, -1),
                        dtype=self.dtype,
                    )
                weight = paddle.static.data(
                    "weight", self.weight.shape, dtype=self.dtype
                )
                if not self.no_bias:
                    bias = paddle.static.data(
                        "bias", self.bias.shape, dtype=self.dtype
                    )
                y = F.conv3d(
                    x,
                    weight,
                    None if self.no_bias else bias,
                    padding=self.padding,
                    stride=self.stride,
                    dilation=self.dilation,
                    groups=self.groups,
                    data_format=self.data_format,
                )

                if self.act == 'sigmoid':
                    y = F.sigmoid(y)

        exe = base.Executor(self.place)
        exe.run(start)
        feed_dict = {"input": self.input, "weight": self.weight}
        if not self.no_bias:
            feed_dict["bias"] = self.bias
        (out,) = exe.run(main, feed=feed_dict, fetch_list=[y])
        return out

    def dygraph_case(self):
        with dg.guard(self.place):
            x = paddle.to_tensor(self.input)
            weight = paddle.to_tensor(self.weight)
            bias = None if self.no_bias else paddle.to_tensor(self.bias)
            y = F.conv3d(
                x,
                weight,
                bias,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                data_format=self.data_format,
            )

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
        self.place = base.CPUPlace()
        self._test_identity()

    @unittest.skipIf(
        not base.core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    def test_identity_gpu(self):
        self.place = base.CUDAPlace(0)
        self._test_identity()


class TestFunctionalConv3DCase2(TestFunctionalConv3D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2, 1]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DCase3(TestFunctionalConv3D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 2, 3, 1, 2, 3]
        self.stride = 2
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DCase4(TestFunctionalConv3D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [1, 1, 2, 2, 3, 3]
        self.stride = 1
        self.dilation = 2
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DCase5(TestFunctionalConv3D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [1, 1], [2, 2], [1, 1], [0, 0]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NDHWC"


class TestFunctionalConv3DCase6(TestFunctionalConv3D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [0, 0], [1, 1], [2, 2], [2, 2]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DCase7(TestFunctionalConv3D):
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
        self.data_format = "NCDHW"


class TestFunctionalConv3DCase8(TestFunctionalConv3D):
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
        self.data_format = "NCDHW"


class TestFunctionalConv3DErrorCase11(TestCase):
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
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                x = paddle.static.data(
                    "input", self.input.shape, dtype=paddle.float32
                )
                y = paddle.static.nn.conv3d(
                    x,
                    self.num_filters,
                    self.filter_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                    param_attr=paddle.nn.initializer.Assign(self.filter),
                    bias_attr=(
                        False
                        if self.bias is None
                        else paddle.nn.initializer.Assign(self.bias)
                    ),
                    act=None,
                    data_format=self.data_format,
                )
        exe = base.Executor()
        exe.run(start)
        (out,) = exe.run(main, feed={"input": self.input}, fetch_list=[y])
        return out

    def test_static_exception(self):
        with self.assertRaises(ValueError):
            self.static_graph_case()


class TestFunctionalConv3DErrorCase12(TestFunctionalConv3DErrorCase11):
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


class TestFunctionalConv3DErrorCase13(TestFunctionalConv3DErrorCase11):
    def setUp(self):
        self.input = np.random.randn(0, 0, 0, 0, 0)
        self.filter = np.random.randn(1, 0, 0, 0, 0)
        self.num_filters = 1
        self.filter_size = 1
        self.bias = None
        self.padding = 0
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.data_format = "NCDHW"


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
