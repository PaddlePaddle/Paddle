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


class TestFunctionalConv3DError(TestCase):
    batch_size = 4
    spatial_shape = (8, 8, 8)
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
        self.data_format = "NDHWC"

    def test_exception(self):
        self.prepare()
        with self.assertRaises(ValueError):
            self.static_graph_case()

    def prepare(self):
        if isinstance(self.filter_shape, int):
            filter_shape = (self.filter_shape,) * 3
        else:
            filter_shape = tuple(self.filter_shape)
        self.weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
            *filter_shape,
        )
        self.bias_shape = (self.out_channels,)

    def static_graph_case(self):
        main = base.Program()
        start = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, start):
                self.channel_last = self.data_format == "NDHWC"
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
                    "weight", self.weight_shape, dtype=self.dtype
                )
                if not self.no_bias:
                    bias = paddle.static.data(
                        "bias", self.bias_shape, dtype=self.dtype
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


class TestFunctionalConv3DErrorCase2(TestFunctionalConv3DError):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [1, 1], [1, 2], [3, 4], [5, 6]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DErrorCase3(TestFunctionalConv3DError):
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
        self.data_format = "not_valid"


class TestFunctionalConv3DErrorCase4(TestFunctionalConv3DError):
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
        self.data_format = "NCDHW"


class TestFunctionalConv3DErrorCase7(TestFunctionalConv3DError):
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
        self.data_format = "not_valid"


class TestFunctionalConv3DErrorCase8(TestFunctionalConv3DError):
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
        self.data_format = "NCDHW"


class TestFunctionalConv3DErrorCase9(TestFunctionalConv3DError):
    def setUp(self):
        self.in_channels = -5
        self.out_channels = 5
        self.filter_shape = 3
        self.padding = [[0, 0], [0, 0], [3, 2], [1, 2], [1, 1]]
        self.stride = 1
        self.dilation = 1
        self.groups = 1
        self.no_bias = False
        self.act = "sigmoid"
        self.data_format = "NCDHW"


class TestFunctionalConv3DErrorCase10(TestFunctionalConv3DError):
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
        self.data_format = "NDHWC"


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

    def dygraph_case(self):
        with dg.guard():
            x = paddle.to_tensor(self.input, dtype=paddle.float32)
            w = paddle.to_tensor(self.filter, dtype=paddle.float32)
            b = (
                None
                if self.bias is None
                else paddle.to_tensor(self.bias, dtype=paddle.float32)
            )
            y = F.conv3d(
                x,
                w,
                b,
                padding=self.padding,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.groups,
                data_format=self.data_format,
            )

    def test_dygraph_exception(self):
        with self.assertRaises(ValueError):
            self.dygraph_case()


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
