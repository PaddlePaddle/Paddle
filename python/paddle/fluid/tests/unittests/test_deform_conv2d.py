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
import paddle.nn.initializer as I
import numpy as np
import unittest
from paddle.fluid.framework import _test_eager_guard
from unittest import TestCase


class TestDeformConv2D(TestCase):
    batch_size = 4
    spatial_shape = (5, 5)
    dtype = "float32"

    def setUp(self):
        self.in_channels = 2
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [0, 0]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = True

    def prepare(self):
        np.random.seed(1)
        paddle.seed(1)
        if isinstance(self.kernel_size, int):
            filter_shape = (self.kernel_size, ) * 2
        else:
            filter_shape = tuple(self.kernel_size)
        self.filter_shape = filter_shape

        self.weight = np.random.uniform(
            -1, 1, (self.out_channels, self.in_channels // self.groups
                    ) + filter_shape).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(-1, 1, (
                self.out_channels, )).astype(self.dtype)

        def out_size(in_size, pad_size, dilation_size, kernel_size,
                     stride_size):
            return (in_size + 2 * pad_size -
                    (dilation_size * (kernel_size - 1) + 1)) / stride_size + 1

        out_h = int(
            out_size(self.spatial_shape[0], self.padding[0], self.dilation[0],
                     self.kernel_size[0], self.stride[0]))
        out_w = int(
            out_size(self.spatial_shape[1], self.padding[1], self.dilation[1],
                     self.kernel_size[1], self.stride[1]))
        out_shape = (out_h, out_w)

        self.input_shape = (self.batch_size, self.in_channels
                            ) + self.spatial_shape

        self.offset_shape = (self.batch_size, self.deformable_groups * 2 *
                             filter_shape[0] * filter_shape[1]) + out_shape

        self.mask_shape = (self.batch_size, self.deformable_groups *
                           filter_shape[0] * filter_shape[1]) + out_shape

        self.input = np.random.uniform(-1, 1,
                                       self.input_shape).astype(self.dtype)

        self.offset = np.random.uniform(-1, 1,
                                        self.offset_shape).astype(self.dtype)

        self.mask = np.random.uniform(-1, 1, self.mask_shape).astype(self.dtype)

    def static_graph_case_dcn(self):
        main = paddle.static.Program()
        start = paddle.static.Program()
        paddle.enable_static()
        with paddle.static.program_guard(main, start):
            x = paddle.static.data(
                "input", (-1, self.in_channels, -1, -1), dtype=self.dtype)
            offset = paddle.static.data(
                "offset", (-1, self.deformable_groups * 2 *
                           self.filter_shape[0] * self.filter_shape[1], -1, -1),
                dtype=self.dtype)
            mask = paddle.static.data(
                "mask", (-1, self.deformable_groups * self.filter_shape[0] *
                         self.filter_shape[1], -1, -1),
                dtype=self.dtype)

            y_v1 = paddle.fluid.layers.deformable_conv(
                input=x,
                offset=offset,
                mask=None,
                num_filters=self.out_channels,
                filter_size=self.filter_shape,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                deformable_groups=self.deformable_groups,
                im2col_step=1,
                param_attr=I.Assign(self.weight),
                bias_attr=False if self.no_bias else I.Assign(self.bias),
                modulated=False)

            y_v2 = paddle.fluid.layers.deformable_conv(
                input=x,
                offset=offset,
                mask=mask,
                num_filters=self.out_channels,
                filter_size=self.filter_shape,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                deformable_groups=self.deformable_groups,
                im2col_step=1,
                param_attr=I.Assign(self.weight),
                bias_attr=False if self.no_bias else I.Assign(self.bias))

        exe = paddle.static.Executor(self.place)
        exe.run(start)
        out_v1, out_v2 = exe.run(main,
                                 feed={
                                     "input": self.input,
                                     "offset": self.offset,
                                     "mask": self.mask
                                 },
                                 fetch_list=[y_v1, y_v2])
        return out_v1, out_v2

    def dygraph_case_dcn(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.input)
        offset = paddle.to_tensor(self.offset)
        mask = paddle.to_tensor(self.mask)

        bias = None if self.no_bias else paddle.to_tensor(self.bias)

        deform_conv2d = paddle.vision.ops.DeformConv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            deformable_groups=self.deformable_groups,
            groups=self.groups,
            weight_attr=I.Assign(self.weight),
            bias_attr=False if self.no_bias else I.Assign(self.bias))

        y_v1 = deform_conv2d(x, offset)
        y_v2 = deform_conv2d(x, offset, mask)

        out_v1 = y_v1.numpy()
        out_v2 = y_v2.numpy()

        return out_v1, out_v2

    def _test_identity(self):
        self.prepare()
        static_dcn_v1, static_dcn_v2 = self.static_graph_case_dcn()
        dy_dcn_v1, dy_dcn_v2 = self.dygraph_case_dcn()
        np.testing.assert_array_almost_equal(static_dcn_v1, dy_dcn_v1)
        np.testing.assert_array_almost_equal(static_dcn_v2, dy_dcn_v2)

    def test_identity(self):
        self.place = paddle.CPUPlace()
        self._test_identity()

        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
            self._test_identity()

    def test_identity_with_eager_guard(self):
        with _test_eager_guard():
            self.test_identity()


class TestDeformConv2DFunctional(TestCase):
    batch_size = 4
    spatial_shape = (5, 5)
    dtype = "float32"

    def setUp(self):
        self.in_channels = 2
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [0, 0]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = True

    def prepare(self):
        np.random.seed(1)
        paddle.seed(1)
        if isinstance(self.kernel_size, int):
            filter_shape = (self.kernel_size, ) * 2
        else:
            filter_shape = tuple(self.kernel_size)
        self.filter_shape = filter_shape

        self.weight = np.random.uniform(
            -1, 1, (self.out_channels, self.in_channels // self.groups
                    ) + filter_shape).astype(self.dtype)
        if not self.no_bias:
            self.bias = np.random.uniform(-1, 1, (
                self.out_channels, )).astype(self.dtype)

        def out_size(in_size, pad_size, dilation_size, kernel_size,
                     stride_size):
            return (in_size + 2 * pad_size -
                    (dilation_size * (kernel_size - 1) + 1)) / stride_size + 1

        out_h = int(
            out_size(self.spatial_shape[0], self.padding[0], self.dilation[0],
                     self.kernel_size[0], self.stride[0]))
        out_w = int(
            out_size(self.spatial_shape[1], self.padding[1], self.dilation[1],
                     self.kernel_size[1], self.stride[1]))
        out_shape = (out_h, out_w)

        self.input_shape = (self.batch_size, self.in_channels
                            ) + self.spatial_shape

        self.offset_shape = (self.batch_size, self.deformable_groups * 2 *
                             filter_shape[0] * filter_shape[1]) + out_shape

        self.mask_shape = (self.batch_size, self.deformable_groups *
                           filter_shape[0] * filter_shape[1]) + out_shape

        self.input = np.random.uniform(-1, 1,
                                       self.input_shape).astype(self.dtype)

        self.offset = np.random.uniform(-1, 1,
                                        self.offset_shape).astype(self.dtype)

        self.mask = np.random.uniform(-1, 1, self.mask_shape).astype(self.dtype)

    def static_graph_case_dcn(self):
        main = paddle.static.Program()
        start = paddle.static.Program()
        paddle.enable_static()
        with paddle.static.program_guard(main, start):
            x = paddle.static.data(
                "input", (-1, self.in_channels, -1, -1), dtype=self.dtype)
            offset = paddle.static.data(
                "offset", (-1, self.deformable_groups * 2 *
                           self.filter_shape[0] * self.filter_shape[1], -1, -1),
                dtype=self.dtype)
            mask = paddle.static.data(
                "mask", (-1, self.deformable_groups * self.filter_shape[0] *
                         self.filter_shape[1], -1, -1),
                dtype=self.dtype)

            y_v1 = paddle.fluid.layers.deformable_conv(
                input=x,
                offset=offset,
                mask=None,
                num_filters=self.out_channels,
                filter_size=self.filter_shape,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                deformable_groups=self.deformable_groups,
                im2col_step=1,
                param_attr=I.Assign(self.weight),
                bias_attr=False if self.no_bias else I.Assign(self.bias),
                modulated=False)

            y_v2 = paddle.fluid.layers.deformable_conv(
                input=x,
                offset=offset,
                mask=mask,
                num_filters=self.out_channels,
                filter_size=self.filter_shape,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
                deformable_groups=self.deformable_groups,
                im2col_step=1,
                param_attr=I.Assign(self.weight),
                bias_attr=False if self.no_bias else I.Assign(self.bias))

        exe = paddle.static.Executor(self.place)
        exe.run(start)
        out_v1, out_v2 = exe.run(main,
                                 feed={
                                     "input": self.input,
                                     "offset": self.offset,
                                     "mask": self.mask
                                 },
                                 fetch_list=[y_v1, y_v2])
        return out_v1, out_v2

    def dygraph_case_dcn(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.input)
        offset = paddle.to_tensor(self.offset)
        mask = paddle.to_tensor(self.mask)
        weight = paddle.to_tensor(self.weight)
        bias = None if self.no_bias else paddle.to_tensor(self.bias)

        y_v1 = paddle.vision.ops.deform_conv2d(
            x=x,
            offset=offset,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            deformable_groups=self.deformable_groups,
            groups=self.groups, )

        y_v2 = paddle.vision.ops.deform_conv2d(
            x=x,
            offset=offset,
            mask=mask,
            weight=weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            deformable_groups=self.deformable_groups,
            groups=self.groups, )

        out_v1 = y_v1.numpy()
        out_v2 = y_v2.numpy()

        return out_v1, out_v2

    def new_api_static_graph_case_dcn(self):
        main = paddle.static.Program()
        start = paddle.static.Program()
        paddle.enable_static()
        with paddle.static.program_guard(main, start):
            x = paddle.static.data(
                "input", (-1, self.in_channels, -1, -1), dtype=self.dtype)
            offset = paddle.static.data(
                "offset", (-1, self.deformable_groups * 2 *
                           self.filter_shape[0] * self.filter_shape[1], -1, -1),
                dtype=self.dtype)
            mask = paddle.static.data(
                "mask", (-1, self.deformable_groups * self.filter_shape[0] *
                         self.filter_shape[1], -1, -1),
                dtype=self.dtype)

            weight = paddle.static.data(
                "weight", list(self.weight.shape), dtype=self.dtype)

            if not self.no_bias:
                bias = paddle.static.data("bias", [-1], dtype=self.dtype)

            y_v1 = paddle.vision.ops.deform_conv2d(
                x=x,
                offset=offset,
                weight=weight,
                bias=None if self.no_bias else bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                deformable_groups=self.deformable_groups,
                groups=self.groups, )

            y_v2 = paddle.vision.ops.deform_conv2d(
                x=x,
                offset=offset,
                mask=mask,
                weight=weight,
                bias=None if self.no_bias else bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                deformable_groups=self.deformable_groups,
                groups=self.groups, )

        exe = paddle.static.Executor(self.place)
        exe.run(start)
        feed_dict = {
            "input": self.input,
            "offset": self.offset,
            "mask": self.mask,
            "weight": self.weight
        }
        if not self.no_bias:
            feed_dict["bias"] = self.bias

        out_v1, out_v2 = exe.run(main, feed=feed_dict, fetch_list=[y_v1, y_v2])
        return out_v1, out_v2

    def _test_identity(self):
        self.prepare()
        static_dcn_v1, static_dcn_v2 = self.static_graph_case_dcn()
        dy_dcn_v1, dy_dcn_v2 = self.dygraph_case_dcn()
        new_static_dcn_v1, new_static_dcn_v2 = self.new_api_static_graph_case_dcn(
        )
        np.testing.assert_array_almost_equal(static_dcn_v1, dy_dcn_v1)
        np.testing.assert_array_almost_equal(static_dcn_v2, dy_dcn_v2)
        np.testing.assert_array_almost_equal(static_dcn_v1, new_static_dcn_v1)
        np.testing.assert_array_almost_equal(static_dcn_v2, new_static_dcn_v2)

    def test_identity(self):
        self.place = paddle.CPUPlace()
        self._test_identity()

        if paddle.is_compiled_with_cuda():
            self.place = paddle.CUDAPlace(0)
            self._test_identity()

    def test_identity_with_eager_guard(self):
        with _test_eager_guard():
            self.test_identity()


# testcases for DeformConv2D
class TestDeformConv2DWithPadding(TestDeformConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [2, 2]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = True


class TestDeformConv2DWithBias(TestDeformConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [2, 2]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DWithAsynPadding(TestDeformConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 2]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DWithDilation(TestDeformConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 1]
        self.stride = [1, 1]
        self.dilation = [3, 3]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DWithStride(TestDeformConv2D):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 1]
        self.stride = [2, 2]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DWithDeformable_Groups(TestDeformConv2D):
    def setUp(self):
        self.in_channels = 5
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 1]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 5
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DWithGroups(TestDeformConv2D):
    def setUp(self):
        self.in_channels = 5
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 1]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 5
        self.no_bias = False


# testcases for deform_conv2d
class TestDeformConv2DFunctionalWithPadding(TestDeformConv2DFunctional):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [2, 2]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = True


class TestDeformConv2DFunctionalWithBias(TestDeformConv2DFunctional):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [2, 2]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DFunctionalWithAsynPadding(TestDeformConv2DFunctional):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 2]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DFunctionalWithDilation(TestDeformConv2DFunctional):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 1]
        self.stride = [1, 1]
        self.dilation = [3, 3]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DFunctionalWithStride(TestDeformConv2DFunctional):
    def setUp(self):
        self.in_channels = 3
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 1]
        self.stride = [2, 2]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DFunctionalWithDeformable_Groups(
        TestDeformConv2DFunctional):
    def setUp(self):
        self.in_channels = 5
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 1]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 5
        self.groups = 1
        self.no_bias = False


class TestDeformConv2DFunctionalWithGroups(TestDeformConv2DFunctional):
    def setUp(self):
        self.in_channels = 5
        self.out_channels = 5
        self.kernel_size = [3, 3]
        self.padding = [1, 1]
        self.stride = [1, 1]
        self.dilation = [1, 1]
        self.deformable_groups = 1
        self.groups = 5
        self.no_bias = False


if __name__ == "__main__":
    unittest.main()
