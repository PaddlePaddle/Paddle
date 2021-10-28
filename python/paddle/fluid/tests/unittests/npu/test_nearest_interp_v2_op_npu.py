#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import sys
sys.path.append("..")
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.nn as nn
import paddle
from paddle.nn.functional import interpolate

from test_nearest_interp_v2_op import nearest_neighbor_interp_np

paddle.enable_static()


class TestNearestInterpOp(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def setUp(self):
        self.set_npu()
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "nearest_interp_v2"
        input_np = np.random.random(self.input_shape).astype("float32")

        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]
        scale_h = 0
        scale_w = 0
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0:
                    scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[1]
                scale_h = self.scale[0]
            output_h = int(in_h * scale_h)
            output_w = int(in_w * scale_w)
        else:
            output_h = self.out_h
            output_w = self.out_w

        output_np = nearest_neighbor_interp_np(
            input_np, output_h, output_w, scale_h, scale_w, self.out_size,
            self.actual_shape, self.align_corners, self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape
        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'data_layout': self.data_layout
        }
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place, ['X'], 'Out', in_place=True, max_relative_error=0.006)

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 3, 4, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = False


class TestNearestNeighborInterpCase1(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.align_corners = False


class TestNearestNeighborInterpCase2(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.align_corners = False


class TestNearestNeighborInterpCase3(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.align_corners = False


class TestNearestNeighborInterpCase4(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")
        self.align_corners = False


class TestNearestNeighborInterpCase5(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11]).astype("int32")
        self.align_corners = False


class TestNearestNeighborInterpCase6(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 129]).astype("int32")
        self.align_corners = False


class TestNearestNeighborInterpSame(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.
        self.align_corners = False


class TestNearestNeighborInterpActualShape(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = False


class TestNearestNeighborInterpScale1(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 7, 5]
        self.out_h = 64
        self.out_w = 32
        self.scale = 2.
        self.out_size = None
        self.align_corners = False


class TestNearestNeighborInterpScale2(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 5, 7]
        self.out_h = 64
        self.out_w = 32
        self.scale = 1.5
        self.out_size = None
        self.align_corners = False


class TestNearestNeighborInterpScale3(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 7, 5]
        self.out_h = 64
        self.out_w = 32
        self.scale = [2.0, 3.0]
        self.out_size = None
        self.align_corners = False


class TestNearestInterpOp_attr_tensor(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def setUp(self):
        self.set_npu()
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "nearest_interp_v2"
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False
        self.attrs = {
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
        }

        input_np = np.random.random(self.input_shape).astype("float32")
        self.inputs = {'X': input_np}

        if self.scale_by_1Dtensor:
            self.inputs['Scale'] = np.array([self.scale]).astype("float32")
        elif self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0:
                    scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[1]
                scale_h = self.scale[0]
            out_h = int(self.input_shape[2] * scale_h)
            out_w = int(self.input_shape[3] * scale_w)
        else:
            out_h = self.out_h
            out_w = self.out_w

        if self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.out_size
        elif self.out_size is not None:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
            self.inputs['SizeTensor'] = size_tensor

        self.attrs['out_h'] = self.out_h
        self.attrs['out_w'] = self.out_w
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        output_np = nearest_neighbor_interp_np(input_np, out_h, out_w, 0, 0,
                                               self.out_size, self.actual_shape,
                                               self.align_corners)
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 5, 4, 4]
        self.out_h = 3
        self.out_w = 3
        self.scale = 0.
        self.out_size = [3, 3]
        self.align_corners = False


# out_size is a tensor list
class TestNearestInterp_attr_tensor_Case1(TestNearestInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = [8, 12]
        self.align_corners = False


# out_size is a 1-D tensor
class TestNearestInterp_attr_tensor_Case2(TestNearestInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = False
        self.shape_by_1Dtensor = True


# scale is a 1-D tensor
class TestNearestInterp_attr_tensor_Case3(TestNearestInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 2.0
        self.out_size = None
        self.align_corners = False
        self.scale_by_1Dtensor = True


class TestNearestInterpOpAPI_dy(unittest.TestCase):
    def test_case(self):
        import paddle
        if core.is_compiled_with_npu():
            place = core.NPUPlace(0)
        else:
            place = core.CPUPlace()
        with fluid.dygraph.guard(place):
            input_data = np.random.random((2, 3, 6, 6)).astype("float32")
            scale_np = np.array([2, 2]).astype("int64")
            input_x = paddle.to_tensor(input_data)
            scale = paddle.to_tensor(scale_np)
            expect_res = nearest_neighbor_interp_np(
                input_data, out_h=12, out_w=12, align_corners=False)
            out = interpolate(
                x=input_x,
                scale_factor=scale,
                mode="nearest",
                align_corners=False)
            self.assertTrue(np.allclose(out.numpy(), expect_res))


if __name__ == "__main__":
    unittest.main()
