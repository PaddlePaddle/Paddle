# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle.nn as nn
import paddle
from paddle.nn.functional import interpolate

paddle.enable_static()


def nearest_interp_test(x,
                        OutSize=None,
                        SizeTensor=None,
                        Scale=None,
                        data_layout='NCHW',
                        out_d=-1,
                        out_h=-1,
                        out_w=-1,
                        scale=[],
                        interp_method='nearest',
                        align_corners=True,
                        align_mode=0):
    if isinstance(scale, float) or isinstance(scale, int):
        scale_list = []
        for _ in range(len(x.shape) - 2):
            scale_list.append(scale)
        scale = list(map(float, scale_list))
    elif isinstance(scale, list) or isinstance(scale, tuple):
        scale = list(map(float, scale))
    if SizeTensor is not None:
        if not isinstance(SizeTensor, list) and not isinstance(
                SizeTensor, tuple):
            SizeTensor = [SizeTensor]
    return paddle._C_ops.nearest_interp(x, OutSize, SizeTensor, Scale,
                                        data_layout, out_d, out_h, out_w, scale,
                                        interp_method, align_corners,
                                        align_mode)


def nearest_neighbor_interp_np(X,
                               out_h,
                               out_w,
                               scale_h=0,
                               scale_w=0,
                               out_size=None,
                               actual_shape=None,
                               align_corners=True,
                               data_layout='NCHW'):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        X = np.transpose(X, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    n, c, in_h, in_w = X.shape

    ratio_h = ratio_w = 0.0
    if (out_h > 1):
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if (out_w > 1):
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w
    out = np.zeros((n, c, out_h, out_w))

    if align_corners:
        for i in range(out_h):
            in_i = int(ratio_h * i + 0.5)
            for j in range(out_w):
                in_j = int(ratio_w * j + 0.5)
                out[:, :, i, j] = X[:, :, in_i, in_j]
    else:
        for i in range(out_h):
            in_i = int(ratio_h * i)
            for j in range(out_w):
                in_j = int(ratio_w * j)
                out[:, :, i, j] = X[:, :, in_i, in_j]

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    # out = np.expand_dims(out, 2)
    return out.astype(X.dtype)


def nearest_neighbor_interp3d_np(X,
                                 out_d,
                                 out_h,
                                 out_w,
                                 scale_d=0,
                                 scale_h=0,
                                 scale_w=0,
                                 out_size=None,
                                 actual_shape=None,
                                 align_corners=True,
                                 data_layout='NCHW'):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        X = np.transpose(X, (0, 4, 1, 2, 3))  # NDHWC => NCDHW
    if out_size is not None:
        out_d = out_size[0]
        out_h = out_size[1]
        out_w = out_size[2]
    if actual_shape is not None:
        out_d = actual_shape[0]
        out_h = actual_shape[1]
        out_w = actual_shape[2]
    n, c, in_d, in_h, in_w = X.shape

    ratio_d = ratio_h = ratio_w = 0.0
    if (out_d > 1):
        if (align_corners):
            ratio_d = (in_d - 1.0) / (out_d - 1.0)
        else:
            if scale_d > 0:
                ratio_d = 1.0 / scale_d
            else:
                ratio_d = 1.0 * in_d / out_d
    if (out_h > 1):
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if (out_w > 1):
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w
    out = np.zeros((n, c, out_d, out_h, out_w))

    if align_corners:
        for d in range(out_d):
            in_d = int(ratio_d * d + 0.5)
            for i in range(out_h):
                in_i = int(ratio_h * i + 0.5)
                for j in range(out_w):
                    in_j = int(ratio_w * j + 0.5)
                    out[:, :, d, i, j] = X[:, :, in_d, in_i, in_j]
    else:
        for d in range(out_d):
            in_d = int(ratio_d * d)
            for i in range(out_h):
                in_i = int(ratio_h * i)
                for j in range(out_w):
                    in_j = int(ratio_w * j)
                    out[:, :, d, i, j] = X[:, :, in_d, in_i, in_j]

    if data_layout == "NDHWC":
        out = np.transpose(out, (0, 2, 3, 4, 1))  # NCDHW => NDHWC
    return out.astype(X.dtype)


class TestNearestInterpOp(OpTest):

    def setUp(self):
        self.python_api = nearest_interp_test
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "nearest_interp_v2"
        input_np = np.random.random(self.input_shape).astype("float64")

        if self.data_layout == "NCHW" and len(self.input_shape) == 4:
            in_d = 1
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_d = 1
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]

        if self.data_layout == "NCDHW" and len(self.input_shape) == 5:
            in_d = self.input_shape[2]
            in_h = self.input_shape[3]
            in_w = self.input_shape[4]
        else:
            in_d = self.input_shape[1]
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        scale_d = 0
        scale_h = 0
        scale_w = 0
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0:
                    scale_d = scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_d = scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                if len(self.scale) == 5:
                    scale_w = self.scale[2]
                    scale_h = self.scale[1]
                    scale_d = self.scale[0]
                else:
                    scale_w = self.scale[1]
                    scale_h = self.scale[0]

            out_h = int(in_h * scale_h)
            out_w = int(in_w * scale_w)
            out_d = int(in_d * scale_d)
        else:
            if len(self.input_shape) == 5:
                out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        if len(self.input_shape) == 4:
            output_np = nearest_neighbor_interp_np(
                input_np, out_h, out_w, scale_h, scale_w, self.out_size,
                self.actual_shape, self.align_corners, self.data_layout)
        elif len(self.input_shape) == 5:
            output_np = nearest_neighbor_interp3d_np(input_np, out_d, out_h,
                                                     out_w, scale_d, scale_h,
                                                     scale_w, self.out_size,
                                                     self.actual_shape,
                                                     self.align_corners,
                                                     self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape
        if len(self.input_shape) == 5:
            self.attrs = {
                'out_d': self.out_d,
                'out_h': self.out_h,
                'out_w': self.out_w,
                'interp_method': self.interp_method,
                'align_corners': self.align_corners,
                'data_layout': self.data_layout
            }
        else:
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
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True, check_eager=True)

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 3, 4, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = []
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True


class TestNearestNeighborInterpCase1(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 1, 7, 8]
        self.out_d = 1
        self.out_h = 1
        self.out_w = 1
        self.scale = []
        self.align_corners = True


class TestNearestNeighborInterpCase2(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = []
        self.align_corners = True


class TestNearestNeighborInterpCase3(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = []
        self.align_corners = True


class TestNearestNeighborInterpCase4(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = []
        self.out_size = np.array([2, 2]).astype("int32")
        self.align_corners = True


class TestNearestNeighborInterpCase5(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = []
        self.out_size = np.array([11, 11]).astype("int32")
        self.align_corners = True


class TestNearestNeighborInterpCase6(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = []
        self.out_size = np.array([65, 129]).astype("int32")
        self.align_corners = True


class TestNearestNeighborInterpSame(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = []
        self.align_corners = True


class TestNearestNeighborInterpActualShape(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = []
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


class TestNearestNeighborInterpDataLayout(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 4, 4, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = []
        self.out_size = np.array([3, 8]).astype("int32")
        self.align_corners = True
        self.data_layout = "NHWC"


class TestNearestInterpOpUint8(OpTest):

    def setUp(self):
        self.python_api = nearest_interp_test
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "nearest_interp_v2"
        input_np = np.random.randint(low=0, high=256,
                                     size=self.input_shape).astype("uint8")

        if self.scale:
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

        output_np = nearest_neighbor_interp_np(input_np, out_h, out_w, 0, 0,
                                               self.out_size, self.actual_shape,
                                               self.align_corners)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners
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
        self.check_output_with_place(place=core.CPUPlace(),
                                     atol=1,
                                     check_eager=True)

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 3, 9, 6]
        self.out_h = 10
        self.out_w = 9
        self.scale = []
        self.align_corners = True


class TestNearestNeighborInterpCase1Uint8(TestNearestInterpOpUint8):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 80
        self.out_w = 40
        self.scale = []
        self.align_corners = True


class TestNearestNeighborInterpCase2Uint8(TestNearestInterpOpUint8):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 5
        self.out_w = 13
        self.scale = []
        self.out_size = np.array([6, 15]).astype("int32")
        self.align_corners = True


class TestNearestInterpWithoutCorners(TestNearestInterpOp):

    def set_align_corners(self):
        self.align_corners = False


class TestNearestNeighborInterpScale1(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 7, 5]
        self.out_h = 64
        self.out_w = 32
        self.scale = 2.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


class TestNearestNeighborInterpScale2(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 5, 7]
        self.out_h = 64
        self.out_w = 32
        self.scale = 1.5
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


class TestNearestNeighborInterpScale3(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 7, 5]
        self.out_h = 64
        self.out_w = 32
        self.scale = [2.0, 3.0]
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


class TestNearestNeighbor3DInterp(TestNearestInterpOp):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 4, 7, 5]
        self.out_d = 8
        self.out_h = 64
        self.out_w = 32
        self.scale = [4.0, 2.0, 3.0]
        self.out_size = np.array([8, 66, 40]).astype("int32")
        self.align_corners = True


class TestNearestInterpOp_attr_tensor(OpTest):

    def setUp(self):
        self.python_api = nearest_interp_test
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

        input_np = np.random.random(self.input_shape).astype("float64")
        self.inputs = {'X': input_np}

        if self.scale_by_1Dtensor:
            self.inputs['Scale'] = np.array([self.scale]).astype("float64")
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
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True, check_eager=True)

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 5, 4, 4]
        self.out_h = 3
        self.out_w = 3
        self.scale = []
        self.out_size = [3, 3]
        self.align_corners = True


# out_size is a tensor list
class TestNearestInterp_attr_tensor_Case1(TestNearestInterpOp_attr_tensor):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = []
        self.out_size = [8, 12]
        self.align_corners = True


# out_size is a 1-D tensor
class TestNearestInterp_attr_tensor_Case2(TestNearestInterpOp_attr_tensor):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = []
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True
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
        self.align_corners = True
        self.scale_by_1Dtensor = True


class TestNearestAPI(unittest.TestCase):

    def test_case(self):
        x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
        y = fluid.data(name="y", shape=[2, 6, 6, 3], dtype="float32")

        dim = fluid.data(name="dim", shape=[1], dtype="int32")
        shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
        actual_size = fluid.data(name="actual_size", shape=[2], dtype="int32")
        scale_tensor = fluid.data(name="scale_tensor",
                                  shape=[1],
                                  dtype="float32")

        out1 = fluid.layers.resize_nearest(y,
                                           out_shape=[12, 12],
                                           data_format='NHWC')
        out2 = fluid.layers.resize_nearest(x, out_shape=[12, dim])
        out3 = fluid.layers.resize_nearest(x, out_shape=shape_tensor)
        out4 = fluid.layers.resize_nearest(x,
                                           out_shape=[4, 4],
                                           actual_shape=actual_size)
        out5 = fluid.layers.resize_nearest(x, scale=scale_tensor)

        x_data = np.random.random((2, 3, 6, 6)).astype("float32")
        dim_data = np.array([12]).astype("int32")
        shape_data = np.array([12, 12]).astype("int32")
        actual_size_data = np.array([12, 12]).astype("int32")
        scale_data = np.array([2.0]).astype("float32")

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        results = exe.run(fluid.default_main_program(),
                          feed={
                              "x": x_data,
                              "y": np.transpose(x_data, (0, 2, 3, 1)),
                              "dim": dim_data,
                              "shape_tensor": shape_data,
                              "actual_size": actual_size_data,
                              "scale_tensor": scale_data
                          },
                          fetch_list=[out1, out2, out3, out4, out5],
                          return_numpy=True)

        expect_res = nearest_neighbor_interp_np(x_data,
                                                out_h=12,
                                                out_w=12,
                                                align_corners=True)
        np.testing.assert_allclose(results[0],
                                   np.transpose(expect_res, (0, 2, 3, 1)),
                                   rtol=1e-05)
        for i in range(len(results) - 1):
            np.testing.assert_allclose(results[i + 1], expect_res, rtol=1e-05)


class TestNearestInterpOpAPI_dy(unittest.TestCase):

    def test_case(self):
        import paddle
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with fluid.dygraph.guard(place):
            input_data = np.random.random((2, 3, 6, 6)).astype("int64")
            scale_np = np.array([2, 2]).astype("int64")
            input_x = paddle.to_tensor(input_data)
            scale = paddle.to_tensor(scale_np)
            expect_res = nearest_neighbor_interp_np(input_data,
                                                    out_h=12,
                                                    out_w=12,
                                                    align_corners=False)
            out = interpolate(x=input_x,
                              scale_factor=scale,
                              mode="nearest",
                              align_corners=False)
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)


class TestNearestInterp3DOpAPI_dy(unittest.TestCase):

    def test_case(self):
        import paddle
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with fluid.dygraph.guard(place):
            input_data = np.random.random((2, 2, 6, 6, 6)).astype("int64")
            scale_np = np.array([2, 2, 2]).astype("int64")
            input_x = paddle.to_tensor(input_data)
            scale = paddle.to_tensor(scale_np)
            expect_res = nearest_neighbor_interp3d_np(input_data,
                                                      out_d=12,
                                                      out_h=12,
                                                      out_w=12,
                                                      align_corners=False)
            out = interpolate(x=input_x,
                              scale_factor=scale,
                              mode="nearest",
                              align_corners=False,
                              data_format="NCDHW")
            np.testing.assert_allclose(out.numpy(), expect_res, rtol=1e-05)


class TestNearestInterpException(unittest.TestCase):

    def test_exception(self):
        import paddle
        input = fluid.data(name="input", shape=[1, 3, 6, 6], dtype="float32")

        def attr_data_format():
            # for 4-D input, data_format can only be NCHW or NHWC
            out = fluid.layers.resize_nearest(input,
                                              out_shape=[4, 8],
                                              data_format='NDHWC')

        def attr_scale_type():
            out = fluid.layers.resize_nearest(input, scale='scale')

        def attr_scale_value():
            out = fluid.layers.resize_nearest(input, scale=-0.3)

        def input_shape_error():
            x = paddle.randn([1, 3])
            out = paddle.nn.functional.interpolate(x, scale_factor='scale')

        def mode_error():
            x = paddle.randn([1, 3])
            out = paddle.nn.functional.interpolate(x,
                                                   scale_factor='scale',
                                                   mode="BILINEAR")

        self.assertRaises(ValueError, attr_data_format)
        self.assertRaises(TypeError, attr_scale_type)
        self.assertRaises(ValueError, attr_scale_value)
        self.assertRaises(ValueError, input_shape_error)
        self.assertRaises(ValueError, mode_error)


@unittest.skipIf(not fluid.core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestNearestInterp3DOpForFloat16(unittest.TestCase):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 2, 6, 6, 6]
        self.scale = [2, 2, 2]
        self.align_corners = False
        self.data_layout = 'NCDHW'

    def check_main(self, x_np, dtype):
        paddle.disable_static()
        x_np = x_np.astype(dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        y = interpolate(x,
                        scale_factor=self.scale,
                        mode=self.interp_method,
                        align_corners=self.align_corners,
                        data_format=self.data_layout)
        x_g = paddle.grad(y, x)
        y_np = y[0].numpy().astype('float32')
        x_g_np = x_g[0].numpy().astype('float32')
        paddle.enable_static()
        return y_np, x_g_np

    def test_main(self):
        self.init_test_case()
        x_np = np.random.random(self.input_shape).astype("float16")

        y_np_1, x_g_np_1 = self.check_main(x_np, 'float16')
        y_np_2, x_g_np_2 = self.check_main(x_np, 'float32')
        # forward
        np.testing.assert_allclose(y_np_1, y_np_2, rtol=1e-03)
        # backward
        np.testing.assert_allclose(x_g_np_1, x_g_np_2)


@unittest.skipIf(not fluid.core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestNearestInterpOpForFloat16(unittest.TestCase):

    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 2, 6, 6]
        self.scale = [2, 2]
        self.align_corners = False

    def check_main(self, x_np, dtype):
        paddle.disable_static()
        x_np = x_np.astype(dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
        y = interpolate(x,
                        scale_factor=self.scale,
                        mode=self.interp_method,
                        align_corners=self.align_corners)
        x_g = paddle.grad(y, x)
        y_np = y[0].numpy().astype('float32')
        x_g_np = x_g[0].numpy().astype('float32')
        paddle.enable_static()
        return y_np, x_g_np

    def test_main(self):
        self.init_test_case()
        x_np = np.random.random(self.input_shape).astype("float16")

        y_np_1, x_g_np_1 = self.check_main(x_np, 'float16')
        y_np_2, x_g_np_2 = self.check_main(x_np, 'float32')
        # forward
        np.testing.assert_allclose(y_np_1, y_np_2)
        # backward
        np.testing.assert_allclose(x_g_np_1, x_g_np_2)


if __name__ == "__main__":
    unittest.main()
