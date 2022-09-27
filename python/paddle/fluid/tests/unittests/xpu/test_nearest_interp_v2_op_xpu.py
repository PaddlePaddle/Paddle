#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")

import paddle

from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


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


class XPUNearestInterpOpWrapper(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'nearest_interp_v2'
        self.use_dynamic_create_class = False

    class TestNearestInterpOp(XPUOpTest):

        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()

            self.out_size = None
            self.actual_shape = None
            self.data_layout = 'NCHW'

            self.interp_method = 'nearest'
            self.scale = 0.
            self.align_corners = True

            self.init_test_case()
            self.op_type = "nearest_interp_v2"
            input_np = np.random.random(self.input_shape).astype(self.dtype)

            # in
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

            # scale
            scale_d = 0
            scale_h = 0
            scale_w = 0
            if self.scale:
                if isinstance(self.scale, float) or isinstance(self.scale, int):
                    if self.scale > 0:
                        scale_d = scale_h = scale_w = float(self.scale)
                        self.scale = [self.scale]
                if isinstance(self.scale, list) and len(self.scale) == 1:
                    scale_d = scale_w = scale_h = self.scale[0]
                    self.scale = [self.scale[0], self.scale[0]]
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

            # output_np
            if len(self.input_shape) == 4:
                output_np = nearest_neighbor_interp_np(
                    input_np, out_h, out_w, scale_h, scale_w, self.out_size,
                    self.actual_shape, self.align_corners, self.data_layout)
            elif len(self.input_shape) == 5:
                output_np = nearest_neighbor_interp3d_np(
                    input_np, out_d, out_h, out_w, scale_d, scale_h, scale_w,
                    self.out_size, self.actual_shape, self.align_corners,
                    self.data_layout)
            self.outputs = {'Out': output_np}

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
                self.attrs['scale'] = self.scale

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out', in_place=True)

        def init_test_case(self):
            self.input_shape = [2, 3, 4, 5]
            self.out_h = 2
            self.out_w = 2
            self.out_size = np.array([3, 3]).astype("int32")

    """
    # case copied form gpu but disabled in xpu: not support 5-dim input_shape
    class TestNearestNeighborInterpCase1(TestNearestInterpOp):
        def init_test_case(self):
            self.interp_method = 'nearest'
            self.input_shape = [4, 1, 1, 7, 8]
            self.out_d = 1
            self.out_h = 1
            self.out_w = 1
            self.scale = 0.
            self.align_corners = True
    """

    class TestNearestNeighborInterpCase2(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [3, 3, 9, 6]
            self.out_h = 12
            self.out_w = 12

    class TestNearestNeighborInterpCase3(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [1, 1, 32, 64]
            self.out_h = 64
            self.out_w = 32

    class TestNearestNeighborInterpCase4(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [4, 1, 7, 8]
            self.out_h = 1
            self.out_w = 1
            self.out_size = np.array([2, 2]).astype("int32")

    class TestNearestNeighborInterpCase5(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [3, 3, 9, 6]
            self.out_h = 12
            self.out_w = 12
            self.out_size = np.array([11, 11]).astype("int32")

    class TestNearestNeighborInterpCase6(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [1, 1, 32, 64]
            self.out_h = 64
            self.out_w = 32
            self.out_size = np.array([65, 129]).astype("int32")

    class TestNearestNeighborInterpSame(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [2, 3, 32, 64]
            self.out_h = 32
            self.out_w = 64

    class TestNearestNeighborInterpActualShape(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [3, 2, 32, 16]
            self.out_h = 64
            self.out_w = 32
            self.out_size = np.array([66, 40]).astype("int32")

    """
    # case copied form gpu but disabled in xpu: not support NHWC data_layout
    class TestNearestNeighborInterpDataLayout(TestNearestInterpOp):
        def init_test_case(self):
            self.interp_method = 'nearest'
            self.input_shape = [2, 4, 4, 5]
            self.out_h = 2
            self.out_w = 2
            self.scale = 0.
            self.out_size = np.array([3, 8]).astype("int32")
            self.align_corners = True
            self.data_layout = "NHWC"
    """

    class TestNearestInterpWithoutCorners(TestNearestInterpOp):

        def set_align_corners(self):
            self.align_corners = False

    class TestNearestNeighborInterpScale1(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [3, 2, 7, 5]
            self.out_h = 64
            self.out_w = 32
            self.scale = 2.
            self.out_size = np.array([66, 40]).astype("int32")

    class TestNearestNeighborInterpScale2(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [3, 2, 5, 7]
            self.out_h = 64
            self.out_w = 32
            self.scale = 1.5
            self.out_size = np.array([66, 40]).astype("int32")

    class TestNearestNeighborInterpScale3(TestNearestInterpOp):

        def init_test_case(self):
            self.input_shape = [3, 2, 7, 5]
            self.out_h = 64
            self.out_w = 32
            self.scale = [2.0, 3.0]
            self.out_size = np.array([66, 40]).astype("int32")

    """
    # case copied form gpu but disabled in xpu: not support 5-dim input_shape
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
    """

    class TestNearestInterpOp_attr_tensor(XPUOpTest):

        def setUp(self):
            self.place = paddle.XPUPlace(0)
            self.init_dtype()

            self.out_size = None
            self.actual_shape = None

            self.interp_method = 'nearest'
            self.scale = 0.
            self.align_corners = True

            self.init_test_case()
            self.op_type = "nearest_interp_v2"
            self.shape_by_1Dtensor = False
            self.scale_by_1Dtensor = False
            self.attrs = {
                'interp_method': self.interp_method,
                'align_corners': self.align_corners,
            }

            input_np = np.random.random(self.input_shape).astype(self.dtype)
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
                                                   self.out_size,
                                                   self.actual_shape,
                                                   self.align_corners)
            self.outputs = {'Out': output_np}

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(self.place, ['X'], 'Out', in_place=True)

        def init_test_case(self):
            self.input_shape = [2, 5, 4, 4]
            self.out_h = 3
            self.out_w = 3
            self.out_size = [3, 3]

    # out_size is a tensor list
    class TestNearestInterp_attr_tensor_Case1(TestNearestInterpOp_attr_tensor):

        def init_test_case(self):
            self.input_shape = [3, 3, 9, 6]
            self.out_h = 12
            self.out_w = 12
            self.out_size = [8, 12]

    # out_size is a 1-D tensor
    class TestNearestInterp_attr_tensor_Case2(TestNearestInterpOp_attr_tensor):

        def init_test_case(self):
            self.input_shape = [3, 2, 32, 16]
            self.out_h = 64
            self.out_w = 32
            self.out_size = np.array([66, 40]).astype("int32")
            self.shape_by_1Dtensor = True

    # scale is a 1-D tensor
    class TestNearestInterp_attr_tensor_Case3(TestNearestInterpOp_attr_tensor):

        def init_test_case(self):
            self.input_shape = [3, 2, 32, 16]
            self.out_h = 64
            self.out_w = 32
            self.scale = 2.0
            self.out_size = None
            self.scale_by_1Dtensor = True


support_types = get_xpu_op_support_types('nearest_interp_v2')
for stype in support_types:
    create_test_class(globals(), XPUNearestInterpOpWrapper, stype)

if __name__ == "__main__":
    unittest.main()
