#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid.core as core
import sys
sys.path.append("..")
from op_test_xpu import XPUOpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

paddle.enable_static()
'''
def nearest_neighbor_interp_np(X,
                               out_h,
                               out_w,
                               out_size=None,
                               actual_shape=None,
                               align_corners=True,
                               data_layout="NCHW"):
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
            ratio_h = 1.0 * in_h / out_h
    if (out_w > 1):
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
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

    return out.astype(X.dtype)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestInterpOp(XPUOpTest):
    def setUp(self):
        self.use_xpu = True
        self.out_size = None
        self.actual_shape = None
        self.data_layout = "NCHW"
        self.init_test_case()
        self.op_type = "nearest_interp"
        input_np = np.random.random(self.input_shape).astype("float32")

        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]

        if self.scale > 0:
            out_h = int(in_h * self.scale)
            out_w = int(in_w * self.scale)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = nearest_neighbor_interp_np(
            input_np, out_h, out_w, self.out_size, self.actual_shape,
            self.align_corners, self.data_layout)
        self.inputs = {"X": input_np}
        if self.out_size is not None:
            self.inputs["OutSize"] = self.out_size
        if self.actual_shape is not None:
            self.inputs["OutSize"] = self.actual_shape
        self.attrs = {
            "out_h": self.out_h,
            "out_w": self.out_w,
            "scale": self.scale,
            "interp_method": self.interp_method,
            "align_corners": self.align_corners,
            "data_layout": self.data_layout
        }
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        place = paddle.XPUPlace(0)
        self.check_grad_with_place(place, ["X"], "Out", in_place=True)

    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [2, 3, 4, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpCase1(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpCase2(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpCase3(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpCase4(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpCase5(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11]).astype("int32")
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpCase6(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 129]).astype("int32")
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpSame(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpActualShape(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpDataLayout(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [2, 4, 4, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 8]).astype("int32")
        self.align_corners = True
        self.data_layout = "NCHW"


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestInterpWithoutCorners(TestNearestInterpOp):
    def set_align_corners(self):
        self.align_corners = False


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpScale1(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 2, 7, 5]
        self.out_h = 64
        self.out_w = 32
        self.scale = 2.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpScale2(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 2, 5, 7]
        self.out_h = 64
        self.out_w = 32
        self.scale = 1.5
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestNeighborInterpScale3(TestNearestInterpOp):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 2, 7, 5]
        self.out_h = 64
        self.out_w = 32
        self.scale = 1.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestInterpOp_attr_tensor(XPUOpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "nearest_interp"
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False
        self.attrs = {
            "interp_method": self.interp_method,
            "align_corners": self.align_corners,
        }

        input_np = np.random.random(self.input_shape).astype("float32")
        self.inputs = {"X": input_np}

        if self.scale_by_1Dtensor:
            self.inputs["Scale"] = np.array([self.scale]).astype("float32")
        elif self.scale > 0:
            out_h = int(self.input_shape[2] * self.scale)
            out_w = int(self.input_shape[3] * self.scale)
            self.attrs["scale"] = self.scale
        else:
            out_h = self.out_h
            out_w = self.out_w

        if self.shape_by_1Dtensor:
            self.inputs["OutSize"] = self.out_size
        elif self.out_size is not None:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(("x" + str(index), np.ones(
                    (1)).astype("int32") * ele))
            self.inputs["SizeTensor"] = size_tensor

        self.attrs["out_h"] = self.out_h
        self.attrs["out_w"] = self.out_w
        output_np = nearest_neighbor_interp_np(input_np, out_h, out_w,
                                               self.out_size, self.actual_shape,
                                               self.align_corners)
        self.outputs = {"Out": output_np}

    def test_check_output(self):
        place = paddle.XPUPlace(0)
        self.check_output_with_place(place)

    def test_check_grad(self):
        place = paddle.XPUPlace(0)
        self.check_grad_with_place(place, ["X"], "Out", in_place=True)

    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [2, 5, 4, 4]
        self.out_h = 3
        self.out_w = 3
        self.scale = 0.
        self.out_size = [3, 3]
        self.align_corners = True


# out_size is a tensor list
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestInterp_attr_tensor_Case1(TestNearestInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = [8, 12]
        self.align_corners = True


# out_size is a 1-D tensor
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestInterp_attr_tensor_Case2(TestNearestInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True
        self.shape_by_1Dtensor = True


# scale is a 1-D tensor
@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestInterp_attr_tensor_Case3(TestNearestInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = "nearest"
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 2.0
        self.out_size = None
        self.align_corners = True
        self.scale_by_1Dtensor = True


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestNearestInterpException(unittest.TestCase):
    def test_exception(self):
        input = fluid.data(name="input", shape=[1, 3, 6, 6], dtype="float32")

        def attr_data_format():
            # for 4-D input, data_format can only be NCHW or NHWC
            out = fluid.layers.resize_nearest(
                input, out_shape=[4, 8], data_format="NDHWC")

        def attr_scale_type():
            out = fluid.layers.resize_nearest(input, scale="scale")

        def attr_scale_value():
            out = fluid.layers.resize_nearest(input, scale=-0.3)

        self.assertRaises(ValueError, attr_data_format)
        self.assertRaises(TypeError, attr_scale_type)
        self.assertRaises(ValueError, attr_scale_value)
'''

if __name__ == "__main__":
    unittest.main()
