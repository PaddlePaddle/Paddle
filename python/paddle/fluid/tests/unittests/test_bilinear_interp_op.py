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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid.core as core
=======
from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

paddle.enable_static()


<<<<<<< HEAD
def bilinear_interp_np(
    input,
    out_h,
    out_w,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    align_mode=0,
    data_layout='NCHW',
):
=======
def bilinear_interp_np(input,
                       out_h,
                       out_w,
                       out_size=None,
                       actual_shape=None,
                       align_corners=True,
                       align_mode=0,
                       data_layout='NCHW'):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    """bilinear interpolation implement in shape [N, C, H, W]"""
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 3, 1, 2))  # NHWC => NCHW
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape

    ratio_h = ratio_w = 0.0
    if out_h > 1:
<<<<<<< HEAD
        if align_corners:
=======
        if (align_corners):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
<<<<<<< HEAD
        if align_corners:
=======
        if (align_corners):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for i in range(out_h):
<<<<<<< HEAD
        if align_mode == 0 and not align_corners:
=======
        if (align_mode == 0 and not align_corners):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            h = int(ratio_h * (i + 0.5) - 0.5)
        else:
            h = int(ratio_h * i)

        h = max(0, h)
        hid = 1 if h < in_h - 1 else 0
<<<<<<< HEAD
        if align_mode == 0 and not align_corners:
=======
        if (align_mode == 0 and not align_corners):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            idx_src_h = max(ratio_h * (i + 0.5) - 0.5, 0)
            h1lambda = idx_src_h - h
        else:
            h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
        for j in range(out_w):
<<<<<<< HEAD
            if align_mode == 0 and not align_corners:
=======
            if (align_mode == 0 and not align_corners):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                w = int(ratio_w * (j + 0.5) - 0.5)
            else:
                w = int(ratio_w * j)
            w = max(0, w)
            wid = 1 if w < in_w - 1 else 0
<<<<<<< HEAD
            if align_mode == 0 and not align_corners:
=======
            if (align_mode == 0 and not align_corners):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
                w1lambda = idx_src_w - w
            else:
                w1lambda = ratio_w * j - w
            w2lambda = 1.0 - w1lambda

<<<<<<< HEAD
            out[:, :, i, j] = h2lambda * (
                w2lambda * input[:, :, h, w]
                + w1lambda * input[:, :, h, w + wid]
            ) + h1lambda * (
                w2lambda * input[:, :, h + hid, w]
                + w1lambda * input[:, :, h + hid, w + wid]
            )
=======
            out[:, :, i, j] = h2lambda*(w2lambda*input[:, :, h, w] +
                                        w1lambda*input[:, :, h, w+wid]) + \
                h1lambda*(w2lambda*input[:, :, h+hid, w] +
                          w1lambda*input[:, :, h+hid, w+wid])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


class TestBilinearInterpOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "bilinear_interp"
        # NOTE(dev): some AsDispensible input is not used under imperative mode.
        # Skip check_eager while found them in Inputs.
        self.check_eager = True
        input_np = np.random.random(self.input_shape).astype("float64")

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

<<<<<<< HEAD
        output_np = bilinear_interp_np(
            input_np,
            out_h,
            out_w,
            self.out_size,
            self.actual_shape,
            self.align_corners,
            self.align_mode,
            self.data_layout,
        )
=======
        output_np = bilinear_interp_np(input_np, out_h, out_w, self.out_size,
                                       self.actual_shape, self.align_corners,
                                       self.align_mode, self.data_layout)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
            self.check_eager = False
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape
            self.check_eager = False

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
<<<<<<< HEAD
            'data_layout': self.data_layout,
=======
            'data_layout': self.data_layout
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=self.check_eager)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ['X'], 'Out', in_place=True, check_eager=self.check_eager
        )
=======
        self.check_grad(['X'],
                        'Out',
                        in_place=True,
                        check_eager=self.check_eager)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 2
        self.out_w = 2
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase1(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase2(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase3(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase4(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = np.array([2, 2]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase5(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = np.array([11, 11]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase6(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = np.array([65, 33]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpSame(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpActualShape(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpDataLayout(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 5, 5, 3]
        self.out_h = 2
        self.out_w = 2
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1
        self.data_layout = "NHWC"


class TestBilinearInterpOpUint8(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "bilinear_interp"
        self.check_eager = True
<<<<<<< HEAD
        input_np = np.random.randint(
            low=0, high=256, size=self.input_shape
        ).astype("uint8")
=======
        input_np = np.random.randint(low=0, high=256,
                                     size=self.input_shape).astype("uint8")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if self.scale > 0:
            out_h = int(self.input_shape[2] * self.scale)
            out_w = int(self.input_shape[3] * self.scale)
        else:
            out_h = self.out_h
            out_w = self.out_w

<<<<<<< HEAD
        output_np = bilinear_interp_np(
            input_np,
            out_h,
            out_w,
            self.out_size,
            self.actual_shape,
            self.align_corners,
            self.align_mode,
        )
=======
        output_np = bilinear_interp_np(input_np, out_h, out_w, self.out_size,
                                       self.actual_shape, self.align_corners,
                                       self.align_mode)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
            self.check_eager = False

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
<<<<<<< HEAD
            'align_mode': self.align_mode,
=======
            'align_mode': self.align_mode
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
<<<<<<< HEAD
        self.check_output_with_place(
            place=core.CPUPlace(), atol=1, check_eager=self.check_eager
        )
=======
        self.check_output_with_place(place=core.CPUPlace(),
                                     atol=1,
                                     check_eager=self.check_eager)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 3, 9, 6]
        self.out_h = 10
        self.out_w = 9
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase1Uint8(TestBilinearInterpOpUint8):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 64
        self.out_w = 32
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase2Uint8(TestBilinearInterpOpUint8):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 5
        self.out_w = 13
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = np.array([6, 15]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpOtherMethod1(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 1


class TestBilinearInterpWithMethod2(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 0


class TestBilinearInterpWithMethod3(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_align_mode(self):
        self.align_corners = True
        self.align_mode = 0


class TestBilinearInterpScale1(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
<<<<<<< HEAD
        self.scale = 2.0
=======
        self.scale = 2.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpScale2(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
<<<<<<< HEAD
        self.scale = 1.0
=======
        self.scale = 1.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpScale3(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.5
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpZero(TestBilinearInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 0.2
        self.align_corners = False
        self.align_mode = 0


class TestBilinearInterpOp_attr_tensor(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "bilinear_interp"
        self.check_eager = True
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False
        self.attrs = {
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
        }

        input_np = np.random.random(self.input_shape).astype("float64")
        self.inputs = {'X': input_np}

        if self.scale_by_1Dtensor:
            self.inputs['Scale'] = np.array([self.scale]).astype("float32")
        elif self.scale > 0:
            out_h = int(self.input_shape[2] * self.scale)
            out_w = int(self.input_shape[3] * self.scale)
            self.attrs['scale'] = self.scale
        else:
            out_h = self.out_h
            out_w = self.out_w

        if self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.out_size
            self.check_eager = False
        elif self.out_size is not None:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
<<<<<<< HEAD
                size_tensor.append(
                    ("x" + str(index), np.ones((1)).astype('int32') * ele)
                )
=======
                size_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.inputs['SizeTensor'] = size_tensor
            self.check_eager = False

        self.attrs['out_h'] = self.out_h
        self.attrs['out_w'] = self.out_w
<<<<<<< HEAD
        output_np = bilinear_interp_np(
            input_np,
            out_h,
            out_w,
            self.out_size,
            self.actual_shape,
            self.align_corners,
        )
=======
        output_np = bilinear_interp_np(input_np, out_h, out_w, self.out_size,
                                       self.actual_shape, self.align_corners)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=self.check_eager)

    def test_check_grad(self):
<<<<<<< HEAD
        self.check_grad(
            ['X'], 'Out', in_place=True, check_eager=self.check_eager
        )
=======
        self.check_grad(['X'],
                        'Out',
                        in_place=True,
                        check_eager=self.check_eager)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 3
        self.out_w = 3
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = [3, 3]
        self.align_corners = True


# out_size is a 1-D tensor
class TestBilinearInterp_attr_tensor_Case1(TestBilinearInterpOp_attr_tensor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = [8, 12]
        self.align_corners = True


# scale is a 1-D tensor
class TestBilinearInterp_attr_tensor_Case2(TestBilinearInterpOp_attr_tensor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
<<<<<<< HEAD
        self.scale = 0.0
=======
        self.scale = 0.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True
        self.shape_by_1Dtensor = True


# scale is a 1-D tensor
class TestBilinearInterp_attr_tensor_Case3(TestBilinearInterpOp_attr_tensor):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 2.0
        self.out_size = None
        self.align_corners = True
        self.scale_by_1Dtensor = True


<<<<<<< HEAD
=======
class TestBilinearInterpOpAPI(unittest.TestCase):

    def test_case(self):
        x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")

        dim = fluid.data(name="dim", shape=[1], dtype="int32")
        shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
        actual_size = fluid.data(name="actual_size", shape=[2], dtype="int32")
        scale_tensor = fluid.data(name="scale_tensor",
                                  shape=[1],
                                  dtype="float32")

        out1 = fluid.layers.resize_bilinear(x, out_shape=[12, 12])
        out2 = fluid.layers.resize_bilinear(x, out_shape=[12, dim])
        out3 = fluid.layers.resize_bilinear(x, out_shape=shape_tensor)
        out4 = fluid.layers.resize_bilinear(x,
                                            out_shape=[4, 4],
                                            actual_shape=actual_size)
        out5 = fluid.layers.resize_bilinear(x, scale=scale_tensor)

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
                              "dim": dim_data,
                              "shape_tensor": shape_data,
                              "actual_size": actual_size_data,
                              "scale_tensor": scale_data
                          },
                          fetch_list=[out1, out2, out3, out4, out5],
                          return_numpy=True)

        expect_res = bilinear_interp_np(x_data,
                                        out_h=12,
                                        out_w=12,
                                        align_corners=True)
        for res in results:
            np.testing.assert_allclose(res, expect_res, rtol=1e-05)


>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
if __name__ == "__main__":
    unittest.main()
