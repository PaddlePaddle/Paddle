#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.nn.functional import interpolate


def bicubic_interp_test(
    x,
    OutSize=None,
    SizeTensor=None,
    Scale=None,
    data_layout='kNCHW',
    out_d=-1,
    out_h=-1,
    out_w=-1,
    scale=[],
    interp_method='bicubic',
    align_corners=True,
    align_mode=0,
):
=======
from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest, _in_eager_without_dygraph_check
import paddle.fluid.core as core
import paddle.fluid as fluid
import paddle
from paddle.fluid import Program, program_guard
from paddle.fluid.framework import _test_eager_guard
from paddle.nn.functional import interpolate


def bicubic_interp_test(x,
                        OutSize=None,
                        SizeTensor=None,
                        Scale=None,
                        data_layout='kNCHW',
                        out_d=-1,
                        out_h=-1,
                        out_w=-1,
                        scale=[],
                        interp_method='bicubic',
                        align_corners=True,
                        align_mode=0):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if isinstance(scale, float) or isinstance(scale, int):
        scale_list = []
        for _ in range(len(x.shape) - 2):
            scale_list.append(scale)
        scale = list(map(float, scale_list))
    elif isinstance(scale, list) or isinstance(scale, tuple):
        scale = list(map(float, scale))
    if SizeTensor is not None:
        if not isinstance(SizeTensor, list) and not isinstance(
<<<<<<< HEAD
            SizeTensor, tuple
        ):
            SizeTensor = [SizeTensor]
    return paddle._C_ops.bicubic_interp(
        x,
        OutSize,
        SizeTensor,
        Scale,
        data_layout,
        out_d,
        out_h,
        out_w,
        scale,
        interp_method,
        align_corners,
        align_mode,
    )
=======
                SizeTensor, tuple):
            SizeTensor = [SizeTensor]
    return paddle._C_ops.bicubic_interp(x, OutSize, SizeTensor, Scale,
                                        data_layout, out_d, out_h, out_w, scale,
                                        interp_method, align_corners,
                                        align_mode)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def cubic_1(x, a):
    return ((a + 2) * x - (a + 3)) * x * x + 1


def cubic_2(x, a):
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def cubic_interp1d(x0, x1, x2, x3, t):
    param = [0, 0, 0, 0]
    a = -0.75
    x_1 = t
    x_2 = 1.0 - t
    param[0] = cubic_2(x_1 + 1.0, a)
    param[1] = cubic_1(x_1, a)
    param[2] = cubic_1(x_2, a)
    param[3] = cubic_2(x_2 + 1.0, a)
    return x0 * param[0] + x1 * param[1] + x2 * param[2] + x3 * param[3]


def value_bound(input, w, h, x, y):
    access_x = int(max(min(x, w - 1), 0))
    access_y = int(max(min(y, h - 1), 0))
    return input[:, :, access_y, access_x]


<<<<<<< HEAD
def bicubic_interp_np(
    input,
    out_h,
    out_w,
    scale_h=0,
    scale_w=0,
    out_size=None,
    actual_shape=None,
    align_corners=True,
    data_layout='kNCHW',
):
=======
def bicubic_interp_np(input,
                      out_h,
                      out_w,
                      scale_h=0,
                      scale_w=0,
                      out_size=None,
                      actual_shape=None,
                      align_corners=True,
                      data_layout='kNCHW'):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    """trilinear interpolation implement in shape [N, C, H, W]"""
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
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
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
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for k in range(out_h):
<<<<<<< HEAD
        if align_corners:
=======
        if (align_corners):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            h = ratio_h * k
        else:
            h = ratio_h * (k + 0.5) - 0.5
        input_y = np.floor(h)
        y_t = h - input_y
        for l in range(out_w):
<<<<<<< HEAD
            if align_corners:
=======
            if (align_corners):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                w = ratio_w * l
            else:
                w = ratio_w * (l + 0.5) - 0.5
            input_x = np.floor(w)
            x_t = w - input_x
            for i in range(batch_size):
                for j in range(channel):
                    coefficients = [0, 0, 0, 0]
                    for ii in range(4):
                        access_x_0 = int(max(min(input_x - 1, in_w - 1), 0))
                        access_x_1 = int(max(min(input_x + 0, in_w - 1), 0))
                        access_x_2 = int(max(min(input_x + 1, in_w - 1), 0))
                        access_x_3 = int(max(min(input_x + 2, in_w - 1), 0))
                        access_y = int(max(min(input_y - 1 + ii, in_h - 1), 0))

                        coefficients[ii] = cubic_interp1d(
                            input[i, j, access_y, access_x_0],
                            input[i, j, access_y, access_x_1],
<<<<<<< HEAD
                            input[i, j, access_y, access_x_2],
                            input[i, j, access_y, access_x_3],
                            x_t,
                        )
                    out[i, j, k, l] = cubic_interp1d(
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                        coefficients[3],
                        y_t,
                    )
=======
                            input[i, j, access_y,
                                  access_x_2], input[i, j, access_y,
                                                     access_x_3], x_t)
                    out[i, j, k,
                        l] = cubic_interp1d(coefficients[0], coefficients[1],
                                            coefficients[2], coefficients[3],
                                            y_t)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC
    return out.astype(input.dtype)


class TestBicubicInterpOp(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.python_api = bicubic_interp_test
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "bicubic_interp_v2"
        # NOTE(dev): some AsDispensible input is not used under imperative mode.
        # Skip check_eager while found them in Inputs.
        self.check_eager = True
        input_np = np.random.random(self.input_shape).astype("float64")
        scale_h = 0
        scale_w = 0
        if self.data_layout == "NCHW":
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]
        else:
            in_h = self.input_shape[1]
            in_w = self.input_shape[2]

        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
<<<<<<< HEAD
                if self.scale > 0.0:
=======
                if self.scale > 0.:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    scale_h = scale_w = float(self.scale)
            if isinstance(self.scale, list) and len(self.scale) == 1:
                scale_w = scale_h = self.scale[0]
            elif isinstance(self.scale, list) and len(self.scale) > 1:
                scale_w = self.scale[1]
                scale_h = self.scale[0]
            out_h = int(in_h * scale_h)
            out_w = int(in_w * scale_w)
        else:
            out_h = self.out_h
            out_w = self.out_w

<<<<<<< HEAD
        output_np = bicubic_interp_np(
            input_np,
            out_h,
            out_w,
            scale_h,
            scale_w,
            self.out_size,
            self.actual_shape,
            self.align_corners,
            self.data_layout,
        )
=======
        output_np = bicubic_interp_np(input_np, out_h, out_w, scale_h, scale_w,
                                      self.out_size, self.actual_shape,
                                      self.align_corners, self.data_layout)
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
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
<<<<<<< HEAD
            'data_layout': self.data_layout,
        }
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.0:
=======
            'data_layout': self.data_layout
        }
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
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
        self.interp_method = 'bicubic'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = []
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True


class TestBicubicInterpCase1(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = []
        self.align_corners = True


class TestBicubicInterpCase2(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 10
        self.out_w = 8
        self.scale = []
        self.align_corners = True


class TestBicubicInterpCase3(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = []
        self.align_corners = False


class TestBicubicInterpCase4(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = []
        self.out_size = np.array([2, 2]).astype("int32")
        self.align_corners = True


class TestBicubicInterpCase5(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 11
        self.out_w = 11
        self.scale = []
        self.out_size = np.array([6, 4]).astype("int32")
        self.align_corners = False


class TestBicubicInterpCase6(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0
        self.out_size = np.array([64, 32]).astype("int32")
        self.align_corners = False


class TestBicubicInterpSame(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = []
        self.align_corners = True


class TestBicubicInterpScale(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
<<<<<<< HEAD
        self.scale = [1.0, 1.0]
=======
        self.scale = [1., 1.]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.align_corners = True


class TestBicubicInterpDataLayout(TestBicubicInterpOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [2, 5, 5, 3]
        self.out_h = 2
        self.out_w = 2
        self.scale = []
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.data_layout = "NHWC"


class TestBicubicInterpOpAPI(unittest.TestCase):
<<<<<<< HEAD
    def test_case(self):
=======

    def test_imperative_case(self):
        with _test_eager_guard():
            self.func_case()
        self.func_case()

    def func_case(self):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        np.random.seed(200)
        x_data = np.random.random((2, 3, 6, 6)).astype("float32")
        dim_data = np.array([12]).astype("int32")
        shape_data = np.array([12, 12]).astype("int32")
        actual_size_data = np.array([12, 12]).astype("int32")
        scale_data = np.array([2.0]).astype("float32")

        prog = fluid.Program()
        startup_prog = fluid.Program()
<<<<<<< HEAD
        place = (
            fluid.CUDAPlace(0)
            if fluid.core.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
=======
        place = fluid.CUDAPlace(
            0) if fluid.core.is_compiled_with_cuda() else fluid.CPUPlace()
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        with fluid.program_guard(prog, startup_prog):

            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")

            dim = fluid.data(name="dim", shape=[1], dtype="int32")
<<<<<<< HEAD
            shape_tensor = fluid.data(
                name="shape_tensor", shape=[2], dtype="int32"
            )
            actual_size = fluid.data(
                name="actual_size", shape=[2], dtype="int32"
            )
            scale_tensor = fluid.data(
                name="scale_tensor", shape=[1], dtype="float32"
            )

            out1 = interpolate(
                x, size=[12, 12], mode='bicubic', align_corners=False
            )
            out2 = interpolate(
                x, size=[12, dim], mode='bicubic', align_corners=False
            )
            out3 = interpolate(
                x, size=shape_tensor, mode='bicubic', align_corners=False
            )
            out4 = interpolate(
                x, size=[12, 12], mode='bicubic', align_corners=False
            )
            out5 = interpolate(
                x,
                scale_factor=scale_tensor,
                mode='bicubic',
                align_corners=False,
            )
            out6 = interpolate(
                x, scale_factor=2.0, mode='bicubic', align_corners=False
            )
            out7 = interpolate(
                x, scale_factor=[2.0, 2.0], mode='bicubic', align_corners=False
            )
=======
            shape_tensor = fluid.data(name="shape_tensor",
                                      shape=[2],
                                      dtype="int32")
            actual_size = fluid.data(name="actual_size",
                                     shape=[2],
                                     dtype="int32")
            scale_tensor = fluid.data(name="scale_tensor",
                                      shape=[1],
                                      dtype="float32")

            out1 = interpolate(x,
                               size=[12, 12],
                               mode='bicubic',
                               align_corners=False)
            out2 = interpolate(x,
                               size=[12, dim],
                               mode='bicubic',
                               align_corners=False)
            out3 = interpolate(x,
                               size=shape_tensor,
                               mode='bicubic',
                               align_corners=False)
            out4 = interpolate(x,
                               size=[12, 12],
                               mode='bicubic',
                               align_corners=False)
            out5 = interpolate(x,
                               scale_factor=scale_tensor,
                               mode='bicubic',
                               align_corners=False)
            out6 = interpolate(x,
                               scale_factor=2.0,
                               mode='bicubic',
                               align_corners=False)
            out7 = interpolate(x,
                               scale_factor=[2.0, 2.0],
                               mode='bicubic',
                               align_corners=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            results = exe.run(
                fluid.default_main_program(),
                feed={
                    "x": x_data,
                    "dim": dim_data,
                    "shape_tensor": shape_data,
                    "actual_size": actual_size_data,
<<<<<<< HEAD
                    "scale_tensor": scale_data,
                },
                fetch_list=[out1, out2, out3, out4, out5, out6, out7],
                return_numpy=True,
            )

            expect_res = bicubic_interp_np(
                x_data, out_h=12, out_w=12, align_corners=False
            )
=======
                    "scale_tensor": scale_data
                },
                fetch_list=[out1, out2, out3, out4, out5, out6, out7],
                return_numpy=True)

            expect_res = bicubic_interp_np(x_data,
                                           out_h=12,
                                           out_w=12,
                                           align_corners=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            for res in results:
                np.testing.assert_allclose(res, expect_res, rtol=1e-05)

        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(x_data)
<<<<<<< HEAD
            interp = interpolate(
                x, size=[12, 12], mode='bicubic', align_corners=False
            )
            dy_result = interp.numpy()
            expect = bicubic_interp_np(
                x_data, out_h=12, out_w=12, align_corners=False
            )
=======
            interp = interpolate(x,
                                 size=[12, 12],
                                 mode='bicubic',
                                 align_corners=False)
            dy_result = interp.numpy()
            expect = bicubic_interp_np(x_data,
                                       out_h=12,
                                       out_w=12,
                                       align_corners=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            np.testing.assert_allclose(dy_result, expect, rtol=1e-05)


class TestBicubicOpError(unittest.TestCase):
<<<<<<< HEAD
    def test_imperative_errors(self):
        # the input of interpoalte must be Variable.
        x1 = fluid.create_lod_tensor(
            np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace()
        )
=======

    def test_imperative_errors(self):
        # the input of interpoalte must be Variable.
        x1 = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]],
                                     fluid.CPUPlace())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertRaises(TypeError, interpolate, x1)

        def test_mode_type():
            # mode must be "BILINEAR" "TRILINEAR" "NEAREST" "BICUBIC"
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")

<<<<<<< HEAD
            out = interpolate(
                x, size=[12, 12], mode='UNKONWN', align_corners=False
            )

        def test_input_shape():
            x = fluid.data(name="x", shape=[2], dtype="float32")
            out = interpolate(
                x, size=[12, 12], mode='BICUBIC', align_corners=False
            )
=======
            out = interpolate(x,
                              size=[12, 12],
                              mode='UNKONWN',
                              align_corners=False)

        def test_input_shape():
            x = fluid.data(name="x", shape=[2], dtype="float32")
            out = interpolate(x,
                              size=[12, 12],
                              mode='BICUBIC',
                              align_corners=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_align_corcers():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            interpolate(x, size=[12, 12], mode='BICUBIC', align_corners=3)

        def test_out_shape():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(x, size=[12], mode='bicubic', align_corners=False)

        def test_attr_data_format():
            # for 5-D input, data_format only can be NCDHW or NDHWC
<<<<<<< HEAD
            input = fluid.data(
                name="input", shape=[2, 3, 6, 9, 4], dtype="float32"
            )
            out = interpolate(
                input, size=[4, 8, 4, 5], mode='trilinear', data_format='NHWC'
            )

        def test_actual_shape():
            # the actual_shape  must be Variable.
            x = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace()
            )
            out = interpolate(
                x, size=[12, 12], mode='BICUBIC', align_corners=False
            )
=======
            input = fluid.data(name="input",
                               shape=[2, 3, 6, 9, 4],
                               dtype="float32")
            out = interpolate(input,
                              size=[4, 8, 4, 5],
                              mode='trilinear',
                              data_format='NHWC')

        def test_actual_shape():
            # the actual_shape  must be Variable.
            x = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]],
                                        fluid.CPUPlace())
            out = interpolate(x,
                              size=[12, 12],
                              mode='BICUBIC',
                              align_corners=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_scale_value():
            # the scale must be greater than zero.
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
<<<<<<< HEAD
            out = interpolate(
                x,
                size=None,
                mode='BICUBIC',
                align_corners=False,
                scale_factor=-2.0,
            )

        def test_attr_5D_input():
            # for 5-D input, data_format only can be NCDHW or NDHWC
            input = fluid.data(
                name="input", shape=[2, 3, 6, 9, 4], dtype="float32"
            )
            out = interpolate(
                input, size=[4, 8, 4, 5], mode='trilinear', data_format='NDHWC'
            )
=======
            out = interpolate(x,
                              size=None,
                              mode='BICUBIC',
                              align_corners=False,
                              scale_factor=-2.0)

        def test_attr_5D_input():
            # for 5-D input, data_format only can be NCDHW or NDHWC
            input = fluid.data(name="input",
                               shape=[2, 3, 6, 9, 4],
                               dtype="float32")
            out = interpolate(input,
                              size=[4, 8, 4, 5],
                              mode='trilinear',
                              data_format='NDHWC')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        def test_scale_type():
            # the scale must be greater than zero.
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
<<<<<<< HEAD
            scale = fluid.create_lod_tensor(
                np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace()
            )
            out = interpolate(
                x,
                size=None,
                mode='bicubic',
                align_corners=False,
                scale_factor=scale,
            )

        def test_align_mode():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(
                x,
                size=None,
                mode='nearest',
                align_corners=False,
                align_mode=2,
                scale_factor=1.0,
            )

        def test_outshape_and_scale():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(
                x,
                size=None,
                mode='bicubic',
                align_corners=False,
                scale_factor=None,
            )

        def test_align_corners_and_nearest():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(
                x,
                size=None,
                mode='nearest',
                align_corners=True,
                scale_factor=None,
            )

        def test_scale_shape():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(
                x,
                size=None,
                mode='nearest',
                align_corners=False,
                scale_factor=[1, 2, 2],
            )

        def test_scale_value_1():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(
                x,
                size=None,
                mode='bicubic',
                align_corners=False,
                scale_factor=[1, 2, 2],
            )

        def test_size_and_scale():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(
                x,
                size=None,
                mode='bicubic',
                align_corners=False,
                scale_factor=None,
            )

        def test_size_and_scale2():
            x = fluid.data(name="input", shape=[2, 3, 6, 9, 4], dtype="float32")
            out = interpolate(
                x,
                size=[2, 2, 2],
                mode='trilinear',
                align_corners=False,
                scale_factor=2.0,
            )

        def test_size_type():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(
                x, size={2, 2}, mode='bicubic', align_corners=False
            )

        def test_input_shape_1():
            x = fluid.data(name="x", shape=[2, 1, 0, 0], dtype="float32")
            out = interpolate(
                x, size=[3, 3], mode="bicubic", align_corners=False
            )
=======
            scale = fluid.create_lod_tensor(np.array([-1, 3, 5, 5]),
                                            [[1, 1, 1, 1]], fluid.CPUPlace())
            out = interpolate(x,
                              size=None,
                              mode='bicubic',
                              align_corners=False,
                              scale_factor=scale)

        def test_align_mode():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(x,
                              size=None,
                              mode='nearest',
                              align_corners=False,
                              align_mode=2,
                              scale_factor=1.0)

        def test_outshape_and_scale():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(x,
                              size=None,
                              mode='bicubic',
                              align_corners=False,
                              scale_factor=None)

        def test_align_corners_and_nearest():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(x,
                              size=None,
                              mode='nearest',
                              align_corners=True,
                              scale_factor=None)

        def test_scale_shape():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(x,
                              size=None,
                              mode='nearest',
                              align_corners=False,
                              scale_factor=[1, 2, 2])

        def test_scale_value_1():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(x,
                              size=None,
                              mode='bicubic',
                              align_corners=False,
                              scale_factor=[1, 2, 2])

        def test_size_and_scale():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(x,
                              size=None,
                              mode='bicubic',
                              align_corners=False,
                              scale_factor=None)

        def test_size_and_scale2():
            x = fluid.data(name="input", shape=[2, 3, 6, 9, 4], dtype="float32")
            out = interpolate(x,
                              size=[2, 2, 2],
                              mode='trilinear',
                              align_corners=False,
                              scale_factor=2.0)

        def test_size_type():
            x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")
            out = interpolate(x,
                              size={2, 2},
                              mode='bicubic',
                              align_corners=False)

        def test_input_shape_1():
            x = fluid.data(name="x", shape=[2, 1, 0, 0], dtype="float32")
            out = interpolate(x,
                              size=[3, 3],
                              mode="bicubic",
                              align_corners=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.assertRaises(ValueError, test_mode_type)
        self.assertRaises(ValueError, test_input_shape)
        self.assertRaises(TypeError, test_align_corcers)
        self.assertRaises(ValueError, test_attr_data_format)
        self.assertRaises(TypeError, test_actual_shape)
        self.assertRaises(ValueError, test_scale_value)
        self.assertRaises(ValueError, test_out_shape)
        self.assertRaises(ValueError, test_attr_5D_input)
        self.assertRaises(TypeError, test_scale_type)
        self.assertRaises(ValueError, test_align_mode)
        self.assertRaises(ValueError, test_outshape_and_scale)
        self.assertRaises(ValueError, test_align_corners_and_nearest)
        self.assertRaises(ValueError, test_scale_shape)
        self.assertRaises(ValueError, test_scale_value_1)
        self.assertRaises(ValueError, test_size_and_scale)
        self.assertRaises(ValueError, test_size_and_scale2)
        self.assertRaises(TypeError, test_size_type)
        self.assertRaises(ValueError, test_input_shape_1)

    def test_errors(self):
        with program_guard(Program(), Program()):
            self.test_imperative_errors()


<<<<<<< HEAD
@unittest.skipIf(
    not fluid.core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestBicubicInterpOpForFloat16(unittest.TestCase):
=======
@unittest.skipIf(not fluid.core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestBicubicInterpOpForFloat16(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.interp_method = 'bicubic'
        self.input_shape = [2, 3, 5, 5]
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.data_layout = 'NCHW'

    def check_main(self, x_np, dtype):
        paddle.disable_static()
        x_np = x_np.astype(dtype)
        x = paddle.to_tensor(x_np)
        x.stop_gradient = False
<<<<<<< HEAD
        y = interpolate(
            x,
            size=self.out_size.tolist(),
            mode=self.interp_method,
            align_corners=self.align_corners,
            data_format=self.data_layout,
        )
=======
        y = interpolate(x,
                        size=self.out_size.tolist(),
                        mode=self.interp_method,
                        align_corners=self.align_corners,
                        data_format=self.data_layout)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

        np.testing.assert_allclose(y_np_1, y_np_2)
        np.testing.assert_allclose(x_g_np_1, x_g_np_2)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
