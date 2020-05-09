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
import platform
import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle.nn.functional import *


def linear_interp_np(input,
                     out_w,
                     out_size=None,
                     actual_shape=None,
                     align_corners=True,
                     align_mode=0,
                     data_layout='NCHW'):
    if data_layout == "NHWC":
        input = np.transpose(input, (0, 2, 1))  # NHWC => NCHW
    if out_size is not None:
        out_w = out_size[0]
    if actual_shape is not None:
        out_w = actual_shape[0]
    batch_size, channel, in_w = input.shape

    ratio_w = 0.0
    if out_w > 1:
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_w))

    for j in range(out_w):
        if (align_mode == 0 and not align_corners):
            w = int(ratio_w * (j + 0.5) - 0.5)
        else:
            w = int(ratio_w * j)
        w = max(0, w)
        wid = 1 if w < in_w - 1 else 0

        if (align_mode == 0 and not align_corners):
            idx_src_w = max(ratio_w * (j + 0.5) - 0.5, 0)
            w1lambda = idx_src_w - w
        else:
            w1lambda = ratio_w * j - w
        w2lambda = 1.0 - w1lambda

        out[:, :, j] = w2lambda * input[:, :, w] + w1lambda * input[:, :, w +
                                                                    wid]

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


class TestLinearInterpOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "linear_interp"
        input_np = np.random.random(self.input_shape).astype("float64")

        if self.data_layout == "NCHW":
            in_w = self.input_shape[2]
        else:
            in_w = self.input_shape[1]

        if self.scale > 0:
            out_w = int(in_w * self.scale)
        else:
            out_w = self.out_w

        output_np = linear_interp_np(input_np, out_w, self.out_size,
                                     self.actual_shape, self.align_corners,
                                     self.align_mode, self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if platform.system() == "Linux":
            self.check_output(atol=1e-7)
        else:
            self.check_output(atol=1e-5)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 100]
        self.out_w = 50
        self.scale = 0.
        self.out_size = np.array([50, ]).astype("int32")
        self.align_corners = False
        self.align_mode = 1


class TestLinearInterpOpDataLayout(TestLinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 100]
        self.out_w = 50
        self.scale = 0.
        self.out_size = np.array([50, ]).astype("int32")
        self.align_corners = False
        self.align_mode = 1
        self.data_layout = 'NHWC'


class TestLinearInterpOpAlignMode(TestLinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 100]
        self.out_w = 50
        self.scale = 0.
        self.out_size = np.array([50, ]).astype("int32")
        self.align_corners = False
        self.align_mode = 0


class TestLinearInterpOpScale(TestLinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [1, 3, 100]
        self.out_w = 50
        self.scale = 0.5
        self.out_size = np.array([50, ]).astype("int32")
        self.align_corners = False
        self.align_mode = 0


class TestLinearInterpOpSizeTensor(TestLinearInterpOp):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "linear_interp"
        input_np = np.random.random(self.input_shape).astype("float64")
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False

        if self.data_layout == "NCHW":
            in_w = self.input_shape[2]
        else:
            in_w = self.input_shape[1]

        if self.scale > 0:
            out_w = int(in_w * self.scale)
        else:
            out_w = self.out_w

        output_np = linear_interp_np(input_np, out_w, self.out_size,
                                     self.actual_shape, self.align_corners,
                                     self.align_mode, self.data_layout)

        self.inputs = {'X': input_np}
        if self.out_size is not None and self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.out_size
        elif self.actual_shape is not None and self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.actual_shape
        else:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
            self.inputs['SizeTensor'] = size_tensor

        self.attrs = {
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout
        }
        self.outputs = {'Out': output_np}


class TestLinearInterpOpAPI(unittest.TestCase):
    def test_case(self):
        x = fluid.data(name="x", shape=[1, 3, 128], dtype="float32")
        shape_tensor = fluid.data(name="shape_tensor", shape=[1], dtype="int32")
        scale_tensor = fluid.data(
            name="scale_tensor", shape=[1], dtype="float32")
        dim = fluid.data(name="dim", shape=[1], dtype="int32")
        actual_size = fluid.data(name='actual_size', shape=[1], dtype='int32')

        out1 = fluid.layers.resize_linear(
            x, out_shape=[256, ], align_mode=1, align_corners=False)
        out2 = fluid.layers.resize_linear(
            x, out_shape=shape_tensor, align_mode=1, align_corners=False)
        out3 = fluid.layers.resize_linear(
            x, scale=scale_tensor, align_mode=1, align_corners=False)
        out4 = fluid.layers.resize_linear(
            x, out_shape=[dim, ], align_mode=1, align_corners=False)
        out5 = fluid.layers.resize_linear(
            x,
            out_shape=[256, ],
            actual_shape=actual_size,
            align_mode=1,
            align_corners=False)

        x_data = np.random.random((1, 3, 128)).astype("float32")
        shape_data = np.array([256, ]).astype("int32")
        scale_data = np.array([2.0, ]).astype("float32")
        dim_data = np.array([256, ]).astype("int32")
        actual_size_data = np.array([256, ]).astype("int32")

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        results = exe.run(fluid.default_main_program(),
                          feed={
                              "x": x_data,
                              "shape_tensor": shape_data,
                              "scale_tensor": scale_data,
                              "dim": dim_data,
                              'actual_size': actual_size_data,
                          },
                          fetch_list=[out1, out2, out3, out4, out5],
                          return_numpy=True)

        expect_res = linear_interp_np(
            x_data, out_w=256, align_mode=1, align_corners=False)

        for res in results:
            self.assertTrue(np.allclose(res, expect_res))


class TestLinearInterpOpAPI2_Func(unittest.TestCase):
    def test_case(self):
        x = fluid.data(name="x", shape=[1, 3, 128], dtype="float32")
        shape_tensor = fluid.data(name="shape_tensor", shape=[1], dtype="int32")
        scale_tensor = fluid.data(
            name="scale_tensor", shape=[1], dtype="float32")
        dim = fluid.data(name="dim", shape=[1], dtype="int32")
        actual_size = fluid.data(name='actual_size', shape=[1], dtype='int32')

        out1 = interpolate(
            x,
            out_shape=[256, ],
            align_mode=1,
            align_corners=False,
            resample='LINEAR')
        out2 = interpolate(
            x,
            out_shape=shape_tensor,
            align_mode=1,
            align_corners=False,
            resample='LINEAR')
        out3 = interpolate(
            x,
            scale=scale_tensor,
            align_mode=1,
            align_corners=False,
            resample='LINEAR')
        out4 = interpolate(
            x,
            out_shape=[dim, ],
            align_mode=1,
            align_corners=False,
            resample='LINEAR')
        out5 = interpolate(
            x,
            out_shape=[256, ],
            actual_shape=actual_size,
            align_mode=1,
            align_corners=False,
            resample='LINEAR')

        x_data = np.random.random((1, 3, 128)).astype("float32")
        shape_data = np.array([256, ]).astype("int32")
        scale_data = np.array([2.0, ]).astype("float32")
        dim_data = np.array([256, ]).astype("int32")
        actual_size_data = np.array([256, ]).astype("int32")

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        results = exe.run(fluid.default_main_program(),
                          feed={
                              "x": x_data,
                              "shape_tensor": shape_data,
                              "scale_tensor": scale_data,
                              "dim": dim_data,
                              'actual_size': actual_size_data,
                          },
                          fetch_list=[out1, out2, out3, out4, out5],
                          return_numpy=True)

        expect_res = linear_interp_np(
            x_data, out_w=256, align_mode=1, align_corners=False)

        for res in results:
            self.assertTrue(np.allclose(res, expect_res))


class TestLinearInterpOpAPI2_0(unittest.TestCase):
    def test_case(self):

        # dygraph 
        x_data = np.random.random((1, 3, 128)).astype("float32")
        us_1 = paddle.nn.UpSample(
            out_shape=[64, ],
            resample='LINEAR',
            align_mode=1,
            align_corners=False)
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(x_data)
            interp = us_1(x)

            expect = linear_interp_np(
                x_data, out_w=64, align_mode=1, align_corners=False)

            self.assertTrue(np.allclose(interp.numpy(), expect))


class TestLinearInterpOpUint8(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "linear_interp"
        input_np = np.random.random(self.input_shape).astype("uint8")

        if self.scale > 0:
            out_w = int(self.input_shape[3] * self.scale)
        else:
            out_w = self.out_w

        output_np = linear_interp_np(input_np, out_w, self.out_size,
                                     self.actual_shape, self.align_corners,
                                     self.align_mode)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size

        self.attrs = {
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        if platform.system() == "Linux":
            self.check_output_with_place(place=core.CPUPlace(), atol=1e-7)
        else:
            self.check_output_with_place(place=core.CPUPlace(), atol=1e-5)

    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [2, 3, 100]
        self.out_w = 50
        self.scale = 0.
        self.out_size = np.array([50, ]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestLinearInterpOpException(unittest.TestCase):
    def test_exception(self):
        def input_shape_error():
            x1 = fluid.data(name="x1", shape=[1], dtype="float32")
            out = fluid.layers.resize_linear(
                x1, out_shape=[256, ], data_format='NCW')

        def data_format_error():
            x2 = fluid.data(name="x2", shape=[1, 3, 128], dtype="float32")
            out = fluid.layers.resize_linear(
                x2, out_shape=[256, ], data_format='NHWCD')

        def out_shape_error():
            x3 = fluid.data(name="x3", shape=[1, 3, 128], dtype="float32")
            out = fluid.layers.resize_linear(
                x3, out_shape=[
                    256,
                    256,
                ], data_format='NHWC')

        self.assertRaises(ValueError, input_shape_error)
        self.assertRaises(ValueError, data_format_error)
        self.assertRaises(ValueError, out_shape_error)


class TestLinearInterpOpError(unittest.TestCase):
    def test_error(self):
        with program_guard(Program(), Program()):

            def input_shape_error():
                x1 = fluid.data(name="x1", shape=[1], dtype="float32")
                out1 = paddle.nn.UpSample(
                    out_shape=[256, ], data_format='NCW', resample='LINEAR')
                out1_res = out1(x1)

            def data_format_error():
                x2 = fluid.data(name="x2", shape=[1, 3, 128], dtype="float32")
                out2 = paddle.nn.UpSample(
                    out_shape=[256, ], data_format='NHWCD', resample='LINEAR')
                out2_res = out2(x2)

            def out_shape_error():
                x3 = fluid.data(name="x3", shape=[1, 3, 128], dtype="float32")
                out3 = paddle.nn.UpSample(
                    out_shape=[
                        256,
                        256,
                    ],
                    data_format='NHWC',
                    resample='LINEAR')
                out3_res = out3(x3)

            self.assertRaises(ValueError, input_shape_error)
            self.assertRaises(ValueError, data_format_error)
            self.assertRaises(ValueError, out_shape_error)


if __name__ == "__main__":
    unittest.main()
