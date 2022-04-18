#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.nn.functional import interpolate


def trilinear_interp_np(input,
                        out_d,
                        out_h,
                        out_w,
                        out_size=None,
                        actual_shape=None,
                        align_corners=True,
                        align_mode=0,
                        data_layout='NCDHW'):
    """trilinear interpolation implement in shape [N, C, D, H, W]"""
    if data_layout == "NDHWC":
        input = np.transpose(input, (0, 4, 1, 2, 3))  # NDHWC => NCDHW
    if out_size is not None:
        out_d = out_size[0]
        out_h = out_size[1]
        out_w = out_size[2]
    if actual_shape is not None:
        out_d = actual_shape[0]
        out_h = actual_shape[1]
        out_w = actual_shape[2]
    batch_size, channel, in_d, in_h, in_w = input.shape

    ratio_d = ratio_h = ratio_w = 0.0
    if out_d > 1:
        if (align_corners):
            ratio_d = (in_d - 1.0) / (out_d - 1.0)
        else:
            ratio_d = 1.0 * in_d / out_d
    if out_h > 1:
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_d, out_h, out_w))

    for i in range(out_d):
        if (align_mode == 0 and not align_corners):
            d = int(ratio_d * (i + 0.5) - 0.5)
        else:
            d = int(ratio_d * i)

        d = max(0, d)
        did = 1 if d < in_d - 1 else 0
        if (align_mode == 0 and not align_corners):
            idx_src_d = max(ratio_d * (i + 0.5) - 0.5, 0)
            d1lambda = idx_src_d - d
        else:
            d1lambda = ratio_d * i - d
        d2lambda = 1.0 - d1lambda

        for j in range(out_h):
            if (align_mode == 0 and not align_corners):
                h = int(ratio_h * (j + 0.5) - 0.5)
            else:
                h = int(ratio_h * j)

            h = max(0, h)
            hid = 1 if h < in_h - 1 else 0
            if (align_mode == 0 and not align_corners):
                idx_src_h = max(ratio_h * (j + 0.5) - 0.5, 0)
                h1lambda = idx_src_h - h
            else:
                h1lambda = ratio_h * j - h
            h2lambda = 1.0 - h1lambda

            for k in range(out_w):
                if (align_mode == 0 and not align_corners):
                    w = int(ratio_w * (k + 0.5) - 0.5)
                else:
                    w = int(ratio_w * k)
                w = max(0, w)
                wid = 1 if w < in_w - 1 else 0
                if (align_mode == 0 and not align_corners):
                    idx_src_w = max(ratio_w * (k + 0.5) - 0.5, 0)
                    w1lambda = idx_src_w - w
                else:
                    w1lambda = ratio_w * k - w
                w2lambda = 1.0 - w1lambda

                out[:, :, i, j, k] = \
                    d2lambda * \
                    (h2lambda * (w2lambda * input[:, :, d, h, w] + \
                              w1lambda * input[:, :, d, h, w+wid]) + \
                    h1lambda * (w2lambda * input[:, :, d, h+hid, w] + \
                              w1lambda * input[:, :, d, h+hid, w+wid])) + \
                    d1lambda * \
                    (h2lambda * (w2lambda * input[:, :, d+did, h, w] + \
                              w1lambda * input[:, :, d+did, h, w+wid]) + \
                    h1lambda * (w2lambda * input[:, :, d+did, h+hid, w] + \
                              w1lambda * input[:, :, d+did, h+hid, w+wid]))
    if data_layout == "NDHWC":
        out = np.transpose(out, (0, 2, 3, 4, 1))  # NCDHW => NDHWC

    return out.astype(input.dtype)


class TestTrilinearInterpOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCDHW'
        self.init_test_case()
        self.op_type = "trilinear_interp"
        # NOTE(dev): some AsDispensible input is not used under imperative mode.
        # Skip check_eager while found them in Inputs.
        self.check_eager = True
        input_np = np.random.random(self.input_shape).astype("float32")

        if self.data_layout == "NCDHW":
            in_d = self.input_shape[2]
            in_h = self.input_shape[3]
            in_w = self.input_shape[4]
        else:
            in_d = self.input_shape[1]
            in_h = self.input_shape[2]
            in_w = self.input_shape[3]

        if self.scale > 0:
            out_d = int(in_d * self.scale)
            out_h = int(in_h * self.scale)
            out_w = int(in_w * self.scale)
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        output_np = trilinear_interp_np(
            input_np, out_d, out_h, out_w, self.out_size, self.actual_shape,
            self.align_corners, self.align_mode, self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
            self.check_eager = False
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape
            self.check_eager = False
        # c++ end treat NCDHW the same way as NCHW
        if self.data_layout == 'NCDHW':
            data_layout = 'NCHW'
        else:
            data_layout = 'NHWC'
        self.attrs = {
            'out_d': self.out_d,
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': data_layout
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=self.check_eager)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', in_place=True, check_eager=self.check_eager)

    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 4, 4, 4]
        self.out_d = 2
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase1(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 1, 7, 8, 9]
        self.out_d = 1
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase2(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 9, 6, 8]
        self.out_d = 12
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase3(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 2, 16, 8, 4]
        self.out_d = 32
        self.out_h = 16
        self.out_w = 8
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase4(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [4, 1, 7, 8, 9]
        self.out_d = 1
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2, 2]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase5(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 3, 9, 6, 8]
        self.out_d = 12
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11, 11]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase6(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 16, 8, 4]
        self.out_d = 8
        self.out_h = 32
        self.out_w = 16
        self.scale = 0.
        self.out_size = np.array([17, 9, 5]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpSame(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 16, 8, 4]
        self.out_d = 16
        self.out_h = 8
        self.out_w = 4
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpSameHW(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 1, 16, 8, 4]
        self.out_d = 8
        self.out_h = 8
        self.out_w = 4
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpActualShape(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 2, 16, 8, 4]
        self.out_d = 64
        self.out_h = 32
        self.out_w = 16
        self.scale = 0.
        self.out_size = np.array([33, 19, 7]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpDatalayout(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 4, 4, 4, 3]
        self.out_d = 2
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1
        self.data_layout = "NDHWC"


class TestTrilinearInterpOpUint8(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "trilinear_interp"
        self.check_eager = True
        input_np = np.random.randint(
            low=0, high=256, size=self.input_shape).astype("uint8")

        if self.scale > 0:
            out_d = int(self.input_shape[2] * self.scale)
            out_h = int(self.input_shape[3] * self.scale)
            out_w = int(self.input_shape[4] * self.scale)
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        output_np = trilinear_interp_np(input_np, out_d, out_h, out_w,
                                        self.out_size, self.actual_shape,
                                        self.align_corners, self.align_mode)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
            self.check_eager = False

        self.attrs = {
            'out_d': self.out_d,
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(
            place=core.CPUPlace(), atol=1, check_eager=self.check_eager)

    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [1, 3, 9, 6, 8]
        self.out_d = 13
        self.out_h = 10
        self.out_w = 9
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase1Uint8(TestTrilinearInterpOpUint8):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 16, 8, 4]
        self.out_d = 13
        self.out_h = 7
        self.out_w = 2
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpCase2Uint8(TestTrilinearInterpOpUint8):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [4, 1, 7, 8, 9]
        self.out_d = 3
        self.out_h = 5
        self.out_w = 13
        self.scale = 0.
        self.out_size = np.array([6, 15, 21]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpOtherMethod1(TestTrilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 1


class TestTrilinearInterpWithMethod2(TestTrilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 0


class TestTrilinearInterpWithMethod3(TestTrilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = True
        self.align_mode = 0


class TestTrilinearInterpScale1(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 82
        self.out_h = 60
        self.out_w = 25
        self.scale = 2.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpScale2(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 60
        self.out_h = 40
        self.out_w = 25
        self.scale = 1.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpScale3(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 60
        self.out_h = 40
        self.out_w = 25
        self.scale = 1.5
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpZero(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 11]
        self.out_d = 60
        self.out_h = 40
        self.out_w = 25
        self.scale = 0.2
        self.align_corners = False
        self.align_mode = 0


class TestTrilinearInterpOp_attr_tensor(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "trilinear_interp"
        self.check_eager = True
        self.shape_by_1Dtensor = False
        self.scale_by_1Dtensor = False
        self.attrs = {
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode
        }

        input_np = np.random.random(self.input_shape).astype("float32")
        self.inputs = {'X': input_np}

        if self.scale_by_1Dtensor:
            self.inputs['Scale'] = np.array([self.scale]).astype("float32")
        elif self.scale > 0:
            out_d = int(self.input_shape[2] * self.scale)
            out_h = int(self.input_shape[3] * self.scale)
            out_w = int(self.input_shape[4] * self.scale)
            self.attrs['scale'] = self.scale
        else:
            out_d = self.out_d
            out_h = self.out_h
            out_w = self.out_w

        if self.shape_by_1Dtensor:
            self.inputs['OutSize'] = self.out_size
            self.check_eager = False
        elif self.out_size is not None:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
            self.inputs['SizeTensor'] = size_tensor
            self.check_eager = False

        self.attrs['out_d'] = self.out_d
        self.attrs['out_h'] = self.out_h
        self.attrs['out_w'] = self.out_w
        output_np = trilinear_interp_np(input_np, out_d, out_h, out_w,
                                        self.out_size, self.actual_shape,
                                        self.align_corners, self.align_mode)
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output(check_eager=self.check_eager)

    def test_check_grad(self):
        self.check_grad(
            ['X'], 'Out', in_place=True, check_eager=self.check_eager)

    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 4, 4, 4]
        self.out_d = 2
        self.out_h = 3
        self.out_w = 3
        self.scale = 0.
        self.out_size = [2, 3, 3]
        self.align_corners = True
        self.align_mode = 1


# out_size is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case1(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [3, 2, 9, 6, 8]
        self.out_d = 32
        self.out_h = 16
        self.out_w = 8
        self.scale = 0.3
        self.out_size = [12, 4, 4]
        self.align_corners = True
        self.align_mode = 1


# scale is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case2(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 8, 8, 4]
        self.out_d = 16
        self.out_h = 12
        self.out_w = 4
        self.scale = 0.
        self.out_size = [16, 4, 10]
        self.align_corners = True
        self.align_mode = 1
        self.shape_by_1Dtensor = True


# scale is a 1-D tensor
class TestTrilinearInterp_attr_tensor_Case3(TestTrilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 8, 8, 4]
        self.out_d = 16
        self.out_h = 16
        self.out_w = 8
        self.scale = 2.0
        self.out_size = None
        self.align_corners = True
        self.align_mode = 1
        self.scale_by_1Dtensor = True


class TestTrilinearInterpAPI(unittest.TestCase):
    def test_case(self):
        x = fluid.data(name="x", shape=[2, 3, 6, 9, 4], dtype="float32")
        y = fluid.data(name="y", shape=[2, 6, 9, 4, 3], dtype="float32")

        dim = fluid.data(name="dim", shape=[1], dtype="int32")
        shape_tensor = fluid.data(name="shape_tensor", shape=[3], dtype="int32")
        actual_size = fluid.data(name="actual_size", shape=[3], dtype="int32")
        scale_tensor = fluid.data(
            name="scale_tensor", shape=[1], dtype="float32")

        out1 = fluid.layers.resize_trilinear(
            y, out_shape=[12, 18, 8], data_format='NDHWC')
        out2 = fluid.layers.resize_trilinear(x, out_shape=[12, dim, 8])
        out3 = fluid.layers.resize_trilinear(x, out_shape=shape_tensor)
        out4 = fluid.layers.resize_trilinear(
            x, out_shape=[4, 4, 8], actual_shape=actual_size)
        out5 = fluid.layers.resize_trilinear(x, scale=scale_tensor)
        out6 = interpolate(
            x, scale_factor=scale_tensor, mode='trilinear', data_format="NCDHW")
        out7 = interpolate(
            x, size=[4, 4, 8], mode='trilinear', data_format="NCDHW")
        out8 = interpolate(
            x, size=shape_tensor, mode='trilinear', data_format="NCDHW")

        x_data = np.random.random((2, 3, 6, 9, 4)).astype("float32")
        dim_data = np.array([18]).astype("int32")
        shape_data = np.array([12, 18, 8]).astype("int32")
        actual_size_data = np.array([12, 18, 8]).astype("int32")
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
                              "y": np.transpose(x_data, (0, 2, 3, 4, 1)),
                              "dim": dim_data,
                              "shape_tensor": shape_data,
                              "actual_size": actual_size_data,
                              "scale_tensor": scale_data
                          },
                          fetch_list=[out1, out2, out3, out4, out5],
                          return_numpy=True)

        expect_res = trilinear_interp_np(
            x_data, out_d=12, out_h=18, out_w=8, align_mode=1)
        self.assertTrue(
            np.allclose(results[0], np.transpose(expect_res, (0, 2, 3, 4, 1))))
        for i in range(len(results) - 1):
            self.assertTrue(np.allclose(results[i + 1], expect_res))


class TestTrilinearInterpOpException(unittest.TestCase):
    def test_exception(self):
        input = fluid.data(name="input", shape=[2, 3, 6, 9, 4], dtype="float32")

        def attr_data_format():
            # for 5-D input, data_format only can be NCDHW or NDHWC
            out = fluid.layers.resize_trilinear(
                input, out_shape=[4, 8, 4], data_format='NHWC')

        self.assertRaises(ValueError, attr_data_format)


if __name__ == "__main__":
    unittest.main()
