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
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.nn.functional import interpolate
import paddle


def bilinear_interp_np(input,
                       out_h,
                       out_w,
                       scale_w=0,
                       scale_h=0,
                       out_size=None,
                       actual_shape=None,
                       align_corners=True,
                       align_mode=0,
                       data_layout='NCHW'):
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
        if (align_corners):
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
        else:
            if scale_h > 0:
                ratio_h = 1.0 / scale_h
            else:
                ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            if scale_w > 0:
                ratio_w = 1.0 / scale_w
            else:
                ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for i in range(out_h):
        if (align_mode == 0 and not align_corners):
            h = int(ratio_h * (i + 0.5) - 0.5)
        else:
            h = int(ratio_h * i)

        h = max(0, h)
        hid = 1 if h < in_h - 1 else 0
        if (align_mode == 0 and not align_corners):
            idx_src_h = max(ratio_h * (i + 0.5) - 0.5, 0)
            h1lambda = idx_src_h - h
        else:
            h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
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

            out[:, :, i, j] = h2lambda*(w2lambda*input[:, :, h, w] +
                                        w1lambda*input[:, :, h, w+wid]) + \
                h1lambda*(w2lambda*input[:, :, h+hid, w] +
                          w1lambda*input[:, :, h+hid, w+wid])

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

    return out.astype(input.dtype)


class TestBilinearInterpOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.data_layout = 'NCHW'
        self.init_test_case()
        self.op_type = "bilinear_interp_v2"
        input_np = np.random.random(self.input_shape).astype("float64")

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
                if self.scale > 0.:
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

        output_np = bilinear_interp_np(
            input_np, out_h, out_w, 0, 0, self.out_size, self.actual_shape,
            self.align_corners, self.align_mode, self.data_layout)
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
            'align_mode': self.align_mode,
            'data_layout': self.data_layout
        }
        if self.scale:
            if isinstance(self.scale, float) or isinstance(self.scale, int):
                if self.scale > 0.:
                    self.scale = [self.scale]
            if isinstance(self.scale, list) and len(self.scale) == 1:
                self.scale = [self.scale[0], self.scale[0]]
            self.attrs['scale'] = self.scale
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase1(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase2(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase3(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase4(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.scale = 0.
        self.out_size = np.array([2, 2]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase5(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = np.array([11, 11]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase6(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([65, 33]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase7(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = [2.0, 0.5]
        self.align_corners = False
        self.align_mode = 1


class TestBilinearInterpSame(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 32
        self.out_w = 64
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpActualShape(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpDataLayout(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 5, 5, 3]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1
        self.data_layout = "NHWC"


class TestBilinearInterpOpUint8(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "bilinear_interp_v2"
        input_np = np.random.randint(
            low=0, high=256, size=self.input_shape).astype("uint8")

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

        output_np = bilinear_interp_np(input_np, out_h, out_w, 0, 0,
                                       self.out_size, self.actual_shape,
                                       self.align_corners, self.align_mode)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode
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
        self.check_output_with_place(place=core.CPUPlace(), atol=1)

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 3, 9, 6]
        self.out_h = 10
        self.out_w = 9
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase1Uint8(TestBilinearInterpOpUint8):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 32, 64]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpCase2Uint8(TestBilinearInterpOpUint8):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 5
        self.out_w = 13
        self.scale = 0.
        self.out_size = np.array([6, 15]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpOtherMethod1(TestBilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 1


class TestBilinearInterpWithMethod2(TestBilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = False
        self.align_mode = 0


class TestBilinearInterpWithMethod3(TestBilinearInterpOp):
    def set_align_mode(self):
        self.align_corners = True
        self.align_mode = 0


class TestBilinearInterpScale1(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 2.
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpScale2(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpScale3(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.5
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpScale4(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = [1.5, 0.5]
        self.align_corners = True
        self.align_mode = 1


class TestBilinearInterpZero(TestBilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 7]
        self.out_h = 60
        self.out_w = 25
        self.scale = 0.2
        self.align_corners = False
        self.align_mode = 0


class TestBilinearInterpOp_attr_tensor(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "bilinear_interp_v2"
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
        output_np = bilinear_interp_np(input_np, out_h, out_w, 0, 0,
                                       self.out_size, self.actual_shape,
                                       self.align_corners)
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 3
        self.out_w = 3
        self.scale = 0.
        self.out_size = [3, 3]
        self.align_corners = True


# out_size is a 1-D tensor
class TestBilinearInterp_attr_tensor_Case1(TestBilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.scale = 0.
        self.out_size = [8, 12]
        self.align_corners = True


# scale is a 1-D tensor
class TestBilinearInterp_attr_tensor_Case2(TestBilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 0.
        self.out_size = np.array([66, 40]).astype("int32")
        self.align_corners = True
        self.shape_by_1Dtensor = True


# scale is a 1-D tensor
class TestBilinearInterp_attr_tensor_Case3(TestBilinearInterpOp_attr_tensor):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.scale = 2.0
        self.out_size = None
        self.align_corners = True
        self.scale_by_1Dtensor = True


class TestBilinearInterpOpAPI(unittest.TestCase):
    def test_case(self):
        x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")

        dim = fluid.data(name="dim", shape=[1], dtype="int32")
        shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
        actual_size = fluid.data(name="actual_size", shape=[2], dtype="int32")
        scale_tensor = fluid.data(
            name="scale_tensor", shape=[1], dtype="float32")

        out1 = fluid.layers.resize_bilinear(x, out_shape=[12, 12])
        out2 = fluid.layers.resize_bilinear(x, out_shape=[12, dim])
        out3 = fluid.layers.resize_bilinear(x, out_shape=shape_tensor)
        out4 = fluid.layers.resize_bilinear(
            x, out_shape=[4, 4], actual_shape=actual_size)
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

        expect_res = bilinear_interp_np(
            x_data, out_h=12, out_w=12, align_corners=True)
        for res in results:
            self.assertTrue(np.allclose(res, expect_res))


class TestUpsampleBilinear2dInterpOpAPI2_0(unittest.TestCase):
    def test_case(self):

        # dygraph
        x_data = np.random.random((1, 3, 6, 6)).astype("float32")
        upsample = paddle.nn.UpsamplingBilinear2d(scale_factor=[2, 2])
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(x_data)
            interp = upsample(x)
            expect = bilinear_interp_np(
                x_data, out_h=12, out_w=12, align_corners=True)
            self.assertTrue(np.allclose(interp.numpy(), expect))


class TestBilinearInterpOpAPI_dy(unittest.TestCase):
    def test_case(self):
        import paddle
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        with fluid.dygraph.guard(place):
            input_data = np.random.random((2, 3, 6, 6)).astype("float32")
            input_x = paddle.to_tensor(input_data)
            expect_res = bilinear_interp_np(
                input_data, out_h=12, out_w=12, align_corners=False)
            out = interpolate(
                x=input_x, size=[12, 12], mode="bilinear", align_corners=False)
            self.assertTrue(np.allclose(out.numpy(), expect_res))


if __name__ == "__main__":
    unittest.main()
