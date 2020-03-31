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


def linear_interp_np(input,
                     out_h,
                     out_w,
                     out_size=None,
                     actual_shape=None,
                     align_corners=True,
                     align_mode=0,
                     data_layout='NCHW'):
    """linear interpolation implement in shape [N, C, H, W]"""
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
            ratio_h = 1.0 * in_h / out_h
    if out_w > 1:
        if (align_corners):
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
        else:
            ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for i in range(out_h):
        if (align_mode == 0 and not align_corners):
            h = int(ratio_h * (i + 0.5) - 0.5)
        else:
            h = int(ratio_h * i)
        h = max(0, h)

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

            out[:, :, i, j] = w2lambda * input[:, :, h,
                                               w] + w1lambda * input[:, :, h, w
                                                                     + wid]

    if data_layout == "NHWC":
        out = np.transpose(out, (0, 2, 3, 1))  # NCHW => NHWC

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

        output_np = linear_interp_np(input_np, out_h, out_w, self.out_size,
                                     self.actual_shape, self.align_corners,
                                     self.align_mode, self.data_layout)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode,
            'data_layout': self.data_layout
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestLinearInterpOpUint8(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "linear_interp"
        input_np = np.random.randint(
            low=0, high=256, size=self.input_shape).astype("uint8")

        if self.scale > 0:
            out_h = int(self.input_shape[2] * self.scale)
            out_w = int(self.input_shape[3] * self.scale)
        else:
            out_h = self.out_h
            out_w = self.out_w

        output_np = linear_interp_np(input_np, out_h, out_w, self.out_size,
                                     self.actual_shape, self.align_corners,
                                     self.align_mode)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size

        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'scale': self.scale,
            'interp_method': self.interp_method,
            'align_corners': self.align_corners,
            'align_mode': self.align_mode
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(place=core.CPUPlace(), atol=1)

    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 2
        self.out_w = 2
        self.scale = 0.
        self.out_size = np.array([3, 3]).astype("int32")
        self.align_corners = True
        self.align_mode = 1


class TestLinearInterpOp_attr_tensor(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "linear_interp"
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
        elif self.out_size is not None:
            size_tensor = []
            for index, ele in enumerate(self.out_size):
                size_tensor.append(("x" + str(index), np.ones(
                    (1)).astype('int32') * ele))
            self.inputs['SizeTensor'] = size_tensor

        self.attrs['out_h'] = self.out_h
        self.attrs['out_w'] = self.out_w
        output_np = linear_interp_np(input_np, out_h, out_w, self.out_size,
                                     self.actual_shape, self.align_corners)
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'linear'
        self.input_shape = [2, 3, 5, 5]
        self.out_h = 3
        self.out_w = 3
        self.scale = 0.
        self.out_size = [3, 3]
        self.align_corners = True


class TestLinearInterpOpAPI(unittest.TestCase):
    def test_case(self):
        x = fluid.data(name="x", shape=[2, 3, 6, 6], dtype="float32")

        dim = fluid.data(name="dim", shape=[1], dtype="int32")
        shape_tensor = fluid.data(name="shape_tensor", shape=[2], dtype="int32")
        actual_size = fluid.data(name="actual_size", shape=[2], dtype="int32")
        scale_tensor = fluid.data(
            name="scale_tensor", shape=[1], dtype="float32")

        out1 = fluid.layers.resize_linear(x, out_shape=[12, 12])
        out2 = fluid.layers.resize_linear(x, out_shape=[12, dim])
        out3 = fluid.layers.resize_linear(x, out_shape=shape_tensor)
        out4 = fluid.layers.resize_linear(
            x, out_shape=[4, 4], actual_shape=actual_size)
        out5 = fluid.layers.resize_linear(x, scale=scale_tensor)

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

        expect_res = linear_interp_np(
            x_data, out_h=12, out_w=12, align_corners=True)
        for res in results:
            self.assertTrue(np.allclose(res, expect_res))


if __name__ == "__main__":
    unittest.main()
