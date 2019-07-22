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


def trilinear_interp_np(input,
                        out_d,
                        out_h,
                        out_w,
                        out_size=None,
                        actual_shape=None,
                        align_corners=True,
                        align_mode=0):
    """trilinear interpolation implement in shape [N, C, D, H, W]"""
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
    return out.astype(input.dtype)


class TestTrilinearInterpOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "trilinear_interp"
        input_np = np.random.random(self.input_shape).astype("float32")

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
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape

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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

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


class TestTrilinearInterpOpUint8(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "trilinear_interp"
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
        self.check_output_with_place(place=core.CPUPlace(), atol=1)

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
        self.out_d = 82
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpScale3(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 9]
        self.out_d = 82
        self.out_h = 60
        self.out_w = 25
        self.scale = 1.5
        self.align_corners = True
        self.align_mode = 1


class TestTrilinearInterpZero(TestTrilinearInterpOp):
    def init_test_case(self):
        self.interp_method = 'trilinear'
        self.input_shape = [2, 3, 5, 7, 11]
        self.out_d = 82
        self.out_h = 60
        self.out_w = 25
        self.scale = 0.2
        self.align_corners = False
        self.align_mode = 0


if __name__ == "__main__":
    unittest.main()
