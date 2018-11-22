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


def nearest_neighbor_interp_np(X,
                               out_h,
                               out_w,
                               out_size=None,
                               actual_shape=None):
    """nearest neighbor interpolation implement in shape [N, C, H, W]"""
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    n, c, in_h, in_w = X.shape

    ratio_h = ratio_w = 0.0
    if out_h > 1:
        ratio_h = (in_h - 1.0) / (out_h - 1.0)
    if out_w > 1:
        ratio_w = (in_w - 1.0) / (out_w - 1.0)

    out = np.zeros((n, c, out_h, out_w))
    for i in range(out_h):
        in_i = int(ratio_h * i + 0.5)
        for j in range(out_w):
            in_j = int(ratio_w * j + 0.5)
            out[:, :, i, j] = X[:, :, in_i, in_j]

    return out.astype(X.dtype)


def bilinear_interp_np(input, out_h, out_w, out_size=None, actual_shape=None):
    """bilinear interpolation implement in shape [N, C, H, W]"""
    if out_size is not None:
        out_h = out_size[0]
        out_w = out_size[1]
    if actual_shape is not None:
        out_h = actual_shape[0]
        out_w = actual_shape[1]
    batch_size, channel, in_h, in_w = input.shape
    if out_h > 1:
        ratio_h = (in_h - 1.0) / (out_h - 1.0)
    else:
        ratio_h = 0.0
    if out_w > 1:
        ratio_w = (in_w - 1.0) / (out_w - 1.0)
    else:
        ratio_w = 0.0

    out = np.zeros((batch_size, channel, out_h, out_w))
    for i in range(out_h):
        h = int(ratio_h * i)
        hid = 1 if h < in_h - 1 else 0
        h1lambda = ratio_h * i - h
        h2lambda = 1.0 - h1lambda
        for j in range(out_w):
            w = int(ratio_w * j)
            wid = 1 if w < in_w - 1 else 0
            w1lambda = ratio_w * j - w
            w2lambda = 1.0 - w1lambda

            out[:, :, i, j] = h2lambda*(w2lambda*input[:, :, h, w] +
                                        w1lambda*input[:, :, h, w+wid]) + \
                h1lambda*(w2lambda*input[:, :, h+hid, w] +
                          w1lambda*input[:, :, h+hid, w+wid])
    return out.astype(input.dtype)


INTERPOLATE_FUNCS = {
    'bilinear': bilinear_interp_np,
    'nearest': nearest_neighbor_interp_np,
}


class TestInterpolateOp(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "interpolate"
        input_np = np.random.random(self.input_shape).astype("float32")

        output_np = INTERPOLATE_FUNCS[self.interp_method](
            input_np, self.out_h, self.out_w, self.out_size, self.actual_shape)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        if self.actual_shape is not None:
            self.inputs['OutSize'] = self.actual_shape
        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 4, 4]
        self.out_h = 2
        self.out_w = 2
        self.out_size = np.array([3, 3]).astype("int32")


class TestBilinearInterpCase1(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1


class TestBilinearInterpCase2(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12


class TestBilinearInterpCase3(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 128, 64]
        self.out_h = 64
        self.out_w = 128


class TestBilinearInterpCase4(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.out_size = np.array([2, 2]).astype("int32")


class TestBilinearInterpCase5(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.out_size = np.array([11, 11]).astype("int32")


class TestBilinearInterpCase6(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 1, 128, 64]
        self.out_h = 64
        self.out_w = 128
        self.out_size = np.array([65, 129]).astype("int32")


class TestBilinearInterpActualShape(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.out_size = np.array([66, 40]).astype("int32")


class TestBilinearInterpBigScale(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 4, 64, 32]
        self.out_h = 100
        self.out_w = 50
        self.out_size = np.array([101, 51]).astype('int32')


class TestInterpolateOpUint8(OpTest):
    def setUp(self):
        self.out_size = None
        self.actual_shape = None
        self.init_test_case()
        self.op_type = "interpolate"
        input_np = np.random.randint(
            low=0, high=256, size=self.input_shape).astype("uint8")
        output_np = INTERPOLATE_FUNCS[self.interp_method](
            input_np, self.out_h, self.out_w, self.out_size, self.actual_shape)
        self.inputs = {'X': input_np}
        if self.out_size is not None:
            self.inputs['OutSize'] = self.out_size
        self.attrs = {
            'out_h': self.out_h,
            'out_w': self.out_w,
            'interp_method': self.interp_method
        }
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output_with_place(place=core.CPUPlace(), atol=1)

    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [1, 3, 9, 6]
        self.out_h = 10
        self.out_w = 9


class TestBilinearInterpCase1Uint8(TestInterpolateOpUint8):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [2, 3, 128, 64]
        self.out_h = 120
        self.out_w = 50


class TestBilinearInterpCase2Uint8(TestInterpolateOpUint8):
    def init_test_case(self):
        self.interp_method = 'bilinear'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 5
        self.out_w = 13
        self.out_size = np.array([6, 15]).astype("int32")


class TestNearestNeighborInterpCase1(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1


class TestNearestNeighborInterpCase2(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12


class TestNearestNeighborInterpCase3(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 128, 64]
        self.out_h = 64
        self.out_w = 128


class TestNearestNeighborInterpCase4(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1
        self.out_size = np.array([2, 2]).astype("int32")


class TestNearestNeighborInterpCase5(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12
        self.out_size = np.array([11, 11]).astype("int32")


class TestNearestNeighborInterpCase6(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [1, 1, 128, 64]
        self.out_h = 64
        self.out_w = 128
        self.out_size = np.array([65, 129]).astype("int32")


class TestNearestNeighborInterpActualShape(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [3, 2, 32, 16]
        self.out_h = 64
        self.out_w = 32
        self.out_size = np.array([66, 40]).astype("int32")


class TestNearestNeighborInterpBigScale(TestInterpolateOp):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 4, 64, 32]
        self.out_h = 100
        self.out_w = 50
        self.out_size = np.array([103, 51]).astype('int32')


class TestNearestNeighborInterpCase1Uint8(TestInterpolateOpUint8):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [2, 3, 128, 64]
        self.out_h = 120
        self.out_w = 50


class TestNearestNeighborInterpCase2Uint8(TestInterpolateOpUint8):
    def init_test_case(self):
        self.interp_method = 'nearest'
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 5
        self.out_w = 13
        self.out_size = np.array([6, 15]).astype("int32")


if __name__ == "__main__":
    unittest.main()
