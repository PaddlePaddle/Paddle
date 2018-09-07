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

import unittest
import numpy as np
from op_test import OpTest


def bilinear_interp_np(input, out_h, out_w):
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
    return out.astype("float32")


class TestBilinearInterpOp(OpTest):
    def setUp(self):
        self.init_test_case()
        self.op_type = "bilinear_interp"
        input_np = np.random.random(self.input_shape).astype("float32")
        output_np = bilinear_interp_np(input_np, self.out_h, self.out_w)

        self.inputs = {'X': input_np}
        self.attrs = {'out_h': self.out_h, 'out_w': self.out_w}
        self.outputs = {'Out': output_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', in_place=True)

    def init_test_case(self):
        self.input_shape = [2, 3, 4, 4]
        self.out_h = 2
        self.out_w = 2


class TestCase1(TestBilinearInterpOp):
    def init_test_case(self):
        self.input_shape = [4, 1, 7, 8]
        self.out_h = 1
        self.out_w = 1


class TestCase2(TestBilinearInterpOp):
    def init_test_case(self):
        self.input_shape = [3, 3, 9, 6]
        self.out_h = 12
        self.out_w = 12


class TestCase3(TestBilinearInterpOp):
    def init_test_case(self):
        self.input_shape = [1, 1, 128, 64]
        self.out_h = 64
        self.out_w = 128


if __name__ == "__main__":
    unittest.main()
