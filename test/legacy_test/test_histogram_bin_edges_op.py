#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestHistogramBinEdgesOp(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(5, 20).astype('float32')
        self.bin = 10
        self.range = None
        self.out = np.histogram_bin_edges(
            self.x, bins=self.bin, range=self.range
        )

    def check_with_place(self, place):
        x_var = paddle.Tensor(self.x, place=place)
        out_var = paddle.histogram_bin_edges(
            x_var, bins=self.bin, min=self.range[0], max=self.range[1]
        )
        out = np.array(out_var)
        np.testing.assert_allclose(self.out, out, rtol=1e-5, atol=1e-5)

    def test_case(self):
        self.check_with_place(paddle.CPUPlace())
        if paddle.is_compiled_with_cuda():
            self.check_with_place(paddle.CUDAPlace(0))


class TestHistogramBinEdgesOp(TestHistogramBinEdgesOp):
    def setUp(self):
        self.x = np.random.randn(5, 20).astype('float32')
        self.bin = 10
        self.range = (0, 1)
        self.out = np.histogram_bin_edges(
            self.x, bins=self.bin, range=self.range
        )


class TestHistogramBinEdgesOpTest1(TestHistogramBinEdgesOp):
    def setUp(self):
        self.x = np.random.randn(5, 20).astype('float32')
        self.bin = 10
        self.range = (1, 1)
        self.out = np.histogram_bin_edges(
            self.x, bins=self.bin, range=self.range
        )


class TestHistogramBinEdgesOpTest2(TestHistogramBinEdgesOp):
    def setUp(self):
        self.x = np.random.randn(5, 20).astype('float32')
        self.bin = 10
        self.range = (0.2, 0.9)
        self.out = np.histogram_bin_edges(
            self.x, bins=self.bin, range=self.range
        )


if __name__ == "__main__":
    unittest.main()
