# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid.core as core
from paddle import _C_ops
from paddle.fluid.framework import _test_eager_guard


class TestMaxPool3DFunc(unittest.TestCase):
    def setInput(self):
        paddle.seed(0)
        self.dense_x = paddle.randn((1, 4, 4, 4, 4))

    def setKernelSize(self):
        self.kernel_sizes = [3, 3, 3]

    def setStride(self):
        self.strides = [1, 1, 1]

    def setPadding(self):
        self.paddings = [0, 0, 0]

    def setUp(self):
        self.setInput()
        self.setKernelSize()
        self.setStride()
        self.setPadding()

    def test(self):
        with _test_eager_guard():
            self.setUp()
            sparse_x = self.dense_x.to_sparse_coo(4)
            out = paddle.sparse.functional.max_pool3d(
                sparse_x,
                self.kernel_sizes,
                stride=self.strides,
                padding=self.paddings)
            out = out.to_dense()

            dense_out = paddle.nn.functional.max_pool3d(
                self.dense_x,
                self.kernel_sizes,
                stride=self.strides,
                padding=self.paddings,
                data_format='NDHWC')
            #compare with dense
            assert np.allclose(dense_out.flatten().numpy(),
                               out.flatten().numpy())


class TestStride(TestMaxPool3DFunc):
    def setStride(self):
        self.strides = 1


class TestPadding(TestMaxPool3DFunc):
    def setPadding(self):
        self.paddings = 1

    def setInput(self):
        self.dense_x = paddle.randn((1, 5, 6, 8, 3))


class TestKernelSize(TestMaxPool3DFunc):
    def setKernelSize(self):
        self.kernel_sizes = [5, 5, 5]

    def setInput(self):
        paddle.seed(0)
        self.dense_x = paddle.randn((1, 6, 9, 6, 3))


class TestInput(TestMaxPool3DFunc):
    def setInput(self):
        paddle.seed(0)
        self.dense_x = paddle.randn((2, 6, 7, 9, 3))
        dropout = paddle.nn.Dropout(0.8)
        self.dense_x = dropout(self.dense_x)


class TestMaxPool3DAPI(unittest.TestCase):
    def test(self):
        with _test_eager_guard():
            dense_x = paddle.randn((2, 3, 6, 6, 3))
            sparse_x = dense_x.to_sparse_coo(4)
            max_pool3d = paddle.sparse.MaxPool3D(
                kernel_size=3, data_format='NDHWC')
            out = max_pool3d(sparse_x)
            out = out.to_dense()

            dense_out = paddle.nn.functional.max_pool3d(
                dense_x, 3, data_format='NDHWC')
            assert np.allclose(dense_out.numpy(), out.numpy())


if __name__ == "__main__":
    unittest.main()
