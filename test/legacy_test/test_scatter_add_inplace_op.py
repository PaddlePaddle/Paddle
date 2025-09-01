# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest

import numpy as np
from op_test import get_places

import paddle
from paddle.framework import core


class TestScatterAddInplaceAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [10, 10]
        self.index_shape = [10, 10]
        self.index_np = np.random.randint(0, 10, (10, 10)).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.place = get_places()
        self.axis = 0
        self.value_np = np.random.randint(0, 10, (10, 10)).astype(np.float32)
        self.value_shape = [10, 10]

    def test_inplace_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)

            x_tensor.scatter_add_(self.axis, index_tensor, value_tensor)

            out_ref = copy.deepcopy(self.x_np)
            for i in range(10):
                for j in range(10):
                    out_ref[self.index_np[i, j], j] += self.value_np[i, j]

            np.testing.assert_allclose(x_tensor.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)


@unittest.skipIf(
    not core.is_compiled_with_cuda(),
    "core is not compiled with CUDA",
)
class TestScatterAddInplaceAPILargeCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [64, 102400]
        self.index_shape = [64, 102400]
        self.index_np = np.random.randint(0, 64, (64, 102400)).astype('int64')
        self.x_np = np.random.random(self.shape).astype(np.float32)
        self.axis = 1
        self.value_np = np.random.randint(0, 50, (64, 102400)).astype(
            np.float32
        )
        self.place = [paddle.CUDAPlace(0)]

    def test_inplace_dygraph(self):
        def run(place):
            paddle.disable_static(place)
            x_tensor = paddle.to_tensor(self.x_np)
            index_tensor = paddle.to_tensor(self.index_np)
            value_tensor = paddle.to_tensor(self.value_np)

            x_tensor.scatter_add_(self.axis, index_tensor, value_tensor)

            out_ref = copy.deepcopy(self.x_np)
            for i in range(64):
                for j in range(102400):
                    out_ref[i, self.index_np[i, j]] += self.value_np[i, j]

            np.testing.assert_allclose(x_tensor.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run(place)


class TestScatterAddInplaceAPIOtherCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 5]
        self.index1_shape = [1, 4]
        self.index_np1 = np.array([[0, 1, 2, 0]]).astype('int64')
        self.index2_shape = [2, 3]
        self.index_np2 = np.array([[0, 1, 2], [0, 1, 4]]).astype('int64')
        self.x_np = np.zeros((3, 5)).astype(np.float32)
        self.value_shape = [2, 5]
        self.value = (
            np.arange(1, 11).reshape(self.value_shape).astype(np.float32)
        )
        self.place = get_places()

    def test_api_dygraph(self):
        def run_inplace(place):
            paddle.disable_static(place)
            out1 = paddle.to_tensor(self.x_np)
            index_tensor1 = paddle.to_tensor(self.index_np1)
            value_tensor = paddle.to_tensor(self.value)
            out1.scatter_add_(0, index_tensor1, value_tensor)
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index1_shape[0]):
                for j in range(self.index1_shape[1]):
                    out_ref[self.index_np1[i, j], j] += self.value[i, j]
            np.testing.assert_allclose(out1.numpy(), out_ref, rtol=0.001)

            index_tensor2 = paddle.to_tensor(self.index_np2)
            out2 = paddle.to_tensor(self.x_np)
            out2.scatter_add_(1, index_tensor2, value_tensor)
            out_ref = copy.deepcopy(self.x_np)
            for i in range(self.index2_shape[0]):
                for j in range(self.index2_shape[1]):
                    out_ref[i, self.index_np2[i, j]] += self.value[i, j]
            np.testing.assert_allclose(out2.numpy(), out_ref, rtol=0.001)

            paddle.enable_static()

        for place in self.place:
            run_inplace(place)

    def test_error(self):
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([[1, 0, 1], [0, 1, 1]]).astype("int32")
        values = paddle.to_tensor([1])

        try:
            tensorx.scatter_add_(0, indices, values)
        except Exception as error:
            self.assertIsInstance(error, ValueError)

        indices = paddle.to_tensor([1]).astype("int32")
        values = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])

        try:
            tensorx.scatter_add_(0, indices, values)
        except Exception as error:
            self.assertIsInstance(error, ValueError)

        indices = paddle.to_tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        ).astype("int32")
        # indices too large
        try:
            tensorx.scatter_add_(0, indices, values)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)

        indices = paddle.to_tensor([[3, 0, 4], [0, 5, 10]]).astype("int32")
        # the element of indices out of range
        try:
            tensorx.scatter_add_(0, indices, values)
        except Exception as error:
            self.assertIsInstance(error, RuntimeError)

    def test_index_type_error(self):
        tensorx = paddle.to_tensor([[1, 2, 3], [4, 5, 6]]).astype("float32")
        indices = paddle.to_tensor([[1, 0, 1], [0, 1, 1]]).astype("float32")
        values = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(TypeError):
            tensorx.scatter_add_(0, indices, values)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
