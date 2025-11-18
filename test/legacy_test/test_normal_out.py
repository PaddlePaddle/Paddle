#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

paddle.enable_static()

np.random.seed(10)
paddle.seed(10)


class TestNormalAPI_out_parameter(unittest.TestCase):
    def test_out_with_shape(self):
        paddle.disable_static()
        shape = [2, 3]

        out_tensor = paddle.empty(shape, dtype='float32')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=0.0, std=1.0, shape=shape, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)

        paddle.enable_static()

    def test_out_with_mean_tensor(self):
        paddle.disable_static()

        mean_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        shape = [3]

        out_tensor = paddle.empty(shape, dtype='float32')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=mean_tensor, std=1.0, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)

        paddle.enable_static()

    def test_out_with_std_tensor(self):
        paddle.disable_static()

        std_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        shape = [3]

        out_tensor = paddle.empty(shape, dtype='float32')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=0.0, std=std_tensor, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)

        paddle.enable_static()

    def test_out_with_mean_std_tensors(self):
        paddle.disable_static()

        mean_tensor = paddle.to_tensor([1.0, 2.0, 3.0])
        std_tensor = paddle.to_tensor([0.5, 1.0, 1.5])
        shape = [3]

        out_tensor = paddle.empty(shape, dtype='float32')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=mean_tensor, std=std_tensor, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)

        paddle.enable_static()

    def test_out_with_complex_mean(self):
        paddle.disable_static()

        shape = [2, 3]

        out_tensor = paddle.empty(shape, dtype='complex64')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(
            mean=1.0 + 1.0j, std=1.0, shape=shape, out=out_tensor
        )

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)
        self.assertEqual(result.dtype, paddle.complex64)

        paddle.enable_static()

    def test_out_with_complex_mean_tensor(self):
        paddle.disable_static()

        mean_tensor = paddle.to_tensor([1.0 + 1.0j, 2.0 + 2.0j, 3.0 + 3.0j])
        shape = [3]

        out_tensor = paddle.empty(shape, dtype='complex64')
        original_ptr = out_tensor.data_ptr()

        result = paddle.normal(mean=mean_tensor, std=1.0, out=out_tensor)

        self.assertEqual(result.data_ptr(), original_ptr)
        self.assertEqual(result.data_ptr(), out_tensor.data_ptr())
        self.assertEqual(list(result.shape), shape)
        self.assertEqual(result.dtype, paddle.complex64)

        paddle.enable_static()


class TestNormalAPI_size_alias(unittest.TestCase):
    def test_size_alias_basic(self):
        paddle.disable_static()
        shape = [2, 3]
        out = paddle.normal(mean=0.0, std=1.0, size=shape)
        self.assertEqual(list(out.shape), shape)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
