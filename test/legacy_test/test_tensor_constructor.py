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

import unittest

import numpy as np

import paddle


class TestTensorConstructor(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        paddle.seed(2025)
        self.shape = [10, 20, 30]

    def test_construct_from_list_and_tuple(self):
        x = np.random.random(size=self.shape)
        res = paddle.Tensor(list(x))
        np.testing.assert_allclose(x, res.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(res.dtype, paddle.float32)
        res = paddle.Tensor(tuple(x))
        np.testing.assert_allclose(x, res.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(res.dtype, paddle.float32)

    def test_empty_construct(self):
        target = paddle.empty([0])
        res = paddle.Tensor()
        self.assertEqual(res.shape, target.shape)

        target = paddle.empty(self.shape, dtype=paddle.float32)
        res = paddle.Tensor(*self.shape)
        self.assertEqual(res.dtype, paddle.float32)
        self.assertEqual(res.shape, self.shape)

    def test_error_construct(self):
        with self.assertRaises(ValueError):
            a = paddle.tensor([1])
            paddle.Tensor(1, 2, 3, a)

    def test_kwargs(self):
        x1 = paddle.Tensor(device="cpu")
        self.assertEqual(x1.place, paddle.CPUPlace())
        x2 = paddle.Tensor(*self.shape, device="cpu")
        self.assertEqual(x2.place, paddle.CPUPlace())

        x = np.random.random(size=self.shape)
        x3 = paddle.Tensor(data=x)
        np.testing.assert_allclose(x, x3.numpy(), rtol=1e-6, atol=1e-6)
        x4 = paddle.Tensor(list(x), device="cpu")
        x5 = paddle.Tensor(data=list(x), device="cpu")
        np.testing.assert_allclose(x4.numpy(), x5.numpy(), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(x, x4.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(x4.place, x5.place)
        self.assertEqual(x4.place, paddle.CPUPlace())


class TestFloatTensor(unittest.TestCase):
    def setUp(self):
        np.random.seed(2025)
        paddle.seed(2025)
        self.shape = [10, 20, 30]
        self.set_api_and_type()

    def set_api_and_type(self):
        self.dtype = paddle.float32
        self.np_dtype = "float32"
        self.api = paddle.FloatTensor

    def test_empty_construct(self):
        target = paddle.empty([0], dtype=self.dtype)
        res = self.api()
        self.assertEqual(res.shape, target.shape)

        target = paddle.empty(self.shape, dtype=self.dtype)
        res = self.api(*self.shape)
        self.assertEqual(res.dtype, self.dtype)
        self.assertEqual(res.shape, self.shape)

    def test_construct_from_list_and_tuple(self):
        x = np.random.random(size=self.shape).astype(self.np_dtype)
        res = self.api(tuple(x))
        np.testing.assert_allclose(x, res.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(res.dtype, self.dtype)
        res = self.api(list(x))
        np.testing.assert_allclose(x, res.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(res.dtype, self.dtype)

    def test_construct_from_tensor_and_numpy(self):
        x = np.random.random(size=self.shape).astype(self.np_dtype)
        x_tensor = paddle.to_tensor(x, dtype=self.dtype)
        res = self.api(x_tensor)
        np.testing.assert_allclose(x, res.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(res.dtype, self.dtype)
        res = self.api(x)
        np.testing.assert_allclose(x, res.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(res.dtype, self.dtype)

    def test_error_construct(self):
        with self.assertRaises(ValueError):
            a = paddle.tensor([1])
            self.api(1, 2, 3, a)


class TestDoubleTensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.float64
        self.np_dtype = "float64"
        self.api = paddle.DoubleTensor


class TestHalfTensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.float16
        self.np_dtype = "float16"
        self.api = paddle.HalfTensor


class TestBFloat16Tensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.bfloat16
        self.np_dtype = "float16"
        self.api = paddle.BFloat16Tensor

    def test_construct_from_list_and_tuple(self):
        x = np.random.random(size=self.shape).astype(self.np_dtype)
        x_target = paddle.to_tensor(x, dtype=self.dtype)
        res = self.api(tuple(x))
        np.testing.assert_allclose(
            x_target.numpy(), res.numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(res.dtype, self.dtype)
        res = self.api(list(x))
        np.testing.assert_allclose(
            x_target.numpy(), res.numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(res.dtype, self.dtype)

    def test_construct_from_tensor_and_numpy(self):
        x_tensor = paddle.randn(self.shape, dtype=self.dtype)
        res = self.api(x_tensor)
        np.testing.assert_allclose(
            x_tensor.numpy(), res.numpy(), rtol=1e-6, atol=1e-6
        )
        self.assertEqual(res.dtype, self.dtype)


class TestByteTensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.uint8
        self.np_dtype = "uint8"
        self.api = paddle.ByteTensor


class TestCharTensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.int8
        self.np_dtype = "int8"
        self.api = paddle.CharTensor


class TestShortTensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.int16
        self.np_dtype = "int16"
        self.api = paddle.ShortTensor


class TestIntTensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.int32
        self.np_dtype = "int32"
        self.api = paddle.IntTensor


class TestLongTensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.int64
        self.np_dtype = "int64"
        self.api = paddle.LongTensor


class TestBoolTensor(TestFloatTensor):
    def set_api_and_type(self):
        self.dtype = paddle.bool
        self.np_dtype = "bool"
        self.api = paddle.BoolTensor


if __name__ == "__main__":
    unittest.main()
