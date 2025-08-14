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

    def test_construct_random_tensor_with_float32(self):
        res = paddle.Tensor(*self.shape)
        self.assertEqual(res.dtype, paddle.float32)
        self.assertEqual(res.shape, self.shape)

    def test_construct_from_list(self):
        x = np.random.random(size=self.shape)
        res = paddle.Tensor(list(x))
        np.testing.assert_allclose(x, res.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(res.dtype, paddle.float32)

    def test_construct_from_tuple(self):
        x = np.random.random(size=self.shape)
        res = paddle.Tensor(tuple(x))
        np.testing.assert_allclose(x, res.numpy(), rtol=1e-6, atol=1e-6)
        self.assertEqual(res.dtype, paddle.float32)

    def test_empty_construct(self):
        target = paddle.empty([0])
        res = paddle.Tensor()
        self.assertEqual(res.shape, target.shape)


if __name__ == "__main__":
    unittest.main()
