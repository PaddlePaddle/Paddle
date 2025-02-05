# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class TestPositiveApi(unittest.TestCase):

    def setUp(self):
        paddle.disable_static()
        self.shape = [2, 3, 4, 5]
        self.low = -100
        self.high = 100

    def test_positive_int16(self):
        x = np.random.randint(self.low, self.high, self.shape, dtype=np.int16)
        expected_out = np.positive(x)
        x_tensor = paddle.to_tensor(x)
        out = paddle.positive(x_tensor).numpy()
        np.testing.assert_allclose(out, expected_out, atol=1e-5)

    def test_positive_int32(self):
        x = np.random.randint(self.low, self.high, self.shape, dtype=np.int32)
        expected_out = np.positive(x)
        x_tensor = paddle.to_tensor(x)
        out = paddle.positive(x_tensor).numpy()
        np.testing.assert_allclose(out, expected_out, atol=1e-5)

    def test_positive_int64(self):
        x = np.random.randint(self.low, self.high, self.shape, dtype=np.int64)
        expected_out = np.positive(x)
        x_tensor = paddle.to_tensor(x)
        out = paddle.positive(x_tensor).numpy()
        np.testing.assert_allclose(out, expected_out, atol=1e-5)

    def test_positive_float16(self):
        x = np.random.uniform(self.low, self.high, self.shape).astype(
            np.float16
        )
        expected_out = np.positive(x)
        x_tensor = paddle.to_tensor(x)
        out = paddle.positive(x_tensor).numpy()
        np.testing.assert_allclose(out, expected_out, atol=1e-3)

    def test_positive_float32(self):
        x = np.random.uniform(self.low, self.high, self.shape).astype(
            np.float32
        )
        expected_out = np.positive(x)
        x_tensor = paddle.to_tensor(x)
        out = paddle.positive(x_tensor).numpy()
        np.testing.assert_allclose(out, expected_out, atol=1e-3)

    def test_positive_float64(self):
        x = np.random.uniform(self.low, self.high, self.shape).astype(
            np.float64
        )
        expected_out = np.positive(x)
        x_tensor = paddle.to_tensor(x)
        out = paddle.positive(x_tensor).numpy()
        np.testing.assert_allclose(out, expected_out, atol=1e-3)

    def test_positive_bool(self):
        x = np.random.choice([True, False], size=self.shape)
        x_tensor = paddle.to_tensor(x, dtype=paddle.bool)

        with self.assertRaises(TypeError):
            paddle.positive(x_tensor)


if __name__ == '__main__':
    unittest.main()
