#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import numpy as np

import paddle


class TestPaddleAddNewFeatures(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([3, 5], dtype='float32')
        self.y_np = np.array([2, 3], dtype='float32')

    def test_paddle_add_with_alpha(self):
        """test paddle.add alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.add(x, y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_paddle_add_with_out(self):
        """test paddle.add out"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out_buffer = paddle.zeros_like(x)

        paddle.add(x, y, alpha=2, out=out_buffer)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out_buffer.numpy(), expected)
        self.assertIs(out_buffer, out_buffer)

    def test_tensor_add_with_alpha(self):
        """test paddle.Tensor.add alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = x.add(y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_tensor_add_inplace_with_alpha(self):
        """test Tensor.add_ alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x.add_(y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_consistency_between_apis(self):
        """test different APIs consistency for add with alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)

        out1 = paddle.add(x, y, alpha=2)
        out2 = x.add(y, alpha=2)
        x.add_(y, alpha=2)

        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out1.numpy(), expected)
        np.testing.assert_array_equal(out2.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
