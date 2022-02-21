#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestQuantile(unittest.TestCase):
    def setUp(self):
        np.random.seed(678)
        self.input_data = np.random.rand(6, 7, 8, 9, 10)

    def test_quantile_single_q(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=0.5, axis=2)
        np_res = np.quantile(self.input_data, q=0.5, axis=2)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_quantile_with_no_axis(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=0.35)
        np_res = np.quantile(self.input_data, q=0.35)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_quantile_with_multi_axis(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=0.75, axis=[0, 2, 3])
        np_res = np.quantile(self.input_data, q=0.75, axis=[0, 2, 3])
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_quantile_with_keepdim(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=0.35, axis=4, keepdim=True)
        np_res = np.quantile(self.input_data, q=0.35, axis=4, keepdims=True)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_quantile_with_keepdim_and_multiple_axis(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=0.1, axis=[1, 4], keepdim=True)
        np_res = np.quantile(self.input_data, q=0.1, axis=[1, 4], keepdims=True)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_quantile_with_boundary_q(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=0, axis=3)
        np_res = np.quantile(self.input_data, q=0, axis=3)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_quantile_include_NaN(self):
        input_data = np.random.randn(2, 3, 4)
        input_data[0, 1, 1] = np.nan
        x = paddle.to_tensor(input_data)
        paddle_res = paddle.quantile(x, q=0.35, axis=0)
        self.assertTrue(paddle.isnan(paddle_res[1, 1]))


class TestQuantileMuitlpleQ(unittest.TestCase):
    def setUp(self):
        np.random.seed(678)
        self.input_data = np.random.rand(10, 3, 4, 5, 4)

    def test_quantile(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=[0.3, 0.44], axis=-2)
        np_res = np.quantile(self.input_data, q=[0.3, 0.44], axis=-2)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_quantile_multiple_axis(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(x, q=[0.2, 0.67], axis=[1, -1])
        np_res = np.quantile(self.input_data, q=[0.2, 0.67], axis=[1, -1])
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_quantile_multiple_axis_keepdim(self):
        x = paddle.to_tensor(self.input_data)
        paddle_res = paddle.quantile(
            x, q=[0.1, 0.2, 0.3], axis=[1, 2], keepdim=True)
        np_res = np.quantile(
            self.input_data, q=[0.1, 0.2, 0.3], axis=[1, 2], keepdims=True)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res))


class TestQuantileError(unittest.TestCase):
    def setUp(self):
        self.x = paddle.randn((2, 3, 4))

    def test_errors(self):
        def test_q_range_error_1():
            paddle_res = paddle.quantile(self.x, q=1.5)

        self.assertRaises(ValueError, test_q_range_error_1)

        def test_q_range_error_2():
            paddle_res = paddle.quantile(self.x, q=[0.2, -0.3])

        self.assertRaises(ValueError, test_q_range_error_2)

        def test_q_range_error_3():
            paddle_res = paddle.quantile(self.x, q=[])

        self.assertRaises(ValueError, test_q_range_error_3)

        def test_x_type_error():
            x = [1, 3, 4]
            paddle_res = paddle.quantile(x, q=0.9)

        self.assertRaises(TypeError, test_x_type_error)

        def test_axis_type_error_1():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=0.4)

        self.assertRaises(ValueError, test_axis_type_error_1)

        def test_axis_type_error_2():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[1, 0.4])

        self.assertRaises(ValueError, test_axis_type_error_2)

        def test_axis_value_error_1():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=10)

        self.assertRaises(ValueError, test_axis_value_error_1)

        def test_axis_value_error_2():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[1, -10])

        self.assertRaises(ValueError, test_axis_value_error_2)

        def test_axis_value_error_3():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[])

        self.assertRaises(ValueError, test_axis_value_error_3)


if __name__ == '__main__':
    unittest.main()
