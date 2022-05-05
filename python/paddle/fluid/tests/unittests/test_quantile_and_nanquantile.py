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

API_list = [(paddle.quantile, np.quantile),
            (paddle.nanquantile, np.nanquantile)]


class TestQuantileAndNanquantile(unittest.TestCase):
    """
        This class is used for numerical precision testing. If there is
        a corresponding numpy API, the precision comparison can be performed directly.
        Otherwise, it needs to be verified by numpy implementated function.
    """

    def setUp(self):
        self.input_data = np.random.rand(4, 7, 6)

    # Test correctness when q and axis are set.
    def test_single_q(self):
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.5, axis=2)
            np_res = res_func(inp, q=0.5, axis=2)
            self.assertTrue(np.allclose(paddle_res.numpy(), np_res))
            inp[0, 1, 2] = np.nan

    # Test correctness for default axis.
    def test_with_no_axis(self):
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.35)
            np_res = res_func(inp, q=0.35)
            self.assertTrue(np.allclose(paddle_res.numpy(), np_res))
            inp[0, 2, 1] = np.nan
            inp[0, 1, 2] = np.nan

    # Test correctness for multiple axis.
    def test_with_multi_axis(self):
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.75, axis=[0, 2])
            np_res = res_func(inp, q=0.75, axis=[0, 2])
            self.assertTrue(np.allclose(paddle_res.numpy(), np_res))
            inp[0, 5, 3] = np.nan
            inp[0, 6, 2] = np.nan

    # Test correctness when keepdim is set.
    def test_with_keepdim(self):
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.35, axis=2, keepdim=True)
            np_res = res_func(inp, q=0.35, axis=2, keepdims=True)
            self.assertTrue(np.allclose(paddle_res.numpy(), np_res))
            inp[0, 3, 4] = np.nan

    # Test correctness when all parameters are set.
    def test_with_keepdim_and_multiple_axis(self):
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0.1, axis=[1, 2], keepdim=True)
            np_res = res_func(inp, q=0.1, axis=[1, 2], keepdims=True)
            self.assertTrue(np.allclose(paddle_res.numpy(), np_res))
            inp[0, 6, 3] = np.nan

    # Test correctness when q = 0.
    def test_with_boundary_q(self):
        inp = self.input_data
        for (func, res_func) in API_list:
            x = paddle.to_tensor(inp)
            paddle_res = func(x, q=0, axis=1)
            np_res = res_func(inp, q=0, axis=1)
            self.assertTrue(np.allclose(paddle_res.numpy(), np_res))
            inp[0, 2, 5] = np.nan

    # Test correctness when input includes NaN.
    def test_quantile_include_NaN(self):
        input_data = np.random.randn(2, 3, 4)
        input_data[0, 1, 1] = np.nan
        x = paddle.to_tensor(input_data)
        paddle_res = paddle.quantile(x, q=0.35, axis=0)
        np_res = np.quantile(input_data, q=0.35, axis=0)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res, equal_nan=True))

    # Test correctness when input filled with NaN.
    def test_nanquantile_all_NaN(self):
        input_data = np.full(shape=[2, 3], fill_value=np.nan)
        input_data[0, 2] = 0
        x = paddle.to_tensor(input_data)
        paddle_res = paddle.nanquantile(x, q=0.35, axis=0)
        np_res = np.nanquantile(input_data, q=0.35, axis=0)
        self.assertTrue(np.allclose(paddle_res.numpy(), np_res, equal_nan=True))


class TestMuitlpleQ(unittest.TestCase):
    """
        This class is used to test multiple input of q.
    """

    def setUp(self):
        self.input_data = np.random.rand(5, 3, 4)

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


class TestError(unittest.TestCase):
    """
        This class is used to test that exceptions are thrown correctly.
        Validity of all parameter values and types should be considered.
    """

    def setUp(self):
        self.x = paddle.randn((2, 3, 4))

    def test_errors(self):
        # Test error when q > 1
        def test_q_range_error_1():
            paddle_res = paddle.quantile(self.x, q=1.5)

        self.assertRaises(ValueError, test_q_range_error_1)

        # Test error when q < 0
        def test_q_range_error_2():
            paddle_res = paddle.quantile(self.x, q=[0.2, -0.3])

        self.assertRaises(ValueError, test_q_range_error_2)

        # Test error with no valid q
        def test_q_range_error_3():
            paddle_res = paddle.quantile(self.x, q=[])

        self.assertRaises(ValueError, test_q_range_error_3)

        # Test error when x is not Tensor
        def test_x_type_error():
            x = [1, 3, 4]
            paddle_res = paddle.quantile(x, q=0.9)

        self.assertRaises(TypeError, test_x_type_error)

        # Test error when scalar axis is not int
        def test_axis_type_error_1():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=0.4)

        self.assertRaises(ValueError, test_axis_type_error_1)

        # Test error when axis in List is not int
        def test_axis_type_error_2():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[1, 0.4])

        self.assertRaises(ValueError, test_axis_type_error_2)

        # Test error when axis not in [-D, D)
        def test_axis_value_error_1():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=10)

        self.assertRaises(ValueError, test_axis_value_error_1)

        # Test error when axis not in [-D, D)
        def test_axis_value_error_2():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[1, -10])

        self.assertRaises(ValueError, test_axis_value_error_2)

        # Test error with no valid axis
        def test_axis_value_error_3():
            paddle_res = paddle.quantile(self.x, q=0.4, axis=[])

        self.assertRaises(ValueError, test_axis_value_error_3)


class TestQuantileRuntime(unittest.TestCase):
    """
        This class is used to test the API could run correctly with
        different devices, different data types, and dygraph/static mode.
    """

    def setUp(self):
        self.input_data = np.random.rand(4, 7)
        self.dtypes = ['float32', 'float64']
        self.devices = ['cpu']
        if paddle.device.is_compiled_with_cuda():
            self.devices.append('gpu')

    def test_dygraph(self):
        paddle.disable_static()
        for (func, res_func) in API_list:
            for device in self.devices:
                # Check different devices
                paddle.set_device(device)
                for dtype in self.dtypes:
                    # Check different dtypes
                    np_input_data = self.input_data.astype(dtype)
                    x = paddle.to_tensor(np_input_data, dtype=dtype)
                    paddle_res = func(x, q=0.5, axis=1)
                    np_res = res_func(np_input_data, q=0.5, axis=1)
                    self.assertTrue(np.allclose(paddle_res.numpy(), np_res))

    def test_static(self):
        paddle.enable_static()
        for (func, res_func) in API_list:
            for device in self.devices:
                x = paddle.static.data(
                    name="x", shape=self.input_data.shape, dtype=paddle.float32)
                x_fp64 = paddle.static.data(
                    name="x_fp64",
                    shape=self.input_data.shape,
                    dtype=paddle.float64)

                results = func(x, q=0.5, axis=1)
                np_input_data = self.input_data.astype('float32')
                results_fp64 = func(x_fp64, q=0.5, axis=1)
                np_input_data_fp64 = self.input_data.astype('float64')

                exe = paddle.static.Executor(device)
                paddle_res, paddle_res_fp64 = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": np_input_data,
                          "x_fp64": np_input_data_fp64},
                    fetch_list=[results, results_fp64])
                np_res = res_func(np_input_data, q=0.5, axis=1)
                np_res_fp64 = res_func(np_input_data_fp64, q=0.5, axis=1)
                self.assertTrue(
                    np.allclose(paddle_res, np_res) and
                    np.allclose(paddle_res_fp64, np_res_fp64))


if __name__ == '__main__':
    unittest.main()
