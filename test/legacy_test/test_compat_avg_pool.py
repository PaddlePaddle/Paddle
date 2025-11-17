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
from op_test import get_places
from test_pool1d_api import avg_pool1D_forward_naive
from test_pool2d_api import avg_pool2D_forward_naive
from test_pool3d_op import avg_pool3D_forward_naive

import paddle


class TestCompatAvgPool1DAPI(unittest.TestCase):
    def setUp(self):
        self.places = get_places()
        self.input_np = np.random.random([2, 3, 32]).astype("float32")

    def run_test_case(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
    ):
        for place in self.places:
            paddle.disable_static(place)
            input_pd = paddle.to_tensor(self.input_np)
            pool_layer = paddle.compat.nn.AvgPool1D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
            result_pd = pool_layer(input_pd)

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size]
            if stride is None:
                stride = kernel_size
            if isinstance(stride, int):
                stride = [stride]
            if isinstance(padding, int):
                padding = [padding]

            result_np = avg_pool1D_forward_naive(
                self.input_np,
                kernel_size,
                stride,
                padding,
                ceil_mode=ceil_mode,
                exclusive=not count_include_pad,
            )
            np.testing.assert_allclose(result_pd.numpy(), result_np, rtol=1e-05)

    @unittest.skipIf(
        paddle.is_compiled_with_xpu(),
        "XPU Kernel has accuracy issue.",
    )
    def test_all_cases(self):
        self.run_test_case(2, 2, 0, False, True)
        self.run_test_case(3, 1, 1, False, True)
        self.run_test_case(3, 2, 1, True, False)
        self.run_test_case(3, None, 0, False, True)

    def test_errors(self):
        with self.assertRaises(TypeError):
            pool = paddle.compat.nn.AvgPool1D(2, exclusive=False, name="test")


class TestCompatAvgPool2DAPI(unittest.TestCase):
    def setUp(self):
        self.places = get_places()
        self.input_np = np.random.random([2, 3, 32, 32]).astype("float32")

    def run_test_case(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    ):
        for place in self.places:
            paddle.disable_static(place)
            input_pd = paddle.to_tensor(self.input_np)
            pool_layer = paddle.compat.nn.AvgPool2D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )
            result_pd = pool_layer(input_pd)

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if stride is None:
                stride = kernel_size
            if isinstance(stride, int):
                stride = [stride, stride]
            if isinstance(padding, int):
                padding = [padding, padding]

            result_np = avg_pool2D_forward_naive(
                self.input_np,
                kernel_size,
                stride,
                padding,
                ceil_mode=ceil_mode,
                exclusive=not count_include_pad,
            )
            if divisor_override is not None:
                result_np = (
                    result_np
                    * (kernel_size[0] * kernel_size[1])
                    / divisor_override
                )
            np.testing.assert_allclose(result_pd.numpy(), result_np, rtol=1e-05)

    @unittest.skipIf(
        paddle.is_compiled_with_xpu(),
        "XPU Kernel has accuracy issue.",
    )
    def test_all_cases(self):
        self.run_test_case(2, 2, 0, False, True, None)
        self.run_test_case([3, 3], [1, 1], [1, 1], False, True, None)
        self.run_test_case(3, 2, 1, True, False, None)
        self.run_test_case(3, None, 0, False, True, None)
        self.run_test_case(3, 2, 1, False, False, 5)

    def test_errors(self):
        with self.assertRaises(TypeError):
            pool = paddle.compat.nn.AvgPool2D(
                2, exclusive=True, data_format="NHWC", name="test"
            )


class TestCompatAvgPool3DAPI(unittest.TestCase):
    def setUp(self):
        self.places = get_places()
        self.input_np = np.random.random([2, 3, 16, 16, 16]).astype("float32")

    def run_test_case(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    ):
        for place in self.places:
            paddle.disable_static(place)
            input_pd = paddle.to_tensor(self.input_np)
            pool_layer = paddle.compat.nn.AvgPool3D(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                divisor_override=divisor_override,
            )
            result_pd = pool_layer(input_pd)

            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size, kernel_size]
            if stride is None:
                stride = kernel_size
            if isinstance(stride, int):
                stride = [stride, stride, stride]
            if isinstance(padding, int):
                padding = [padding, padding, padding]

            result_np = avg_pool3D_forward_naive(
                self.input_np,
                kernel_size,
                stride,
                padding,
                ceil_mode=ceil_mode,
                exclusive=not count_include_pad,
            )
            if divisor_override is not None:
                result_np = (
                    result_np
                    * (kernel_size[0] * kernel_size[1] * kernel_size[2])
                    / divisor_override
                )
            np.testing.assert_allclose(result_pd.numpy(), result_np, rtol=1e-05)

    @unittest.skipIf(
        paddle.is_compiled_with_xpu(),
        "XPU Kernel has accuracy issue.",
    )
    def test_all_cases(self):
        self.run_test_case(2, 2, 0, False, True, None)
        self.run_test_case([3, 3, 3], [1, 1, 1], [1, 1, 1], False, True, None)
        self.run_test_case(3, 2, 1, True, False, None)
        self.run_test_case(3, None, 0, False, True, None)
        self.run_test_case(3, 2, 1, False, False, 5)

    def test_errors(self):
        with self.assertRaises(TypeError):
            pool = paddle.compat.nn.AvgPool3D(
                2, exclusive=True, data_format="NDHWC", name="test"
            )


if __name__ == '__main__':
    unittest.main()
