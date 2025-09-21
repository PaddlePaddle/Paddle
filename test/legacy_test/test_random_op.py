#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from utils import dygraph_guard

import paddle


class TestRandomFromToOp(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.from_val = 1
        self.to_val = 10
        self.dtypes = [
            paddle.float32,
            paddle.float64,
            paddle.int32,
            paddle.int64,
            paddle.float16,
            paddle.bfloat16,
        ]

    def test_random_op(self):
        def test_value_range(tensor, min_val=None, max_val=None, dtype=None):
            tensor_np = tensor.numpy()
            if min_val is not None:
                self.assertTrue(np.all(tensor_np >= min_val))
            if max_val is not None:
                self.assertTrue(np.all(tensor_np <= max_val))

        def get_expected_range(dtype):
            if dtype in [paddle.int32, paddle.int64]:
                if dtype == paddle.int32:
                    return 0, 2**31 - 1
                else:  # int64
                    return 0, 2**63 - 1
            else:
                if dtype == paddle.float32:
                    return 0, 2**24
                elif dtype == paddle.float64:
                    return 0, 2**53
                elif dtype == paddle.float16:
                    return 0, 2**11

        def test_random_from_to(dtype, place):
            paddle.set_device(place)
            tensor = paddle.ones(self.shape, dtype=dtype)
            tensor.random_(self.from_val, self.to_val)
            self.assertEqual(tensor.dtype, dtype)

            if dtype != paddle.bfloat16:
                test_value_range(tensor, self.from_val, self.to_val - 1)

        def test_random_from(dtype, place):
            paddle.set_device(place)
            tensor = paddle.ones(self.shape, dtype=dtype)
            tensor.random_(self.from_val)
            self.assertEqual(tensor.dtype, dtype)

            if dtype != paddle.bfloat16:
                test_value_range(tensor, 0, self.from_val - 1)

        def test_random(dtype, place):
            paddle.set_device(place)
            tensor = paddle.ones(self.shape, dtype=dtype)
            tensor.random_()
            self.assertEqual(tensor.dtype, dtype)

            if dtype != paddle.bfloat16:
                min_val, max_val = get_expected_range(dtype)
                test_value_range(tensor, min_val, max_val)

        places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))

        for place in places:
            for dtype in self.dtypes:
                with self.subTest(place=str(place), dtype=str(dtype)):
                    test_random_from_to(dtype, place)
                    test_random_from(dtype, place)
                    test_random(dtype, place)

    def test_random_value_error(self):
        tensor = paddle.ones(self.shape, dtype=paddle.float32)
        with self.assertRaises(ValueError) as context:
            tensor.random_(from_=10, to=5)
        self.assertIn(
            "random_ expects 'from' to be less than 'to'",
            str(context.exception),
        )

    def test_random_update_to(self):
        dtype = paddle.float16
        place = paddle.CPUPlace()
        paddle.set_device(place)

        from_val = 2048
        to_val = 2148
        tensor = paddle.ones([10], dtype=dtype)
        tensor.random_(from_val, to_val)

    def test_pir_random_(self):
        devices = [paddle.device.get_device()]
        if "gpu:" in devices and not paddle.device.is_compiled_with_rocm():
            devices.append("cpu")
        for device in devices:
            with paddle.device.device_guard(device), dygraph_guard():
                st_x = paddle.ones(self.shape, dtype=paddle.float32)

                def func(x):
                    x.random_(self.from_val, self.to_val)
                    return x

                st_func = paddle.jit.to_static(func, full_graph=True)
                st_func(st_x)
                st_out = st_x.numpy()
                self.assertTrue(np.all(st_out >= self.from_val))
                self.assertTrue(np.all(st_out <= self.to_val - 1))


class TestRandomGrad(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.from_val = 0
        self.to_val = 10

    def run_(self, places):
        def test_random_from_to_grad():
            tensor_a = paddle.ones(self.shape)
            tensor_a.stop_gradient = False
            tensor_b = tensor_a * 0.5
            tensor_b.retain_grads()
            tensor_b.random_(self.from_val, self.to_val)
            loss = tensor_b.sum()
            loss.backward()
            random_grad = tensor_b.grad.numpy()
            self.assertTrue((random_grad == 0).all())

        def test_random_grad():
            tensor_a = paddle.ones(self.shape)
            tensor_a.stop_gradient = False
            tensor_b = tensor_a * 0.5
            tensor_b.retain_grads()
            tensor_b.random_()
            loss = tensor_b.sum()
            loss.backward()
            random_grad = tensor_b.grad.numpy()
            self.assertTrue((random_grad == 0).all())

        for place in places:
            paddle.set_device(place)
            test_random_from_to_grad()
            test_random_grad()

    def test_random_from_to_grad(self):
        self.run_(get_places())


if __name__ == '__main__':
    unittest.main()
