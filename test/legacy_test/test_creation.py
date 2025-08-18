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
from itertools import product

import numpy as np
from utils import dygraph_guard

import paddle


class TestTensorCreation(unittest.TestCase):
    def setUp(self):
        self.devices = [paddle.CPUPlace(), "cpu"]
        if paddle.device.is_compiled_with_cuda():
            self.devices.append(paddle.CUDAPlace(0))
            self.devices.append("gpu")
            self.devices.append("gpu:0")
        if paddle.device.is_compiled_with_xpu():
            self.devices.append(paddle.device.XPUPlace(0))
        if paddle.device.is_compiled_with_ipu():
            self.devices.append(paddle.device.IPUPlace())

        self.requires_grads = [True, False]
        self.dtypes = ["float32", paddle.float32, "int32", paddle.int32]

    def test_ones(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.ones(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.ones, full_graph=True, backend=None
                )
                x = st_f(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_zeros(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.zeros(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.zeros, full_graph=True, backend=None
                )
                x = st_f(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_full(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.full(
                    [2],
                    fill_value=3.14,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.full, full_graph=True, backend=None
                )
                x = st_f(
                    [2],
                    fill_value=3.14,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_empty(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.empty(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.empty, full_graph=True, backend=None
                )
                x = st_f(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_eye(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.eye(
                    3,
                    3,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.eye, full_graph=True, backend=None
                )
                x = st_f(
                    3,
                    3,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if (
                    isinstance(device, paddle.framework.core.Place)
                    # skip xpu for unknown reason
                    and not isinstance(device, paddle.framework.core.XPUPlace)
                ):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_ones_like(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.ones_like(
                    paddle.randn([2, 2]),
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.ones_like, full_graph=True, backend=None
                )
                x = st_f(
                    paddle.randn([2, 2]),
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_zeros_like(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.zeros_like(
                    paddle.randn([2, 2]),
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.zeros_like, full_graph=True, backend=None
                )
                x = st_f(
                    paddle.randn([2, 2]),
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_full_like(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.full_like(
                    paddle.randn([2, 2]),
                    3.14,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.full_like, full_graph=True, backend=None
                )
                x = st_f(
                    paddle.randn([2, 2]),
                    3.14,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_empty_like(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.empty_like(
                    paddle.randn([2, 2]),
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.empty_like, full_graph=True, backend=None
                )
                x = st_f(
                    paddle.randn([2, 2]),
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_arange(self):
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.arange(
                    3.14,
                    5.9,
                    1.11,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                st_f = paddle.jit.to_static(
                    paddle.arange, full_graph=True, backend=None
                )
                x = st_f(
                    3.14,
                    5.9,
                    1.11,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if not paddle.device.is_compiled_with_xpu() and isinstance(
                    device, paddle.framework.core.Place
                ):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)


class TestTensorPatchMethod(unittest.TestCase):
    def setUp(self):
        self.devices = [None, paddle.CPUPlace(), "cpu"]
        if paddle.device.is_compiled_with_cuda():
            self.devices.append(paddle.CUDAPlace(0))
            self.devices.append("gpu")
            self.devices.append("gpu:0")
        if paddle.device.is_compiled_with_xpu():
            self.devices.append(paddle.device.XPUPlace(0))
        if paddle.device.is_compiled_with_ipu():
            self.devices.append(paddle.device.IPUPlace())

        self.requires_grads = [True, False]
        self.shapes = [
            [4, 4],
        ]
        self.dtypes = ["float32", paddle.float32, "int32", paddle.int32]

    def test_Tensor_new_ones(self):
        for shape, device, requires_grad, dtype in product(
            self.shapes, self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.ones(
                    [1],
                ).new_ones(
                    shape,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

                def new_ones(x, shape, dtype, requires_grad, device):
                    return x.new_ones(
                        shape,
                        dtype=dtype,
                        requires_grad=requires_grad,
                        device=device,
                    )

                st_f = paddle.jit.to_static(
                    new_ones, full_graph=True, backend=None
                )
                x = st_f(
                    paddle.randn([1]),
                    shape,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_Tensor_new_zeros(self):
        for shape, device, requires_grad, dtype in product(
            self.shapes, self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.zeros(
                    [1],
                ).new_zeros(
                    shape,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

                def new_zeros(x, shape, dtype, requires_grad, device):
                    return x.new_zeros(
                        shape,
                        dtype=dtype,
                        requires_grad=requires_grad,
                        device=device,
                    )

                st_f = paddle.jit.to_static(
                    new_zeros, full_graph=True, backend=None
                )
                x = st_f(
                    paddle.randn([1]),
                    shape,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_Tensor_new_full(self):
        for shape, device, requires_grad, dtype in product(
            self.shapes, self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.full(
                    [1],
                    3.14,
                ).new_full(
                    shape,
                    2.0,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                np.testing.assert_allclose(
                    x.numpy(), paddle.full(shape, 2.0).numpy(), 1e-6, 1e-6
                )

                def new_full(
                    x, shape, fill_value, dtype, requires_grad, device
                ):
                    return x.new_full(
                        shape,
                        fill_value,
                        dtype=dtype,
                        requires_grad=requires_grad,
                        device=device,
                    )

                st_f = paddle.jit.to_static(
                    new_full, full_graph=True, backend=None
                )
                x = st_f(
                    paddle.randn([1]),
                    shape,
                    2.0,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)
                np.testing.assert_allclose(
                    x.numpy(), paddle.full(shape, 2.0).numpy(), 1e-6, 1e-6
                )

    def test_Tensor_new_empty(self):
        for shape, device, requires_grad, dtype in product(
            self.shapes, self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                x = paddle.empty(
                    [1],
                ).new_empty(
                    shape,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

                def new_empty(x, shape, dtype, requires_grad, device):
                    return x.new_empty(
                        shape,
                        dtype=dtype,
                        requires_grad=requires_grad,
                        device=device,
                    )

                st_f = paddle.jit.to_static(
                    new_empty, full_graph=True, backend=None
                )
                x = st_f(
                    paddle.randn([1]),
                    shape,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)


if __name__ == '__main__':
    unittest.main()
