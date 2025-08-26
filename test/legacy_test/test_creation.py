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
from paddle.static import InputSpec


class TestTensorCreation(unittest.TestCase):
    def setUp(self):
        self.devices = [paddle.CPUPlace(), "cpu"]
        if paddle.device.is_compiled_with_cuda():
            self.devices.append(paddle.CUDAPlace(0))
            self.devices.append("gpu")
            self.devices.append("gpu:0")
        if paddle.device.is_compiled_with_xpu():
            self.devices.append(paddle.XPUPlace(0))
        if paddle.device.is_compiled_with_ipu():
            self.devices.append(paddle.device.IPUPlace())

        self.requires_grads = [True, False]
        self.dtypes = [None, "float32", paddle.float32, "int32", paddle.int32]

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

                def wrapped_ones(
                    shape,
                    dtype=None,
                    name=None,
                    *,
                    out=None,
                    device=None,
                    requires_grad=False,
                ):
                    return paddle.ones(
                        shape,
                        dtype,
                        name,
                        out=out,
                        device=device,
                        requires_grad=requires_grad,
                    )

                st_f = paddle.jit.to_static(
                    wrapped_ones, full_graph=True, backend=None
                )
                x = st_f(
                    [2],
                    out=None,
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

                def wrapped_zeros(
                    shape,
                    dtype=None,
                    name=None,
                    *,
                    out=None,
                    device=None,
                    requires_grad=False,
                ):
                    return paddle.zeros(
                        shape,
                        dtype,
                        name,
                        out=out,
                        device=device,
                        requires_grad=requires_grad,
                    )

                st_f = paddle.jit.to_static(
                    wrapped_zeros, full_graph=True, backend=None
                )
                x = st_f(
                    [2],
                    out=None,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

    def test_randn(self):
        types = [
            None,
            "float32",
            paddle.float32,
            "float64",
            paddle.float64,
        ]
        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, types
        ):
            with dygraph_guard():
                x = paddle.randn(
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

                def wrapped_randn(
                    shape,
                    dtype=None,
                    name=None,
                    *,
                    out=None,
                    device=None,
                    requires_grad=False,
                ):
                    return paddle.randn(
                        shape,
                        dtype,
                        name,
                        out=out,
                        device=device,
                        requires_grad=requires_grad,
                    )

                st_f = paddle.jit.to_static(
                    wrapped_randn, full_graph=True, backend=None
                )
                x = st_f(
                    [2],
                    out=None,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

                y = paddle.empty_like(x)
                x = paddle.randn(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                    out=y,
                )
                self.assertEqual(x.data_ptr(), y.data_ptr())

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
        # empty has extra arg: pin_memory
        pin_memorys = [False]
        if (
            paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_xpu()
        ):
            pin_memorys.append(True)
        for device, requires_grad, dtype, pin_memory in product(
            self.devices,
            self.requires_grads,
            self.dtypes,
            pin_memorys,
        ):
            if device not in [
                "gpu",
                "gpu:0",
                paddle.CUDAPlace(0)
                if paddle.device.is_compiled_with_cuda()
                else None,
                paddle.XPUPlace(0)
                if paddle.device.is_compiled_with_xpu()
                else None,
            ]:
                pin_memory = False
            with dygraph_guard():
                x = paddle.empty(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                    pin_memory=pin_memory,
                )
                if (
                    isinstance(device, paddle.framework.core.Place)
                    and not pin_memory
                ):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

                def wrapped_empty(
                    shape,
                    dtype=None,
                    name=None,
                    *,
                    out=None,
                    device=None,
                    requires_grad=False,
                    pin_memory=False,
                ):
                    return paddle.empty(
                        shape,
                        dtype,
                        name,
                        out=out,
                        device=device,
                        requires_grad=requires_grad,
                        pin_memory=pin_memory,
                    )

                st_f = paddle.jit.to_static(
                    wrapped_empty, full_graph=True, backend=None
                )
                x = st_f(
                    [2],
                    out=None,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                    pin_memory=pin_memory,
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

    def test_range(self):
        def range_manual(start, end, step, dtype, device, requires_grad):
            if end is None:
                end = start
                start = 0
            if dtype is None:
                dtype = paddle.get_default_dtype()
            size_ = int(np.abs(np.trunc((end - start) / step))) + 1
            out = paddle.empty([size_])

            for i in range(size_):
                out[i] = start + i * step

            out = out.to(device=device, dtype=dtype)
            out.stop_gradient = not requires_grad
            return out

        for device, requires_grad, dtype in product(
            self.devices, self.requires_grads, self.dtypes
        ):
            with dygraph_guard():
                for start, end, step in [
                    (0, 5, 1),
                    (2, 7, 2),
                    (5, None, 1),
                    (0, 1, 0.1),
                    (-1.1, -3.7, -0.09),
                    (-1.1, -3.7, -0.10001),
                    (-1.1, -3.7, -0.9999),
                ]:
                    if np.abs(step) < 1 and dtype in [
                        paddle.int32,
                        "int32",
                        paddle.int64,
                        "int64",
                    ]:
                        with self.assertRaises(ValueError):
                            x = paddle.range(
                                start,
                                end,
                                step,
                                dtype=dtype,
                                device=device,
                                requires_grad=requires_grad,
                            )
                            continue
                    else:
                        x = paddle.range(
                            start,
                            end,
                            step,
                            dtype=dtype,
                            device=device,
                            requires_grad=requires_grad,
                        )
                        x_ref = range_manual(
                            start, end, step, dtype, device, requires_grad
                        )
                        self.assertEqual(x.place, x_ref.place)
                        self.assertEqual(x.dtype, x_ref.dtype)
                        self.assertEqual(x.stop_gradient, x_ref.stop_gradient)
                        np.testing.assert_allclose(
                            x.numpy(),
                            x_ref.numpy(),
                            1e-6,
                            1e-6,
                            err_msg=f"[FAILED] wrong result when testing: range({start},{end},{step})",
                        )

                        def wrapped_range(
                            start, end, step, dtype, device, requires_grad
                        ):
                            return paddle.range(
                                start,
                                end,
                                step,
                                dtype,
                                device=device,
                                requires_grad=requires_grad,
                            )

                        st_f = paddle.jit.to_static(
                            wrapped_range, full_graph=True, backend=None
                        )
                        x = st_f(
                            start,
                            end,
                            step,
                            dtype,
                            device=device,
                            requires_grad=requires_grad,
                        )
                        if (
                            isinstance(device, paddle.framework.core.Place)
                            # skip xpu for unknown reason
                            and not isinstance(
                                device, paddle.framework.core.XPUPlace
                            )
                        ):
                            self.assertEqual(x.place, x_ref.place)
                        self.assertEqual(x.dtype, x_ref.dtype)
                        self.assertEqual(x.stop_gradient, x_ref.stop_gradient)
                        np.testing.assert_allclose(
                            x.numpy(),
                            x_ref.numpy(),
                            1e-6,
                            1e-6,
                            err_msg=f"[FAILED] wrong result when testing: range({start},{end},{step})",
                        )

                        def wrapped_range(start, end, step):
                            return paddle.range(
                                start,
                                end,
                                step,
                                dtype,
                                device=device,
                                requires_grad=requires_grad,
                            )

                        if end is None:
                            st_f = paddle.jit.to_static(
                                wrapped_range,
                                input_spec=[
                                    InputSpec([-1]),
                                    None,
                                    InputSpec([-1]),
                                ],
                                full_graph=True,
                                backend=None,
                            )
                        else:
                            st_f = paddle.jit.to_static(
                                wrapped_range,
                                input_spec=[
                                    InputSpec([-1]),
                                    InputSpec([-1]),
                                    InputSpec([-1]),
                                ],
                                full_graph=True,
                                backend=None,
                            )

                        x = st_f(
                            paddle.to_tensor(start),
                            paddle.to_tensor(end) if end is not None else None,
                            paddle.to_tensor(step),
                        )
                        if (
                            isinstance(device, paddle.framework.core.Place)
                            # skip xpu for unknown reason
                            and not isinstance(
                                device, paddle.framework.core.XPUPlace
                            )
                        ):
                            self.assertEqual(x.place, x_ref.place)
                        self.assertEqual(x.dtype, x_ref.dtype)
                        self.assertEqual(x.stop_gradient, x_ref.stop_gradient)
                        np.testing.assert_allclose(
                            x.numpy(),
                            x_ref.numpy(),
                            1e-6,
                            1e-6,
                            err_msg=f"[FAILED] wrong result when testing: range({start},{end},{step})",
                        )


class TestTensorPatchMethod(unittest.TestCase):
    def setUp(self):
        self.devices = [None, paddle.CPUPlace(), "cpu"]
        if paddle.device.is_compiled_with_cuda():
            self.devices.append(paddle.CUDAPlace(0))
            self.devices.append("gpu")
            self.devices.append("gpu:0")
        if paddle.device.is_compiled_with_xpu():
            self.devices.append(paddle.XPUPlace(0))
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
        # empty has extra arg: pin_memory
        pin_memorys = [False]
        if (
            paddle.device.is_compiled_with_cuda()
            or paddle.device.is_compiled_with_xpu()
        ):
            pin_memorys.append(True)
        for shape, device, requires_grad, dtype, pin_memory in product(
            self.shapes,
            self.devices,
            self.requires_grads,
            self.dtypes,
            pin_memorys,
        ):
            if device not in [
                "gpu",
                "gpu:0",
                paddle.CUDAPlace(0)
                if paddle.device.is_compiled_with_cuda()
                else None,
                paddle.XPUPlace(0)
                if paddle.device.is_compiled_with_xpu()
                else None,
            ]:
                pin_memory = False
            with dygraph_guard():
                x = paddle.empty(
                    [1],
                ).new_empty(
                    shape,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                    pin_memory=pin_memory,
                )
                if (
                    isinstance(device, paddle.framework.core.Place)
                    and not pin_memory
                ):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)

                def new_empty(
                    x, shape, dtype, requires_grad, device, pin_memory
                ):
                    return x.new_empty(
                        shape,
                        dtype=dtype,
                        requires_grad=requires_grad,
                        device=device,
                        pin_memory=pin_memory,
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
                    pin_memory=pin_memory,
                )
                if isinstance(device, paddle.framework.core.Place):
                    self.assertEqual(x.place, device)
                self.assertEqual(x.stop_gradient, not requires_grad)
                if isinstance(dtype, paddle.dtype):
                    self.assertEqual(x.dtype, dtype)


class TestCreationOut(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.rand(3, 4).astype(np.float32)
        self.constant = 3.14

    def test_full(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.full(x.shape, self.constant, out=t)
        np.testing.assert_allclose(t.numpy(), np.full(x.shape, self.constant))
        np.testing.assert_allclose(y.numpy(), np.full(x.shape, self.constant))
        self.assertEqual(t.data_ptr(), y.data_ptr())

    def test_ones(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.ones(x.shape, out=t)
        np.testing.assert_allclose(t.numpy(), np.ones(x.shape))
        np.testing.assert_allclose(y.numpy(), np.ones(x.shape))
        self.assertEqual(t.data_ptr(), y.data_ptr())

    def test_zeros(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.zeros(x.shape, out=t)
        np.testing.assert_allclose(t.numpy(), np.zeros(x.shape))
        np.testing.assert_allclose(y.numpy(), np.zeros(x.shape))
        self.assertEqual(t.data_ptr(), y.data_ptr())

    def test_randn(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.randn(x.shape, out=t)
        self.assertEqual(t.data_ptr(), y.data_ptr())

    def test_empty(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.empty(x.shape, out=t)
        self.assertEqual(t.data_ptr(), y.data_ptr())

    @unittest.skipIf(
        paddle.device.is_compiled_with_cuda()
        and paddle.device.is_compiled_with_rocm(),
        reason="Skip for paddle.eye in dcu is not correct",
    )
    def test_eye(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.eye(x.shape[0], x.shape[1], out=t)
        np.testing.assert_allclose(t.numpy(), np.eye(x.shape[0], x.shape[1]))
        np.testing.assert_allclose(y.numpy(), np.eye(x.shape[0], x.shape[1]))
        self.assertEqual(t.data_ptr(), y.data_ptr())

    def test_arange(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.arange(-1.1, 3.4, 0.1, out=t)
        np.testing.assert_allclose(
            t.numpy(), np.arange(-1.1, 3.4, 0.1), 1e-6, 1e-6
        )
        np.testing.assert_allclose(
            y.numpy(), np.arange(-1.1, 3.4, 0.1), 1e-6, 1e-6
        )
        self.assertEqual(t.data_ptr(), y.data_ptr())

    def test_range(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.range(-1.1, 3.4, 0.1, out=t)
        self.assertEqual(t.data_ptr(), y.data_ptr())


if __name__ == '__main__':
    unittest.main()
