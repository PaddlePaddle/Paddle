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
        self.dtypes = [None, paddle.float32]
        self.pin_memorys = [False]
        if (
            paddle.device.is_compiled_with_cuda()
            and not paddle.device.is_compiled_with_rocm()
        ):
            self.pin_memorys.append(True)

    def test_arange(self):
        for device, requires_grad, dtype, pin_memory in product(
            self.devices, self.requires_grads, self.dtypes, self.pin_memorys
        ):
            if (
                device
                not in [
                    "gpu",
                    "gpu:0",
                    paddle.CUDAPlace(0)
                    if paddle.device.is_compiled_with_cuda()
                    else None,
                    paddle.XPUPlace(0)
                    if paddle.device.is_compiled_with_xpu()
                    else None,
                ]
                and pin_memory
            ):
                continue  # skip

            with dygraph_guard():
                x = paddle.arange(
                    3.14,
                    5.9,
                    1.11,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                    pin_memory=pin_memory,
                )
                if pin_memory:
                    self.assertTrue("pinned" in str(x.place))
                if (
                    not paddle.device.is_compiled_with_xpu()
                    and isinstance(device, paddle.framework.core.Place)
                    and not pin_memory
                ):
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


class TestCreationOut(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.rand(3, 4).astype(np.float32)
        self.constant = 3.14

    def test_arange(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.arange(-1.1, 3.4, 0.1, out=t, requires_grad=True)
        np.testing.assert_allclose(
            t.numpy(), np.arange(-1.1, 3.4, 0.1), 1e-6, 1e-6
        )
        np.testing.assert_allclose(
            y.numpy(), np.arange(-1.1, 3.4, 0.1), 1e-6, 1e-6
        )
        self.assertEqual(t.data_ptr(), y.data_ptr())
        self.assertEqual(y.stop_gradient, False)
        self.assertEqual(t.stop_gradient, False)

    def test_range(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.range(-1.1, 3.4, 0.1, out=t, requires_grad=True)
        self.assertEqual(t.data_ptr(), y.data_ptr())
        self.assertEqual(y.stop_gradient, False)
        self.assertEqual(t.stop_gradient, False)


if __name__ == '__main__':
    unittest.main()
