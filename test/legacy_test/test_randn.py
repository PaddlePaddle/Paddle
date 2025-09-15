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
from op_test import get_device, get_device_place, is_custom_device
from utils import dygraph_guard

import paddle


class TestTensorCreation(unittest.TestCase):
    def setUp(self):
        self.devices = [paddle.CPUPlace(), "cpu"]
        if paddle.device.is_compiled_with_cuda() or is_custom_device():
            self.devices.append(get_device_place())
            self.devices.append(get_device())
            self.devices.append("gpu:0")
        if paddle.device.is_compiled_with_xpu():
            self.devices.append(paddle.XPUPlace(0))
        if paddle.device.is_compiled_with_ipu():
            self.devices.append(paddle.device.IPUPlace())

        self.requires_grads = [True, False]
        self.dtypes = [None, paddle.float32]
        self.pin_memorys = [False]
        if (
            paddle.device.is_compiled_with_cuda() or is_custom_device()
        ) and not paddle.device.is_compiled_with_rocm():
            self.pin_memorys.append(True)

    @unittest.skipIf(paddle.device.is_compiled_with_xpu(), "skip xpu")
    def test_randn(self):
        types = [
            None,
            "float32",
            paddle.float32,
            "float64",
            paddle.float64,
        ]
        for device, requires_grad, dtype, pin_memory in product(
            self.devices, self.requires_grads, types, self.pin_memorys
        ):
            if (
                device
                not in [
                    get_device(),
                    "gpu:0",
                    get_device_place()
                    if (
                        paddle.device.is_compiled_with_cuda()
                        or is_custom_device()
                    )
                    else None,
                    paddle.XPUPlace(0)
                    if paddle.device.is_compiled_with_xpu()
                    else None,
                ]
                and pin_memory
            ):
                continue  # skip

            with dygraph_guard():
                x = paddle.randn(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                    pin_memory=pin_memory,
                )
                if pin_memory:
                    self.assertTrue("pinned" in str(x.place))

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
                    pin_memory=False,
                ):
                    return paddle.randn(
                        shape,
                        dtype,
                        name,
                        out=out,
                        device=device,
                        requires_grad=requires_grad,
                        pin_memory=pin_memory,
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

                y = paddle.empty_like(x)
                x = paddle.randn(
                    [2],
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                    out=y,
                )
                self.assertEqual(x.data_ptr(), y.data_ptr())


class TestCreationOut(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.rand(3, 4).astype(np.float32)
        self.constant = 3.14

    def test_randn(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.randn(x.shape, out=t)
        self.assertEqual(t.data_ptr(), y.data_ptr())


if __name__ == '__main__':
    unittest.main()
