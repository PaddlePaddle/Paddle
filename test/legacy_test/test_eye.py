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

    def test_eye(self):
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
                x = paddle.eye(
                    3,
                    3,
                    dtype=dtype,
                    requires_grad=requires_grad,
                    device=device,
                    pin_memory=pin_memory,
                )
                if pin_memory:
                    self.assertTrue("pinned" in str(x.place))

                if (
                    isinstance(device, paddle.framework.core.Place)
                    and not pin_memory
                ):
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


class TestCreationOut(unittest.TestCase):
    def setUp(self):
        self.x_np = np.random.rand(3, 4).astype(np.float32)
        self.constant = 3.14

    @unittest.skipIf(
        paddle.device.is_compiled_with_cuda()
        and paddle.device.is_compiled_with_rocm(),
        reason="Skip for paddle.eye in dcu is not correct",
    )
    def test_eye(self):
        x = paddle.randn([2, 2])
        t = paddle.empty_like(x)
        y = paddle.eye(x.shape[0], x.shape[1], out=t, requires_grad=True)
        np.testing.assert_allclose(t.numpy(), np.eye(x.shape[0], x.shape[1]))
        np.testing.assert_allclose(y.numpy(), np.eye(x.shape[0], x.shape[1]))
        self.assertEqual(t.data_ptr(), y.data_ptr())
        self.assertEqual(y.stop_gradient, False)
        self.assertEqual(t.stop_gradient, False)


if __name__ == '__main__':
    unittest.main()
