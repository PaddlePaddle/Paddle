# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


@unittest.skipIf(
    paddle.device.is_compiled_with_cuda()
    and paddle.device.is_compiled_with_rocm(),
    reason="Skip dcu for error occurs when running on dcu",
)
class TestSlogDet(unittest.TestCase):
    def setUp(self) -> None:
        self.shapes = [
            [2, 2, 5, 5],
            [10, 10],
            [0, 5, 5],
            [0, 0, 0],
            [3, 3, 5, 5],
            [6, 5, 5],
        ]
        self.dtypes = [
            "float32",
            "float64",
            "complex64",
            "complex128",
        ]

    def compiled_with_cuda(self):
        return (
            paddle.device.is_compiled_with_cuda()
            and not paddle.device.is_compiled_with_rocm()
        )

    def slogdet_backward(self, x, _, grad_logabsdet):
        x_inv_T = np.swapaxes(np.linalg.inv(x).conj(), -1, -2)
        grad_x = grad_logabsdet * x_inv_T
        return grad_x

    def test_compat_slogdet(self):
        devices = [paddle.device.get_device()]
        if (
            any(device.startswith("gpu:") for device in devices)
            and not paddle.device.is_compiled_with_rocm()
        ):
            devices.append("cpu")
        for device in devices:
            with paddle.device.device_guard(device), dygraph_guard():
                for shape, dtype in product(self.shapes, self.dtypes):
                    err_msg = f"shape = {shape}, dtype = {dtype}"

                    # test eager
                    x = paddle.randn(shape, dtype)
                    x.stop_gradient = False
                    out = paddle.compat.slogdet(x)
                    self.assertTrue(hasattr(out, "sign"))
                    self.assertTrue(hasattr(out, "logabsdet"))
                    sign, logabsdet = out
                    self.assertEqual(sign.dtype, x.dtype)
                    self.assertFalse(logabsdet.is_complex())
                    logdet_grad = paddle.randn_like(logabsdet)
                    sign_ref, logdet_ref = np.linalg.slogdet(x.numpy())

                    np.testing.assert_allclose(
                        sign.numpy(), sign_ref, 1e-5, 1e-5, err_msg=err_msg
                    )
                    np.testing.assert_allclose(
                        logabsdet.numpy(),
                        logdet_ref,
                        1e-5,
                        1e-5,
                        err_msg=err_msg,
                    )

                    (x_grad,) = paddle.grad(logabsdet, x, logdet_grad)
                    x_grad_ref = self.slogdet_backward(
                        x.numpy(),
                        sign.numpy(),
                        logdet_grad.numpy()[..., None, None],
                    )
                    np.testing.assert_allclose(
                        x_grad.numpy(), x_grad_ref, 1e-4, 1e-4, err_msg=err_msg
                    )

                    # test pir
                    st_f = paddle.jit.to_static(
                        paddle.compat.slogdet,
                        full_graph=True,
                    )
                    sign, logabsdet = st_f(x)
                    self.assertTrue(hasattr(out, "sign"))
                    self.assertTrue(hasattr(out, "logabsdet"))
                    self.assertEqual(sign.dtype, x.dtype)
                    self.assertFalse(logabsdet.is_complex())

                    np.testing.assert_allclose(
                        sign.numpy(), sign_ref, 1e-5, 1e-5, err_msg=err_msg
                    )
                    np.testing.assert_allclose(
                        logabsdet.numpy(),
                        logdet_ref,
                        1e-5,
                        1e-5,
                        err_msg=err_msg,
                    )

                    # test pir + dynamic shape
                    st_f = paddle.jit.to_static(
                        paddle.compat.slogdet,
                        full_graph=True,
                        input_spec=[
                            paddle.static.InputSpec(
                                shape=[-1] * len(shape), dtype=dtype
                            ),
                        ],
                    )
                    sign, logabsdet = st_f(x)
                    self.assertTrue(hasattr(out, "sign"))
                    self.assertTrue(hasattr(out, "logabsdet"))
                    self.assertEqual(sign.dtype, x.dtype)
                    self.assertFalse(logabsdet.is_complex())

                    np.testing.assert_allclose(
                        sign.numpy(), sign_ref, 1e-5, 1e-5, err_msg=err_msg
                    )
                    np.testing.assert_allclose(
                        logabsdet.numpy(),
                        logdet_ref,
                        1e-5,
                        1e-5,
                        err_msg=err_msg,
                    )

    def test_error(self):
        x = paddle.randn([5], "float32")
        with self.assertRaises(ValueError):
            sign, logabsdet = paddle.compat.slogdet(x)

    def test_out(self):
        x = paddle.randn([5, 5], "float32")
        sign_, logabsdet_ = paddle.randn([]), paddle.randn([])

        sign, logabsdet = paddle.compat.slogdet(x, out=(sign_, logabsdet_))

        # skip until multiple outputs are supported for out
        # self.assertEqual(sign_.data_ptr(), sign.data_ptr())
        # self.assertEqual(logabsdet_.data_ptr(), logabsdet.data_ptr())

    def test_singular_matrix(self):
        x = paddle.to_tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
            ],
            dtype="float32",
        )
        sign, logabsdet = paddle.compat.slogdet(x)
        self.assertEqual(sign.item(), 0)
        self.assertEqual(logabsdet.item(), -np.inf)

        if self.compiled_with_cuda():
            with paddle.device.device_guard("cpu"):
                x = paddle.to_tensor(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2],
                    ],
                    dtype="float32",
                )
                sign, logabsdet = paddle.compat.slogdet(x)
                self.assertEqual(sign.item(), 0)
                self.assertEqual(logabsdet.item(), -np.inf)

    def test_invertible_matrix_backward(self):
        with paddle.device.device_guard("cpu"):
            x = paddle.to_tensor(
                [
                    [0.5, 0, 0],
                    [0, 0.6, 0],
                    [0, 0, 0.7],
                ],
                dtype="float32",
                place="cpu",
                stop_gradient=False,
            )
            out = paddle.compat.slogdet(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            sign, logabsdet = out
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            logdet_grad = paddle.randn_like(logabsdet)
            sign_ref, logdet_ref = np.linalg.slogdet(x.numpy())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

            (x_grad,) = paddle.grad(logabsdet, x, logdet_grad)
            x_grad_ref = self.slogdet_backward(
                x.numpy(),
                sign.numpy(),
                logdet_grad.numpy()[..., None, None],
            )
            np.testing.assert_allclose(x_grad.numpy(), x_grad_ref, 1e-5, 1e-5)

            # test pir + dynamic shape
            st_f = paddle.jit.to_static(
                paddle.compat.slogdet,
                full_graph=True,
                input_spec=[
                    paddle.static.InputSpec(shape=[-1, -1], dtype="float32"),
                ],
            )
            sign, logabsdet = st_f(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

    def test_batched_invertible_matrix_backward(self):
        def run():
            x = paddle.to_tensor(
                [
                    [
                        [0.5, 0, 0],
                        [0, 0.6, 0],
                        [0, 0, 0.7],
                    ],
                    [
                        [0.2, 0, 0],
                        [0, 0.3, 0],
                        [0, 0, 0.4],
                    ],
                ],
                dtype="float32",
                place="cpu",
                stop_gradient=False,
            )
            out = paddle.compat.slogdet(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            sign, logabsdet = out
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            logdet_grad = paddle.randn_like(logabsdet)
            sign_ref, logdet_ref = np.linalg.slogdet(x.numpy())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

            (x_grad,) = paddle.grad(logabsdet, x, logdet_grad)
            x_grad_ref = self.slogdet_backward(
                x.numpy(),
                sign.numpy(),
                logdet_grad.numpy()[..., None, None],
            )
            np.testing.assert_allclose(x_grad.numpy(), x_grad_ref, 1e-5, 1e-5)

            # test pir + dynamic shape
            st_f = paddle.jit.to_static(
                paddle.compat.slogdet,
                full_graph=True,
                input_spec=[
                    paddle.static.InputSpec(shape=[-1, -1], dtype="float32"),
                ],
            )
            sign, logabsdet = st_f(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

        run()

        if self.compiled_with_cuda():
            with paddle.device.device_guard("cpu"):
                run()

    def test_zero_dim_invertible_matrix_backward(self):
        def run():
            x = paddle.zeros(
                shape=[2, 0, 0],
                dtype="float32",
                device="cpu",
                requires_grad=True,
            )
            out = paddle.compat.slogdet(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            sign, logabsdet = out
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            logdet_grad = paddle.randn_like(logabsdet)
            sign_ref, logdet_ref = np.linalg.slogdet(x.numpy())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

            (x_grad,) = paddle.grad(logabsdet, x, logdet_grad)
            x_grad_ref = self.slogdet_backward(
                x.numpy(),
                sign.numpy(),
                logdet_grad.numpy()[..., None, None],
            )
            np.testing.assert_allclose(x_grad.numpy(), x_grad_ref, 1e-5, 1e-5)

            # test pir + dynamic shape
            st_f = paddle.jit.to_static(
                paddle.compat.slogdet,
                full_graph=True,
                input_spec=[
                    paddle.static.InputSpec(shape=[-1, -1], dtype="float32"),
                ],
            )
            sign, logabsdet = st_f(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

        run()
        if self.compiled_with_cuda():
            with paddle.device.device_guard("cpu"):
                run()

    def test_zero_dim_complex_invertible_matrix_backward(self):
        def run():
            x = (
                paddle.zeros(
                    shape=[2, 0, 0],
                    dtype="float32",
                    device="cpu",
                    requires_grad=True,
                )
                + paddle.randn(
                    shape=[2, 0, 0],
                    dtype="float32",
                    device="cpu",
                    requires_grad=True,
                )
                * 1j
            )
            out = paddle.compat.slogdet(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            sign, logabsdet = out
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            logdet_grad = paddle.randn_like(logabsdet)
            sign_ref, logdet_ref = np.linalg.slogdet(x.numpy())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

            (x_grad,) = paddle.grad(logabsdet, x, logdet_grad)
            x_grad_ref = self.slogdet_backward(
                x.numpy(),
                sign.numpy(),
                logdet_grad.numpy()[..., None, None],
            )
            np.testing.assert_allclose(x_grad.numpy(), x_grad_ref, 1e-5, 1e-5)

            # test pir + dynamic shape
            st_f = paddle.jit.to_static(
                paddle.compat.slogdet,
                full_graph=True,
                input_spec=[
                    paddle.static.InputSpec(shape=[-1, -1], dtype="float32"),
                ],
            )
            sign, logabsdet = st_f(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

        run()
        if self.compiled_with_cuda():
            with paddle.device.device_guard("cpu"):
                run()

    def test_det_zero(self):
        def run():
            x = paddle.to_tensor(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ],
                dtype="float32",
                place="cpu",
            )
            out = paddle.compat.slogdet(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            sign, logabsdet = out
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            sign_ref, logdet_ref = np.linalg.slogdet(x.numpy())
            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

        run()

    def test_complex_invertible_matrix_backward(self):
        def run():
            x = (
                paddle.randn(
                    shape=[2, 3, 3],
                    dtype="float32",
                    device="cpu",
                    requires_grad=True,
                )
                + paddle.randn(
                    shape=[2, 3, 3],
                    dtype="float32",
                    device="cpu",
                    requires_grad=True,
                )
                * 1j
            )
            out = paddle.compat.slogdet(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            sign, logabsdet = out
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            logdet_grad = paddle.randn_like(logabsdet)
            sign_ref, logdet_ref = np.linalg.slogdet(x.numpy())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

            (x_grad,) = paddle.grad(logabsdet, x, logdet_grad)
            x_grad_ref = self.slogdet_backward(
                x.numpy(),
                sign.numpy(),
                logdet_grad.numpy()[..., None, None],
            )
            np.testing.assert_allclose(x_grad.numpy(), x_grad_ref, 1e-5, 1e-5)

            # test pir + dynamic shape
            st_f = paddle.jit.to_static(
                paddle.compat.slogdet,
                full_graph=True,
                input_spec=[
                    paddle.static.InputSpec(shape=[-1, -1], dtype="float32"),
                ],
            )
            sign, logabsdet = st_f(x)
            self.assertTrue(hasattr(out, "sign"))
            self.assertTrue(hasattr(out, "logabsdet"))
            self.assertEqual(sign.dtype, x.dtype)
            self.assertFalse(logabsdet.is_complex())

            np.testing.assert_allclose(sign.numpy(), sign_ref, 1e-5, 1e-5)
            np.testing.assert_allclose(
                logabsdet.numpy(),
                logdet_ref,
                1e-5,
                1e-5,
            )

        run()
        if self.compiled_with_cuda():
            with paddle.device.device_guard("cpu"):
                run()


if __name__ == '__main__':
    unittest.main()
