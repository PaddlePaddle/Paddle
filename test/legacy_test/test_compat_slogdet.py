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


class TestSlogDet(unittest.TestCase):
    def setUp(self) -> None:
        self.shapes = [
            [2, 2, 5, 5],
            [10, 10],
            [0, 5, 5],
            [0, 0, 0],
            [3, 3, 5, 5],
            [6, 5, 5],
            [4, 50, 50],
        ]
        self.dtypes = [
            "float32",
            "float64",
            "complex64",
            "complex128",
        ]

    def slogdet_backward(self, x, _, grad_logabsdet):
        x_inv_T = np.swapaxes(np.linalg.inv(x).conj(), -1, -2)
        grad_x = grad_logabsdet * x_inv_T
        return grad_x

    def test_compat_slogdet(self):
        with dygraph_guard():
            for shape, dtype in product(self.shapes, self.dtypes):
                err_msg = f"shape = {shape}, dtype = {dtype}"

                # test eager
                x = paddle.randn(shape, dtype)
                x.stop_gradient = False
                sign, logdet = paddle.compat.slogdet(x)
                logdet_grad = paddle.randn_like(logdet)
                sign_ref, logdet_ref = np.linalg.slogdet(x.numpy())

                np.testing.assert_allclose(
                    sign.numpy(), sign_ref, 1e-5, 1e-5, err_msg=err_msg
                )
                np.testing.assert_allclose(
                    logdet.numpy(), logdet_ref, 1e-5, 1e-5, err_msg=err_msg
                )

                (x_grad,) = paddle.grad(logdet, x, logdet_grad)
                x_grad_ref = self.slogdet_backward(
                    x.numpy(),
                    sign.numpy(),
                    logdet_grad.numpy()[..., None, None],
                )
                np.testing.assert_allclose(
                    x_grad.numpy(), x_grad_ref, 1e-5, 1e-5, err_msg=err_msg
                )

                # test pir
                st_f = paddle.jit.to_static(
                    paddle.compat.slogdet,
                    full_graph=True,
                    backend=None,
                )
                sign, logdet = st_f(x)

                np.testing.assert_allclose(
                    sign.numpy(), sign_ref, 1e-5, 1e-5, err_msg=err_msg
                )
                np.testing.assert_allclose(
                    logdet.numpy(), logdet_ref, 1e-5, 1e-5, err_msg=err_msg
                )


if __name__ == '__main__':
    unittest.main()
