#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestTensorApplyAPI(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor([1, 2, 3, 4, 5], stop_gradient=True)
        self.function = lambda x: 3 * x + 2

    def test_dygraph(self):
        y = self.x.apply(self.function)
        np.testing.assert_allclose(
            self.function(self.x).numpy(), y.numpy(), rtol=1e-05
        )

    def test_error(self):
        self.x.stop_gradient = False

        def fn_inplace(x):
            x.apply_(self.function)

        def fn_outplace(x, func):
            x.apply(func)

        self.assertRaises(RuntimeError, fn_inplace, self.x)
        self.assertRaises(RuntimeError, fn_outplace, self.x, self.function)
        with paddle.jit.api.sot_mode_guard(False):
            self.assertRaises(
                RuntimeError,
                paddle.jit.to_static(fn_outplace),
                self.x,
                self.function,
            )
            with paddle.pir_utils.IrGuard():
                self.assertRaises(
                    RuntimeError,
                    paddle.jit.to_static(fn_outplace),
                    self.x,
                    self.function,
                )

    def test_to_static(self):
        def fn(x, func):
            y = x.apply(func)
            return y

        with paddle.jit.api.sot_mode_guard(False):
            paddle.disable_static()
            jit_g = paddle.jit.to_static(fn)
            out_legacy_ir = jit_g(self.x, self.function)
            with paddle.pir_utils.IrGuard():
                jit_g = paddle.jit.to_static(fn)
                out_pir = jit_g(self.x, self.function)
        np.testing.assert_allclose(
            self.function(self.x).numpy(), out_legacy_ir.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            self.function(self.x).numpy(), out_pir.numpy(), rtol=1e-05
        )


if __name__ == "__main__":
    unittest.main()
