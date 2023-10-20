# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.framework import core


class TestDy2staticNewIR(unittest.TestCase):
    def test_basic_network_backward(self):
        core._set_prim_all_enabled(True)

        def func(x):
            x1 = paddle.mean(x)
            out = paddle.nn.functional.gelu(x1, False)
            return out

        # ==== dygraph computation ====
        static_func = paddle.jit.to_static(func, full_graph=True)
        x = paddle.randn((8, 16, 64))
        x.stop_gradient = False
        ref_out = func(x) * 2
        ref_out.backward()
        ref_grad = x.grad.numpy()
        x.clear_gradient()

        # ==== to static compuatation ====
        out = static_func(x)
        actual_out = out * 2
        actual_out.backward()
        actual_grad = x.grad
        core._set_prim_all_enabled(True)

        np.testing.assert_allclose(
            ref_out, actual_out.numpy(), atol=1e-6, rtol=1e-6
        )

        np.testing.assert_allclose(
            ref_grad, actual_grad.numpy(), atol=1e-6, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
