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
from paddle.base import core

np.random.seed(10)


def ref_vander(x, N=None, increasing=False):
    return np.vander(x, N, increasing)


class TestVanderAPI(unittest.TestCase):
    # test paddle.tensor.math.vander

    def setUp(self):
        self.shape = [5]
        self.x = np.random.uniform(-1, 1, self.shape).astype(np.float32)
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def api_case(self, N=None, increasing=False):
        paddle.enable_static()
        out_ref = ref_vander(self.x, N, increasing)

        def test_static_or_pir_mode():
            with paddle.static.program_guard(paddle.static.Program()):
                x = paddle.static.data('X', self.shape)
                out = paddle.vander(x, N, increasing)
                exe = paddle.static.Executor(self.place)
                res = exe.run(feed={'X': self.x}, fetch_list=[out])
            if N != 0:
                np.testing.assert_allclose(res[0], out_ref, rtol=1e-05)
            else:
                np.testing.assert_allclose(
                    res[0].size, out_ref.size, rtol=1e-05
                )

        test_static_or_pir_mode()

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out = paddle.vander(x, N, increasing)
        np.testing.assert_allclose(out.numpy(), out_ref, rtol=1e-05)
        paddle.enable_static()

    def test_api(self):
        self.api_case()
        N = list(range(9))
        for n in N:
            self.api_case(n)
            self.api_case(n, increasing=True)

    def test_complex(self):
        paddle.disable_static(self.place)
        real = np.random.rand(5)
        imag = np.random.rand(5)
        complex_np = real + 1j * imag
        complex_paddle = paddle.complex(
            paddle.to_tensor(real), paddle.to_tensor(imag)
        )

        def test_api_case(N, increasing=False):
            for n in N:
                res_np = np.vander(complex_np, n, increasing)
                res_paddle = paddle.vander(complex_paddle, n, increasing)
                np.testing.assert_allclose(
                    res_paddle.numpy(), res_np, rtol=1e-05
                )

        N = [0, 1, 2, 3, 4]
        test_api_case(N)
        test_api_case(N, increasing=True)
        paddle.enable_static()

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            self.assertRaises(TypeError, paddle.vander, 1)
            x = paddle.static.data('X', [10, 12], 'int32')
            self.assertRaises(ValueError, paddle.vander, x)
            x1 = paddle.static.data('X1', [10], 'int32')
            self.assertRaises(ValueError, paddle.vander, x1, n=-1)


if __name__ == "__main__":
    unittest.main()
