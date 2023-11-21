#  Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
from scipy import special

import paddle


def ref_multigammaln(x, p):
    return special.multigammaln(x, p)


def ref_multigammaln_grad(x, p):
    def single_multigammaln_grad(x, p):
        return special.psi(x - 0.5 * np.arange(0, p)).sum()

    vectorized_multigammaln_grad = np.vectorize(single_multigammaln_grad)
    return vectorized_multigammaln_grad(x, p)


class TestMultigammalnAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(1024)
        self.x = np.random.rand(10, 20).astype('float32') + 1.0
        self.p = 2
        self.init_input()
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def init_input(self):
        pass

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('x', self.x.shape, dtype=self.x.dtype)
            out = paddle.multigammaln(x, self.p)
            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x,
                },
                fetch_list=[out],
            )
            out_ref = ref_multigammaln(self.x, self.p)
            np.testing.assert_allclose(out_ref, res[0], rtol=1e-6, atol=1e-6)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        out = paddle.multigammaln(x, self.p)
        out_ref = ref_multigammaln(self.x, self.p)
        np.testing.assert_allclose(out_ref, out.numpy(), rtol=1e-6, atol=1e-6)
        paddle.enable_static()


class TestMultigammalnAPICase1(TestMultigammalnAPI):
    def init_input(self):
        self.x = np.random.rand(10, 20).astype('float64') + 1.0


class TestMultigammalnGrad(unittest.TestCase):
    def setUp(self):
        np.random.seed(1024)
        self.dtype = 'float32'
        self.x = np.array([2, 3, 4, 5, 6, 7, 8]).astype(dtype=self.dtype)
        self.p = 3
        self.place = (
            paddle.CUDAPlace(0)
            if paddle.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_backward(self):
        expected_x_grad = ref_multigammaln_grad(self.x, self.p)
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x, dtype=self.dtype, place=self.place)
        x.stop_gradient = False
        out = x.multigammaln(self.p)
        loss = out.sum()
        loss.backward()

        np.testing.assert_allclose(
            x.grad.numpy().astype('float32'),
            expected_x_grad,
            rtol=1e-6,
            atol=1e-6,
        )
        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
