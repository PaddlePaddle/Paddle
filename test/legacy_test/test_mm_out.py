# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestMmOutAndGrad(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_shape = [3, 4]
        self.y_shape = [4, 5]
        self.x_np = np.random.rand(*self.x_shape).astype(np.float32)
        self.y_np = np.random.rand(*self.y_shape).astype(np.float32)

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        out = paddle.empty((3, 5), dtype='float32')
        out.stop_gradient = False

        if test_type == 'raw':
            result = paddle.mm(x, y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'out':
            paddle.mm(x, y, out=out)
            out.mean().backward()
            return out, x.grad, y.grad

    def test_mm_out(self):
        out_std, x_grad_std, y_grad_std = self.do_test('raw')
        out, x_grad, y_grad = self.do_test('out')
        np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-20)
        np.testing.assert_allclose(
            x_grad.numpy(), x_grad_std.numpy(), rtol=1e-20
        )
        np.testing.assert_allclose(
            y_grad.numpy(), y_grad_std.numpy(), rtol=1e-20
        )


class TestMmOutScenarios(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_mm_out_scenarios(self):
        def run_mm(test_type):
            x = paddle.to_tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], stop_gradient=False
            )
            y = paddle.to_tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], stop_gradient=False
            )
            out = paddle.zeros([3, 3], dtype='float32')
            out.stop_gradient = False

            if test_type == "return":
                out = paddle.mm(x, y)
            elif test_type == "input_out":
                paddle.mm(x, y, out=out)
            elif test_type == "both_return":
                out = paddle.mm(x, y, out=out)
            elif test_type == "both_input_out":
                tmp = paddle.mm(x, y, out=out)

            expected = np.array(x.numpy()) @ np.array(y.numpy())
            np.testing.assert_allclose(
                out.numpy(), expected, rtol=1e-5, atol=1e-5
            )

            loss = out.sum()
            loss.backward()
            return out, x.grad, y.grad, out.grad

        out1, x1, y1, o1 = run_mm("return")
        out2, x2, y2, o2 = run_mm("input_out")
        out3, x3, y3, o3 = run_mm("both_return")
        out4, x4, y4, o4 = run_mm("both_input_out")

        np.testing.assert_allclose(
            out1.numpy(), out2.numpy(), rtol=1e-20, atol=1e-20
        )
        np.testing.assert_allclose(
            out1.numpy(), out3.numpy(), rtol=1e-20, atol=1e-20
        )
        np.testing.assert_allclose(
            out1.numpy(), out4.numpy(), rtol=1e-20, atol=1e-20
        )

        np.testing.assert_allclose(
            x1.numpy(), x2.numpy(), rtol=1e-20, atol=1e-20
        )
        np.testing.assert_allclose(
            x1.numpy(), x3.numpy(), rtol=1e-20, atol=1e-20
        )
        np.testing.assert_allclose(
            x1.numpy(), x4.numpy(), rtol=1e-20, atol=1e-20
        )

        np.testing.assert_allclose(
            y1.numpy(), y2.numpy(), rtol=1e-20, atol=1e-20
        )
        np.testing.assert_allclose(
            y1.numpy(), y3.numpy(), rtol=1e-20, atol=1e-20
        )
        np.testing.assert_allclose(
            y1.numpy(), y4.numpy(), rtol=1e-20, atol=1e-20
        )

        np.testing.assert_equal(o1, None)
        np.testing.assert_equal(o2, None)
        np.testing.assert_equal(o3, None)
        np.testing.assert_equal(o4, None)


class TestMmOutBatchedAndShapes(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def _check_out(self, x_np, y_np):
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.to_tensor(y_np, stop_gradient=False)

        expected = paddle.mm(x, y)
        out = paddle.empty(expected.shape, dtype=expected.dtype)
        paddle.mm(x, y, out=out)
        np.testing.assert_allclose(
            out.numpy(), expected.numpy(), rtol=1e-5, atol=1e-5
        )

    def test_2d(self):
        self._check_out(
            np.random.rand(3, 4).astype(np.float32),
            np.random.rand(4, 5).astype(np.float32),
        )

    def test_3d_batched(self):
        self._check_out(
            np.random.rand(2, 3, 4).astype(np.float32),
            np.random.rand(2, 4, 5).astype(np.float32),
        )

    def test_broadcast_3d_2d(self):
        self._check_out(
            np.random.rand(2, 3, 4).astype(np.float32),
            np.random.rand(4, 5).astype(np.float32),
        )

    def test_1d_2d(self):
        self._check_out(
            np.random.rand(4).astype(np.float32),
            np.random.rand(4, 5).astype(np.float32),
        )

    def test_2d_1d(self):
        self._check_out(
            np.random.rand(3, 4).astype(np.float32),
            np.random.rand(4).astype(np.float32),
        )


if __name__ == "__main__":
    unittest.main()
