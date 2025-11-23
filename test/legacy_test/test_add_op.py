#  Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_device_place

import paddle
from paddle.base import core


class TestPaddleAddBackward(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()
        self.x_np_f32 = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        self.y_np_f32 = np.array(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32
        )
        self.N = self.x_np_f32.size
        self.expected_grad = np.full(
            self.x_np_f32.shape, 1.0 / self.N, dtype=np.float32
        )

    def test_backward(self):
        x = paddle.to_tensor(self.x_np_f32, stop_gradient=False)
        y = paddle.to_tensor(self.y_np_f32, stop_gradient=False)

        out = paddle.add(x, y)
        out.mean().backward()

        np.testing.assert_allclose(
            x.grad.numpy(), self.expected_grad, rtol=1e-6
        )
        np.testing.assert_allclose(
            y.grad.numpy(), self.expected_grad, rtol=1e-6
        )

    def test_backward_broadcast(self):
        x_np = self.x_np_f32
        y_np = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.to_tensor(y_np, stop_gradient=False)

        out = paddle.add(x, y)
        loss = out.mean()
        loss.backward()

        N = out.numel()
        expected_x_grad = np.full(x_np.shape, 1.0 / N, dtype=np.float32)
        expected_y_grad = np.full(y_np.shape, 2.0 / N, dtype=np.float32)

        np.testing.assert_allclose(x.grad.numpy(), expected_x_grad, rtol=1e-6)
        np.testing.assert_allclose(y.grad.numpy(), expected_y_grad, rtol=1e-6)

    @unittest.skipUnless(
        core.is_float16_supported(get_device_place()), "Skip float16 test"
    )
    def test_backward_mixed_precision_f16(self):
        # X: float32, Y: float16
        x_np = self.x_np_f32
        y_np = self.y_np_f32.astype(np.float16)
        x = paddle.to_tensor(x_np, stop_gradient=False)
        y = paddle.to_tensor(y_np, stop_gradient=False)

        out = paddle.add(x, y)
        out.mean().backward()

        N = out.numel()
        expected_x_grad = np.full(x_np.shape, 1.0 / N, dtype=np.float32)
        expected_y_grad = np.full(y_np.shape, 1.0 / N, dtype=np.float16)

        rtol, atol = 1e-3, 1e-3
        actual_x_grad = x.grad.numpy()
        np.testing.assert_allclose(
            actual_x_grad, expected_x_grad, rtol=rtol, atol=atol
        )
        assert actual_x_grad.dtype == expected_x_grad.dtype, (
            f"x.grad dtype mismatch: expected {expected_x_grad.dtype}, got {actual_x_grad.dtype}"
        )
        actual_y_grad = y.grad.numpy()
        np.testing.assert_allclose(
            actual_y_grad, expected_y_grad, rtol=rtol, atol=atol
        )
        assert actual_y_grad.dtype == expected_y_grad.dtype, (
            f"y.grad dtype mismatch: expected {expected_y_grad.dtype}, got {actual_y_grad.dtype}"
        )

    def test_backward_with_grad(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x_np_f32, stop_gradient=False)
        y = paddle.to_tensor(self.y_np_f32, stop_gradient=False)
        out = paddle.add(x, y)

        out_grad_np = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
        )
        out_grad = paddle.to_tensor(out_grad_np)
        out.backward(grad_tensor=out_grad)

        expected_grad = out_grad_np
        np.testing.assert_allclose(x.grad.numpy(), expected_grad, rtol=1e-6)
        np.testing.assert_allclose(y.grad.numpy(), expected_grad, rtol=1e-6)


class TestPaddleAddNewFeatures(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([3, 5], dtype='float32')
        self.y_np = np.array([2, 3], dtype='float32')
        self.scalar = 2.0
        self.place = get_device_place()

    def test_paddle_add_with_alpha(self):
        """test paddle.add alpha"""
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        out = paddle.add(x, y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out.numpy(), expected)

        out.mean().backward()
        expected_x_grad = np.array([0.5, 0.5], dtype='float32')
        expected_y_grad = np.array([1.0, 1.0], dtype='float32')  # alpha=2
        np.testing.assert_array_equal(x.grad.numpy(), expected_x_grad)
        np.testing.assert_array_equal(y.grad.numpy(), expected_y_grad)

    def test_tensor_add_with_alpha(self):
        """test paddle.Tensor.add alpha"""
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        out = x.add(y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out.numpy(), expected)

        out.mean().backward()
        expected_x_grad = np.array([0.5, 0.5], dtype='float32')
        expected_y_grad = np.array([1.0, 1.0], dtype='float32')  # alpha=2
        np.testing.assert_array_equal(x.grad.numpy(), expected_x_grad)
        np.testing.assert_array_equal(y.grad.numpy(), expected_y_grad)

    def test_tensor_add_inplace_with_alpha(self):
        """test Tensor.add_ alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x.add_(y, alpha=2)
        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_consistency_between_apis(self):
        """test different APIs consistency for add with alpha"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)

        out1 = paddle.add(x, y, alpha=2)
        out2 = x.add(y, alpha=2)
        x.add_(y, alpha=2)

        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out1.numpy(), expected)
        np.testing.assert_array_equal(out2.numpy(), expected)
        np.testing.assert_array_equal(x.numpy(), expected)

    def test_static_graph_add_with_alpha(self):
        """test static graph add with alpha and parameter aliases"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 2], dtype='float32')
            out1 = paddle.add(x, y, alpha=2)
            out2 = paddle.add(input=x, other=y, alpha=2)

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x_np.reshape(1, 2),
                    'y': self.y_np.reshape(1, 2),
                },
                fetch_list=[out1, out2],
            )

            expected = self.x_np + self.y_np * 2
            for result in res:
                np.testing.assert_array_equal(result.flatten(), expected)
        paddle.disable_static()

    def test_param_alias_input_other(self):
        """test parameter alias input/other in dynamic graph"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)

        out1 = paddle.add(input=x, other=y, alpha=2)
        out2 = x.add(other=y, alpha=2)
        x_clone = x.clone()
        x_clone.add_(other=y, alpha=2)

        expected = self.x_np + self.y_np * 2
        np.testing.assert_array_equal(out1.numpy(), expected)
        np.testing.assert_array_equal(out2.numpy(), expected)
        np.testing.assert_array_equal(x_clone.numpy(), expected)

    # Note: y does not support scalars separately, but will support them uniformly in the future.
    # def test_scalar_addition(self):
    #     """test scalar addition"""
    #     x = paddle.to_tensor(self.x_np)

    #     out1 = paddle.add(x, self.scalar)
    #     expected1 = self.x_np + self.scalar
    #     np.testing.assert_array_equal(out1.numpy(), expected1)

    #     out2 = x.add(self.scalar)
    #     np.testing.assert_array_equal(out2.numpy(), expected1)

    #     out3 = paddle.add(x, self.scalar, alpha=2)
    #     expected3 = self.x_np + self.scalar * 2
    #     np.testing.assert_array_equal(out3.numpy(), expected3)

    # def test_scalar_addition_inplace(self):
    #     """test inplace scalar addition"""
    #     x = paddle.to_tensor(self.x_np)
    #     x_clone = x.clone()

    #     x_clone.add_(self.scalar)
    #     expected = self.x_np + self.scalar
    #     np.testing.assert_array_equal(x_clone.numpy(), expected)

    #     x_clone2 = x.clone()
    #     x_clone2.add_(self.scalar, alpha=2)
    #     expected2 = self.x_np + self.scalar * 2
    #     np.testing.assert_array_equal(x_clone2.numpy(), expected2)

    # def test_different_dtype_scalar(self):
    #     """test different dtype scalar addition"""
    #     x = paddle.to_tensor(self.x_np)

    #     out1 = x.add(2)
    #     expected1 = self.x_np + 2
    #     np.testing.assert_array_equal(out1.numpy(), expected1)

    #     out2 = x.add(2.5)
    #     expected2 = self.x_np + 2.5
    #     np.testing.assert_array_equal(out2.numpy(), expected2)

    # def test_scalar_addition_static_graph(self):
    #     """test static graph scalar addition"""
    #     paddle.enable_static()
    #     with paddle.static.program_guard(paddle.static.Program()):
    #         x = paddle.static.data(name='x', shape=[-1, 2], dtype='float32')
    #         out1 = paddle.add(x, self.scalar)
    #         out2 = paddle.add(x, self.scalar, alpha=2)

    #         exe = paddle.static.Executor(self.place)
    #         res = exe.run(
    #             feed={'x': self.x_np.reshape(1, 2)},
    #             fetch_list=[out1, out2],
    #         )

    #         expected1 = self.x_np + self.scalar
    #         expected2 = self.x_np + self.scalar * 2
    #         np.testing.assert_array_equal(res[0].flatten(), expected1)
    #         np.testing.assert_array_equal(res[1].flatten(), expected2)
    #     paddle.disable_static()


class TestAddOut(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.place = get_device_place()

    def test_add_with_alpha_out(self):
        def run_add_with_alpha(test_type):
            x = paddle.to_tensor([1.0, 2.0, 3.0], stop_gradient=False)
            y = paddle.to_tensor([4.0, 5.0, 6.0], stop_gradient=False)
            out = paddle.zeros_like(x)
            out.stop_gradient = False
            alpha = 2.0

            if test_type == "return":
                out = paddle.add(x, y, alpha=alpha)
            elif test_type == "input_out":
                paddle.add(x, y, alpha=alpha, out=out)
            elif test_type == "both_return":
                out = paddle.add(x, y, alpha=alpha, out=out)
            elif test_type == "both_input_out":
                tmp = paddle.add(x, y, alpha=alpha, out=out)

            expected = x + y * alpha
            np.testing.assert_allclose(
                out.numpy(),
                expected.numpy(),
                rtol=1e-20,
                atol=1e-20,
            )

            loss = out.sum()
            loss.backward()

            return out, x.grad, y.grad, out.grad

        out1, x1, y1, o1 = run_add_with_alpha("return")
        out2, x2, y2, o2 = run_add_with_alpha("input_out")
        out3, x3, y3, o3 = run_add_with_alpha("both_return")
        out4, x4, y4, o4 = run_add_with_alpha("both_input_out")

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


if __name__ == "__main__":
    unittest.main()
