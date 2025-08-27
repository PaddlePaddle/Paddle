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

import paddle
from paddle.base import core


class TestPaddleDivide(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([4, 9, 16], dtype='float32')
        self.y_np = np.array([2, 3, 4], dtype='float32')
        self.scalar = 2.0
        self.place = (
            core.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else core.CPUPlace()
        )

    def test_paddle_divide(self):
        """Test paddle.divide"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.divide(x, y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_paddle_divide_with_param_names(self):
        """Test paddle.divide with input= and other="""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.divide(input=x, other=y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    # def test_paddle_divide_with_scalar(self):
    #     """Test paddle.divide with scalar"""
    #     x = paddle.to_tensor(self.x_np)
    #     out = paddle.divide(x, self.scalar)
    #     expected = self.x_np / self.scalar
    #     np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_paddle_divide_rounding_modes(self):
        """Test paddle.divide with different rounding modes"""
        x = paddle.to_tensor([5, -5, 3.5, -3.5], dtype='float32')
        y = paddle.to_tensor([2, 2, 2, 2], dtype='float32')

        # Trunc mode
        out1 = paddle.divide(x, y, rounding_mode='trunc')
        expected1 = np.array([2.0, -2.0, 1.0, -1.0])
        np.testing.assert_allclose(out1.numpy(), expected1, rtol=1e-6)

        # Floor mode
        out2 = paddle.divide(x, y, rounding_mode='floor')
        expected2 = np.array([2.0, -3.0, 1.0, -2.0])
        np.testing.assert_allclose(out2.numpy(), expected2, rtol=1e-6)

    def test_divide_with_out_and_rounding_modes(self):
        """Test paddle.divide with out parameter and rounding modes"""
        x = paddle.to_tensor([5.0, -5.0, 3.5, -3.5], dtype='float32')
        y = paddle.to_tensor([2.0, 2.0, 2.0, 2.0], dtype='float32')
        out = paddle.zeros_like(x)

        # Test trunc mode with out
        paddle.divide(x, y, rounding_mode='trunc', out=out)
        expected_trunc = np.array([2.0, -2.0, 1.0, -1.0])
        np.testing.assert_allclose(out.numpy(), expected_trunc, rtol=1e-20)

        # Test floor mode with out
        paddle.divide(x, y, rounding_mode='floor', out=out)
        expected_floor = np.array([2.0, -3.0, 1.0, -2.0])
        np.testing.assert_allclose(out.numpy(), expected_floor, rtol=1e-20)

    def test_paddle_divide_mixed_dtypes(self):
        """Test paddle.divide with mixed dtypes (int/float combinations)"""
        test_cases = [
            # (x_dtype, y_dtype, expected_dtype, rounding_mode)
            # ('int8', 'float16', 'float16', None),
            # ('int16', 'float32', 'float32', None),
            # ('uint8', 'float64', 'float64', None),
            # ('int32', 'bfloat16', 'bfloat16', None),
            # ('float16', 'int64', 'float16', None),
            # ('bfloat16', 'uint8', 'bfloat16', None),
            # ('float64', 'int8', 'float64', None),
            # ('int8', 'int32', 'int32', 'trunc'),
            # ('int32', 'int64', 'int64', 'trunc'),
            ('int32', 'int32', 'int32', 'trunc'),
            ('int64', 'int64', 'int64', 'trunc'),
            ('int16', 'int16', 'int16', 'trunc'),
            ('int8', 'int8', 'int8', 'trunc'),
            ('uint8', 'uint8', 'uint8', 'trunc'),
        ]

        for x_dtype, y_dtype, expected_dtype, rounding_mode in test_cases:
            with self.subTest(x_dtype=x_dtype, y_dtype=y_dtype):
                x = paddle.to_tensor([1, 2, 3], dtype=x_dtype)
                y = paddle.to_tensor([2, 1, 3], dtype=y_dtype)

                out = paddle.divide(x, y, rounding_mode=rounding_mode)

                self.assertEqual(
                    out.dtype,
                    getattr(paddle, expected_dtype),
                    f'Dtype mismatch: {x_dtype}/{y_dtype} should be {expected_dtype}',
                )

    def test_paddle_divide_static_graph(self):
        """Test paddle.divide in static graph"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 3], dtype='float32')
            out1 = paddle.divide(x, y)
            out2 = paddle.divide(input=x, other=y)

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x_np.reshape(1, 3),
                    'y': self.y_np.reshape(1, 3),
                },
                fetch_list=[out1, out2],
            )

            expected = self.x_np / self.y_np
            for result in res:
                np.testing.assert_allclose(
                    result.flatten(), expected, rtol=1e-6
                )
        paddle.disable_static()

    def test_paddle_divide_static_graph_rounding_modes(self):
        """Test paddle.divide in static graph with rounding modes"""
        paddle.enable_static()

        # Test trunc mode
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 4], dtype='float32')
            out = paddle.divide(x, y, rounding_mode='trunc')

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': np.array([5, -5, 3.5, -3.5], dtype='float32').reshape(
                        1, 4
                    ),
                    'y': np.array([2, 2, 2, 2], dtype='float32').reshape(1, 4),
                },
                fetch_list=[out],
            )

            expected = np.array([2.0, -2.0, 1.0, -1.0])
            np.testing.assert_allclose(res[0].flatten(), expected, rtol=1e-6)

        # Test floor mode
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 4], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 4], dtype='float32')
            out = paddle.divide(x, y, rounding_mode='floor')

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': np.array([5, -5, 3.5, -3.5], dtype='float32').reshape(
                        1, 4
                    ),
                    'y': np.array([2, 2, 2, 2], dtype='float32').reshape(1, 4),
                },
                fetch_list=[out],
            )

            expected = np.array([2.0, -3.0, 1.0, -2.0])
            np.testing.assert_allclose(res[0].flatten(), expected, rtol=1e-6)

        paddle.disable_static()

    def test_divide_with_out_static_graph(self):
        """Test paddle.divide with out parameter in static graph"""
        paddle.enable_static()

        # Test with out parameter
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 3], dtype='float32')
            out = paddle.static.data(name='out', shape=[-1, 3], dtype='float32')
            result = paddle.divide(x, y, out=out)

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x_np.reshape(1, 3),
                    'y': self.y_np.reshape(1, 3),
                    'out': np.zeros((1, 3), dtype='float32'),
                },
                fetch_list=[result],
            )

            expected = self.x_np / self.y_np
            np.testing.assert_allclose(res[0].flatten(), expected, rtol=1e-20)

        paddle.disable_static()


class TestPaddleDiv(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([4, 9, 16], dtype='float32')
        self.y_np = np.array([2, 3, 4], dtype='float32')
        self.scalar = 2.0
        self.place = (
            core.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else core.CPUPlace()
        )

    def test_paddle_div(self):
        """Test paddle.div"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.div(x, y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_paddle_div_with_param_names(self):
        """Test paddle.div with input= and other="""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.div(input=x, other=y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    # def test_paddle_div_with_scalar(self):
    #     """Test paddle.div with scalar"""
    #     x = paddle.to_tensor(self.x_np)
    #     out = paddle.div(x, self.scalar)
    #     expected = self.x_np / self.scalar
    #     np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_paddle_div_rounding_modes(self):
        """Test paddle.div with different rounding modes"""
        x = paddle.to_tensor([5, -5, 3.5, -3.5], dtype='float32')
        y = paddle.to_tensor([2, 2, 2, 2], dtype='float32')

        # Trunc mode
        out1 = paddle.div(x, y, rounding_mode='trunc')
        expected1 = np.array([2.0, -2.0, 1.0, -1.0])
        np.testing.assert_allclose(out1.numpy(), expected1, rtol=1e-6)

        # Floor mode
        out2 = paddle.div(x, y, rounding_mode='floor')
        expected2 = np.array([2.0, -3.0, 1.0, -2.0])
        np.testing.assert_allclose(out2.numpy(), expected2, rtol=1e-6)

    def test_paddle_div_with_out_and_rounding_modes(self):
        """Test paddle.div with out parameter and rounding modes"""
        x = paddle.to_tensor([5.0, -5.0, 3.5, -3.5], dtype='float32')
        y = paddle.to_tensor([2.0, 2.0, 2.0, 2.0], dtype='float32')
        out = paddle.zeros_like(x)

        # Test trunc mode with out
        paddle.div(x, y, rounding_mode='trunc', out=out)
        expected_trunc = np.array([2.0, -2.0, 1.0, -1.0])
        np.testing.assert_allclose(out.numpy(), expected_trunc, rtol=1e-20)

        # Test floor mode with out
        paddle.div(x, y, rounding_mode='floor', out=out)
        expected_floor = np.array([2.0, -3.0, 1.0, -2.0])
        np.testing.assert_allclose(out.numpy(), expected_floor, rtol=1e-20)

    def test_paddle_div_static_graph(self):
        """Test paddle.div in static graph"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 3], dtype='float32')
            out = paddle.div(x, y)

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x_np.reshape(1, 3),
                    'y': self.y_np.reshape(1, 3),
                },
                fetch_list=[out],
            )

            expected = self.x_np / self.y_np
            np.testing.assert_allclose(res[0].flatten(), expected, rtol=1e-6)
        paddle.disable_static()

    def test_div_with_out_static_graph(self):
        """Test paddle.div with out parameter in static graph"""
        paddle.enable_static()

        # Test with out parameter
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 3], dtype='float32')
            out = paddle.static.data(name='out', shape=[-1, 3], dtype='float32')
            result = paddle.div(x, y, out=out)

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x_np.reshape(1, 3),
                    'y': self.y_np.reshape(1, 3),
                    'out': np.zeros((1, 3), dtype='float32'),
                },
                fetch_list=[result],
            )

            expected = self.x_np / self.y_np
            np.testing.assert_allclose(res[0].flatten(), expected, rtol=1e-20)

        paddle.disable_static()


class TestPaddleDivideInplace(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([4, 9, 16], dtype='float32')
        self.y_np = np.array([2, 3, 4], dtype='float32')
        self.scalar = 2.0

    def test_paddle_divide_(self):
        """Test paddle.divide_"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x.divide_(y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6)

    def test_paddle_divide__with_param_names(self):
        """Test paddle.divide_ with input= and other="""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x.divide_(other=y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6)

    # def test_paddle_divide__with_scalar(self):
    #     """Test paddle.divide_ with scalar"""
    #     x = paddle.to_tensor(self.x_np)
    #     x.divide_(self.scalar)
    #     expected = self.x_np / self.scalar
    #     np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6)

    def test_paddle_divide__rounding_modes(self):
        """Test paddle.divide_ with different rounding modes"""
        x = paddle.to_tensor([5, -5, 3.5, -3.5], dtype='float32')
        y = paddle.to_tensor([2, 2, 2, 2], dtype='float32')

        # Trunc mode
        x_clone = x.clone()
        x_clone.divide_(y, rounding_mode='trunc')
        expected1 = np.array([2.0, -2.0, 1.0, -1.0])
        np.testing.assert_allclose(x_clone.numpy(), expected1, rtol=1e-6)

        # Floor mode
        x_clone = x.clone()
        x_clone.divide_(y, rounding_mode='floor')
        expected2 = np.array([2.0, -3.0, 1.0, -2.0])
        np.testing.assert_allclose(x_clone.numpy(), expected2, rtol=1e-6)

    def test_paddle_divide__mixed_dtypes(self):
        """Test paddle.divide_ with mixed dtypes (int/float combinations)"""
        test_cases = [
            # (x_dtype, y_dtype, expected_dtype, rounding_mode)
            # ('int8', 'float16', 'float16', None),
            # ('int16', 'float32', 'float32', None),
            # ('uint8', 'float64', 'float64', None),
            # ('int32', 'bfloat16', 'bfloat16', None),
            # ('float16', 'int64', 'float16', None),
            # ('bfloat16', 'uint8', 'bfloat16', None),
            # ('float64', 'int8', 'float64', None),
            # ('int8', 'int32', 'int32', 'trunc'),
            # ('int32', 'int64', 'int64', 'trunc'),
            ('int32', 'int32', 'int32', 'trunc'),
            ('int64', 'int64', 'int64', 'trunc'),
            ('int16', 'int16', 'int16', 'trunc'),
            ('int8', 'int8', 'int8', 'trunc'),
            ('uint8', 'uint8', 'uint8', 'trunc'),
        ]

        for x_dtype, y_dtype, expected_dtype, rounding_mode in test_cases:
            with self.subTest(x_dtype=x_dtype, y_dtype=y_dtype):
                x = paddle.to_tensor([1, 2, 3], dtype=x_dtype)
                y = paddle.to_tensor([2, 1, 3], dtype=y_dtype)

                x.divide_(y, rounding_mode=rounding_mode)

                self.assertEqual(
                    x.dtype,
                    getattr(paddle, expected_dtype),
                    f'Dtype mismatch: {x_dtype}/{y_dtype} should be {expected_dtype}',
                )


class TestPaddleDivInplace(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([4, 9, 16], dtype='float32')
        self.y_np = np.array([2, 3, 4], dtype='float32')
        self.scalar = 2.0

    def test_paddle_div_(self):
        """Test paddle.div_"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x.div_(y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6)

    def test_paddle_div__with_param_names(self):
        """Test paddle.div_ with input= and other="""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x.div_(other=y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6)

    # def test_paddle_div__with_scalar(self):
    #     """Test paddle.div_ with scalar"""
    #     x = paddle.to_tensor(self.x_np)
    #     x.div_(self.scalar)
    #     expected = self.x_np / self.scalar
    #     np.testing.assert_allclose(x.numpy(), expected, rtol=1e-6)

    def test_paddle_div__rounding_modes(self):
        """Test paddle.div_ with different rounding modes"""
        x = paddle.to_tensor([5, -5, 3.5, -3.5], dtype='float32')
        y = paddle.to_tensor([2, 2, 2, 2], dtype='float32')

        # Trunc mode
        x_clone = x.clone()
        x_clone.div_(y, rounding_mode='trunc')
        expected1 = np.array([2.0, -2.0, 1.0, -1.0])
        np.testing.assert_allclose(x_clone.numpy(), expected1, rtol=1e-6)

        # Floor mode
        x_clone = x.clone()
        x_clone.div_(y, rounding_mode='floor')
        expected2 = np.array([2.0, -3.0, 1.0, -2.0])
        np.testing.assert_allclose(x_clone.numpy(), expected2, rtol=1e-6)


class TestPaddleTrueDivide(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([4, 9, 16], dtype='float32')
        self.y_np = np.array([2, 3, 4], dtype='float32')
        self.scalar = 2.0
        self.place = (
            core.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else core.CPUPlace()
        )

    def test_paddle_true_divide(self):
        """Test paddle.true_divide"""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.true_divide(x, y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_paddle_true_divide_with_param_names(self):
        """Test paddle.true_divide with input= and other="""
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        out = paddle.true_divide(input=x, other=y)
        expected = self.x_np / self.y_np
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    # def test_paddle_true_divide_with_scalar(self):
    #     """Test paddle.true_divide with scalar"""
    #     x = paddle.to_tensor(self.x_np)
    #     out = paddle.true_divide(x, self.scalar)
    #     expected = self.x_np / self.scalar
    #     np.testing.assert_allclose(out.numpy(), expected, rtol=1e-6)

    def test_paddle_true_divide_static_graph(self):
        """Test paddle.true_divide in static graph"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[-1, 3], dtype='float32')
            y = paddle.static.data(name='y', shape=[-1, 3], dtype='float32')
            out1 = paddle.true_divide(x, y)
            out2 = paddle.true_divide(input=x, other=y)

            exe = paddle.static.Executor(self.place)
            res = exe.run(
                feed={
                    'x': self.x_np.reshape(1, 3),
                    'y': self.y_np.reshape(1, 3),
                },
                fetch_list=[out1, out2],
            )

            expected = self.x_np / self.y_np
            for result in res:
                np.testing.assert_allclose(
                    result.flatten(), expected, rtol=1e-6
                )
        paddle.disable_static()


class TestPaddleDivWithOut(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([4.0, 9.0, 16.0], dtype='float32')
        self.y_np = np.array([2.0, 3.0, 4.0], dtype='float32')
        self.place = (
            core.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else core.CPUPlace()
        )

    def run_div_test(self, test_type):
        """Helper function to test different out parameter scenarios"""
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        out = paddle.zeros_like(x)
        out.stop_gradient = False

        if test_type == "return":
            out = paddle.div(x, y)
        elif test_type == "input_out":
            paddle.div(x, y, out=out)
        elif test_type == "both_return":
            out = paddle.div(x, y, out=out)
        elif test_type == "both_input_out":
            tmp = paddle.div(x, y, out=out)

        expected = self.x_np / self.y_np
        np.testing.assert_allclose(
            out.numpy(),
            expected,
            rtol=1e-20,
            atol=1e-20,
        )

        loss = out.sum()
        loss.backward()

        return out, x.grad, y.grad, out.grad

    def test_div_with_out(self):
        """Test paddle.div with out parameter variations"""
        out1, x1, y1, o1 = self.run_div_test("return")
        out2, x2, y2, o2 = self.run_div_test("input_out")
        out3, x3, y3, o3 = self.run_div_test("both_return")
        out4, x4, y4, o4 = self.run_div_test("both_input_out")

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


class TestPaddleDivideWithOut(unittest.TestCase):
    def setUp(self):
        self.x_np = np.array([4.0, 9.0, 16.0], dtype='float32')
        self.y_np = np.array([2.0, 3.0, 4.0], dtype='float32')
        self.place = (
            core.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else core.CPUPlace()
        )

    def run_divide_test(self, test_type):
        """Helper function to test different out parameter scenarios"""
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        out = paddle.zeros_like(x)
        out.stop_gradient = False

        if test_type == "return":
            out = paddle.divide(x, y)
        elif test_type == "input_out":
            paddle.divide(x, y, out=out)
        elif test_type == "both_return":
            out = paddle.divide(x, y, out=out)
        elif test_type == "both_input_out":
            tmp = paddle.divide(x, y, out=out)

        expected = self.x_np / self.y_np
        np.testing.assert_allclose(
            out.numpy(),
            expected,
            rtol=1e-20,
            atol=1e-20,
        )

        loss = out.sum()
        loss.backward()

        return out, x.grad, y.grad, out.grad

    def test_divide_with_out(self):
        """Test paddle.divide with out parameter variations"""
        out1, x1, y1, o1 = self.run_divide_test("return")
        out2, x2, y2, o2 = self.run_divide_test("input_out")
        out3, x3, y3, o3 = self.run_divide_test("both_return")
        out4, x4, y4, o4 = self.run_divide_test("both_input_out")

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
