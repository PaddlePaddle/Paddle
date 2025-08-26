# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_devices

import paddle
from paddle.static import Program, program_guard

DYNAMIC = 1
STATIC = 2


def _run_power(mode, x, y, device='cpu'):
    # dynamic mode
    if mode == DYNAMIC:
        paddle.disable_static()
        # Set device
        paddle.set_device(device)
        # y is scalar
        if isinstance(y, (int, float)):
            x_ = paddle.to_tensor(x)
            y_ = y
            res = paddle.pow(x_, y_)
            return res.numpy()
        # y is tensor
        else:
            x_ = paddle.to_tensor(x)
            y_ = paddle.to_tensor(y)
            res = paddle.pow(x_, y_)
            return res.numpy()
    # static graph mode
    elif mode == STATIC:
        paddle.enable_static()
        # y is scalar
        if isinstance(y, (int, float)):
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = y
                res = paddle.pow(x_, y_)
                place = (
                    paddle.CPUPlace()
                    if device == 'cpu'
                    else paddle.CUDAPlace(0)
                )
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x}, fetch_list=[res])
                return outs[0]
        # y is tensor
        else:
            with program_guard(Program(), Program()):
                x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
                y_ = paddle.static.data(name="y", shape=y.shape, dtype=y.dtype)
                res = paddle.pow(x_, y_)
                place = (
                    paddle.CPUPlace()
                    if device == 'cpu'
                    else paddle.CUDAPlace(0)
                )
                exe = paddle.static.Executor(place)
                outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[res])
                return outs[0]


class TestPowerAPI(unittest.TestCase):
    """TestPowerAPI."""

    def setUp(self):
        self.places = get_devices()

    def test_power(self):
        """test_power."""
        np.random.seed(7)
        for place in self.places:
            # test 1-d float tensor ** float scalar
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = np.random.rand() * 10
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d float tensor ** int scalar
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = int(np.random.rand() * 10)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            x = (np.random.rand(*dims) * 10).astype(np.int64)
            y = int(np.random.rand() * 10)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d float tensor ** 1-d float tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.rand(*dims) * 10).astype(np.float64)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d int tensor ** 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int64)
            y = (np.random.rand(*dims) * 10).astype(np.int64)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d int tensor ** 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.int32)
            y = (np.random.rand(*dims) * 10).astype(np.int32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 1-d int tensor ** 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = (np.random.rand(*dims) * 10).astype(np.float32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test float scalar ** 2-d float tensor
            dims = (np.random.randint(2, 10), np.random.randint(5, 10))
            x = np.random.rand() * 10
            y = (np.random.rand(*dims) * 10).astype(np.float32)
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test 2-d float tensor ** float scalar
            dims = (np.random.randint(2, 10), np.random.randint(5, 10))
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = np.random.rand() * 10
            res = _run_power(DYNAMIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y, place)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)

            # test broadcast
            dims = (
                np.random.randint(1, 10),
                np.random.randint(5, 10),
                np.random.randint(5, 10),
            )
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.rand(dims[-1]) * 10).astype(np.float64)
            res = _run_power(DYNAMIC, x, y)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            res = _run_power(STATIC, x, y)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)


class TestPowerError(unittest.TestCase):
    """TestPowerError."""

    def test_errors(self):
        """test_errors."""
        np.random.seed(7)

        # test dynamic computation graph: inputs must be broadcastable
        dims = (
            np.random.randint(1, 10),
            np.random.randint(5, 10),
            np.random.randint(5, 10),
        )
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.float64)
        self.assertRaises(ValueError, _run_power, DYNAMIC, x, y)
        self.assertRaises(ValueError, _run_power, STATIC, x, y)

        # test dynamic computation graph: inputs must be broadcastable
        dims = (
            np.random.randint(1, 10),
            np.random.randint(5, 10),
            np.random.randint(5, 10),
        )
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.rand(dims[-1] + 1) * 10).astype(np.int8)
        self.assertRaises(TypeError, paddle.pow, x, y)

        # test 1-d float tensor ** int string
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = int(np.random.rand() * 10)
        self.assertRaises(TypeError, paddle.pow, x, str(y))

    def test_pir_error(self):
        with paddle.pir_utils.IrGuard():

            def x_dtype_error():
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    x = paddle.static.data('x', [2, 2], dtype='int8')
                    out = paddle.pow(x, 2)

            self.assertRaises(TypeError, x_dtype_error)


class TestPowerAPI_ZeroSize(unittest.TestCase):
    """TestPowerAPI."""

    def setUp(self):
        self.places = get_devices()

    def _test_power(self, shape):
        np.random.seed(7)
        for place in self.places:
            dims = shape
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = np.random.rand() * 10
            paddle.disable_static()
            paddle.set_device(place)
            x_ = paddle.to_tensor(x)
            x_.stop_gradient = False
            y_ = y
            res = paddle.pow(x_, y_)
            np.testing.assert_allclose(res, np.power(x, y), rtol=1e-05)
            loss = paddle.sum(res)
            loss.backward()
            np.testing.assert_allclose(x_.grad.shape, x_.shape)

    def test_power(self):
        self._test_power((0, 2))
        self._test_power((0, 0))


class TestPowerAPI_Specialization(unittest.TestCase):
    """TestPowerAPI."""

    def setUp(self):
        self.places = get_devices()

    def _test_power(self, factor: float):
        np.random.seed(7)
        inputs = [
            np.random.rand(10, 10) * 10,
            np.complex128(
                np.random.rand(10, 10) * 10 + 1j * np.random.rand(10, 10)
            ),
        ]
        for x in inputs:
            for place in self.places:
                paddle.disable_static()
                paddle.set_device(place)
                x_ = paddle.to_tensor(x)
                x_.stop_gradient = False
                res = paddle.pow(x_, factor)
                np.testing.assert_allclose(res, np.power(x, factor), rtol=1e-05)
                loss = paddle.sum(res)
                loss.backward()
                np.testing.assert_allclose(x_.grad.shape, x_.shape)

    def test_power(self):
        self._test_power(0)
        self._test_power(0.5)
        self._test_power(1.5)
        self._test_power(1)
        self._test_power(2)
        self._test_power(3)
        self._test_power(4)
        self._test_power(-0.5)
        self._test_power(-1)
        self._test_power(-2)


class TestPowerAPI_Alias(unittest.TestCase):
    """
    Test the alias of pow function.
    ``pow(input=2, exponent=1.1)`` is equivalent to ``pow(x=2, y=1.1)``
    """

    def setUp(self):
        self.places = get_devices()
        self.test_cases = [
            ([1.0, 2.0, 3.0], [1.1]),  # 1D tensor
            ([[1, 2], [3, 4]], 2),  # 2D tensor with scalar exponent
            (3.0, [2.0]),  # Scalar input
        ]

    def test_powxy(self):
        for alias_param_1 in ["x", "input"]:
            for alias_param_2 in ["y", "exponent"]:
                for place in self.places:
                    paddle.set_device(place)
                    paddle.disable_static(place)
                    for input_data, exp_data in self.test_cases:
                        input_tensor = paddle.to_tensor(input_data)
                        exp_tensor = paddle.to_tensor(exp_data)
                        output_alias = paddle.pow(
                            **{
                                alias_param_1: input_tensor,
                                alias_param_2: exp_tensor,
                            }
                        )
                        output_std = paddle.pow(x=input_tensor, y=exp_tensor)
                        self.assertTrue(
                            paddle.allclose(output_alias, output_std),
                            msg=f"Alias {alias_param_1}/{alias_param_2} failed on {place} with input {input_data}, exp {exp_data}",
                        )

    def test_xpowy(self):
        for alias_param_2 in ["y", "exponent"]:
            for place in self.places:
                paddle.set_device(place)
                paddle.disable_static(place)
                for input_data, exp_data in self.test_cases:
                    input_tensor = paddle.to_tensor(input_data)
                    exp_tensor = paddle.to_tensor(exp_data)
                    output_alias = input_tensor.pow(
                        **{alias_param_2: exp_tensor}
                    )
                    output_std = input_tensor.pow(y=exp_tensor)
                    self.assertTrue(
                        paddle.allclose(output_alias, output_std),
                        msg=f"Alias {alias_param_2} failed on {place} with input {input_data}, exp {exp_data}",
                    )


class TestPowOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.x_np = np.random.uniform(0.1, 1, [3, 4]).astype(np.float32)
        self.y_np = np.random.uniform(1, 3, [3, 4]).astype(np.float32)
        self.test_types = [
            "decorator_input",
            "decorator_exponent",
            "decorator_both",
            "out",
            "out_decorator",
        ]

    def do_test(self, test_type):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        y = paddle.to_tensor(self.y_np, stop_gradient=False)
        if test_type == 'raw':
            result = paddle.pow(x, y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'decorator_input':
            result = paddle.pow(input=x, y=y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'decorator_exponent':
            result = paddle.pow(x, exponent=y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'decorator_both':
            result = paddle.pow(input=x, exponent=y)
            result.mean().backward()
            return result, x.grad, y.grad
        elif test_type == 'out':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.pow(x, y, out=out)
            out.mean().backward()
            return out, x.grad, y.grad
        elif test_type == 'out_decorator':
            out = paddle.empty_like(x)
            out.stop_gradient = False
            paddle.pow(input=x, exponent=y, out=out)
            out.mean().backward()
            return out, x.grad, y.grad
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def test_all(self):
        out_std, x_grad_std, y_grad_std = self.do_test('raw')
        for test_type in self.test_types:
            out, x_grad, y_grad = self.do_test(test_type)
            np.testing.assert_allclose(out.numpy(), out_std.numpy(), rtol=1e-6)
            np.testing.assert_allclose(
                x_grad.numpy(), x_grad_std.numpy(), rtol=1e-6
            )
            np.testing.assert_allclose(
                y_grad.numpy(), y_grad_std.numpy(), rtol=1e-6
            )


if __name__ == '__main__':
    unittest.main()
