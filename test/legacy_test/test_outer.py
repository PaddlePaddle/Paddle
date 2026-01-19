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
from op_test import (
    convert_float_to_uint16,
    convert_uint16_to_float,
    get_device_place,
)

import paddle


class TestMultiplyApi(unittest.TestCase):
    def _run_static_graph_case(self, x_data, y_data):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            paddle.enable_static()
            x = paddle.static.data(
                name='x', shape=x_data.shape, dtype=x_data.dtype
            )
            y = paddle.static.data(
                name='y', shape=y_data.shape, dtype=y_data.dtype
            )
            res = paddle.outer(x, y)

            place = get_device_place()
            exe = paddle.static.Executor(place)
            outs = exe.run(
                paddle.static.default_main_program(),
                feed={'x': x_data, 'y': y_data},
                fetch_list=[res],
            )
            res = outs[0]
            return res

    def _run_dynamic_graph_case(self, x_data, y_data):
        paddle.disable_static()
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        res = paddle.outer(x, y)
        return res.numpy()

    def test_multiply_static(self):
        np.random.seed(7)

        # test static computation graph: 3-d array
        x_data = np.random.rand(2, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 5, 10).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test static computation graph: 2-d array
        x_data = np.random.rand(200, 5).astype(np.float64)
        y_data = np.random.rand(50, 5).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test static computation graph: 1-d array
        x_data = np.random.rand(50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test static computation graph: 1-d int32 array
        x_data = np.random.rand(50).astype(np.int32)
        y_data = np.random.rand(50).astype(np.int32)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test static computation graph: 1-d int64 array
        x_data = np.random.rand(50).astype(np.int64)
        y_data = np.random.rand(50).astype(np.int64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test static computation graph: 3-d int32 big array
        x_data = np.random.randint(-80000, 80000, [5, 10, 10]).astype(np.int32)
        y_data = np.random.randint(-80000, 80000, [2, 10]).astype(np.int32)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test static computation graph: 3-d int64 big array
        x_data = np.random.randint(-80000, 80000, [5, 10, 10]).astype(np.int64)
        y_data = np.random.randint(-80000, 80000, [2, 10]).astype(np.int64)
        res = self._run_static_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

    def test_multiply_dynamic(self):
        # test dynamic computation graph: 3-d array
        x_data = np.random.rand(5, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 10).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: 2-d array
        x_data = np.random.rand(20, 50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: Scalar
        x_data = np.random.rand(20, 10).astype(np.float32)
        y_data = np.random.rand(1).astype(np.float32).item()
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=10000.0)

        # test dynamic computation graph: 2-d array Complex
        x_data = np.random.rand(20, 50).astype(
            np.float64
        ) + 1j * np.random.rand(20, 50).astype(np.float64)
        y_data = np.random.rand(50).astype(np.float64) + 1j * np.random.rand(
            50
        ).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: 3-d array Complex
        x_data = np.random.rand(5, 10, 10).astype(
            np.float64
        ) + 1j * np.random.rand(5, 10, 10).astype(np.float64)
        y_data = np.random.rand(2, 10).astype(np.float64) + 1j * np.random.rand(
            2, 10
        ).astype(np.float64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: 3-d int32 array
        x_data = np.random.rand(5, 10, 10).astype(np.int32)
        y_data = np.random.rand(2, 10).astype(np.int32)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: 3-d int64 array
        x_data = np.random.rand(5, 10, 10).astype(np.int64)
        y_data = np.random.rand(2, 10).astype(np.int64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: 3-d int32 big array
        x_data = np.random.randint(-80000, 80000, [5, 10, 10]).astype(np.int32)
        y_data = np.random.randint(-80000, 80000, [2, 10]).astype(np.int32)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)

        # test dynamic computation graph: 3-d int64 big array
        x_data = np.random.randint(-80000, 80000, [5, 10, 10]).astype(np.int64)
        y_data = np.random.randint(-80000, 80000, [2, 10]).astype(np.int64)
        res = self._run_dynamic_graph_case(x_data, y_data)
        np.testing.assert_allclose(res, np.outer(x_data, y_data), rtol=1e-05)


class TestMultiplyError(unittest.TestCase):
    def test_errors_dynamic(self):
        np.random.seed(7)

        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float64)
        y_data = np.random.randn(200).astype(np.float64)
        y = paddle.to_tensor(y_data)
        self.assertRaisesRegex(
            ValueError,
            r"multiply\(\): argument 'x' \(position 0\) must be Tensor, but got numpy.ndarray ",
            paddle.outer,
            x_data,
            y,
        )

        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        x = paddle.to_tensor(x_data)
        self.assertRaisesRegex(
            ValueError,
            r"multiply\(\): argument 'y' \(position 1\) must be Tensor, but got numpy.ndarray ",
            paddle.outer,
            x,
            y_data,
        )

        # test dynamic computation graph: dtype must be Tensor type
        x_data = np.random.randn(200).astype(np.float32)
        y_data = np.random.randn(200).astype(np.float32)
        self.assertRaisesRegex(
            ValueError,
            r"multiply\(\): argument 'x' \(position 0\) must be Tensor, but got numpy.ndarray",
            paddle.outer,
            x_data,
            y_data,
        )


class TestMultiplyApi_ZeroSize(unittest.TestCase):
    def test_multiply_dynamic(self):
        x_data = np.random.rand(5, 10, 0).astype(np.float64)
        y_data = np.random.rand(0, 10).astype(np.float64)
        paddle.disable_static()
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        x.stop_gradient = False
        y.stop_gradient = False
        res = paddle.outer(x, y)
        np.testing.assert_allclose(
            res.numpy(), np.outer(x_data, y_data), rtol=1e-05
        )
        loss = paddle.sum(res)
        loss.backward()
        np.testing.assert_allclose(x.grad.shape, x.shape)

    def test_multiply_dynamic1(self):
        x_data = np.random.rand(0).astype(np.float32)
        y_data = np.random.rand(1).astype(np.float32)
        paddle.disable_static()
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        res = paddle.outer(x, y)
        np_res = np.outer(x_data, y_data)
        np.testing.assert_allclose(res.numpy(), np_res, rtol=1e-05)


class TestOuterOutAndParamDecorator(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        self.shape = [3]
        self.out_shape = [self.shape[0], self.shape[0]]
        self.x_np = np.random.rand(*self.shape).astype("float32")
        self.y_np = np.random.rand(*self.shape).astype("float32")

        self.apis = [paddle.outer, paddle.ger]

        self.test_types = ["decorator1", "decorator2", "out", "out_decorator"]

    def do_test(self, api, test_type):
        x = paddle.to_tensor(self.x_np)
        y = paddle.to_tensor(self.y_np)
        x.stop_gradient = y.stop_gradient = False
        out = paddle.zeros(self.out_shape, dtype="float32")
        out.stop_gradient = False

        if test_type == "raw":
            out = api(x, y)
            loss = out.mean()
            loss.backward()
            x_grad, y_grad = x.grad, y.grad
            return out, x_grad, y_grad
        elif test_type == "decorator1":
            res = api(x, vec2=y)
            loss = res.mean()
            loss.backward()
            x_grad, y_grad = x.grad, y.grad
            return res, x_grad, y_grad
        elif test_type == "decorator2":
            out = api(vec2=y, input=x)
            loss = out.mean()
            loss.backward()
            x_grad, y_grad = x.grad, y.grad
            return out, x_grad, y_grad
        elif test_type == "out":
            res = api(x, y, out=out)
            loss = out.mean()
            loss.backward()
            x_grad, y_grad = x.grad, y.grad
            return out, x_grad, y_grad
        elif test_type == "out_decorator":
            res = api(out=out, vec2=y, input=x)
            loss = out.mean()
            loss.backward()
            x_grad, y_grad = x.grad, y.grad
            return out, x_grad, y_grad
        else:
            raise NotImplementedError(
                f"Test type {test_type} is not implemented."
            )

    def test_outer_out_decorator(self):
        out_std, x_grad_std, y_grad_std = self.do_test(paddle.outer, "raw")
        for api in self.apis:
            for test_type in self.test_types:
                out, x_grad, y_grad = self.do_test(api, test_type)
                np.testing.assert_allclose(
                    out.numpy(), out_std.numpy(), rtol=1e-20
                )
                np.testing.assert_allclose(
                    x_grad.numpy(), x_grad_std.numpy(), rtol=1e-20
                )
                np.testing.assert_allclose(
                    y_grad.numpy(), y_grad_std.numpy(), rtol=1e-20
                )


class TestOuterAlias(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

    def test_outer_alias(self):
        """
        Test the alias of outer function.
        ``outer(input=x, vec2=y)`` is equivalent to ``outer(x=x, y=y)``
        """
        shape_cases = [
            [2],
            [2, 4],
            [2, 4, 8],
        ]
        dtype_cases = [
            "float32",
            "float64",
            "int32",
            "int64",
        ]
        if paddle.is_compiled_with_cuda():
            dtype_cases.extend(["float16", "bfloat16"])

        for shape in shape_cases:
            for dtype in dtype_cases:
                x = paddle.rand(shape).astype(dtype)
                y = paddle.rand(shape).astype(dtype)

                # Test all alias combinations
                combinations = [
                    {"x": x, "y": y},
                    {"input": x, "y": y},
                    {"x": x, "vec2": y},
                    {"input": x, "vec2": y},
                ]

                x_numpy = x.numpy()
                y_numpy = y.numpy()

                # Get baseline result
                if dtype == "bfloat16":
                    x_numpy = convert_uint16_to_float(x_numpy)
                    y_numpy = convert_uint16_to_float(y_numpy)
                expected = np.outer(x_numpy, y_numpy)
                if dtype == "bfloat16":
                    expected = convert_float_to_uint16(expected)

                rtol = 1e-5 if dtype != "bfloat16" else 1e-4

                for params in combinations:
                    out = paddle.outer(**params)
                    np.testing.assert_allclose(out.numpy(), expected, rtol=rtol)


if __name__ == '__main__':
    unittest.main()
