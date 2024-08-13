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

import os
import unittest

import numpy as np
from op_test import convert_float_to_uint16, convert_uint16_to_float

import paddle
from paddle import base
from paddle.base import core


def np_sinc(x: np.ndarray):
    tmp = np.sinc(x)
    return np.where(~np.isnan(tmp), tmp, np.full_like(x, 1.0))


def np_sinc_gradient(x: np.ndarray):
    x = np.pi * np.where(x == 0, 1.0e-20, x)
    s = np.sin(x)
    c = np.cos(x)
    tmp = np.pi * (x * c - s) / x**2
    return np.where(~np.isnan(tmp), tmp, np.full_like(x, 0.0))


class TestSincAPI(unittest.TestCase):
    def setUp(self):
        self.support_dtypes = [
            'float32',
            'float64',
        ]
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))
        self.shapes = [[6], [16, 64]]

    def test_dtype(self):
        def run_dygraph(place):
            paddle.disable_static(place)
            for dtype in self.support_dtypes:
                for shape in self.shapes:
                    x_data = np.random.rand(*shape).astype(dtype)
                    x = paddle.to_tensor(x_data)
                    x.stop_gradient = False
                    out = paddle.sinc(x)
                    out.backward()
                    x_grad = x.grad
                    out_expected = np_sinc(x_data)
                    np_grad_expected = np_sinc_gradient(x_data)
                    np.testing.assert_allclose(
                        out.numpy(), out_expected, rtol=1e-6, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        x_grad.numpy(), np_grad_expected, rtol=1e-6, atol=0.02
                    )

        def run_static(place):
            paddle.enable_static()
            for dtype in self.support_dtypes:
                for shape in self.shapes:
                    x_data = np.random.rand(*shape).astype(dtype)
                    startup_program = paddle.static.Program()
                    main_program = paddle.static.Program()
                    exe = base.Executor(place)
                    with paddle.static.program_guard(
                        main_program, startup_program
                    ):
                        x = paddle.static.data(
                            name='x', shape=shape, dtype=dtype
                        )
                        x.stop_gradient = False
                        res = paddle.sinc(x)
                        x_grad = paddle.static.gradients(res, x)
                        [static_result, static_grad_result] = exe.run(
                            feed={'x': x_data}, fetch_list=[res, x_grad]
                        )
                        out_expected = np_sinc(x_data)
                        np_grad_expected = np_sinc_gradient(x_data)
                    np.testing.assert_allclose(
                        static_result, out_expected, rtol=1e-6, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        static_grad_result,
                        np_grad_expected,
                        rtol=1e-6,
                        atol=0.02,
                    )

        for place in self.place:
            run_dygraph(place)
            run_static(place)

    def test_zero(self):
        def run_dygraph(place):
            paddle.disable_static(place)
            for dtype in self.support_dtypes:
                for shape in self.shapes:
                    x_data = np.random.rand(*shape).astype(dtype)
                    mask = (
                        (np.random.rand(*shape) > 0.5)
                        .astype('int')
                        .astype(dtype)
                    )
                    x_data = x_data * mask
                    x = paddle.to_tensor(x_data)
                    x.stop_gradient = False
                    out = paddle.sinc(x)
                    out.backward()
                    x_grad = x.grad
                    out_expected = np_sinc(x_data)
                    np_grad_expected = np_sinc_gradient(x_data)
                    np.testing.assert_allclose(
                        out.numpy(), out_expected, rtol=1e-6, atol=1e-6
                    )
                    np.testing.assert_allclose(
                        x_grad.numpy(), np_grad_expected, rtol=1e-6, atol=0.02
                    )

        for place in self.place:
            run_dygraph(place)

    def test_input_type_error(self):
        with self.assertRaises(TypeError):
            x = np.random.rand(6).astype('float32')
            x = paddle.sinc(x)

    def test_input_dype_error(self):
        paddle.enable_static()
        place = paddle.CPUPlace()
        with self.assertRaises(TypeError):
            x_data = np.random.rand(6).astype('int32')
            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()
            exe = base.Executor(place)
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name='x', shape=[6], dtype='int32')
                res = paddle.sinc(x)
                static_result = exe.run(feed={'x': x_data}, fetch_list=[res])[0]

        with self.assertRaises(TypeError):
            x_data = np.random.rand(6).astype('int64')
            startup_program = paddle.static.Program()
            main_program = paddle.static.Program()
            exe = base.Executor(place)
            with paddle.static.program_guard(main_program, startup_program):
                x = paddle.static.data(name='x', shape=[6], dtype='int64')
                res = paddle.sinc(x)
                static_result = exe.run(feed={'x': x_data}, fetch_list=[res])[0]


class TestSincInplaceAPI(unittest.TestCase):
    def setUp(self):
        self.support_dtypes = [
            'float32',
            'float64',
        ]
        self.place = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.place.append(paddle.CPUPlace())
        if core.is_compiled_with_cuda():
            self.place.append(paddle.CUDAPlace(0))
        self.shapes = [[6], [16, 64]]

    def test_inplace(self):
        def run_dygraph(place):
            paddle.disable_static(place)
            for dtype in self.support_dtypes:
                for shape in self.shapes:
                    x_data = np.random.rand(*shape).astype(dtype)
                    x = paddle.to_tensor(x_data)
                    paddle.sinc_(x)
                    out_expected = np_sinc(x_data)
                    np.testing.assert_allclose(
                        x.numpy(), out_expected, rtol=1e-6, atol=1e-6
                    )

        for place in self.place:
            run_dygraph(place)

    def test_inplace_input_type_error(self):
        with self.assertRaises(TypeError):
            x = np.random.rand(6).astype('float32')
            paddle.sinc_(x)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_float16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the float16",
)
class TestSincAPIFP16(unittest.TestCase):
    def setUp(self):
        self.shapes = [[6], [16, 64]]
        self.dtype = 'float16'
        self.place = paddle.CUDAPlace(0)

    def test_dtype(self):
        def run_static(place):
            paddle.enable_static()
            for shape in self.shapes:
                x_data = np.random.rand(*shape).astype(self.dtype)
                startup_program = paddle.static.Program()
                main_program = paddle.static.Program()
                exe = base.Executor(place)
                with paddle.static.program_guard(main_program, startup_program):
                    x = paddle.static.data(
                        name='x', shape=shape, dtype=self.dtype
                    )
                    x.stop_gradient = False
                    res = paddle.sinc(x)
                    x_grad = paddle.static.gradients(res, x)
                    [static_result, static_grad_result] = exe.run(
                        feed={'x': x_data}, fetch_list=[res, x_grad]
                    )
                    out_expected = np_sinc(x_data)
                    np_grad_expected = np_sinc_gradient(x_data)
                np.testing.assert_allclose(
                    static_result, out_expected, rtol=1e-6, atol=1e-6
                )
                np.testing.assert_allclose(
                    static_grad_result, np_grad_expected, rtol=0.1, atol=0.1
                )

        run_static(self.place)

    def test_zero(self):
        def run_static(place):
            paddle.enable_static()
            for shape in self.shapes:
                x_data = np.random.rand(*shape).astype(self.dtype)
                mask = (
                    (np.random.rand(*shape) > 0.5)
                    .astype('int')
                    .astype(self.dtype)
                )
                x_data = x_data * mask
                startup_program = paddle.static.Program()
                main_program = paddle.static.Program()
                exe = base.Executor(place)
                with paddle.static.program_guard(main_program, startup_program):
                    x = paddle.static.data(
                        name='x', shape=shape, dtype=self.dtype
                    )
                    x.stop_gradient = False
                    res = paddle.sinc(x)
                    x_grad = paddle.static.gradients(res, x)
                    [static_result, static_grad_result] = exe.run(
                        feed={'x': x_data}, fetch_list=[res, x_grad]
                    )
                    out_expected = np_sinc(x_data)
                    np_grad_expected = np_sinc_gradient(x_data)
                np.testing.assert_allclose(
                    static_result, out_expected, rtol=1e-6, atol=1e-6
                )
                np.testing.assert_allclose(
                    static_grad_result, np_grad_expected, rtol=0.1, atol=0.1
                )

        run_static(self.place)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestSincAPIBF16(unittest.TestCase):
    def setUp(self):
        self.shapes = [[6], [16, 64]]
        self.dtype = 'uint16'
        self.place = paddle.CUDAPlace(0)

    def test_dtype(self):
        def run(place):
            paddle.enable_static()
            for shape in self.shapes:
                x_data_np = np.random.rand(*shape).astype('float32')
                x_data = convert_float_to_uint16(x_data_np)
                startup_program = paddle.static.Program()
                main_program = paddle.static.Program()
                exe = base.Executor(place)
                with paddle.static.program_guard(main_program, startup_program):
                    x = paddle.static.data(
                        name='x', shape=shape, dtype=self.dtype
                    )
                    x.stop_gradient = False
                    res = paddle.sinc(x)
                    x_grad = paddle.static.gradients(res, x)
                    [static_result, static_grad_result] = exe.run(
                        feed={'x': x_data}, fetch_list=[res, x_grad]
                    )
                    out_expected = np_sinc(x_data_np)
                    np_grad_expected = np_sinc_gradient(x_data_np)
                result = convert_uint16_to_float(static_result)
                grad_result = convert_uint16_to_float(static_grad_result)
                np.testing.assert_allclose(
                    result, out_expected, rtol=1e-3, atol=1e-2
                )
                np.testing.assert_allclose(
                    grad_result, np_grad_expected, atol=0.2
                )

        run(self.place)

    def test_zero(self):
        def run(place):
            paddle.enable_static()
            for shape in self.shapes:
                x_data_np = np.random.rand(*shape).astype('float32')
                mask = (
                    (np.random.rand(*shape) > 0.5)
                    .astype('int')
                    .astype('float32')
                )
                x_data_np = x_data_np * mask
                x_data = convert_float_to_uint16(x_data_np)
                startup_program = paddle.static.Program()
                main_program = paddle.static.Program()
                exe = base.Executor(place)
                with paddle.static.program_guard(main_program, startup_program):
                    x = paddle.static.data(
                        name='x', shape=shape, dtype=self.dtype
                    )
                    x.stop_gradient = False
                    res = paddle.sinc(x)
                    x_grad = paddle.static.gradients(res, x)
                    [static_result, static_grad_result] = exe.run(
                        feed={'x': x_data}, fetch_list=[res, x_grad]
                    )
                    out_expected = np_sinc(x_data_np)
                    np_grad_expected = np_sinc_gradient(x_data_np)
                result = convert_uint16_to_float(static_result)
                grad_result = convert_uint16_to_float(static_grad_result)
                np.testing.assert_allclose(
                    result, out_expected, rtol=1e-3, atol=1e-2
                )
                np.testing.assert_allclose(
                    grad_result, np_grad_expected, atol=0.2
                )

        run(self.place)


if __name__ == "__main__":
    unittest.main()
