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

import os
import unittest

import numpy as np

import paddle
from paddle.base import core


def _run_ldexp_dynamic(x, y, device='cpu'):
    # dynamic mode
    paddle.disable_static()
    # Set device
    paddle.set_device(device)
    x_ = paddle.to_tensor(x)
    # y is scalar
    if isinstance(y, (int)):
        y_ = y
    # y is tensor
    else:
        y_ = paddle.to_tensor(y)
    res = paddle.ldexp(x_, y_)
    return res.numpy()


def _run_ldexp_static(x, y, device='cpu'):
    # static graph mode
    paddle.enable_static()
    # y is scalar
    if isinstance(y, (int)):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
            y_ = y
            res = paddle.ldexp(x_, y_)
            place = (
                paddle.CPUPlace() if device == 'cpu' else paddle.CUDAPlace(0)
            )
            exe = paddle.static.Executor(place)
            outs = exe.run(
                paddle.static.default_main_program(),
                feed={'x': x, 'y': y},
                fetch_list=[res],
            )
            return outs[0]
    # y is tensor
    else:
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x_ = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
            y_ = paddle.static.data(name="y", shape=y.shape, dtype=y.dtype)
            res = paddle.ldexp(x_, y_)
            place = (
                paddle.CPUPlace() if device == 'cpu' else paddle.CUDAPlace(0)
            )
            exe = paddle.static.Executor(place)
            outs = exe.run(
                paddle.static.default_main_program(),
                feed={'x': x, 'y': y},
                fetch_list=[res],
            )
            return outs[0]


def check_dtype(input, desired_dtype):
    if input.dtype != desired_dtype:
        raise ValueError(
            f"The expected data type to be obtained is {desired_dtype}, but got {input.dtype}"
        )


class TestLdexpAPIWithDynamic(unittest.TestCase):
    def setUp(self):
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append('cpu')
        if core.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_ldexp_dynamic(self):
        np.random.seed(7)
        for place in self.places:
            # test 1-d float tensor and 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp_dynamic(x, y, place)
            check_dtype(res, np.float64)
            np.testing.assert_allclose(res, np.ldexp(x, y))

            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp_dynamic(x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))

            # test 1-d int tensor and 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.randint(-10, 10, dims)).astype(np.int64)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp_dynamic(x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))

            dims = (np.random.randint(200, 300),)
            x = (np.random.randint(-10, 10, dims)).astype(np.int32)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp_dynamic(x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))

            # test broadcast
            dims = (
                np.random.randint(1, 10),
                np.random.randint(5, 10),
                np.random.randint(5, 10),
            )
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.randint(-10, 10, dims[-1])).astype(np.int32)
            res = _run_ldexp_dynamic(x, y)
            check_dtype(res, np.float64)
            np.testing.assert_allclose(res, np.ldexp(x, y))


class TestLdexpAPIWithStatic(unittest.TestCase):
    def setUp(self):
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append('cpu')
        if core.is_compiled_with_cuda():
            self.places.append('gpu')

    def test_ldexp_static(self):
        np.random.seed(7)
        for place in self.places:
            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp_static(x, y, place)
            check_dtype(res, np.float64)
            np.testing.assert_allclose(res, np.ldexp(x, y))

            dims = (np.random.randint(200, 300),)
            x = (np.random.rand(*dims) * 10).astype(np.float32)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp_static(x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))

            # test 1-d int tensor and 1-d int tensor
            dims = (np.random.randint(200, 300),)
            x = (np.random.randint(-10, 10, dims)).astype(np.int64)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp_static(x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))

            dims = (np.random.randint(200, 300),)
            x = (np.random.randint(-10, 10, dims)).astype(np.int32)
            y = (np.random.randint(-10, 10, dims)).astype(np.int32)
            res = _run_ldexp_static(x, y, place)
            check_dtype(res, np.float32)
            np.testing.assert_allclose(res, np.ldexp(x, y))

            # test broadcast
            dims = (
                np.random.randint(1, 10),
                np.random.randint(5, 10),
                np.random.randint(5, 10),
            )
            x = (np.random.rand(*dims) * 10).astype(np.float64)
            y = (np.random.randint(-10, 10, dims[-1])).astype(np.int32)
            res = _run_ldexp_static(x, y)
            check_dtype(res, np.float64)
            np.testing.assert_allclose(res, np.ldexp(x, y))


class TestLdexpError(unittest.TestCase):
    """TestLdexpError."""

    def test_errors(self):
        """test_errors."""
        np.random.seed(7)

        # test 1-d float and int tensor
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.randint(-10, 10, dims)).astype(np.int32)
        self.assertRaises(TypeError, paddle.ldexp, x, paddle.to_tensor(y))

        # test 1-d float tensor and int
        dims = (np.random.randint(200, 300),)
        x = (np.random.rand(*dims) * 10).astype(np.float64)
        y = (np.random.randint(-10, 10, dims)).astype(np.int32)
        self.assertRaises(TypeError, paddle.ldexp, paddle.to_tensor(x), y)


if __name__ == '__main__':
    unittest.main()
