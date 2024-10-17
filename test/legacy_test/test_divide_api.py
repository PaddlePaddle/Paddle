# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.pir_utils import test_with_pir_api


def ref_divide(x, y, rounding_mode):
    out = np.divide(x, y)
    if rounding_mode == "trunc":
        out = np.trunc(out)
    elif rounding_mode == "floor":
        out = np.floor(out)
    if x.dtype in [np.int32, np.int64] and y.dtype in [np.int32, np.int64]:
        if rounding_mode is None:
            out = out.astype(np.float32)  # numpy cast to float64 by default
        else:
            out = out.astype(x.dtype)
    return out


class TestDivideAPI(unittest.TestCase):
    def setUp(self):
        self.rounding_mode = ["trunc", "floor", None]
        self.dtype = np.float32
        self.x = np.random.randn(2, 3, 4)
        self.y = np.random.randn(2, 3, 4)
        self.devices = ['cpu']
        if paddle.device.is_compiled_with_cuda():
            self.devices.append('gpu')

    def test_dygraph(self):
        paddle.disable_static()
        for device in self.devices:
            for rounding_mode in self.rounding_mode:
                x = paddle.to_tensor(
                    self.x.copy(), dtype=self.dtype, place=device
                )
                y = paddle.to_tensor(
                    self.y.copy(), dtype=self.dtype, place=device
                )
                paddle_res = paddle.divide(x, y, rounding_mode).numpy()
                paddle_inplace_res = paddle.divide_(x, y, rounding_mode).numpy()
                np_res = ref_divide(
                    self.x.astype(self.dtype),
                    self.y.astype(self.dtype),
                    rounding_mode,
                )
                np.testing.assert_allclose(
                    paddle_res, np_res, err_msg=f'{rounding_mode}'
                )
                np.testing.assert_allclose(
                    paddle_inplace_res, np_res, err_msg=f'{rounding_mode}'
                )
                assert (
                    paddle_res.dtype == np_res.dtype
                ), f'{paddle_res.dtype, np_res.dtype, rounding_mode}'

    def test_static(self):
        paddle.enable_static()
        s_p = paddle.static.Program()
        m_p = paddle.static.Program()
        with paddle.static.program_guard(m_p, s_p):
            for device in self.devices:
                for round_mode in self.rounding_mode:
                    x = paddle.static.data(
                        name="x", shape=self.x.shape, dtype=self.dtype
                    )
                    y = paddle.static.data(
                        name="y", shape=self.y.shape, dtype=self.dtype
                    )

                    results = paddle.divide(x, y, round_mode)
                    inplace_results = paddle.divide_(x, y, round_mode)
                    x_np = self.x.astype(self.dtype)
                    y_np = self.y.astype(self.dtype)
                    np_res = ref_divide(x_np, y_np, round_mode)

                    exe = paddle.static.Executor(device)
                    paddle_res, paddle_inplace_res = exe.run(
                        paddle.static.default_main_program(),
                        feed={"x": x_np, "y": y_np},
                        fetch_list=[results, inplace_results],
                    )
                    assert (
                        paddle_res.dtype == np_res.dtype
                    ), f'{paddle_res.dtype, np_res.dtype, round_mode}'


class TestDivideFloat64(TestDivideAPI):
    def setUp(self):
        super().setUp()
        self.dtype = np.float64


class TestDivideInt32(TestDivideAPI):
    def setUp(self):
        super().setUp()
        self.dtype = np.int32
        self.x = np.random.randint(-10, 10, (2, 3, 5, 6))
        self.y = np.random.randint(-10, 10, (2, 3, 5, 6))
        self.y[self.y == 0] = 4


class TestDivideInt64(TestDivideInt32):
    def setUp(self):
        super().setUp()
        self.dtype = np.int64


class TestDivideError(unittest.TestCase):
    def test_error(self):
        def error():
            paddle_res = paddle.divide(
                paddle.to_tensor(1), paddle.to_tensor(2), "error"
            )

        def error1():
            paddle_res = paddle.divide_(
                paddle.to_tensor(1), paddle.to_tensor(2), "error"
            )

        self.assertRaises(ValueError, error)
        self.assertRaises(ValueError, error1)


class TestDividePIR(TestDivideAPI):
    @test_with_pir_api
    def test_with_pir(self):
        paddle.enable_static()
        for device in self.devices:
            for round_mode in self.rounding_mode:
                x = paddle.static.data(
                    name="x", shape=self.x.shape, dtype=self.dtype
                )
                y = paddle.static.data(
                    name="x", shape=self.y.shape, dtype=self.dtype
                )

                results = paddle.divide(x, y, round_mode)
                inplace_results = paddle.divide_(x, y, round_mode)
                x_np = self.x.astype(self.dtype)
                y_np = self.y.astype(self.dtype)
                np_res = ref_divide(x_np, y_np, round_mode)

                exe = paddle.static.Executor(device)
                paddle_res, paddle_inplace_res = exe.run(
                    paddle.static.default_main_program(),
                    feed={"x": x_np, "y": y_np},
                    fetch_list=[results, inplace_results],
                )
                assert (
                    paddle_res.dtype == np_res.dtype
                ), f'{paddle_res.dtype, np_res.dtype}'
                assert (
                    paddle_inplace_res.dtype == np_res.dtype
                ), f'{paddle_inplace_res.dtype, np_res.dtype}'


if __name__ == '__main__':
    unittest.main()
