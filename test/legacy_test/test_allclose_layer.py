#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base

paddle.enable_static()


class TestAllcloseLayer(unittest.TestCase):
    def allclose_check(self, use_cuda, dtype='float32'):
        a = paddle.static.data(name="a", shape=[2], dtype=dtype)
        b = paddle.static.data(name="b", shape=[2], dtype=dtype)

        result = paddle.allclose(
            a, b, rtol=1e-05, atol=1e-08, equal_nan=False, name="ignore_nan"
        )
        result_nan = paddle.allclose(
            a, b, rtol=1e-05, atol=1e-08, equal_nan=True, name="equal_nan"
        )
        result_corner = paddle.allclose(
            a, b, rtol=0.01, atol=0.0, name="corner_case"
        )

        place = base.CUDAPlace(0) if use_cuda else base.CPUPlace()
        exe = base.Executor(place)
        exe.run(base.default_startup_program())

        x = np.array([10000.0, 1e-07]).astype(dtype)
        y = np.array([10000.1, 1e-08]).astype(dtype)
        result_v, result_nan_v, result_c = exe.run(
            feed={'a': x, 'b': y},
            fetch_list=[result, result_nan, result_corner],
        )
        self.assertEqual(result_v, False)
        self.assertEqual(result_nan_v, False)

        x = np.array([10000.0, 1e-08]).astype(dtype)
        y = np.array([10000.1, 1e-09]).astype(dtype)
        result_v, result_nan_v, result_c = exe.run(
            feed={'a': x, 'b': y},
            fetch_list=[result, result_nan, result_corner],
        )
        self.assertEqual(result_v, True)
        self.assertEqual(result_nan_v, True)

        x = np.array([1.0, float('nan')]).astype(dtype)
        y = np.array([1.0, float('nan')]).astype(dtype)
        result_v, result_nan_v, result_c = exe.run(
            feed={'a': x, 'b': y},
            fetch_list=[result, result_nan, result_corner],
        )
        self.assertEqual(result_v, False)
        self.assertEqual(result_nan_v, True)

        # for corner case
        x = np.array([10.1, 10.1]).astype(dtype)
        y = np.array([10, 10]).astype(dtype)
        result_v, result_nan_v, result_c = exe.run(
            feed={'a': x, 'b': y},
            fetch_list=[result, result_nan, result_corner],
        )
        corner_res = dtype == 'float64'
        self.assertEqual(result_c, corner_res)

    def test_allclose_cpu_fp32(self):
        main = base.Program()
        startup = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, startup):
                self.allclose_check(use_cuda=False, dtype='float32')

    def test_allclose_cpu_fp64(self):
        main = base.Program()
        startup = base.Program()
        with base.unique_name.guard():
            with base.program_guard(main, startup):
                self.allclose_check(use_cuda=False, dtype='float64')

    def test_allclose_gpu_fp32(self):
        if base.core.is_compiled_with_cuda():
            main = base.Program()
            startup = base.Program()
            with base.unique_name.guard():
                with base.program_guard(main, startup):
                    self.allclose_check(use_cuda=True, dtype='float32')

    def test_allclose_gpu_fp64(self):
        if base.core.is_compiled_with_cuda():
            main = base.Program()
            startup = base.Program()
            with base.unique_name.guard():
                with base.program_guard(main, startup):
                    self.allclose_check(use_cuda=True, dtype='float64')

    def test_dygraph_mode(self):
        x_1 = np.array([10000.0, 1e-07]).astype("float32")
        y_1 = np.array([10000.1, 1e-08]).astype("float32")
        x_2 = np.array([10000.0, 1e-08]).astype("float32")
        y_2 = np.array([10000.1, 1e-09]).astype("float32")
        x_3 = np.array([1.0, float('nan')]).astype("float32")
        y_3 = np.array([1.0, float('nan')]).astype("float32")
        x_4 = np.array([10.1]).astype("float32")
        y_4 = np.array([10]).astype("float32")
        x_5 = np.array([10.1]).astype("float64")
        y_5 = np.array([10]).astype("float64")

        with base.dygraph.guard():
            x_v_1 = paddle.to_tensor(x_1)
            y_v_1 = paddle.to_tensor(y_1)
            ret_1 = paddle.allclose(
                x_v_1,
                y_v_1,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False,
                name='test_1',
            )
            self.assertEqual(ret_1.numpy(), False)
            ret_1 = paddle.allclose(
                x_v_1,
                y_v_1,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=True,
                name='test_2',
            )
            self.assertEqual(ret_1.numpy(), False)
            x_v_2 = paddle.to_tensor(x_2)
            y_v_2 = paddle.to_tensor(y_2)
            ret_2 = paddle.allclose(
                x_v_2,
                y_v_2,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False,
                name='test_3',
            )
            self.assertEqual(ret_2.numpy(), True)
            ret_2 = paddle.allclose(
                x_v_2,
                y_v_2,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=True,
                name='test_4',
            )
            self.assertEqual(ret_2.numpy(), True)
            x_v_3 = paddle.to_tensor(x_3)
            y_v_3 = paddle.to_tensor(y_3)
            ret_3 = paddle.allclose(
                x_v_3,
                y_v_3,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=False,
                name='test_5',
            )
            self.assertEqual(ret_3.numpy(), False)
            ret_3 = paddle.allclose(
                x_v_3,
                y_v_3,
                rtol=1e-05,
                atol=1e-08,
                equal_nan=True,
                name='test_6',
            )
            self.assertEqual(ret_3.numpy(), True)
            # for corner case
            x_v_4 = paddle.to_tensor(x_4)
            y_v_4 = paddle.to_tensor(y_4)
            ret_4 = paddle.allclose(
                x_v_4, y_v_4, rtol=0.01, atol=0.0, name='test_7'
            )
            self.assertEqual(ret_4.numpy(), False)
            x_v_5 = paddle.to_tensor(x_5)
            y_v_5 = paddle.to_tensor(y_5)
            ret_5 = paddle.allclose(
                x_v_5, y_v_5, rtol=0.015, atol=0.0, name='test_8'
            )
            self.assertEqual(ret_5.numpy(), True)


if __name__ == "__main__":
    unittest.main()
