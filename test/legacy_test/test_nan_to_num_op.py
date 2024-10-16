#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest

import numpy as np

import paddle
from paddle.base import core

# from op_test import OpTest


def np_nan_to_num(
    x: np.ndarray,
    nan: float = 0.0,
    posinf: float | None = None,
    neginf: float | None = None,
) -> np.ndarray:
    return np.nan_to_num(x, True, nan=nan, posinf=posinf, neginf=neginf)


def np_nan_to_num_op(
    x: np.ndarray,
    nan: float,
    replace_posinf_with_max: bool,
    posinf: float,
    replace_neginf_with_min: bool,
    neginf: float,
) -> np.ndarray:
    if replace_posinf_with_max:
        posinf = None
    if replace_neginf_with_min:
        neginf = None
    return np.nan_to_num(x, True, nan=nan, posinf=posinf, neginf=neginf)


def np_nan_to_num_grad(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    dx = np.copy(dout)
    dx[np.isnan(x) | (x == np.inf) | (x == -np.inf)] = 0
    return dx


class TestNanToNum(unittest.TestCase):
    def setUp(self):
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )

    def test_static(self):
        x_np = np.array([[1, np.nan, -2], [np.inf, 0, -np.inf]]).astype(
            np.float32
        )
        out1_np = np_nan_to_num(x_np)
        out2_np = np_nan_to_num(x_np, 1.0)
        out3_np = np_nan_to_num(x_np, 1.0, 9.0)
        out4_np = np_nan_to_num(x_np, 1.0, 9.0, -12.0)
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', x_np.shape)
            out1 = paddle.nan_to_num(x)
            out2 = paddle.nan_to_num(x, 1.0)
            out3 = paddle.nan_to_num(x, 1.0, 9.0)
            out4 = paddle.nan_to_num(x, 1.0, 9.0, -12.0)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': x_np}, fetch_list=[out1, out2, out3, out4])

        np.testing.assert_allclose(out1_np, res[0])
        np.testing.assert_allclose(out2_np, res[1])
        np.testing.assert_allclose(out3_np, res[2])
        np.testing.assert_allclose(out4_np, res[3])

    def test_dygraph(self):
        paddle.disable_static(place=self.place)

        with paddle.base.dygraph.guard():
            # NOTE(tiancaishaonvjituizi): float64 input fails the test
            x_np = np.array([[1, np.nan, -2], [np.inf, 0, -np.inf]]).astype(
                np.float32
                # np.float64
            )
            x_tensor = paddle.to_tensor(x_np, stop_gradient=False)

            out_tensor = paddle.nan_to_num(x_tensor)
            out_np = np_nan_to_num(x_np)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)

            out_tensor = paddle.nan_to_num(x_tensor, 1.0, None, None)
            out_np = np_nan_to_num(x_np, 1, None, None)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)

            out_tensor = paddle.nan_to_num(x_tensor, 1.0, 2.0, None)
            out_np = np_nan_to_num(x_np, 1, 2, None)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)

            out_tensor = paddle.nan_to_num(x_tensor, 1.0, None, -10.0)
            out_np = np_nan_to_num(x_np, 1, None, -10)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)

            out_tensor = paddle.nan_to_num(x_tensor, 1.0, 100.0, -10.0)
            out_np = np_nan_to_num(x_np, 1, 100, -10)
            np.testing.assert_allclose(out_tensor.numpy(), out_np)

        paddle.enable_static()

    def test_check_grad(self):
        paddle.disable_static(place=self.place)
        x_np = np.array([[1, np.nan, -2], [np.inf, 0, -np.inf]]).astype(
            np.float32
        )
        x_tensor = paddle.to_tensor(x_np, stop_gradient=False)

        y = paddle.nan_to_num(x_tensor)
        dx = paddle.grad(y, x_tensor)[0].numpy()

        np_grad = np_nan_to_num_grad(x_np, np.ones_like(x_np))
        np.testing.assert_allclose(np_grad, dx)

        paddle.enable_static()


# class BaseTestCases:
#
#     class BaseOpTest(OpTest):
#
#         def setUp(self):
#             self.op_type = "nan_to_num"
#             input = np.arange(100, dtype=np.float64)
#             input[5] = np.nan
#             input[29] = np.inf
#             input[97] = -np.inf
#             self.inputs = {'X': input}
#             self.attrs = self._attrs()
#             self.outputs = {
#                 'Out': np_nan_to_num_op(self.inputs['X'], **self.attrs)
#             }
#             paddle.enable_static()
#
#         def test_check_output(self):
#             self.check_output()
#
#         def test_check_grad(self):
#             input = self.inputs['X']
#             dout = np.ones_like(input) / input.size
#             self.check_grad(
#                 ['X'],
#                 'Out',
#                 user_defined_grads=[np_nan_to_num_grad(self.inputs['X'], dout)])
#
#         def _attrs(self):
#             raise NotImplementedError()
#
#
# class TestNanToNumOp1(BaseTestCases.BaseOpTest):
#
#     def _attrs(self):
#         return {
#             'nan': 0.0,
#             'replace_posinf_with_max': True,
#             'posinf': -1,
#             'replace_neginf_with_min': True,
#             'neginf': -10
#         }
#
#
# class TestNanToNumOp2(BaseTestCases.BaseOpTest):
#
#     def _attrs(self):
#         return {
#             'nan': 2.0,
#             'replace_posinf_with_max': False,
#             'posinf': -1,
#             'replace_neginf_with_min': True,
#             'neginf': -10
#         }
#
#
# class TestNanToNumOp3(BaseTestCases.BaseOpTest):
#
#     def _attrs(self):
#         return {
#             'nan': 0.0,
#             'replace_posinf_with_max': False,
#             'posinf': -1,
#             'replace_neginf_with_min': False,
#             'neginf': -10
#         }

if __name__ == "__main__":
    unittest.main()
