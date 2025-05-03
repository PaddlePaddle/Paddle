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

from __future__ import annotations

import unittest

import numpy as np
from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


@check_no_breakgraph
def unary_api(func, x: np.ndarray, *args, **kwargs):
    return func(x, *args, **kwargs)


@check_no_breakgraph
def binary_api(
    func: np.ufunc,
    x: np.ndarray | list,
    y: np.ndarray | list,
    *args,
    **kwargs,
):
    return func(x, y, *args, **kwargs)


@check_no_breakgraph
def binary_operator(op: str, x: np.ndarray, y: np.ndarray):
    if op == "+":
        return x + y
    elif op == "-":
        return x - y
    elif op == "*":
        return x * y
    elif op == "/":
        return x / y


@check_no_breakgraph
def get_item(x: np.ndarray, index: int | tuple | np.ndarray):
    return x[index]


@check_no_breakgraph
def set_item(
    x: np.ndarray,
    index: int | list | np.ndarray,
    value: int | list | np.ndarray,
):
    x[index] = value
    return x


@check_no_breakgraph
def grad_fn(pd_x, np_y, np_y2):
    pd_y = paddle.to_tensor(np_y)
    pd_z = paddle.add(pd_x, pd_y)
    pd_y2 = paddle.to_tensor(np_y2).astype("float32")
    pd_z2 = paddle.matmul(pd_z, pd_y2)
    return pd_z2


@check_no_breakgraph
def demo(pd_x):
    np_y = np.array([[1, 2, 3], [4, 5, 6]])
    np_y2 = np.transpose(np_y)
    pd_y2 = paddle.to_tensor(np_y2)
    pd_z = paddle.matmul(pd_x, pd_y2)
    np_z = pd_z.numpy()
    np_z2 = np.mean(np_z)
    np_z3 = np.add(pd_x.numpy(), np_z2)
    np_z4 = np_z * np_z3[:, :2]
    pd_z2 = paddle.subtract(pd_z, paddle.to_tensor(np_z4))
    return pd_z2


def absolute(x):
    return np.absolute(x)


class TestNumpyArray(TestCaseBase):
    def test_guard(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(
                binary_api, np.add, np.array([1, 2]), np.array([3, 4])
            )
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(
                binary_api,
                np.add,
                np.array([1, 2], dtype=np.int32),
                np.array([3, 4], dtype=np.int32),
            )
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(
                binary_api, np.add, np.array([1]), np.array([3])
            )
            self.assertEqual(ctx.translate_count, 3)
            self.assert_results(
                binary_api, np.add, np.array([4, 3]), np.array([2, 1])
            )
            self.assertEqual(ctx.translate_count, 3)

    def test_gradient(self):
        pd_x = paddle.randn([2, 3], dtype="float32")
        pd_x.stop_gradient = False
        self.assert_results_with_grad(
            pd_x,
            grad_fn,
            pd_x,
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
            np.ones(shape=(3, 2)),
        )

    def test_add_01(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        self.assert_results(binary_api, np.add, x, y)

    @unittest.skip("Not supported yet")
    def test_add_02(self):
        x = [1, 2]
        y = [3, 4]
        self.assert_results(binary_api, np.add, x, y)

    @unittest.skip("Not supported yet")
    def test_sub(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        self.assert_results(binary_api, np.subtract, x, y)
        self.assert_results(binary_api, np.subtract, [1, 2], [3, 4])

    @unittest.skip("Not supported yet")
    def test_mul(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        self.assert_results(binary_api, np.multiply, x, y)
        self.assert_results(binary_api, np.multiply, [1, 2], [3, 4])
        self.assert_results_with_grad(binary_api, np.multiply, [1, 2], [3, 4])

    @unittest.skip("Not supported yet")
    def test_div(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        self.assert_results(binary_api, np.divide, x, y)
        self.assert_results(binary_api, np.divide, [1, 2], [3, 4])

    @unittest.skip("Not supported yet")
    def test_operator(self):
        x = np.array([1, 2])
        y = np.array([3, 4])
        self.assert_results(binary_operator, '+', x, y)
        self.assert_results(binary_operator, '-', x, y)
        self.assert_results(binary_operator, '*', x, y)
        self.assert_results(binary_operator, '/', x, y)

    @unittest.skip("Not supported yet")
    def test_argmax(self):
        x = np.array([[1, 2], [3, 4]])
        self.assert_results(unary_api, np.argmax, x)
        self.assert_results(unary_api, np.argmax, x, axis=0)

    @unittest.skip("Not supported yet")
    def test_sum(self):
        x = np.array([[1, 2], [3, 4]])
        self.assert_results(unary_api, np.sum, x)
        self.assert_results(unary_api, np.sum, x, axis=0)

    @unittest.skip("Not supported yet")
    def test_getitem(self):
        x = np.array([[1, 2], [3, 4]])
        self.assert_results(get_item, x, 0)
        i = tuple(0, 0)
        self.assert_results(get_item, x, i)
        i2 = np.array([0, 1])
        self.assert_results(get_item, x, i2)

    @unittest.skip("Not supported yet")
    def test_setitem(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([5, 6])
        self.assert_results(set_item, x, 0, y)
        i = np.array([0])
        self.assert_results(set_item, x, i, y)
        self.assert_results(set_item, x, [0], [5, 6])

    @unittest.skip("Not supported yet")
    def test_demo(self):
        pd_x = paddle.randn([2, 3], dtype="float32")
        pd_x.stop_gradient = False
        self.assert_results_with_grad(pd_x, demo, pd_x)

    def test_absolute(self):
        x = np.array([[1, -2], [-3, 4]])
        self.assert_results(absolute, x)


if __name__ == "__main__":
    unittest.main()
