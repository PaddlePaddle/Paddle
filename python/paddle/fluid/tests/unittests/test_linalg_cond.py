#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.static as static
from paddle.fluid.framework import _test_eager_guard

p_list_n_n = ("fro", "nuc", 1, -1, np.inf, -np.inf)
p_list_m_n = (None, 2, -2)


def test_static_assert_true(self, x_list, p_list):
    for p in p_list:
        for x in x_list:
            with static.program_guard(static.Program(), static.Program()):
                input_data = static.data("X", shape=x.shape, dtype=x.dtype)
                output = paddle.linalg.cond(input_data, p)
                exe = static.Executor()
                result = exe.run(feed={"X": x}, fetch_list=[output])
                expected_output = np.linalg.cond(x, p)
                np.testing.assert_allclose(result[0],
                                           expected_output,
                                           rtol=5e-5)


def test_dygraph_assert_true(self, x_list, p_list):
    for p in p_list:
        for x in x_list:
            input_tensor = paddle.to_tensor(x)
            output = paddle.linalg.cond(input_tensor, p)
            expected_output = np.linalg.cond(x, p)
            np.testing.assert_allclose(output.numpy(),
                                       expected_output,
                                       rtol=5e-5)


def gen_input():
    np.random.seed(2021)
    # generate square matrix or batches of square matrices
    input_1 = np.random.rand(5, 5).astype('float32')
    input_2 = np.random.rand(3, 6, 6).astype('float64')
    input_3 = np.random.rand(2, 4, 3, 3).astype('float32')

    # generate non-square matrix or batches of non-square matrices
    input_4 = np.random.rand(9, 7).astype('float64')
    input_5 = np.random.rand(4, 2, 10).astype('float32')
    input_6 = np.random.rand(3, 5, 4, 1).astype('float32')

    list_n_n = (input_1, input_2, input_3)
    list_m_n = (input_4, input_5, input_6)
    return list_n_n, list_m_n


def gen_empty_input():
    # generate square matrix or batches of square matrices which are empty tensor
    input_1 = np.random.rand(0, 7, 7).astype('float32')
    input_2 = np.random.rand(0, 9, 9).astype('float32')
    input_3 = np.random.rand(0, 4, 5, 5).astype('float64')

    # generate non-square matrix or batches of non-square matrices which are empty tensor
    input_4 = np.random.rand(0, 7, 11).astype('float32')
    input_5 = np.random.rand(0, 10, 8).astype('float64')
    input_6 = np.random.rand(5, 0, 4, 3).astype('float32')

    list_n_n = (input_1, input_2, input_3)
    list_m_n = (input_4, input_5, input_6)
    return list_n_n, list_m_n


class API_TestStaticCond(unittest.TestCase):

    def test_out(self):
        paddle.enable_static()
        # test calling results of 'cond' in static mode
        x_list_n_n, x_list_m_n = gen_input()
        test_static_assert_true(self, x_list_n_n, p_list_n_n + p_list_m_n)
        test_static_assert_true(self, x_list_m_n, p_list_m_n)


class API_TestDygraphCond(unittest.TestCase):

    def func_out(self):
        paddle.disable_static()
        # test calling results of 'cond' in dynamic mode
        x_list_n_n, x_list_m_n = gen_input()
        test_dygraph_assert_true(self, x_list_n_n, p_list_n_n + p_list_m_n)
        test_dygraph_assert_true(self, x_list_m_n, p_list_m_n)

    def test_out(self):
        with _test_eager_guard():
            self.func_out()
        self.func_out()


class TestCondAPIError(unittest.TestCase):

    def func_dygraph_api_error(self):
        paddle.disable_static()
        # test raising errors when 'cond' is called in dygraph mode
        p_list_error = ('fro_', '_nuc', -0.7, 0, 1.5, 3)
        x_list_n_n, x_list_m_n = gen_input()
        for p in p_list_error:
            for x in (x_list_n_n + x_list_m_n):
                x_tensor = paddle.to_tensor(x)
                self.assertRaises(ValueError, paddle.linalg.cond, x_tensor, p)

        for p in p_list_n_n:
            for x in x_list_m_n:
                x_tensor = paddle.to_tensor(x)
                self.assertRaises(ValueError, paddle.linalg.cond, x_tensor, p)

    def test_dygraph_api_error(self):
        with _test_eager_guard():
            self.func_dygraph_api_error()
        self.func_dygraph_api_error()

    def test_static_api_error(self):
        paddle.enable_static()
        # test raising errors when 'cond' is called in static mode
        p_list_error = ('f ro', 'fre', 'NUC', -1.6, 0, 5)
        x_list_n_n, x_list_m_n = gen_input()
        for p in p_list_error:
            for x in (x_list_n_n + x_list_m_n):
                with static.program_guard(static.Program(), static.Program()):
                    x_data = static.data("X", shape=x.shape, dtype=x.dtype)
                    self.assertRaises(ValueError, paddle.linalg.cond, x_data, p)

        for p in p_list_n_n:
            for x in x_list_m_n:
                with static.program_guard(static.Program(), static.Program()):
                    x_data = static.data("X", shape=x.shape, dtype=x.dtype)
                    self.assertRaises(ValueError, paddle.linalg.cond, x_data, p)

    # it's not supported when input is an empty tensor in static mode
    def test_static_empty_input_error(self):
        paddle.enable_static()

        x_list_n_n, x_list_m_n = gen_empty_input()
        for p in (p_list_n_n + p_list_m_n):
            for x in x_list_n_n:
                with static.program_guard(static.Program(), static.Program()):
                    x_data = static.data("X", shape=x.shape, dtype=x.dtype)
                    self.assertRaises(ValueError, paddle.linalg.cond, x_data, p)

        for p in (p_list_n_n + p_list_m_n):
            for x in x_list_n_n:
                with static.program_guard(static.Program(), static.Program()):
                    x_data = static.data("X", shape=x.shape, dtype=x.dtype)
                    self.assertRaises(ValueError, paddle.linalg.cond, x_data, p)


class TestCondEmptyTensorInput(unittest.TestCase):

    def func_dygraph_empty_tensor_input(self):
        paddle.disable_static()
        # test calling results of 'cond' when input is an empty tensor in dynamic mode
        x_list_n_n, x_list_m_n = gen_empty_input()
        test_dygraph_assert_true(self, x_list_n_n, p_list_n_n + p_list_m_n)
        test_dygraph_assert_true(self, x_list_m_n, p_list_m_n)

    def test_dygraph_empty_tensor_input(self):
        with _test_eager_guard():
            self.func_dygraph_empty_tensor_input()
        self.func_dygraph_empty_tensor_input()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
