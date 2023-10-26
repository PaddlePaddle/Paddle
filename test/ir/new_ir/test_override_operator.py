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

import unittest

import numpy as np

import paddle

paddle.enable_static()


class TestOperatorBase(unittest.TestCase):
    def setUp(self):
        self.shape = [4, 16]

    def check_operator(self, operator_func, expected_out):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.ones(self.shape, dtype='float32') * 2
            out = operator_func(x)

            exe = paddle.static.Executor(paddle.CPUPlace())
            res = exe.run(main_program, fetch_list=[out])
            np.testing.assert_almost_equal(res[0], expected_out)


class TestOperator(TestOperatorBase):
    def test_add(self):
        operator_func = lambda x: x + x
        expected_out = np.ones(self.shape, dtype='float32') * 4
        self.check_operator(operator_func, expected_out)

    def test_sub(self):
        operator_func = lambda x: x - x
        expected_out = np.ones(self.shape, dtype='float32') * 0
        self.check_operator(operator_func, expected_out)

    def test_mul(self):
        operator_func = lambda x: x * x
        expected_out = np.ones(self.shape, dtype='float32') * 4
        self.check_operator(operator_func, expected_out)

    def test_div(self):
        operator_func = lambda x: x / x
        expected_out = np.ones(self.shape, dtype='float32') * 1
        self.check_operator(operator_func, expected_out)


class TestOperatorWithScale(TestOperatorBase):
    def test_add(self):
        operator_func = lambda x: x + 1
        expected_out = np.ones(self.shape, dtype='float32') * 3
        self.check_operator(operator_func, expected_out)

    def test_sub(self):
        operator_func = lambda x: x - 1.0
        expected_out = np.ones(self.shape, dtype='float32')
        self.check_operator(operator_func, expected_out)

    def test_mul(self):
        operator_func = lambda x: x * 2
        expected_out = np.ones(self.shape, dtype='float32') * 4
        self.check_operator(operator_func, expected_out)

    def test_div(self):
        operator_func = lambda x: x / 2.0
        expected_out = np.ones(self.shape, dtype='float32') * 1
        self.check_operator(operator_func, expected_out)


class TestCompareOperator(TestOperatorBase):
    def test_lt(self):
        operator_func = lambda x: x < x - 1
        expected_out = np.zeros(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_gt(self):
        operator_func = lambda x: x > x - 1
        expected_out = np.ones(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_le(self):
        operator_func = lambda x: x <= x
        expected_out = np.ones(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_ge(self):
        operator_func = lambda x: x >= x + 1
        expected_out = np.zeros(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)


class TestCompareOpWithFull(TestOperatorBase):
    def test_lt(self):
        operator_func = lambda x: x < 1
        expected_out = np.zeros(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_gt(self):
        operator_func = lambda x: x > 1.0
        expected_out = np.ones(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_le(self):
        operator_func = lambda x: x <= 2
        expected_out = np.ones(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)

    def test_ge(self):
        operator_func = lambda x: x >= 3.0
        expected_out = np.zeros(self.shape, dtype='bool')
        self.check_operator(operator_func, expected_out)


if __name__ == '__main__':
    unittest.main()
