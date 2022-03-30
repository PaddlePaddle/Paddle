# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.distribution import constraint

import config
import parameterize as param


@param.param_cls((param.TEST_CASE_NAME, 'value'),
                 [('NotImplement', np.random.rand(2, 3))])
class TestConstraint(unittest.TestCase):
    def setUp(self):
        self._constraint = constraint.Constraint()

    def test_costraint(self):
        with self.assertRaises(NotImplementedError):
            self._constraint(self.value)


@param.param_cls((param.TEST_CASE_NAME, 'value', 'expect'),
                 [('real', 1., True)])
class TestReal(unittest.TestCase):
    def setUp(self):
        self._constraint = constraint.Real()

    def test_costraint(self):
        self.assertEqual(self._constraint(self.value), self.expect)


@param.param_cls((param.TEST_CASE_NAME, 'lower', 'upper', 'value', 'expect'),
                 [('in_range', 0, 1, 0.5, True), ('out_range', 0, 1, 2, False)])
class TestRange(unittest.TestCase):
    def setUp(self):
        self._constraint = constraint.Range(self.lower, self.upper)

    def test_costraint(self):
        self.assertEqual(self._constraint(self.value), self.expect)


@param.param_cls((param.TEST_CASE_NAME, 'value', 'expect'),
                 [('positive', 1, True), ('negative', -1, False)])
class TestPositive(unittest.TestCase):
    def setUp(self):
        self._constraint = constraint.Positive()

    def test_costraint(self):
        self.assertEqual(self._constraint(self.value), self.expect)


@param.param_cls((param.TEST_CASE_NAME, 'value', 'expect'),
                 [('simplex', paddle.to_tensor([0.5, 0.5]), True),
                  ('non_simplex', paddle.to_tensor([-0.5, 0.5]), False)])
class TestSimplex(unittest.TestCase):
    def setUp(self):
        self._constraint = constraint.Simplex()

    def test_costraint(self):
        self.assertEqual(self._constraint(self.value), self.expect)


if __name__ == '__main__':
    unittest.main()
