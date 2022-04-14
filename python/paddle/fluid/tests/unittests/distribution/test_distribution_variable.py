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
from paddle.distribution import variable
from paddle.distribution import constraint

import config
import parameterize as param


@param.param_cls(
    (param.TEST_CASE_NAME, 'is_discrete', 'event_rank', 'constraint'),
    [('NotImplement', False, 0, constraint.Constraint())])
class TestVariable(unittest.TestCase):
    def setUp(self):
        self._var = variable.Variable(self.is_discrete, self.event_rank,
                                      self.constraint)

    @param.param_func([(1, )])
    def test_costraint(self, value):
        with self.assertRaises(NotImplementedError):
            self._var.constraint(value)


@param.param_cls((param.TEST_CASE_NAME, 'base', 'rank'),
                 [('real_base', variable.real, 10)])
class TestIndependent(unittest.TestCase):
    def setUp(self):
        self._var = variable.Independent(self.base, self.rank)

    @param.param_func([(paddle.rand([2, 3, 4]), ValueError), ])
    def test_costraint(self, value, expect):
        with self.assertRaises(expect):
            self._var.constraint(value)


@param.param_cls((param.TEST_CASE_NAME, 'vars', 'axis'),
                 [('real_base', [variable.real], 10)])
class TestStack(unittest.TestCase):
    def setUp(self):
        self._var = variable.Stack(self.vars, self.axis)

    def test_is_discrete(self):
        self.assertEqual(self._var.is_discrete, False)

    @param.param_func([(paddle.rand([2, 3, 4]), ValueError), ])
    def test_costraint(self, value, expect):
        with self.assertRaises(expect):
            self._var.constraint(value)


if __name__ == '__main__':
    unittest.main()
