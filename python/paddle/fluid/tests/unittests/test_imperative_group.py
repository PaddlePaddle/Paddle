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

from __future__ import print_function

import contextlib
import unittest
import numpy as np
import six
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph.nn import Linear
import paddle.fluid.core as core
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.framework import _test_eager_guard, _in_legacy_dygraph, in_dygraph_mode


class TestDataParallelGroup(unittest.TestCase):
    def create_varbase(self, dtype, shape):
        return paddle.rand(shape=shape, dtype=dtype)

    def assign_group_by_size(self, *args):
        if in_dygraph_mode():
            return core.eager_assign_group_by_size(*args)
        elif _in_legacy_dygraph():
            return core.assign_group_by_size(*args)

    def test_construct_group0(self):
        # one dtype & one limit capability
        var_list = []
        var_list.append(self.create_varbase("float32", [2, 50]))
        var_list.append(self.create_varbase("float32", [2, 100]))
        var_list.append(self.create_varbase("float32", [2, 50]))
        var_list.append(self.create_varbase("float32", [2, 25]))
        res = self.assign_group_by_size(var_list, [False, False, False, False],
                                        [400])
        self.assertEqual([[0], [1], [2], [3]], res)

    def test_construct_group1(self):
        # multi dtype & one limit capability
        var_list = []
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        res = self.assign_group_by_size(
            var_list, [False, False, False, False, False, False], [400])
        self.assertEqual([[0, 2], [1, 3], [4], [5]], res)

    def test_construct_group2(self):
        # one dtype & multi limit capability
        var_list = []
        var_list.append(self.create_varbase("float32", [2, 50]))
        var_list.append(self.create_varbase("float32", [2, 50]))
        var_list.append(self.create_varbase("float32", [2, 50]))
        var_list.append(self.create_varbase("float32", [2, 50]))
        res = self.assign_group_by_size(var_list, [False, False, False, False],
                                        [400, 800])
        self.assertEqual([[0], [1, 2], [3]], res)

    def test_construct_group3(self):
        # multi dtype & multi limit capability
        var_list = []
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        res = self.assign_group_by_size(
            var_list, [False, False, False, False, False, False], [200, 400])
        self.assertEqual([[0], [1], [2, 4], [3, 5]], res)

    def test_construct_group4(self):
        # multi dtype & zero limit capability
        var_list = []
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        res = self.assign_group_by_size(
            var_list, [False, False, False, False, False, False], [0])
        self.assertEqual([[0], [1], [2], [3], [4], [5]], res)

    def test_construct_group5(self):
        # multi dtype & infinite capability
        var_list = []
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        res = self.assign_group_by_size(
            var_list, [False, False, False, False, False, False], [10000])
        self.assertEqual([[0, 2, 4], [1, 3, 5]], res)

    def test_construct_group6(self):
        # multi dtype & limit capability & multi tensor type
        var_list = []
        var_list.append(self.create_varbase(
            "float32",
            [1, 50], ))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        res = self.assign_group_by_size(
            var_list, [True, False, False, False, False, True], [400])
        self.assertEqual([[0], [1, 3], [2, 4], [5]], res)

    def test_construct_group7(self):
        # multi dtype & multi limit capability & multi tensor type
        var_list = []
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        var_list.append(self.create_varbase("float32", [1, 50]))
        var_list.append(self.create_varbase("float64", [1, 25]))
        res = self.assign_group_by_size(
            var_list, [True, False, False, False, False, True], [200, 400])
        self.assertEqual([[0], [1], [2], [3], [4], [5]], res)

    def test_construct_group8(self):
        # one dtype & one limit capability & have tensor_indices
        var_list = []
        var_list.append(self.create_varbase("float32", [2, 25]))
        var_list.append(self.create_varbase("float32", [2, 100]))
        var_list.append(self.create_varbase("float32", [2, 50]))
        var_list.append(self.create_varbase("float32", [2, 25]))
        res = self.assign_group_by_size(var_list, [False, False, False, False],
                                        [400], [3, 0, 1, 2])
        self.assertEqual([[3, 0], [1], [2]], res)

    def test_construct_group9(self):
        # one dtype & one limit capability & have tensor_indices
        var_list = []
        var_list.append(self.create_varbase("float32", [2, 25]))
        var_list.append(self.create_varbase("float32", [2, 25]))
        var_list.append(self.create_varbase("float32", [2, 25]))
        var_list.append(self.create_varbase("float32", [2, 1000]))
        res = self.assign_group_by_size(var_list, [False, False, False, True],
                                        [300], [1, 0, 2, 3])
        self.assertEqual([[1, 0], [3], [2]], res)

    def test_construct_group_in_legacy_mode(self):
        with _test_eager_guard():
            pass
        self.test_construct_group0()
        self.test_construct_group1()
        self.test_construct_group2()
        self.test_construct_group3()
        self.test_construct_group4()
        self.test_construct_group5()
        self.test_construct_group6()
        self.test_construct_group7()
        self.test_construct_group8()
        self.test_construct_group9()


if __name__ == '__main__':
    unittest.main()
