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

<<<<<<< HEAD
import random
import unittest

import paddle.fluid as fluid
from paddle.fluid.transpiler.distribute_transpiler import slice_variable


class TestSliceVar(unittest.TestCase):
=======
from __future__ import print_function

import math
import unittest
from paddle.fluid.transpiler.distribute_transpiler import slice_variable
import paddle.fluid as fluid
import paddle.fluid.core as core
import random


class TestSliceVar(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def check_slice_output(self, shapes, expected_sizes, min_size):
        var_list = []
        program = fluid.Program()
        for shape in shapes:
<<<<<<< HEAD
            var = program.global_block().create_var(
                name=str(random.randint(10000, 99999)),
                persistable=True,
                shape=shape,
            )
=======
            var = program.global_block().create_var(name=str(
                random.randint(10000, 99999)),
                                                    persistable=True,
                                                    shape=shape)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            var_list.append(var)
        blocks = slice_variable(var_list, 10, min_size)
        all_sizes = []
        for s in expected_sizes:
            for s2 in s:
                all_sizes.append(s2)
        for i, block_str in enumerate(blocks):
            varname, block_id, size = block_str.split(":")
            self.assertEqual(int(size), all_sizes[i])

    def test_1k(self):
        shapes = [[3, 5], [1024], [28, 784], [8, 1020], [800, 10]]
<<<<<<< HEAD
        expected_sizes = [
            [15],
            [1024],
            [2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352, 784],
            [2040, 2040, 2040, 2040],
            [1150, 1150, 1150, 1150, 1150, 1150, 1100],
        ]
=======
        expected_sizes = [[15], [1024],
                          [
                              2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352,
                              2352, 784
                          ], [2040, 2040, 2040, 2040],
                          [1150, 1150, 1150, 1150, 1150, 1150, 1100]]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.check_slice_output(shapes, expected_sizes, 1024)

    def test_check_output_8k(self):
<<<<<<< HEAD
        shapes = [
            [3, 5],
            [1024],
            [28, 784],
            [8, 1020],
            [800, 10],
            [6, 33, 33, 33],
        ]
        expected_sizes = [
            [15],
            [1024],
            [10976, 10976],
            [8160],
            [8000],
            [35937, 35937, 35937, 35937, 35937, 35937],
        ]
=======
        shapes = [[3, 5], [1024], [28, 784], [8, 1020], [800, 10],
                  [6, 33, 33, 33]]
        expected_sizes = [[15], [1024], [10976, 10976], [8160], [8000],
                          [35937, 35937, 35937, 35937, 35937, 35937]]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.check_slice_output(shapes, expected_sizes, 8192)


if __name__ == '__main__':
    unittest.main()
