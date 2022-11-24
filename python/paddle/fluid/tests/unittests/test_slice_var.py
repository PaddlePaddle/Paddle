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
from paddle.fluid.transpiler.distribute_transpiler import slice_variable
import paddle.fluid as fluid
import random


class TestSliceVar(unittest.TestCase):

    def check_slice_output(self, shapes, expected_sizes, min_size):
        var_list = []
        program = fluid.Program()
        for shape in shapes:
<<<<<<< HEAD
            var = program.global_block().create_var(name=str(
                random.randint(10000, 99999)),
                                                    persistable=True,
                                                    shape=shape)
=======
            var = program.global_block().create_var(
                name=str(random.randint(10000, 99999)),
                persistable=True,
                shape=shape,
            )
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
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
        expected_sizes = [[15], [1024],
                          [
                              2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352,
                              2352, 784
                          ], [2040, 2040, 2040, 2040],
                          [1150, 1150, 1150, 1150, 1150, 1150, 1100]]
=======
        expected_sizes = [
            [15],
            [1024],
            [2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352, 784],
            [2040, 2040, 2040, 2040],
            [1150, 1150, 1150, 1150, 1150, 1150, 1100],
        ]
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f

        self.check_slice_output(shapes, expected_sizes, 1024)

    def test_check_output_8k(self):
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

        self.check_slice_output(shapes, expected_sizes, 8192)


if __name__ == '__main__':
    unittest.main()
