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

import math
import unittest
from paddle.fluid.distribute_transpiler import split_dense_variable
import paddle.fluid as fluid
import paddle.fluid.core as core
import random


class TestSplitVar(unittest.TestCase):
    def test_check_output(self):
        # split below shapes to 10 servers
        shapes = [[3, 5], [1024], [28, 784], [8, 1020], [800, 10]]
        expected_sizes = [
            [15], [1024],
            [2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352, 2352, 784],
            [2040, 2040, 2040, 2040],
            [1150, 1150, 1150, 1150, 1150, 1150, 1100]
        ]
        var_list = []
        program = fluid.Program()
        for shape in shapes:
            var = program.global_block().create_var(
                name=str(random.randint(10000, 99999)),
                persistable=True,
                # dtype=core.VarDesc.VarType.LOD_TENSOR,
                shape=shape)
            var_list.append(var)
        blocks = split_dense_variable(var_list, 10)
        all_sizes = []
        for s in expected_sizes:
            for s2 in s:
                all_sizes.append(s2)
        for i, block_str in enumerate(blocks):
            varname, block_id, size = block_str.split(":")
            self.assertEqual(int(size), all_sizes[i])


if __name__ == '__main__':
    unittest.main()
