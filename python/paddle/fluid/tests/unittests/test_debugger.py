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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import debugger
from paddle.fluid.framework import Program


class TestDebugger(unittest.TestCase):

    def test_debug_str(self):
        p = Program()
        b = p.current_block()

        #selected_rows
        b.create_var(name='selected_rows',
                     dtype="float32",
                     shape=[5, 10],
                     type=core.VarDesc.VarType.SELECTED_ROWS)

        #tensor array
        b.create_var(name='tensor_array',
                     shape=[5, 10],
                     type=core.VarDesc.VarType.LOD_TENSOR_ARRAY)

        #operator
        mul_x = b.create_parameter(dtype="float32",
                                   shape=[5, 10],
                                   lod_level=0,
                                   name="mul.x")
        mul_y = b.create_var(dtype="float32",
                             shape=[10, 8],
                             lod_level=0,
                             name="mul.y")
        mul_out = b.create_var(dtype="float32",
                               shape=[5, 8],
                               lod_level=0,
                               name="mul.out")
        b.append_op(type="mul",
                    inputs={
                        "X": mul_x,
                        "Y": mul_y
                    },
                    outputs={"Out": mul_out},
                    attrs={"x_num_col_dims": 1})

        print(debugger.pprint_program_codes(p))

        debugger.draw_block_graphviz(p.block(0), path="./test.dot")


if __name__ == '__main__':
    unittest.main()
