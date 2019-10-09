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

from __future__ import print_function

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
        b.create_var(
            name='selected_rows',
            dtype="float32",
            shape=[5, 10],
            type=core.VarDesc.VarType.SELECTED_ROWS)

        #tensor array
        b.create_var(
            name='tensor_array',
            shape=[5, 10],
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY)

        #operator
        mul_x = b.create_parameter(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
        mul_y = b.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = b.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        b.append_op(
            type="mul",
            inputs={"X": mul_x,
                    "Y": mul_y},
            outputs={"Out": mul_out},
            attrs={"x_num_col_dims": 1})

        print(debugger.pprint_program_codes(p))

        debugger.draw_block_graphviz(p.block(0), path="./test.dot")

    def test_fast_nan_debug(self):
        p = Program()
        with fluid.program_guard(p):
            with fluid.unique_name.guard():
                x = fluid.layers.data(name="x", shape=[4], dtype='float32')
                y = fluid.layers.data(name="y", shape=[1], dtype='float32')

                hidden = x
                for i in range(10):
                    hidden = fluid.layers.fc(input=hidden,
                                             size=200,
                                             act="sigmoid")

                y_predict = fluid.layers.fc(input=hidden, size=1, act=None)

            debugger.prepare_fast_nan_inf_debug(p)

            for _block in p.blocks:
                _vars_in_prog = [_var_name for _var_name in _block.vars]
                for _var_name in _vars_in_prog:
                    is_good = _var_name.startswith("debug_var_") or (
                        ("debug_var_" + _var_name + "_0") in _vars_in_prog)
                    assert is_good

    def test_fast_nan_debug_skiplist(self):
        q = Program()
        with fluid.program_guard(q):
            with fluid.unique_name.guard():
                x = fluid.layers.data(name="x", shape=[4], dtype='float32')
                y = fluid.layers.data(name="y", shape=[1], dtype='float32')

                hidden = x
                for i in range(10):
                    hidden = fluid.layers.fc(input=hidden,
                                             size=200,
                                             act="sigmoid",
                                             name="fc_" + str(i))

                y_predict = fluid.layers.fc(input=hidden,
                                            size=1,
                                            act=None,
                                            name="final")

            debugger.prepare_fast_nan_inf_debug(q, skip_list=["final"])

            for _block in q.blocks:
                _vars_in_prog = [_var_name for _var_name in _block.vars]
                for _var_name in _vars_in_prog:
                    if "final" not in _var_name:
                        is_good = _var_name.startswith("debug_var_") or (
                            ("debug_var_" + _var_name + "_1") in _vars_in_prog)
                        assert is_good

                    else:
                        assert "debug_var_" not in _var_name


if __name__ == '__main__':
    unittest.main()
