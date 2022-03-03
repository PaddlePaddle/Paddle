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

import paddle
from paddle import _C_ops
from paddle.fluid.framework import _test_eager_guard
from paddle.fluid import core

import unittest


class TestRunProgram(unittest.TestCase):
    def test_eager(self):
        paddle.set_device('cpu')
        paddle.enable_static()
        # step 1: construct program
        x = paddle.static.data(shape=[2, 4], name='x')
        y = paddle.static.data(shape=[4, 2], name='y')
        out = paddle.matmul(x, y)
        program = paddle.static.default_main_program()

        # step 1: call run_program in eager mode
        paddle.disable_static('cpu')

        with _test_eager_guard():
            x_t = paddle.ones([2, 4])
            x_t.name = "x"
            y_t = paddle.ones([4, 2])
            y_t.name = "y"
            fake_var = paddle.zeros([1])
            fake_var.name = 'Fake_var'

            out_t = paddle.zeros([2, 2])
            out_t.name = out.name

            scope = core.Scope()
            attrs = ('global_block', program.desc.block(0), 'start_op_index', 0,
                     'end_op_index', program.desc.block(0).op_size(), 'is_test',
                     True, 'program_id', id(program))

            _C_ops.run_program([x_t, y_t], [fake_var], [out_t], [scope],
                               [fake_var], *attrs)

            print(out_t)


if __name__ == '__main__':
    unittest.main()
