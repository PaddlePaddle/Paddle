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
import numpy as np
from paddle import _C_ops
from paddle.fluid.framework import _test_eager_guard, Variable
from paddle.fluid import core

import unittest


def _append_backward_desc(main_program, outs):
    # make sure all status of is_test are False in train mode.
    program = main_program.clone()
    targets = []
    for out in outs:
        if isinstance(out, Variable):
            targets.append(program.global_block().var(out.name))

    if targets:
        paddle.fluid.backward.gradients(targets=targets, inputs=[])

    return program


def _create_out(var):
    assert isinstance(var, Variable)
    var_desc = var.desc
    varbase = None
    if not core._in_eager_mode():
        var_base = core.VarBase(var_desc.dtype(),
                                var_desc.shape(),
                                var_desc.name(), var_desc.type(), False)
    else:
        var_base = core.eager.Tensor(var_desc.dtype(),
                                     var_desc.shape(),
                                     var_desc.name(), var_desc.type(), False)
    return var_base


class TestRunProgram(unittest.TestCase):
    def test_eager(self):
        paddle.set_device('cpu')
        paddle.enable_static()
        # step 1: construct program
        x = paddle.static.data(shape=[2, 4], name='x')
        x.stop_gradient = False
        y = paddle.static.data(shape=[4, 2], name='y')
        y.stop_gradient = False
        out = paddle.matmul(x, y)

        main_program = paddle.static.default_main_program()
        program = _append_backward_desc(main_program, [out])
        print(program)

        # step 1: call run_program in eager mode
        paddle.disable_static('cpu')

        with _test_eager_guard():
            x_t = paddle.ones([2, 4])
            x_t.name = "x"
            x_t.stop_gradient = False
            y_t = paddle.ones([4, 2])
            y_t.name = "y"
            y_t.stop_gradient = False

            fake_var = paddle.zeros([1])
            fake_var.name = 'Fake_var'

            # out_t = paddle.zeros([2, 2])
            # out_t.name = out.name
            # out_t.stop_gradient = False
            out_t = _create_out(out)
            print(out_t.name)

            scope = core.Scope()
            attrs = ('global_block', program.desc.block(0), 'start_op_index', 0,
                     'end_op_index', main_program.desc.block(0).op_size(),
                     'is_test', False, 'program_id', id(program))

            _C_ops.run_program([x_t, y_t], [fake_var], [out_t], [scope],
                               [fake_var], *attrs)

            loss = paddle.mean(out_t)
            loss.backward()

            self.assertTrue(np.array_equal(np.ones([2, 2]) * 4, out_t.numpy()))
            print(out_t, out_t.name)
            print(x_t.grad)
            print(y_t.grad)
            self.assertTrue(np.array_equal(np.ones([2, 4]), x_t.grad.numpy()))
            self.assertTrue(
                np.array_equal(np.ones([4, 2]) * 0.5, y_t.grad.numpy()))


if __name__ == '__main__':
    unittest.main()
