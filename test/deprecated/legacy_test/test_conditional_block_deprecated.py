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

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.static import Executor, append_backward
from paddle.static.nn.control_flow import ConditionalBlock


class ConditionalBlockTest(unittest.TestCase):

    def test_forward(self):
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            data = paddle.static.data(name='X', shape=[-1, 1], dtype='float32')
            data.stop_gradient = False
            data.persistable = True
            cond = ConditionalBlock(inputs=[data])
            out = paddle.tensor.fill_constant(
                [10, 10], dtype='float32', value=0.0
            )
            out.stop_gradient = False
            with cond.block():
                hidden = paddle.static.nn.fc(x=data, size=10)
                paddle.assign(hidden, out)

            cpu = core.CPUPlace()
            exe = Executor(cpu)
            exe.run(startup_program)

            x = np.random.random(size=(10, 1)).astype('float32')

            loss = paddle.mean(out)
            grad_list = append_backward(loss=loss)
            if paddle.framework.in_pir_mode():
                outs = exe.run(
                    main_program,
                    feed={'X': x},
                    fetch_list=[out, grad_list[0][1]],
                )
            else:
                outs = exe.run(
                    main_program,
                    feed={'X': x},
                    fetch_list=[
                        out,
                        main_program.block(0).var(data.name + "@GRAD"),
                    ],
                )


class TestConditionalBlockOpInferShape(unittest.TestCase):
    def test_infer_shape(self):
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            global_block = main_program.global_block()
            sub_block = main_program._create_block()
            main_program._rollback()
            step_scope = global_block.create_var(
                type=core.VarDesc.VarType.STEP_SCOPES
            )
            cond_var = paddle.tensor.fill_constant(
                shape=[1], dtype='bool', value=False
            )

            op = global_block.append_op(
                type='conditional_block',
                inputs={
                    'Cond': [cond_var],
                    'Input': [],
                },
                outputs={'Out': [], 'Scope': [step_scope]},
                attrs={'sub_block': sub_block, 'is_scalar_condition': True},
            )
            op.desc.infer_shape(global_block.desc)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
