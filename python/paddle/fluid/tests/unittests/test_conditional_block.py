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
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.layers as layers
from paddle.fluid.backward import append_backward
from paddle.fluid.executor import Executor
from paddle.fluid.layers.control_flow import ConditionalBlock


class ConditionalBlockTest(unittest.TestCase):
    def test_forward(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            data = layers.data(name='X', shape=[1], dtype='float32')
            data.stop_gradient = False
            cond = ConditionalBlock(inputs=[data])
            out = paddle.tensor.create_tensor(dtype='float32')
            with cond.block():
                hidden = layers.fc(input=data, size=10)
                layers.assign(hidden, out)

            cpu = core.CPUPlace()
            exe = Executor(cpu)
            exe.run(startup_program)

            x = np.random.random(size=(10, 1)).astype('float32')

            outs = exe.run(main_program, feed={'X': x}, fetch_list=[out])[0]
            print(outs)
            loss = paddle.mean(out)
            append_backward(loss=loss)
            outs = exe.run(
                main_program,
                feed={'X': x},
                fetch_list=[main_program.block(0).var(data.name + "@GRAD")],
            )[0]
            print(outs)


class TestConditionalBlockOpInferShape(unittest.TestCase):
    def test_infer_shape(self):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            global_block = main_program.global_block()
            sub_block = main_program._create_block()
            main_program._rollback()
            step_scope = global_block.create_var(
                type=core.VarDesc.VarType.STEP_SCOPES
            )
            cond_var = layers.fill_constant(
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
    unittest.main()
