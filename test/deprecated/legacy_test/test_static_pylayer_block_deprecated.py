# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.static.nn.static_pylayer import StaticPyLayerBlock


class StaticPyLayerBlockTest(unittest.TestCase):
    def test_forward_and_backward(self):
        paddle.enable_static()
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            data = paddle.static.data(name='X', shape=[10, 1], dtype='float32')
            data.stop_gradient = False
            static_pylayer_manager = StaticPyLayerBlock(inputs=[data])
            fwd_out = paddle.tensor.create_tensor(dtype='float32')
            with static_pylayer_manager.block(is_backward_block=False) as mgr:
                hidden_fwd = paddle.static.nn.fc(x=data, size=10)
                paddle.assign(hidden_fwd, fwd_out)
                mgr.fwd_outputs = [fwd_out]

            grad_name = data.name + core.grad_var_suffix()
            with static_pylayer_manager.block(is_backward_block=True) as mgr:
                constant_tensor = paddle.tensor.fill_constant(
                    shape=[10, 1], dtype="float32", value=2.0
                )
                mgr.var_old_to_new[constant_tensor.name] = grad_name

            cpu = core.CPUPlace()
            exe = Executor(cpu)
            exe.run(startup_program)

            x = np.random.random(size=(10, 1)).astype('float32')
            outs = exe.run(main_program, feed={'X': x}, fetch_list=[fwd_out])[0]
            print(outs)
            loss = paddle.mean(fwd_out)
            append_backward(loss=loss)
            outs = exe.run(
                main_program,
                feed={'X': x},
                fetch_list=[data.grad_name],
            )[0]
            print(outs)


if __name__ == '__main__':
    unittest.main()
