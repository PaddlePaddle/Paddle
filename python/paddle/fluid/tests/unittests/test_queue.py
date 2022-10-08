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

import numpy as np
import os
import unittest

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core


class TestQueue(unittest.TestCase):

    def test_eq(self):
        """
        test queue_generator op, enqueue op and dequeue op.
        """

        main_program = fluid.Program()
        startup_program = fluid.Program()
        value = np.random.rand(1)
        with fluid.program_guard(main_program, startup_program):
            data_in = layers.create_global_var(shape=[2, 3],
                                               value=value,
                                               dtype="float32",
                                               persistable=True,
                                               name='var_in')
            data_out = layers.create_global_var(shape=[2, 3],
                                                value=value - 1.0,
                                                dtype="float32",
                                                persistable=True,
                                                name='var_out')
        startup_block = startup_program.block(0)
        queue_name = 'blocking_queue'
        startup_block.create_var(name=queue_name,
                                 persistable=True,
                                 type=core.VarDesc.VarType.RAW)
        startup_block.append_op(type="queue_generator",
                                attrs={'names': [queue_name]})
        block = main_program.block(0)
        block.append_op(type='enqueue',
                        inputs={'X': data_in},
                        attrs={'queue_name': queue_name})
        block.append_op(type='dequeue',
                        outputs={'Out': [data_out]},
                        attrs={'queue_name': queue_name})

        place = fluid.CUDAPlace(
            0) if core.is_compiled_with_cuda() else fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)
        ret, = exe.run(main_program, fetch_list=[data_out.name])
        np.testing.assert_allclose(np.asarray(ret),
                                   np.full((2, 3), value, np.float32),
                                   rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
