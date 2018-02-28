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
from paddle.fluid.executor import Executor
from paddle.fluid.layers import fill_constant


class TestRoutineOp(unittest.TestCase):
    def test_simple_routine(self):
        ch = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)

        with fluid.Go():
            input_value = fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.FP64, value=99)
            fluid.channel_send(ch, input_value, dtype=core.VarDesc.VarType.LOD_TENSOR)

        result, status = fluid.channel_recv(ch, dtype=core.VarDesc.VarType.LOD_TENSOR)
        fluid.channel_close(ch)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        outs = exe.run(fetch_list=[result])
        self.assertEqual(outs[0], 99)

    def test_daisy_chain(self):
        n = 50

        cond = fluid.layers.less_than(
            x=fluid.layers.zeros(shape=[1], dtype='int64'),
            y=self._create_one_dim_tensor(n))

        leftmost = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
        left = leftmost

        for i in range(n):
            right = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
            with fluid.Go():
                one_tensor = self._create_one_dim_tensor(1)
                result, status = fluid.channel_recv(right, dtype=core.VarDesc.VarType.LOD_TENSOR)
                import pdb; pdb.set_trace()
                one_added = fluid.layers.elementwise_add(x=one_tensor, y=result)
                fluid.channel_send(left, one_added, dtype=core.VarDesc.VarType.LOD_TENSOR)
            left = right

        #fluid.layers.assign(leftmost, right)
        # fluid.layers.assign(leftmost, left)

        # with fluid.While(steps=n):

        # while_op = fluid.layers.While(cond=cond)
        # with while_op.block():
        #     right = fluid.make_channel(dtype=core.VarDesc.VarType.LOD_TENSOR)
        #
        #     with fluid.Go():
        #         result, status = fluid.channel_recv(right, dtype=core.VarDesc.VarType.LOD_TENSOR)
        #         one_added = fluid.layers.elementwise_add(x=one_tensor, y=result, act='relu')
        #         fluid.channel_send(left, one_added, dtype=core.VarDesc.VarType.LOD_TENSOR)
        #
        #     fluid.layers.assign(right, left)
        #     fluid.layers.increment(x=i, in_place=True)

        with fluid.Go():
            one_tensor = self._create_one_dim_tensor(1)
            fluid.channel_send(right, one_tensor, dtype=core.VarDesc.VarType.LOD_TENSOR)

        leftmost_received, status = fluid.channel_recv(leftmost, dtype=core.VarDesc.VarType.LOD_TENSOR)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        outs = exe.run(fetch_list=[leftmost_received])
        self.assertEqual(outs[0][0], 51)

    def _create_one_dim_tensor(self, value):
        one_dim_tensor = fill_constant(shape=[1], dtype=core.VarDesc.VarType.INT64, value=value)
        one_dim_tensor.stop_gradient = True
        return one_dim_tensor

if __name__ == '__main__':
    unittest.main()
