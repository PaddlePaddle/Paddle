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


class TestRoutineOp(unittest.TestCase):
    def test_simple_routine(self):
        ch = fluid.make_channel(dtype='bool')

        with fluid.Go():
            fluid.channel_send(ch, True, dtype='bool')

        result = fluid.channel_recv(ch, dtype='bool')
        fluid.channel_close(ch)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        outs = exe.run(fetch_list=[result])
        self.assertEqual(outs[0], True)

    def test_daisy_chain(self):
        n = 10000

        i = fluid.layers.zeros(shape=[1], dtype='int64')
        array_len = fluid.layers.fill_constant(shape=[1], dtype='int64', value=n)
        array_len.stop_gradient = True

        cond = fluid.layers.less_than(x=i, y=array_len)

        leftmost = fluid.make_channel(dtype='int32')
        right = leftmost
        left = leftmost

        # with fluid.While(steps=n):

        while_op = fluid.layers.While(cond=cond)
        with while_op.block():
            right = fluid.make_channel(dtype='int32')

            with fluid.Go():
                fluid.channel_send(left, 1 + fluid.channel_recv(
                    right, dtype='int32')[0], dtype='int32')

            left = right

            i = fluid.layers.increment(x=i, in_place=True)

        with fluid.Go():
            fluid.channel_send(right, 1)

        leftmost_received = fluid.channel_recv(leftmost, dtype='int32')

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        outs = exe.run(fetch_list=[leftmost_received])
        self.assertEqual(outs[0][1], True)


if __name__ == '__main__':
    unittest.main()
