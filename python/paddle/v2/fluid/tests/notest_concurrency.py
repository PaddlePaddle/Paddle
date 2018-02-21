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
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
from paddle.v2.fluid.executor import Executor
import numpy
import time

class TestRoutineOp(unittest.TestCase):
    def test_simple_routine(self):
        ch = fluid.make_channel(dtype=bool)
        d0 = fluid.layers.data(
            "d0", shape=[10], append_batch_size=False, dtype='float32')
        i = fluid.layers.zeros(shape=[1], dtype='int64')
        data_array = fluid.layers.array_write(x=d0, i=i)

        with fluid.Go():
            d = fluid.layers.array_read(array=data_array, i=i)

            fluid.channel_send(ch, True)

        result = fluid.channel_recv(ch)
        fluid.channel_close(ch)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        d = []
        for i in xrange(3):
            d.append(numpy.random.random(size=[10]).astype('float32'))

        outs = exe.run(
            feed={'d0': d[0]},
            fetch_list=[]
        )

        while True:
            time.sleep(10)

        #self.assertEqual(outs[0], True)


if __name__ == '__main__':
    unittest.main()
