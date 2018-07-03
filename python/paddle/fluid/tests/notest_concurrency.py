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
        ch = fluid.make_channel(
            dtype=core.VarDesc.VarType.BOOL, name="CreateChannel")
        with fluid.Go():
            fluid.channel_send(ch, True)

        result = fluid.channel_recv(ch)
        fluid.channel_close(ch)

        cpu = core.CPUPlace()
        exe = Executor(cpu)

        outs = exe.run(fetch_list=[result])
        self.assertEqual(outs[0], True)


if __name__ == '__main__':
    unittest.main()
