#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.v2.fluid.core as core
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.framework as framework
from paddle.v2.fluid.executor import Executor
from paddle.v2.fluid.framework import default_startup_program


class TestSwitch(unittest.TestCase):
    def check_switch(self, value):

        x = layers.fill_constant(shape=[1], dtype='float32', value=value)

        zero_var = layers.fill_constant(shape=[1], dtype='float32', value=0.0)
        one_var = layers.fill_constant(shape=[1], dtype='float32', value=1.0)
        two_var = layers.fill_constant(shape=[1], dtype='float32', value=2.0)

        cond1 = layers.less_than(x, zero_var)
        cond2 = layers.less_than(x, one_var)
        cond3 = layers.less_than(x, two_var)

        result = layers.create_global_var(
            shape=[1], value=-1.0, dtype='float32', persistable=True)

        with layers.Switch() as switch:
            with switch.case(cond1):
                layers.assign(result, zero_var)
            with switch.case(cond2):
                layers.assign(result, one_var)
            with switch.case(cond3):
                layers.assign(result, two_var)

        cpu = core.CPUPlace()
        exe = Executor(cpu)
        exe.run(default_startup_program())

        out = exe.run(feed={}, fetch_list=[result])[0]
        return out

    def test_switch(self):
        main_program = framework.Program()
        startup_program = framework.Program()
        with framework.program_guard(main_program, startup_program):
            result = self.check_switch(0.0)
            self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
