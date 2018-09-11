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

from __future__ import print_function

import unittest

import paddle.fluid.core as core
import paddle.fluid.layers as layers
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor
from paddle.fluid.framework import default_startup_program


class TestSwitch(unittest.TestCase):
    def check_switch(self, value):
        x = layers.fill_constant(shape=[1], dtype='float32', value=value)

        zero_var = layers.fill_constant(shape=[1], dtype='float32', value=0.0)
        one_var = layers.fill_constant(shape=[1], dtype='float32', value=1.0)
        two_var = layers.fill_constant(shape=[1], dtype='float32', value=2.0)
        three_var = layers.fill_constant(shape=[1], dtype='float32', value=3.0)

        result = layers.create_global_var(
            shape=[1], value=-1.0, dtype='float32', persistable=True)

        with layers.Switch() as switch:
            with switch.case(layers.less_than(x, zero_var)):
                layers.assign(zero_var, result)
            with switch.case(layers.less_than(x, one_var)):
                layers.assign(one_var, result)
            with switch.case(layers.less_than(x, two_var)):
                layers.assign(two_var, result)
            with switch.default():
                layers.assign(three_var, result)

        cpu = core.CPUPlace()
        exe = Executor(cpu)
        exe.run(default_startup_program())

        out = exe.run(feed={}, fetch_list=[result])[0][0]
        return out

    def test_switch(self):
        test_data = {(-0.1, 0), (0.1, 1), (1.1, 2), (2.1, 3)}
        for x, expected_result in test_data:
            main_program = framework.Program()
            startup_program = framework.Program()
            with framework.program_guard(main_program, startup_program):
                result = self.check_switch(x)
                self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
