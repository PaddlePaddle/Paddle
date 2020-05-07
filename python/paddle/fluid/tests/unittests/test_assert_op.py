#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import unittest


class TestAssertOp(unittest.TestCase):
    def run_network(self, net_func):
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            net_func()
        exe = fluid.Executor()
        exe.run(main_program)

    def test_assert_true(self):
        def net_func():
            condition = layers.fill_constant(
                shape=[1], dtype='bool', value=True)
            layers.Assert(condition, [])

        self.run_network(net_func)

    def test_assert_false(self):
        def net_func():
            condition = layers.fill_constant(
                shape=[1], dtype='bool', value=False)
            layers.Assert(condition)

        with self.assertRaises(fluid.core.EnforceNotMet):
            self.run_network(net_func)

    def test_assert_cond_numel_error(self):
        def net_func():
            condition = layers.fill_constant(
                shape=[1, 2], dtype='bool', value=True)
            layers.Assert(condition, [])

        with self.assertRaises(fluid.core.EnforceNotMet):
            self.run_network(net_func)

    def test_assert_print_data(self):
        def net_func():
            zero = layers.fill_constant(shape=[1], dtype='int64', value=0)
            one = layers.fill_constant(shape=[1], dtype='int64', value=1)
            condition = layers.less_than(one, zero)  # False
            layers.Assert(condition, [zero, one])

        print("test_assert_print_data")
        with self.assertRaises(fluid.core.EnforceNotMet):
            self.run_network(net_func)

    def test_assert_summary(self):
        def net_func():
            x = layers.fill_constant(shape=[10], dtype='float32', value=2.0)
            condition = layers.reduce_max(x) < 1.0
            layers.Assert(condition, (x, ), 5)

        print("test_assert_summary")
        with self.assertRaises(fluid.core.EnforceNotMet):
            self.run_network(net_func)

    def test_assert_summary_greater_than_size(self):
        def net_func():
            x = layers.fill_constant(shape=[2, 3], dtype='float32', value=2.0)
            condition = layers.reduce_max(x) < 1.0
            layers.Assert(condition, [x], 10, name="test")

        print("test_assert_summary_greater_than_size")
        with self.assertRaises(fluid.core.EnforceNotMet):
            self.run_network(net_func)


if __name__ == '__main__':
    unittest.main()
