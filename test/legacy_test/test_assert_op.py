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

import unittest

import paddle
from paddle import base
from paddle.static.nn.control_flow import Assert


class TestAssertOp(unittest.TestCase):
    def run_network(self, net_func):
        main_program = base.Program()
        startup_program = base.Program()
        with base.program_guard(main_program, startup_program):
            net_func()
        exe = base.Executor()
        exe.run(main_program)

    def test_assert_true(self):
        def net_func():
            condition = paddle.tensor.fill_constant(
                shape=[1], dtype='bool', value=True
            )
            Assert(condition, [])

        self.run_network(net_func)

    def test_assert_false(self):
        def net_func():
            condition = paddle.tensor.fill_constant(
                shape=[1], dtype='bool', value=False
            )
            Assert(condition)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_cond_numel_error(self):
        def net_func():
            condition = paddle.tensor.fill_constant(
                shape=[1, 2], dtype='bool', value=True
            )
            Assert(condition, [])

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_print_data(self):
        def net_func():
            zero = paddle.tensor.fill_constant(
                shape=[1], dtype='int64', value=0
            )
            one = paddle.tensor.fill_constant(shape=[1], dtype='int64', value=1)
            condition = paddle.less_than(one, zero)  # False
            Assert(condition, [zero, one])

        print("test_assert_print_data")
        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_summary(self):
        def net_func():
            x = paddle.tensor.fill_constant(
                shape=[10], dtype='float32', value=2.0
            )
            condition = paddle.max(x) < 1.0
            Assert(condition, (x,), 5)

        print("test_assert_summary")
        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_summary_greater_than_size(self):
        def net_func():
            x = paddle.tensor.fill_constant(
                shape=[2, 3], dtype='float32', value=2.0
            )
            condition = paddle.max(x) < 1.0
            Assert(condition, [x], 10, name="test")

        print("test_assert_summary_greater_than_size")
        with self.assertRaises(ValueError):
            self.run_network(net_func)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
