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

import paddle

paddle.enable_static()


class TestBuildModuleWithAssertOp(unittest.TestCase):
    def test_assert_construct(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[2, 8], dtype="float32")
            condition = paddle.all(x > 0)
            paddle.static.nn.control_flow.Assert(condition, [x], 20)
        assert_op = main_program.global_block().ops[-1]
        self.assertEqual(assert_op.name(), "pd_op.assert")
        self.assertEqual(len(assert_op.results()), 0)

    def run_network(self, net_func):
        with paddle.pir_utils.IrGuard():
            main = paddle.static.Program()
            startup = paddle.static.Program()
            with paddle.static.program_guard(main, startup):
                net_func()

            exe = paddle.static.Executor()
            exe.run(main)

    def test_assert_true(self):
        def net_func():
            condition = paddle.tensor.fill_constant(
                shape=[1], dtype='bool', value=True
            )
            paddle.static.nn.control_flow.Assert(condition, [])

        self.run_network(net_func)

    def test_assert_false(self):
        def net_func():
            condition = paddle.tensor.fill_constant(
                shape=[1], dtype='bool', value=False
            )
            paddle.static.nn.control_flow.Assert(condition)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    # TODO(MarioLulab): May lead `test_assert_construct` construct empty main_program. Fix it soon.
    # def test_assert_cond_numel_error(self):
    #     def net_func():
    #         condition = paddle.tensor.fill_constant(
    #             shape=[1, 2], dtype='bool', value=True
    #         )
    #         paddle.static.nn.control_flow.Assert(condition, [])

    #     with self.assertRaises(ValueError):
    #         self.run_network(net_func)

    def test_assert_print_data(self):
        def net_func():
            zero = paddle.tensor.fill_constant(
                shape=[5], dtype='int64', value=0
            )
            one = paddle.tensor.fill_constant(shape=[5], dtype='int64', value=1)
            condition = paddle.less_than(one, zero).all()  # False
            paddle.static.nn.control_flow.Assert(
                condition, [zero, one], summarize=8
            )

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_summary(self):
        def net_func():
            x = paddle.tensor.fill_constant(
                shape=[10], dtype='float32', value=2.0
            )
            condition = paddle.max(x) < 1.0
            paddle.static.nn.control_flow.Assert(condition, (x,), 5)

        with self.assertRaises(ValueError):
            self.run_network(net_func)

    def test_assert_summary_greater_than_size(self):
        def net_func():
            x = paddle.tensor.fill_constant(
                shape=[2, 3], dtype='float32', value=2.0
            )
            condition = paddle.max(x) < 1.0
            paddle.static.nn.control_flow.Assert(
                condition, [x], 10, name="test"
            )

        with self.assertRaises(ValueError):
            self.run_network(net_func)


if __name__ == "__main__":
    unittest.main()
