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

import numpy as np

import paddle

paddle.enable_static()


class TestBuildModuleWithAssertOp(unittest.TestCase):
    def construct_program_with_assert(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[6, 8], dtype="float32")
            condition = paddle.all(x > 0)
            paddle.static.nn.control_flow.Assert(condition, [x], 6)
        return main_program

    def test_if_with_single_output(self):
        main_program = self.construct_program_with_assert()
        assert_op = main_program.global_block().ops[-1]
        self.assertEqual(assert_op.name(), "pd_op.assert")
        self.assertEqual(len(assert_op.results()), 0)

    def test_run(self):
        feed_dict = {"x": np.random.randn(6, 8).astype("float32")}
        with paddle.pir_utils.IrGuard():
            main = self.construct_program_with_assert()
            exe = paddle.static.Executor(paddle.CPUPlace())
            exe.run(main, feed=feed_dict)


if __name__ == "__main__":
    unittest.main()
