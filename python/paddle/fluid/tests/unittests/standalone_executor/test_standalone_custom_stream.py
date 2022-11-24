# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid import core
from test_standalone_executor import build_program

paddle.enable_static()


class TestCustomStream(unittest.TestCase):
    """
    fill_constant(cpu)     gaussian_random
      |     |      |              |
      |     | matmul_v2(s1) fill_constant
      |     |      |              |    |
      |     |     elementwise_add(s1)  |
      |     |           |              |
      |  elementwise_sub(cpu)          |
      |     |           |              |
      |  tanh(cpu)     elementwise_add(s2)
      |     |                  |
    elementwise_sub(s1)      tanh(s2)
                 |             |
                elementwise_add(s2)
                        |
                  reduce_mean(s2)
    """

    def setUp(self):
        self.steps = 3

    def set_custom_stream(self, prog):
        op_index_for_stream1 = [2, 4, 9]
        op_index_for_stream2 = [7, 8, 10, 11]
        ops = prog.global_block().ops
        for op_index in op_index_for_stream1:
            ops[op_index].dist_attr.execution_stream = "s1"
        for op_index in op_index_for_stream2:
            ops[op_index].dist_attr.execution_stream = "s2"

    def run_program(self, apply_custom_stream=False):
        paddle.seed(2022)
        main_program, startup_program, fetch_list = build_program()
        self.assertEqual(len(startup_program.global_block().ops), 0)

        if apply_custom_stream:
            self.set_custom_stream(main_program)

        with paddle.static.program_guard(main_program, startup_program):
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            scope = core.Scope()
            outs = []
            for i in range(self.steps):
                outs.append(
                    exe.run(main_program, scope=scope, fetch_list=fetch_list)
                )
        return outs

    def test_result(self):
        if not core.is_compiled_with_cuda():
            return

        baselines = self.run_program()
        outs = self.run_program(apply_custom_stream=True)
        for bl, out in zip(baselines, outs):
            self.assertEqual(bl[0], out[0])


if __name__ == "__main__":
    unittest.main()
