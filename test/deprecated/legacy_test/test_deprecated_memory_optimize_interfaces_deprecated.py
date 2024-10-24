# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from simple_nets import simple_fc_net

import paddle
from paddle import base
from paddle.distributed import transpiler


class DeprecatedMemoryOptimizationInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.method = transpiler.memory_optimize

    def build_network(self, call_interface):
        startup_prog = base.Program()
        main_prog = base.Program()
        with base.program_guard(main_prog, startup_prog):
            with base.unique_name.guard():
                loss = simple_fc_net()
                opt = paddle.optimizer.Adam(learning_rate=1e-3)
                opt.minimize(loss)

                if call_interface:
                    self.method(main_prog)

        return main_prog

    def assert_program_equal(self, prog1, prog2):
        block_num = prog1.num_blocks
        self.assertEqual(block_num, prog2.num_blocks)

        for block_id in range(block_num):
            block1 = prog1.block(block_id)
            block2 = prog2.block(block_id)
            self.assertEqual(len(block1.ops), len(block2.ops))
            for op1, op2 in zip(block1.ops, block2.ops):
                self.assertEqual(op1.input_arg_names, op2.input_arg_names)
                self.assertEqual(op1.output_arg_names, op2.output_arg_names)

            self.assertEqual(len(block1.vars), len(block2.vars))
            for var1 in block1.vars.values():
                self.assertTrue(var1.name in block2.vars)
                var2 = block2.vars.get(var1.name)
                self.assertEqual(var1.name, var2.name)

    def test_main(self):
        prog1 = self.build_network(False)
        prog2 = self.build_network(True)
        self.assert_program_equal(prog1, prog2)


class ReleaseMemoryTest(DeprecatedMemoryOptimizationInterfaceTest):
    def setUp(self):
        self.method = transpiler.release_memory


if __name__ == '__main__':
    unittest.main()
