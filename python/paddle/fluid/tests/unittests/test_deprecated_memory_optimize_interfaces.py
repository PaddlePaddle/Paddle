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

<<<<<<< HEAD
import unittest

from simple_nets import simple_fc_net

import paddle.fluid as fluid


class DeprecatedMemoryOptimizationInterfaceTest(unittest.TestCase):
=======
import paddle.fluid as fluid
import unittest
from simple_nets import simple_fc_net


class DeprecatedMemoryOptimizationInterfaceTest(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.method = fluid.memory_optimize

    def build_network(self, call_interface):
        startup_prog = fluid.Program()
        main_prog = fluid.Program()
        with fluid.program_guard(main_prog, startup_prog):
            with fluid.unique_name.guard():
                loss = simple_fc_net()
                opt = fluid.optimizer.Adam(learning_rate=1e-3)
                opt.minimize(loss)

                if call_interface:
                    self.method(main_prog)

        return main_prog

    def assert_program_equal(self, prog1, prog2):
        block_num = prog1.num_blocks
<<<<<<< HEAD
        self.assertEqual(block_num, prog2.num_blocks)
=======
        self.assertEquals(block_num, prog2.num_blocks)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        for block_id in range(block_num):
            block1 = prog1.block(block_id)
            block2 = prog2.block(block_id)
<<<<<<< HEAD
            self.assertEqual(len(block1.ops), len(block2.ops))
            for op1, op2 in zip(block1.ops, block2.ops):
                self.assertEqual(op1.input_arg_names, op2.input_arg_names)
                self.assertEqual(op1.output_arg_names, op2.output_arg_names)

            self.assertEqual(len(block1.vars), len(block2.vars))
            for var1 in block1.vars.values():
                self.assertTrue(var1.name in block2.vars)
                var2 = block2.vars.get(var1.name)
                self.assertEqual(var1.name, var2.name)
=======
            self.assertEquals(len(block1.ops), len(block2.ops))
            for op1, op2 in zip(block1.ops, block2.ops):
                self.assertEquals(op1.input_arg_names, op2.input_arg_names)
                self.assertEquals(op1.output_arg_names, op2.output_arg_names)

            self.assertEquals(len(block1.vars), len(block2.vars))
            for var1 in block1.vars.values():
                self.assertTrue(var1.name in block2.vars)
                var2 = block2.vars.get(var1.name)
                self.assertEquals(var1.name, var2.name)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def test_main(self):
        prog1 = self.build_network(False)
        prog2 = self.build_network(True)
        self.assert_program_equal(prog1, prog2)


class ReleaseMemoryTest(DeprecatedMemoryOptimizationInterfaceTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.method = fluid.release_memory


if __name__ == '__main__':
    unittest.main()
