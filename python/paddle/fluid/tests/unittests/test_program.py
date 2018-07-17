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

from paddle.fluid.framework import Program, default_main_program, program_guard, grad_var_name
import paddle.fluid.layers as layers

main_program = default_main_program()


class TestProgram(unittest.TestCase):
    def test_program(self):
        b = main_program.current_block()
        self.assertEqual(-1, b.parent_idx)
        self.assertEqual(0, b.idx)

        b = main_program.create_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = main_program.create_block()
        self.assertEqual(2, b.idx)
        self.assertEqual(1, b.parent_idx)

        main_program.rollback()

        b = main_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = main_program.create_block()
        self.assertEqual(3, b.idx)
        self.assertEqual(1, b.parent_idx)

        main_program.rollback()
        b = main_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

    def test_program_clone(self):
        prog = Program()

        x = prog.global_block().create_var(
            name='X', shape=[1000, 784], dtype='float32')

        y = prog.global_block().create_var(
            name='Y', shape=[784, 100], dtype='float32')
        out = prog.global_block().create_var(name='Out', dtype='float32')
        prog.global_block().append_op(
            type="mul", inputs={'X': [x],
                                'Y': [y]}, outputs={'Out': [out]})

        # FIXME(yuyang18): We manual compare the output string, since the order
        # of variable could be changed.
        print(prog)
        print(prog.clone())

    def test_parse_program_from_string(self):
        prog = Program()

        x = prog.global_block().create_var(
            name='X', shape=[1000, 784], dtype='float32')

        y = prog.global_block().create_var(
            name='Y', shape=[784, 100], dtype='float32')
        out = prog.global_block().create_var(name='Out', dtype='float32')
        prog.global_block().append_op(
            type="mul", inputs={'X': [x],
                                'Y': [y]}, outputs={'Out': [out]})

        binary_str = prog.desc.serialize_to_string()
        prog_restored = Program.parse_from_string(binary_str)

        print(prog)
        print(prog_restored)

    def test_program_clone_with_parameter(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            d = layers.data(name='x', shape=[784], dtype='float32')
            hidden = layers.fc(input=d, size=100)
            layers.fc(input=hidden, size=100)

        new_program = main_program.clone()
        self.assertNotEqual(0, len(new_program.blocks[0].all_parameters()))


if __name__ == '__main__':
    unittest.main()
