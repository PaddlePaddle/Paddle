# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import os

import paddle
from paddle import base


class TestSaveModuleWithCommonOp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "test_jit_save_load/model"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                input = paddle.full(
                    shape=[1, 512, 64], fill_value=0.5, dtype='float32'
                )
                weight = paddle.full(
                    shape=[64, 64], fill_value=0.5, dtype='float32'
                )
                bias = paddle.full(shape=[64], fill_value=1.0, dtype='float32')
                x = paddle.matmul(input, weight)
                y = paddle.add(x, bias)

            file_path = os.path.join(
            self.temp_dir.name,"test_save_program1.json")
            pir_version = 1
            base.core.serialize_pir_program(
                main_program, file_path, pir_version
            )

            recover_program = paddle.static.Program()
            base.core.deserialize_pir_program(
                file_path, recover_program, pir_version
            )

            self.assertEqual(
                len(main_program.global_block().ops),
                len(recover_program.global_block().ops),
            )
            for i in range(len(main_program.global_block().ops)):
                self.assertEqual(
                    main_program.global_block().ops[i].name(),
                    recover_program.global_block().ops[i].name(),
                )

    def test_save_no_trainable(self):
        # check save with trainable=False, no stopgradient info
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                input = paddle.full(
                    shape=[1, 512, 64], fill_value=0.5, dtype='float32'
                )
                weight = paddle.full(
                    shape=[64, 64], fill_value=0.5, dtype='float32'
                )
                input.stop_gradient = False
                bias = paddle.full(shape=[64], fill_value=1.0, dtype='float32')
                x = paddle.matmul(input, weight)
                y = paddle.add(x, bias)

            file_path = os.path.join(
                self.temp_dir.name,"test_save_program1_0.json")
            pir_version = 1
            base.core.serialize_pir_program(
                main_program, file_path, pir_version, True, True, False
            )

            recover_program = paddle.static.Program()
            base.core.deserialize_pir_program(
                file_path, recover_program, pir_version
            )

            self.assertEqual(
                main_program.global_block().ops[-1].result(0).stop_gradient,
                False,
            )
            self.assertEqual(
                recover_program.global_block().ops[-1].result(0).stop_gradient,
                True,
            )

    def test_builtin_save(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.static.Program()
            with paddle.static.program_guard(main_program):
                x_2 = paddle.static.data(
                    shape=[4, 5], dtype='int32', name='x_2'
                )
                out1, out2 = paddle.split(x=x_2, num_or_sections=2, axis=0)
                out = paddle.concat([out1, out2], axis=1)

            file_path = os.path.join(
                self.temp_dir.name,"test_save_program2.json")
            pir_version = 1
            base.core.serialize_pir_program(
                main_program, file_path, pir_version, True, True, True
            )

            recover_program = paddle.static.Program()
            base.core.deserialize_pir_program(
                file_path, recover_program, pir_version
            )

            self.assertEqual(
                len(main_program.global_block().ops),
                len(recover_program.global_block().ops),
            )
            for i in range(len(main_program.global_block().ops)):
                self.assertEqual(
                    main_program.global_block().ops[i].name(),
                    recover_program.global_block().ops[i].name(),
                )

def true_func():
    a = paddle.full(shape=[1, 2], dtype='float32', fill_value=1)
    b = paddle.full(shape=[2, 3], dtype='int64', fill_value=1)
    return a, b


def false_func():
    a = paddle.full(shape=[1, 2], dtype='float32', fill_value=3)
    b = paddle.full(shape=[2, 3], dtype='int64', fill_value=2)
    return a, b


class TestSaveModuleWithIfOp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "test_jit_save_load/model"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def construct_program_with_if(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(name="x", shape=[6, 1], dtype="float32")
            y = paddle.static.data(name="y", shape=[6, 1], dtype="float32")
            x.stop_gradient = False
            y.stop_gradient = False
            paddle.static.nn.cond(x < y, lambda: x + y, lambda: x - y)
        return main_program
    
    def check_block(self, org_block, load_block):
        self.assertEqual(len(org_block.ops), len(load_block.ops))
        for i in range(len(org_block.ops)):
            org_op = org_block.ops[i]
            load_op = load_block.ops[i]
            self.assertEqual(
                org_op.name(),
                load_op.name(),
            )

            for org_block_in, load_block_in in zip(org_op.blocks(), load_op.blocks()):
                self.check_block(org_block_in, load_block_in)

    def test_if_with_single_output(self):
        with paddle.pir_utils.IrGuard():
            main_program = self.construct_program_with_if()
            file_path = os.path.join(
                self.temp_dir.name,"test_save_program_if.json")
            pir_version = 1
            base.core.serialize_pir_program(
                main_program, file_path, pir_version
            )
            
            recover_program = paddle.static.Program()
            base.core.deserialize_pir_program(
                file_path, recover_program, pir_version
            )
    
            self.check_block(main_program.global_block(), recover_program.global_block())    

    def test_if_with_multiple_output(self):
        with paddle.pir_utils.IrGuard():
            main_program = self.construct_program_with_if()
            cond_value = main_program.global_block().ops[-1].operand_source(0)
            with paddle.pir.core.program_guard(main_program):
                paddle.static.nn.cond(cond_value, true_func, false_func)
            
            file_path = os.path.join(
            self.temp_dir.name,"test_save_program_if2.json")
            pir_version = 1
            base.core.serialize_pir_program(
                main_program, file_path, pir_version
            )
            
            recover_program = paddle.static.Program()
            base.core.deserialize_pir_program(
                file_path, recover_program, pir_version
            )

            self.check_block(main_program.global_block(), recover_program.global_block())    

if __name__ == '__main__':
    unittest.main()
