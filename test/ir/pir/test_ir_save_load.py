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

import paddle
from paddle import base


class TestA(unittest.TestCase):
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

            file_path = "test_save_program1.json"
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

            file_path = "test_save_program1_0.json"
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

            file_path = "test_save_program2.json"
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


if __name__ == '__main__':
    unittest.main()
