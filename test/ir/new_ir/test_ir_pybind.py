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
from paddle.fluid import core

paddle.enable_static()


def get_ir_program():
    x = paddle.randn([4, 4])
    main_program, start_program = (
        paddle.static.Program(),
        paddle.static.Program(),
    )
    with paddle.static.program_guard(main_program, start_program):
        x_s = paddle.static.data('x', [4, 4], x.dtype)
        x_s.stop_gradient = False
        y_s = paddle.matmul(x_s, x_s)
        y_s = paddle.add(x_s, y_s)
        y_s = paddle.tanh(y_s)
    newir_program = core.translate_newirprogram(main_program.desc)
    return newir_program


class TestPybind(unittest.TestCase):
    def test_program(self):
        newir_program = get_ir_program()
        newir_program.print()
        ops = newir_program.block().get_op_list()
        self.assertTrue(
            len(ops), 4
        )  # ir program add "builtin.get_parameter" by default, so size is 4
        for op in ops:
            # check op.name function
            if op.name() == 'pd.tanh':
                self.assertTrue(True)
                return
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
