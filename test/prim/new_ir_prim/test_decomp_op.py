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
from paddle import ir
from paddle.decomposition import decompose

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
        y_s = paddle.mean(y_s)
        y_s = paddle.tanh(y_s)
    newir_program = ir.translate_to_new_ir(main_program.desc)
    return newir_program


class TestBuildOp(unittest.TestCase):
    def test_build_op(self):
        newir_program = get_ir_program()
        paddle.framework.set_flags({"FLAGS_enable_new_ir_api": True})
        decompose(newir_program)
        op_name_list = [op.name() for op in newir_program.block().ops]
        self.assertEqual(
            op_name_list,
            [
                'builtin.get_parameter',
                'pd.matmul',
                'pd.add',
                'pd.full_int_array',
                'pd.sum',
                'pd.full',
                'pd.divide',
                'pd.tanh',
            ],
        )


if __name__ == "__main__":
    unittest.main()
