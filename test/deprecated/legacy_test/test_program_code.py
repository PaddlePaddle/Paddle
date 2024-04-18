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

import unittest

import paddle
from paddle import base


class TestProgramToReadableCode(unittest.TestCase):
    def setUp(self):
        self.program = base.Program()
        self.block = self.program.current_block()
        self.var = self.block.create_var(
            name="X", shape=[-1, 23, 48], dtype='float32'
        )
        self.param = self.block.create_parameter(
            name="W", shape=[23, 48], dtype='float32', trainable=True
        )
        self.op = self.block.append_op(
            type="abs", inputs={"X": [self.var]}, outputs={"Out": [self.var]}
        )
        # add control flow op and sub block
        self.append_cond_op(self.program)

    def append_cond_op(self, program):
        def true_func():
            return paddle.tensor.fill_constant(
                shape=[2, 3], dtype='int32', value=2
            )

        def false_func():
            return paddle.tensor.fill_constant(
                shape=[3, 2], dtype='int32', value=-1
            )

        with base.program_guard(program):
            x = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.1
            )
            y = paddle.tensor.fill_constant(
                shape=[1], dtype='float32', value=0.23
            )
            pred = paddle.less_than(y, x)
            out = paddle.static.nn.cond(pred, true_func, false_func)

    def test_program_code(self):
        self.var._to_readable_code()
        self.param._to_readable_code()
        self.op._to_readable_code()
        self.block._to_readable_code()
        self.program._to_readable_code()

    def test_program_print(self):
        print(self.var)
        print(self.param)
        print(self.op)
        print(self.block)
        print(self.program)


if __name__ == "__main__":
    unittest.main()
