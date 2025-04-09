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

from paddle.base import core


class TestInferShape(unittest.TestCase):
    def test_sum_op(self):
        prog = core.ProgramDesc()
        self.assertIsNotNone(prog)
        block = prog.block(0)
        self.assertIsNotNone(block)

        shape = [10, 20]

        # prepare input/output
        x1 = block.var(b'x1')
        x1.set_type(core.VarDesc.VarType.DENSE_TENSOR)
        x1.set_shape(shape)
        x2 = block.var(b'x2')
        x2.set_type(core.VarDesc.VarType.DENSE_TENSOR)
        x2.set_shape(shape)

        out = block.var(b'out')
        out.set_type(core.VarDesc.VarType.DENSE_TENSOR)

        # prepare the operator
        sum_op_desc = block.append_op()
        sum_op_desc.set_type("sum")
        sum_op_desc.set_input("X", ["x1", "x2"])
        sum_op_desc.set_output("Out", ["out"])

        sum_op_desc.check_attrs()
        sum_op_desc.infer_shape(block)
        self.assertEqual(out.shape(), shape)

    def test_mul_op(self):
        prog = core.ProgramDesc()
        self.assertIsNotNone(prog)
        block = prog.block(0)
        self.assertIsNotNone(block)

        x_shape = [10, 20]
        y_shape = [20, 30]

        # prepare input/output
        x1 = block.var(b'x')
        x1.set_type(core.VarDesc.VarType.DENSE_TENSOR)
        x1.set_shape(x_shape)
        x2 = block.var(b'y')
        x2.set_type(core.VarDesc.VarType.DENSE_TENSOR)
        x2.set_shape(y_shape)

        out = block.var(b'out')
        out.set_type(core.VarDesc.VarType.DENSE_TENSOR)

        # prepare the operator
        mul_op_desc = block.append_op()
        mul_op_desc.set_type("mul")
        mul_op_desc.set_input("X", ["x"])
        mul_op_desc.set_input("Y", ["y"])
        mul_op_desc.set_output("Out", ["out"])
        mul_op_desc._set_attr("x_num_col_dims", 1)
        mul_op_desc._set_attr("y_num_col_dims", 1)

        mul_op_desc.check_attrs()
        mul_op_desc.infer_shape(block)
        self.assertEqual(out.shape(), [x_shape[0], y_shape[1]])


if __name__ == '__main__':
    unittest.main()
