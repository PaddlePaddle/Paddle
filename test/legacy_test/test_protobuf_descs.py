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
from paddle.base.framework import Program


class TestOpDesc(unittest.TestCase):
    def test_op_desc(self):
        program_desc = core.ProgramDesc()
        self.assertIsNotNone(program_desc)
        block = program_desc.block(0)
        self.assertIsNotNone(block)
        op = block.append_op()
        self.assertIsNotNone(op)
        op.set_type("test")
        self.assertEqual("test", op.type())
        op.set_input("X", ["a", "b", "c"])
        self.assertEqual(["a", "b", "c"], op.input("X"))
        self.assertEqual(["X"], op.input_names())

        op.set_output("Out", ["z"])
        self.assertEqual(['z'], op.output("Out"))
        self.assertEqual(["Out"], op.output_names())

        op._set_attr("int_attr", 1)
        self.assertEqual(1, op.attr("int_attr"))
        self.assertTrue(op.has_attr("int_attr"))
        self.assertEqual(core.AttrType.INT, op.attr_type("int_attr"))

        op._set_attr("float_attr", -1.32)
        self.assertAlmostEqual(-1.32, op.attr("float_attr"), delta=1e-4)
        self.assertTrue(op.has_attr("float_attr"))

        op._set_attr("bool_attr", False)
        self.assertFalse(op.attr("bool_attr"))

        op._set_attr("string_attr", "abc")
        self.assertEqual("abc", op.attr("string_attr"))
        self.assertTrue(op.has_attr("string_attr"))

        op._set_attr("ints_attr", [1, 2, 3])
        self.assertEqual([1, 2, 3], op.attr("ints_attr"))

        expected = [1.2, 2.3, 3.4]
        op._set_attr("floats_attr", expected)
        for e, a in zip(expected, op.attr("floats_attr")):
            self.assertAlmostEqual(e, a, delta=1e-4)

        op._set_attr("strings_attr", ["a", "b", "c"])
        self.assertEqual(["a", "b", "c"], op.attr("strings_attr"))

        op._set_attr("bools_attr", [True, False, True])
        self.assertEqual([True, False, True], op.attr("bools_attr"))

        self.assertEqual(8, len(op.attr_names()))

        op.set_block_attr("_block_attr", program_desc.block(0))
        self.assertEqual(0, op._block_attr_id("_block_attr"))

        mul_op = block.append_op()
        mul_op.set_type("mul")
        mul_op.check_attrs()
        self.assertEqual(mul_op.attr("x_num_col_dims"), 1)
        self.assertEqual(mul_op.attr("y_num_col_dims"), 1)


class TestProgramDesc(unittest.TestCase):
    def test_instance(self):
        program_desc = core.ProgramDesc()
        self.assertIsNotNone(program_desc)
        del program_desc
        program_desc = core.ProgramDesc()
        self.assertIsNotNone(program_desc)
        self.assertIsNotNone(program_desc.block(0))
        del program_desc

    def test_append_block(self):
        program_desc = core.ProgramDesc()
        self.assertIsNotNone(program_desc)
        block_root = program_desc.block(0)
        self.assertIsNotNone(block_root)
        self.assertEqual(block_root.id, 0)
        block1 = program_desc.append_block(block_root)
        block2 = program_desc.append_block(block1)
        self.assertIsNotNone(block1)
        self.assertEqual(block1.id, block2.parent)
        self.assertEqual(block_root.id, block1.parent)
        block3 = program_desc.append_block(block_root)
        self.assertEqual(block3.parent, block_root.id)
        self.assertEqual(program_desc.block(1).id, 1)
        self.assertEqual(4, program_desc.num_blocks())


class TestVarDesc(unittest.TestCase):
    def test_shape(self):
        program_desc = core.ProgramDesc()
        block = program_desc.block(0)
        var = block.var(b'my_var')
        var.set_type(core.VarDesc.VarType.SELECTED_ROWS)
        src_shape = [3, 2, 10, 8]
        var.set_shape(src_shape)
        res_shape = var.shape()
        self.assertEqual(src_shape, res_shape)
        self.assertEqual(core.VarDesc.VarType.SELECTED_ROWS, var.type())

    def test_multiple_shape(self):
        program_desc = core.ProgramDesc()
        block = program_desc.block(0)
        var = block.var(b'my_reader')
        var.set_type(core.VarDesc.VarType.READER)
        src_shapes = [[2, 3, 3], [4, 5], [6, 7, 8, 9]]
        var.set_shapes(src_shapes)
        res_shapes = var.shapes()
        self.assertEqual(src_shapes, res_shapes)
        self.assertEqual(core.VarDesc.VarType.READER, var.type())

    def test_dtype(self):
        program_desc = core.ProgramDesc()
        block = program_desc.block(0)
        var = block.var(b'my_var')
        var.set_type(core.VarDesc.VarType.DENSE_TENSOR)
        var.set_dtype(core.VarDesc.VarType.INT32)
        self.assertEqual(core.VarDesc.VarType.INT32, var.dtype())
        self.assertEqual(core.VarDesc.VarType.DENSE_TENSOR, var.type())

    def test_multiple_dtype(self):
        program_desc = core.ProgramDesc()
        block = program_desc.block(0)
        var = block.var(b'my_reader')
        var.set_type(core.VarDesc.VarType.READER)
        src_types = [
            core.VarDesc.VarType.INT32,
            core.VarDesc.VarType.FP64,
            core.VarDesc.VarType.FP32,
        ]
        var.set_dtypes(src_types)
        self.assertEqual(src_types, var.dtypes())
        self.assertEqual(core.VarDesc.VarType.READER, var.type())

    def test_multiple_lod_level(self):
        program_desc = core.ProgramDesc()
        block = program_desc.block(0)
        var = block.var(b'my_reader')
        var.set_type(core.VarDesc.VarType.READER)
        src_types = [3, 1, 2]
        var.set_lod_levels(src_types)
        self.assertEqual(src_types, var.lod_levels())
        self.assertEqual(core.VarDesc.VarType.READER, var.type())


class TestBlockDesc(unittest.TestCase):
    def test_add_var(self):
        program_desc = core.ProgramDesc()
        self.assertIsNotNone(program_desc)
        block = program_desc.block(0)
        self.assertIsNotNone(block)
        var1 = block.var(b"var1")
        var2 = block.var(b"var2")
        var3 = block.var(b"var3")
        all_vars = block.all_vars()
        self.assertEqual(set(all_vars), {var1, var2, var3})
        var2_re = block.find_var(b"var2")
        self.assertEqual(var2_re, var2)

    def test_add_op(self):
        program_desc = core.ProgramDesc()
        self.assertIsNotNone(program_desc)
        block = program_desc.block(0)
        self.assertIsNotNone(block)
        op1 = block.append_op()
        op2 = block.append_op()
        op0 = block._prepend_op()
        all_ops = []
        for idx in range(0, block.op_size()):
            all_ops.append(block.op(idx))
        self.assertEqual(all_ops, [op0, op1, op2])

    def test__remove_op(self):
        program = Program()
        program_desc = program.desc
        self.assertIsNotNone(program_desc)
        block = program_desc.block(0)
        self.assertIsNotNone(block)

        op0 = block.append_op()
        op1 = block.append_op()
        op2 = block.append_op()
        op0.set_type("test")
        op1.set_type("test")
        op2.set_type("test")

        block._remove_op(1, 2)
        program._sync_with_cpp()

        all_ops = []
        for idx in range(0, block.op_size()):
            all_ops.append(block.op(idx))
        self.assertEqual(all_ops, [op0, op2])


if __name__ == '__main__':
    unittest.main()
