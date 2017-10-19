import unittest
import paddle.v2.framework.core as core


class TestOpDesc(unittest.TestCase):
    def test_op_desc(self):
        prog = core.ProgramDesc()
        self.assertIsNotNone(prog)
        block = prog.block(0)
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

        op.set_attr("int_attr", 1)
        self.assertEqual(1, op.attr("int_attr"))
        self.assertTrue(op.has_attr("int_attr"))
        self.assertEqual(core.AttrType.INT, op.attr_type("int_attr"))

        op.set_attr("float_attr", -1.32)
        self.assertAlmostEqual(-1.32, op.attr("float_attr"), delta=1e-4)
        self.assertTrue(op.has_attr("float_attr"))

        op.set_attr("bool_attr", False)
        self.assertFalse(op.attr("bool_attr"))

        op.set_attr("string_attr", "abc")
        self.assertEqual("abc", op.attr("string_attr"))
        self.assertTrue(op.has_attr("string_attr"))

        op.set_attr("ints_attr", [1, 2, 3])
        self.assertEqual([1, 2, 3], op.attr("ints_attr"))

        expected = [1.2, 2.3, 3.4]
        op.set_attr("floats_attr", expected)
        for e, a in zip(expected, op.attr("floats_attr")):
            self.assertAlmostEqual(e, a, delta=1e-4)

        op.set_attr("strings_attr", ["a", "b", "c"])
        self.assertEqual(["a", "b", "c"], op.attr("strings_attr"))

        op.set_attr("bools_attr", [True, False, True])
        self.assertEqual([True, False, True], op.attr("bools_attr"))

        self.assertEqual(8, len(op.attr_names()))

        op.set_block_attr("block_attr", prog.block(0))
        self.assertEqual(0, op.block_attr("block_attr"))

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
        prog_desc = core.ProgramDesc()
        self.assertIsNotNone(prog_desc)
        block_root = prog_desc.block(0)
        self.assertIsNotNone(block_root)
        self.assertEqual(block_root.id, 0)
        block1 = prog_desc.append_block(block_root)
        block2 = prog_desc.append_block(block1)
        self.assertIsNotNone(block1)
        self.assertEqual(block1.id, block2.parent)
        self.assertEqual(block_root.id, block1.parent)
        block3 = prog_desc.append_block(block_root)
        self.assertEqual(block3.parent, block_root.id)
        self.assertEqual(prog_desc.block(1).id, 1)
        self.assertEqual(4, prog_desc.num_blocks())


class TestVarDesc(unittest.TestCase):
    def test_shape(self):
        program_desc = core.ProgramDesc()
        block = program_desc.block(0)
        var = block.var('my_var')
        var.set_type(core.VarDesc.VarType.SELECTED_ROWS)
        src_shape = [3, 2, 10, 8]
        var.set_shape(src_shape)
        res_shape = var.shape()
        self.assertEqual(src_shape, res_shape)
        self.assertEqual(core.VarDesc.VarType.SELECTED_ROWS, var.type())

    def test_data_type(self):
        program_desc = core.ProgramDesc()
        block = program_desc.block(0)
        var = block.var('my_var')
        var.set_type(core.VarDesc.VarType.LOD_TENSOR)
        var.set_data_type(core.DataType.INT32)
        self.assertEqual(core.DataType.INT32, var.data_type())
        self.assertEqual(core.VarDesc.VarType.LOD_TENSOR, var.type())


class TestBlockDesc(unittest.TestCase):
    def test_add_var(self):
        prog = core.ProgramDesc()
        self.assertIsNotNone(prog)
        block = prog.block(0)
        self.assertIsNotNone(block)
        var1 = block.var("var1")
        var2 = block.var("var2")
        var3 = block.var("var3")
        all_vars = block.all_vars()
        self.assertEqual(set(all_vars), {var1, var2, var3})
        var2_re = block.find_var("var2")
        self.assertEqual(var2_re, var2)

    def test_add_op(self):
        prog = core.ProgramDesc()
        self.assertIsNotNone(prog)
        block = prog.block(0)
        self.assertIsNotNone(block)
        op1 = block.append_op()
        op2 = block.append_op()
        op0 = block.prepend_op()
        all_ops = []
        for idx in xrange(0, block.op_size()):
            all_ops.append(block.op(idx))
        self.assertEqual(all_ops, [op0, op1, op2])


if __name__ == '__main__':
    unittest.main()
