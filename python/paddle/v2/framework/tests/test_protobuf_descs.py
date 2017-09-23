import unittest
import paddle.v2.framework.core as core


class TestOpDesc(unittest.TestCase):
    def test_op_desc(self):
        prog = core.ProgramDesc.__create_program_desc__()
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


class TestProgramDesc(unittest.TestCase):
    def test_instance(self):
        program_desc = core.ProgramDesc.instance()
        self.assertIsNotNone(program_desc)
        del program_desc
        program_desc = core.ProgramDesc.instance()
        self.assertIsNotNone(program_desc)
        self.assertIsNotNone(program_desc.block(0))
        del program_desc

    def test_append_block(self):
        prog_desc = core.ProgramDesc.__create_program_desc__()
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
        program_desc = core.ProgramDesc.instance()
        block = program_desc.root_block()
        var = block.new_var()
        src_shape = [3, 2, 10, 8]
        var.set_shape(src_shape)
        res_shape = var.shape()
        self.assertEqual(src_shape, res_shape)


if __name__ == '__main__':
    unittest.main()
