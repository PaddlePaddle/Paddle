import unittest
import paddle.v2.framework.core as core


class TestProgramDesc(unittest.TestCase):
    def test_instance(self):
        program_desc = core.ProgramDesc.instance()
        self.assertIsNotNone(program_desc)
        del program_desc
        program_desc = core.ProgramDesc.instance()
        self.assertIsNotNone(program_desc)
        self.assertIsNotNone(program_desc.root_block())
        del program_desc

    def test_append_block(self):
        prog_desc = core.ProgramDesc.__create_program_desc__()
        self.assertIsNotNone(prog_desc)
        block_root = prog_desc.root_block()
        self.assertEqual(block_root.id(), 0)
        block1 = prog_desc.append_block(block_root)
        block2 = prog_desc.append_block(block1)
        self.assertEqual(block1.id(), block2.parent())
        self.assertEqual(block_root.id(), block1.parent())
        block3 = prog_desc.append_block(block_root)
        self.assertEqual(block3.parent(), block_root.id())
        self.assertEqual(prog_desc.block(1).id(), 1)


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
