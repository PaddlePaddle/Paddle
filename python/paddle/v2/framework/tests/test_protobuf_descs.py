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
        block1 = prog_desc.append_block(prog_desc.root_block())
        block2 = prog_desc.append_block(block1)
        self.assertEqual(block1.id(), block2.parent())
        self.assertEqual(prog_desc.root_block().id(), block1.parent())


if __name__ == '__main__':
    unittest.main()
