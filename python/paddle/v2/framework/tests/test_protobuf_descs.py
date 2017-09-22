import unittest
import paddle.v2.framework.core as core


class TestProgramDesc(unittest.TestCase):
    def test_instance(self):
        program_desc = core.ProgramDesc.instance()
        self.assertIsNotNone(program_desc)
        del program_desc
        program_desc = core.ProgramDesc.instance()
        self.assertIsNotNone(program_desc)
        del program_desc


if __name__ == '__main__':
    unittest.main()
