import unittest

import paddle.v2.framework.core as core
from paddle.v2.framework.graph import g_program


class TestProgram(unittest.TestCase):
    def test_program(self):
        b = g_program.current_block()
        self.assertEqual(-1, b.parent_idx)
        self.assertEqual(0, b.idx)

        b = g_program.create_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = g_program.create_block()
        self.assertEqual(2, b.idx)
        self.assertEqual(1, b.parent_idx)

        g_program.rollback()

        b = g_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = g_program.create_block()
        self.assertEqual(3, b.idx)
        self.assertEqual(1, b.parent_idx)

        g_program.rollback()
        b = g_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

    def test_backward(self):
        prog = core.ProgramDesc.__create_program_desc__()
        self.assertIsNotNone(prog)
        block = prog.block(0)
        self.assertIsNotNone(block)

        sum_op_desc = block.append_op()
        sum_op_desc.set_type("sum")
        sum_op_desc.set_input("X", ["x1", "x2"])
        sum_op_desc.set_output("Out", ["out"])

        self.assertEqual(len(block.all_ops()), 1)
        prog.backward(set())
        self.assertEqual(len(block.all_ops()), 3)


if __name__ == '__main__':
    unittest.main()
