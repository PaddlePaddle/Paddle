import unittest

import paddle.v2.framework.core as core
from paddle.v2.framework.framework import g_program


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

    def test_append_backward(self):
        prog = core.ProgramDesc.__create_program_desc__()
        self.assertIsNotNone(prog)
        block = prog.block(0)
        self.assertIsNotNone(block)

        mul_op_desc = block.append_op()
        mul_op_desc.set_type("mul")
        mul_op_desc.set_input("X", ["x1"])
        mul_op_desc.set_input("Y", ["y1"])
        mul_op_desc.set_output("Out", ["out1"])

        sum_op_desc = block.append_op()
        sum_op_desc.set_type("elementwise_add")
        sum_op_desc.set_input("X", ["out1"])
        sum_op_desc.set_input("Y", ["b1"])
        sum_op_desc.set_output("Out", ["out2"])

        target = block.var("out2")

        expect_ops = [
            "mul", "elementwise_add", "fill_constant", "elementwise_add_grad",
            "mul_grad"
        ]

        def grad_name(name):
            return name + "@GRAD"

        actual_ops = []
        param_to_grad = prog.append_backward(target, set())
        for var_name in ("x1", "y1", "out1", "b1"):
            self.assertEqual(param_to_grad[var_name][0], grad_name(var_name))
            self.assertEqual(param_to_grad[var_name][1], 0)

        for op in block.all_ops():
            actual_ops.append(op.type())
        self.assertEqual(actual_ops, expect_ops)


if __name__ == '__main__':
    unittest.main()
