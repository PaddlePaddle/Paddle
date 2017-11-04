import unittest

import paddle.v2.framework.core as core
from paddle.v2.framework.framework import Program
from paddle.v2.framework.framework import g_main_program


class TestProgram(unittest.TestCase):
    def test_program(self):
        b = g_main_program.current_block()
        self.assertEqual(-1, b.parent_idx)
        self.assertEqual(0, b.idx)

        b = g_main_program.create_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = g_main_program.create_block()
        self.assertEqual(2, b.idx)
        self.assertEqual(1, b.parent_idx)

        g_main_program.rollback()

        b = g_main_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

        b = g_main_program.create_block()
        self.assertEqual(3, b.idx)
        self.assertEqual(1, b.parent_idx)

        g_main_program.rollback()
        b = g_main_program.current_block()
        self.assertEqual(1, b.idx)
        self.assertEqual(0, b.parent_idx)

    def test_program_clone(self):
        prog = Program()

        x = prog.global_block().create_var(
            name='X', shape=[1000, 784], dtype='float32')

        y = prog.global_block().create_var(
            name='Y', shape=[784, 100], dtype='float32')
        out = prog.global_block().create_var(name='Out', dtype='float32')
        prog.global_block().append_op(
            type="mul", inputs={'X': [x],
                                'Y': [y]}, outputs={'Out': [out]})

        # FIXME(yuyang18): We manual compare the output string, since the order
        # of variable could be changed.
        print prog
        print prog.clone()

    def test_parse_program_from_string(self):
        prog = Program()

        x = prog.global_block().create_var(
            name='X', shape=[1000, 784], dtype='float32')

        y = prog.global_block().create_var(
            name='Y', shape=[784, 100], dtype='float32')
        out = prog.global_block().create_var(name='Out', dtype='float32')
        prog.global_block().append_op(
            type="mul", inputs={'X': [x],
                                'Y': [y]}, outputs={'Out': [out]})

        binary_str = prog.desc.serialize_to_string()
        prog_restored = Program.parse_from_string(binary_str)

        print prog
        print prog_restored

    def test_append_backward(self):
        prog = Program()
        block = prog.global_block()

        mul_x = block.create_var(
            dtype="float32", shape=[5, 10], lod_level=0, name="mul.x")
        mul_y = block.create_var(
            dtype="float32", shape=[10, 8], lod_level=0, name="mul.y")
        mul_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="mul.out")
        mul_op = block.append_op(
            type="mul",
            inputs={"X": [mul_x],
                    "Y": mul_y},
            outputs={"Out": [mul_out]},
            attrs={"x_num_col_dims": 1})

        add_y = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="add.y")
        add_out = block.create_var(
            dtype="float32", shape=[5, 8], lod_level=0, name="add.out")
        add_op = block.append_op(
            type="elementwise_add",
            inputs={"X": mul_out,
                    "Y": add_y},
            outputs={"Out": add_out},
            attrs={"x_num_col_dims": 1})

        self.assertEqual(mul_op.idx, 0)
        self.assertEqual(add_op.idx, 1)
        param_to_grad = prog.append_backward(add_out, set())

        def grad_name(name):
            return name + "@GRAD"

        for var_name in ("mul.x", "mul.y", "mul.out", "add.y", "add.out"):
            self.assertEqual(param_to_grad[var_name][0], grad_name(var_name))
            self.assertEqual(param_to_grad[var_name][1], 0)

        expect_ops = [
            "mul", "elementwise_add", "fill_constant", "elementwise_add_grad",
            "mul_grad"
        ]
        actual_ops = []
        for op in block.ops:
            actual_ops.append(op.type)
        self.assertEqual(actual_ops, expect_ops)


if __name__ == '__main__':
    unittest.main()
