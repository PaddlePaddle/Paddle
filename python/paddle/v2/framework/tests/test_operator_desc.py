import unittest
from paddle.v2.framework.graph import Variable, g_program
import paddle.v2.framework.core as core


class TestOperator(unittest.TestCase):
    def test_error_type(self):
        block = g_program.create_block()
        try:
            block.append_op(type="no_such_op")
            self.assertFail()
        except AssertionError as err:
            self.assertEqual(
                err.message,
                "Operator with type \"no_such_op\" has not been registered.")

    def test_input_output(self):
        block = g_program.current_block()
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
            outputs={"Out": [mul_out]})
        self.assertEqual(mul_op.type, "mul")
        self.assertEqual(mul_op.input_names, ["X", "Y"])
        self.assertEqual(mul_op.input("X"), ["x"])
        self.assertEqual(mul_op.output_names, ["Out"])
        self.assertEqual(mul_op.output("Out"), ["out"])

    def test_mult_input(self):
        block = g_program.current_block()
        sum_x1 = block.create_var(
            dtype="int", shape=[3, 4], lod_level=0, name="sum.x1")
        sum_x2 = block.create_var(
            dtype="int", shape=[3, 4], lod_level=0, name="sum.x2")
        sum_x3 = block.create_var(
            dtype="int", shape=[3, 4], lod_level=0, name="sum.x3")
        sum_out = block.create_var(
            dtype="int", shape=[3, 4], lod_level=0, name="sum.out")
        sum_op = block.append_op(
            type="sum",
            inputs={"X": [sum_x1, sum_x2, sum_x3]},
            outputs={"Out": sum_out})
        self.assertEqual(sum_op.type, "sum")
        self.assertEqual(sum_op.input_names, ["X"])
        self.assertEqual(sum_op.input("X"), ["x1", "x2", "x3"])
        self.assertEqual(sum_op.output_names, ["Out"])
        self.assertEqual(sum_op.output("Out"), ["out"])


if __name__ == '__main__':
    unittest.main()
