import unittest
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


class TestInferShape(unittest.TestCase):
    def test_sum_op(self):
        prog = core.ProgramDesc.__create_program_desc__()
        self.assertIsNotNone(prog)
        block = prog.block(0)
        self.assertIsNotNone(block)

        # prepare input/output
        x1 = block.new_var("x1")
        x1.set_shape([10, 20])
        x2 = block.new_var("x2")
        x2.set_shape([10, 20])

        out = block.new_var("out")

        # prepare the operator
        sum_op_desc = block.append_op()
        sum_op_desc.set_type("sum")
        sum_op_desc.set_input("X", ["x1", "x2"])
        sum_op_desc.set_output("Out", ["out"])

        sum_op = Operator("sum", X=["x1", "x2"], Out="out")
        sum_op.infer_shape(sum_op_desc, block)
        print(out.shape())
