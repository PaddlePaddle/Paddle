import unittest
import paddle.v2.framework.create_op_creation_methods as creation
import paddle.v2.framework.core as core


class TestMomentumSgdOp(unittest.TestCase):
    def test_plain_input_output(self):
        sgd_op = creation.op_creations.sgd_op(
            param="param", grad="grad", param_out="param_out")
        print str(sgd_op)


if __name__ == "__main__":
    unittest.main()
