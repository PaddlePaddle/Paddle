import unittest

import paddle.v2.framework.core as core
import paddle.v2.framework.create_op_creation_methods as creation


class TestGrad(unittest.TestCase):
    def test_add_grad(self):
        op = creation.op_creations.add_two(X="X", Y="Y")
        backward_op = core.Operator.backward(op, set())
        print(backward_op)


if __name__ == '__main__':
    unittest.main()
