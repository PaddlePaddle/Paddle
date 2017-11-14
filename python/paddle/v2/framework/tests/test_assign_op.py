import op_test
import numpy
import unittest


class TestAssignOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign"
        x = numpy.random.random(size=(100, 10))
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        self.check_output()

    def test_backward(self):
        self.check_grad(['X'], 'Out')


if __name__ == '__main__':
    unittest.main()
