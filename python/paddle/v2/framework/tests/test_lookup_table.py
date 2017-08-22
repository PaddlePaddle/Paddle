import unittest
import numpy as np
from op_test_util import OpTestMeta
from gradient_checker import GradientChecker, create_op


class TestSigmoidOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = 'lookup_table'
        table = np.random.random((17, 31)).astype('float32')
        ids = np.random.randint(0, 17, 4)
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids]}


class TestSigmoidGradOp(GradientChecker):
    def test_grad(self):
        op = create_op('lookup_table')
        table = np.random.random((17, 31)).astype('float32')
        ids = np.random.randint(0, 17, 4)
        inputs = {'W': table, 'Ids': ids}
        # compare gradients between cpu and gpu
        self.compare_grad(op, inputs)
        # check gradients 
        self.check_grad(op, inputs, set('W'), 'Out')


if __name__ == '__main__':
    unittest.main()
