import unittest
import numpy as np
from paddle.v2.framework.op import Operator
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestPadOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "pad"
        self.inputs = {'X': np.random.random((16, 16)).astype("float32"), }
        self.attrs = {}
        self.attrs['paddings'] = [(0, 1), (2, 3)]
        self.attrs['pad_value'] = 0
        self.outputs = {
            'Out': np.pad(self.inputs['X'],
                          self.attrs['paddings'],
                          mode='constant',
                          constant_values=0)
        }


class TestPadGradOp(GradientChecker):
    def setUp(self):
        self.op = Operator(
            type="pad",
            X="X",
            Out="Out",
            paddings=[(0, 1), (2, 3)],
            pad_value=0)
        self.inputs = {'X': np.random.random((16, 16)).astype("float32"), }

    def test_normal(self):
        self.check_grad(
            self.op, self.inputs, set(["X"]), "Out", max_relative_error=0.5)

    def test_cpu_gpu_compare(self):
        self.compare_grad(self.op, self.inputs)


if __name__ == '__main__':
    unittest.main()
