import unittest
import numpy as np
from paddle.v2.framework.op import Operator
from gradient_checker import GradientChecker
from op_test_util import OpTestMeta


class ClipOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        input = np.random.random((16, 16)).astype("float32")
        input[np.abs(input - 0.1) < 0.05] = 0.5
        input[np.abs(input - 0.9) < 0.05] = 0.5
        self.type = "clip"
        self.inputs = {'X': input, }
        self.attrs = {}
        self.attrs['min'] = 0.1
        self.attrs['max'] = 0.9
        self.outputs = {
            'Out': np.clip(self.inputs['X'], self.attrs['min'],
                           self.attrs['max'])
        }


class TestClipGradOp(GradientChecker):
    def setUp(self):
        input = np.random.random((8, 8)).astype("float32")
        print "input: %s" % input
        self.op = Operator(type="clip", X="X", Out="Out", min=0.1, max=0.9)
        self.inputs = {'X': input, }

    def test_normal(self):
        self.check_grad(
            self.op, self.inputs, set(["X"]), "Out", max_relative_error=0.5)

    def t_cpu_gpu_compare(self):
        self.compare_grad(self.op, self.inputs)


if __name__ == '__main__':
    unittest.main()
