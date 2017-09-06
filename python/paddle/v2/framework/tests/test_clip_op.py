import unittest
import numpy as np
from paddle.v2.framework.op import Operator
from gradient_checker import GradientChecker
from op_test_util import OpTestMeta


class TestClipOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        input = np.random.random((16, 16)).astype("float32")
        print "input: %s" % input
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
        self.op = Operator(type="clip", X="X", Out="Out", min=0.1, max=0.9)
        self.inputs = {'X': np.random.random((16, 16)).astype("float32"), }

    def test_normal(self):
        self.check_grad(
            self.op, self.inputs, set(["X"]), "Out", max_relative_error=0.5)

    def test_cpu_gpu_compare(self):
        self.compare_grad(self.op, self.inputs)


if __name__ == '__main__':
    unittest.main()
