import unittest
import numpy as np
from paddle.v2.framework.op import Operator
from gradient_checker import GradientChecker
from op_test_util import OpTestMeta


class TestCropOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "crop"
        self.inputs = {'X': np.random.random((16, 16)).astype("float32"), }
        self.attrs = {}
        self.attrs['offsets'] = [2, 3]
        self.attrs['shape'] = [8, 8]
        self.outputs = {'Out': self.inputs['X'][2:10, 3:11]}


class TestCropGradOp(GradientChecker):
    def setUp(self):
        self.op = Operator(
            type="crop", X="X", Out="Out", offsets=[2, 3], shape=[8, 8])
        self.inputs = {'X': np.random.random((16, 16)).astype("float32"), }

    def test_normal(self):
        self.check_grad(
            self.op, self.inputs, set(["X"]), "Out", max_relative_error=0.5)

    def test_cpu_gpu_compare(self):
        self.compare_grad(self.op, self.inputs)


if __name__ == '__main__':
    unittest.main()
