import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestMulOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "mul"
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((84, 100)).astype("float32")
        }
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}


class TestMulGradOp(GradientChecker):
    def setUp(self):
        self.op = create_op("mul")
        self.inputs = {
            'X': np.random.random((32, 84)).astype("float32"),
            'Y': np.random.random((84, 100)).astype("float32")
        }

    def test_cpu_gpu_compare(self):
        self.compare_grad(self.op, self.inputs)

    def test_normal(self):
        # mul op will enlarge the relative error
        self.check_grad(
            self.op, self.inputs, ["X", "Y"], "Out", max_relative_error=0.5)

    def test_ignore_x(self):
        self.check_grad(
            self.op,
            self.inputs, ["Y"],
            "Out",
            max_relative_error=0.5,
            no_grad_set={"X"})

    def test_ignore_y(self):
        self.check_grad(
            self.op,
            self.inputs, ["X"],
            "Out",
            max_relative_error=0.5,
            no_grad_set={"Y"})


# TODO(dzh,qijun) : mulgrad test case need transpose feature of blas library

if __name__ == '__main__':
    unittest.main()
