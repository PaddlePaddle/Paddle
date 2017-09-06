import unittest
import numpy as np
from gradient_checker import GradientChecker, create_op
from op_test_util import OpTestMeta


class TestCosSimOp(unittest.TestCase):
    __metaclass__ = OpTestMeta

    def setUp(self):
        self.type = "cos_sim"
        self.inputs = {
            'X': np.random.random((32, 64)).astype("float32"),
            'Y': np.random.random((32, 64)).astype("float32")
        }
        expect_x_norm = np.linalg.norm(self.inputs['X'], axis=1)
        expect_y_norm = np.linalg.norm(self.inputs['Y'], axis=1)
        expect_out = (self.inputs['X'] * self.inputs['Y']).sum(axis=1) / \
            expect_x_norm / expect_y_norm
        self.outputs = {
            'XNorm': np.expand_dims(expect_x_norm, 1),
            'YNorm': np.expand_dims(expect_y_norm, 1),
            'Out': np.expand_dims(expect_out, 1)
        }


class TestCosSimGradOp(GradientChecker):
    def setUp(self):
        self.op = create_op("cos_sim")
        self.inputs = {
            'X': np.random.random((10, 5)).astype("float32"),
            'Y': np.random.random((10, 5)).astype("float32")
        }

    def test_cpu_gpu_compare(self):
        self.compare_grad(self.op, self.inputs)

    def test_normal(self):
        self.check_grad(
            self.op, self.inputs, ["X", "Y"], "Out", max_relative_error=0.05)

    def test_ignore_x(self):
        self.check_grad(
            self.op,
            self.inputs, ["Y"],
            "Out",
            max_relative_error=0.05,
            no_grad_set={"X"})

    def test_ignore_y(self):
        self.check_grad(
            self.op,
            self.inputs, ["X"],
            "Out",
            max_relative_error=0.05,
            no_grad_set={"Y"})


if __name__ == '__main__':
    unittest.main()
