import unittest
import numpy as np
from op_test import OpTest


class TestBilinearTensorProductOp(OpTest):
    def setUp(self):
        self.op_type = "bilinear_tensor_product"
        self.inputs = {
            'X': np.random.random(3).astype("float32"),
            'Y': np.random.random(4).astype("float32"),
            'Weight': np.random.random((5, 3, 4)).astype("float32"),
            'Bias': np.random.random(5).astype("float32")
        }
        self.outputs = {
            'Out': np.matmul(
                np.matmul(self.inputs['Weight'], self.inputs['Y']),
                self.inputs['X']) + self.inputs['Bias']
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y', 'Weight', 'Bias'], 'Out', max_relative_error=0.5)


if __name__ == "__main__":
    unittest.main()
