import unittest
import numpy as np
from op_test import OpTest


class PReluTest(OpTest):
    def setUp(self):
        self.op_type = "prelu"
        x_np = np.random.normal(size=(10, 10)).astype("float32")
        x_np_sign = np.sign(x_np)
        x_np = x_np_sign * np.maximum(x_np, .005)
        alpha_np = np.array([.1])
        self.inputs = {'X': x_np, 'Alpha': alpha_np}
        out_np = np.maximum(self.inputs['X'], 0.)
        out_np = out_np + np.minimum(self.inputs['X'],
                                     0.) * self.inputs['Alpha']
        assert out_np is not self.inputs['X']
        self.outputs = {'Out': out_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
