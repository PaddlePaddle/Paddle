import unittest
import numpy as np
from op_test import OpTest


class PreluTest(OpTest):
    def setUp(self):
        self.op_type = "prelu"
        self.inputs = {'X': np.random.normal(size=(10, 10)).astype("float32")}
        self.attrs = {'alpha': 0.1}
        out_np = np.maximum(self.inputs['X'], 0.)
        out_np = out_np + np.minimum(self.inputs['X'], 0.) * self.attrs['alpha']
        assert out_np is not self.inputs['X']
        self.outputs = {'Out': out_np}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


if __name__ == "__main__":
    unittest.main()
