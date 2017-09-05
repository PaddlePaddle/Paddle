import unittest
import numpy as np
from op_test import OpTest
import paddle.v2.framework.core as core


class TestSigmoid(OpTest):
    def setUp(self):
        self.op_type = "sigmoid"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [11, 17]).astype("float32")
        }
        self.outputs = {'Y': 1 / (1 + np.exp(-self.inputs['X']))}

    def test_check_output(self):
        self.check_output(core.CPUPlace())
        self.check_output(core.GPUPlace(0))

    def test_check_grad(self):
        self.check_grad(["X"], "Y", max_relative_error=0.007)


if __name__ == '__main__':
    unittest.main()
