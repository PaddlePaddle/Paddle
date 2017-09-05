import unittest
import numpy as np
from op_test import OpTest
import paddle.v2.framework.core as core


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "sum"
        x0 = np.random.random((3, 4)).astype('float32')
        x1 = np.random.random((3, 4)).astype('float32')
        x2 = np.random.random((3, 4)).astype('float32')
        self.inputs = {"X": {"x0": x0, "x1": x1, "x2": x2}}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}

    def test_check_output(self):
        self.check_output(core.CPUPlace())
        self.check_output(core.GPUPlace(0))

    def test_check_grad(self):
        self.check_grad(["x0"], "Out")


if __name__ == '__main__':
    unittest.main()
