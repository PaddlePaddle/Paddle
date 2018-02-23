import unittest
import numpy as np
from op_test import OpTest


def maxout_forward_naive(input, groups):
    s0, s1, s2, s3 = input.shape
    return np.ndarray([s0, s1 / groups, groups, s2, s3], \
        buffer = input, dtype=input.dtype).max(axis=(2))


class TestMaxOutOp(OpTest):
    def setUp(self):
        self.op_type = "maxout"
        self.init_test_case()
        input = np.random.random(self.shape).astype("float32")
        output = self.MaxOut_forward_naive(input, self.groups).astype("float32")

        self.inputs = {'X': input}
        self.attrs = {'groups': self.groups}

        self.outputs = {'Out': output.astype('float32')}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.MaxOut_forward_naive = maxout_forward_naive
        self.shape = [100, 6, 2, 2]
        self.groups = 2


if __name__ == '__main__':
    unittest.main()
