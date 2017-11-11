import unittest
import numpy as np
from op_test import OpTest



def maxout_forward_naive_2sweetsky(input, groups, num_channels):
    s0, s1, s2, s3 = input.shape
    return np.ndarray([s0, s1 / groups, groups, s2, s3], \
        buffer = input, dtype=input.dtype).max(axis=(2))


def maxout_forward_naive(input, groups,num_channels):
    s0, s1, s2, s3 = input.shape
    return np.ndarray([s0, s1 / groups, groups, s2, s3], \
        buffer = input, dtype=input.dtype).max(axis=(2))




class TestMaxOut_Op(OpTest):
    def setUp(self):
        self.op_type = "maxout"
        self.init_test_case()
        input = np.random.random(self.shape).astype("float32")
        output = self.MaxOut_forward_naive(input, self.groups,
                self.num_channels).astype("float32")

        self.inputs = {'X': input}
        self.attrs = {'groups': self.groups, 'num_channels': self.num_channels}

        self.outputs = {'Out': output.astype('float32')}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        print self.inputs
        print self.outputs
        self.check_grad(['X'], 'Out', max_relative_error=0.5)

    def init_test_case(self):
        self.MaxOut_forward_naive = maxout_forward_naive
        self.shape = [100, 6, 2, 2]
        self.groups=2
        self.num_channels=6




if __name__ == '__main__':
    unittest.main()
