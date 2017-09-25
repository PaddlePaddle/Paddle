import unittest
import numpy as np
from op_test import OpTest


class TestLRNOp(OpTest):
    def setUp(self):
        self.op_type = "lrn"
        N = 2
        C = 3
        H = 4
        W = 5

        n = 5
        k = 2
        alpha = 0.0001
        beta = 0.75
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [N, C, H, W]).astype("float32"),
        }
        self.outputs = {
            'out': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float32")
        }

        start = -(n - 1) / 2
        end = start + n
        print "start", start
        print "end", end
        for m in range(0, N):
            for i in range(0, C):
                for c in range(start, end + 1):
                    cur_channel = i + c
                    if cur_channel >= 0 and cur_channel < N:
                        continue

    def test_check_output(self):
        print "inputs:", self.inputs['X']
        print "slice:", self.inputs['X'][0, 0, :, :]
        #self.check_output()

    '''
    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.005)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))
    '''


if __name__ == "__main__":
    unittest.main()
