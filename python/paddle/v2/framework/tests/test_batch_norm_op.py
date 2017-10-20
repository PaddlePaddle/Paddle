import unittest
import numpy as np
from op_test import OpTest


class TestBatchNormOp(OpTest):
    def setUp(self):
        self.op_type = "batch_norm"
        N = 1
        C = 3
        H = 5
        W = 6

        x = np.random.random((N, C, H, W)).astype('float32')
        scale = np.random.random((C)).astype('float32')
        bias = np.random.random((C)).astype('float32')
        mean = np.zeros((C)).astype('float32')
        variance = np.zeros((C)).astype('float32')

        y = xba
        mean_out = mean
        variance_out = variance
        saved_mean = mean
        saved_variance = variance

        self.inputs = {
            "X": x,
            "Scale": scale,
            "Bias": bias,
            "Mean": mean,
            "Variance": variance
        }
        self.outputs = {
            "Y": y,
            "MeanOut": mean_out,
            "VarianceOut": variance_out,
            "SavedMean": saved_mean,
            "SavedVariance": saved_variance
        }
        self.attrs = {'is_test': False}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
