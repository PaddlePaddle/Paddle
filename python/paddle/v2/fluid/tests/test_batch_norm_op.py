import unittest
import numpy as np
from op_test import OpTest


class TestBatchNormOpNCHW(OpTest):
    def setUp(self):
        self.op_type = "batch_norm"
        N, C, H, W = [np.random.randint(2, 10) for i in range(4)]
        scale_shape = (C)
        x_np = np.random.uniform(
            size=(N, C, H, W), low=-1., high=1.).astype(np.float32)
        scale_np = np.random.uniform(
            size=scale_shape, low=-1., high=1.).astype(np.float32)
        bias_np = np.random.uniform(
            size=scale_shape, low=-1., high=1.).astype(np.float32)

        mean_np = np.zeros(scale_shape).astype(np.float32)
        variance_np = np.ones(scale_shape).astype(np.float32)
        epsilon = 1e-5
        momentum = 0.9

        #forward
        batch_mean = np.mean(x_np, axis=(0, 2, 3))
        batch_var = np.var(x_np, axis=(0, 2, 3), ddof=0)
        batch_std = (batch_var + epsilon)**-0.5
        x_centered = (x_np - np.reshape(
            batch_mean, newshape=(1, C, 1, 1))) * np.reshape(
                batch_std, newshape=(1, C, 1, 1))
        y = x_centered * np.reshape(
            scale_np, newshape=(1, C, 1, 1)) + np.reshape(
                bias_np, newshape=(1, C, 1, 1))
        running_mean = mean_np * momentum + batch_mean * (1. - momentum)
        running_var = variance_np * momentum + batch_var * (1. - momentum)

        self.attrs = {
            "is_test": False,
            "epsilon": epsilon,
            "momentum": momentum,
            "tensor_format": "NCHW"
        }
        self.inputs = {
            "X": x_np,
            "Scale": scale_np,
            "Bias": bias_np,
            "Mean": mean_np,
            "Variance": variance_np
        }
        self.outputs = {
            "Y": y,
            'MeanOut': running_mean,
            "VarianceOut": running_var,
            "SavedMean": batch_mean,
            "SavedVariance": batch_std
        }
        self.in_place_map = {
            "MeanOut": "Mean",
            "Mean": "Mean",
            "VarianceOut": "Variance",
            "Variance": "Variance"
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')


class TestBatchNormOpNHWC(OpTest):
    def setUp(self):
        self.op_type = "batch_norm"
        N, H, W, C = [np.random.randint(2, 10) for i in range(4)]
        scale_shape = (C)
        x_np = np.random.uniform(
            size=(N, H, W, C), low=-1., high=1.).astype(np.float32)
        scale_np = np.random.uniform(
            size=scale_shape, low=-1., high=1.).astype(np.float32)
        bias_np = np.random.uniform(
            size=scale_shape, low=-1., high=1.).astype(np.float32)

        mean_np = np.zeros(scale_shape).astype(np.float32)
        variance_np = np.ones(scale_shape).astype(np.float32)
        epsilon = 1e-5
        momentum = 0.9

        #forward
        batch_mean = np.mean(x_np, axis=(0, 1, 2))
        batch_var = np.var(x_np, axis=(0, 1, 2), ddof=0)
        batch_std = (batch_var + epsilon)**-0.5
        x_centered = (x_np - np.reshape(
            batch_mean, newshape=(1, 1, 1, C))) * np.reshape(
                batch_std, newshape=(1, 1, 1, C))
        y = x_centered * np.reshape(
            scale_np, newshape=(1, 1, 1, C)) + np.reshape(
                bias_np, newshape=(1, 1, 1, C))
        running_mean = mean_np * momentum + batch_mean * (1. - momentum)
        running_var = variance_np * momentum + batch_var * (1. - momentum)

        self.attrs = {
            "is_test": False,
            "epsilon": epsilon,
            "momentum": momentum,
            "tensor_format": "NHWC"
        }
        self.inputs = {
            "X": x_np,
            "Scale": scale_np,
            "Bias": bias_np,
            "Mean": mean_np,
            "Variance": variance_np
        }
        self.outputs = {
            "Y": y,
            'MeanOut': running_mean,
            "VarianceOut": running_var,
            "SavedMean": batch_mean,
            "SavedVariance": batch_std
        }
        self.in_place_map = {
            "MeanOut": "Mean",
            "Mean": "Mean",
            "VarianceOut": "Variance",
            "Variance": "Variance"
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')


if __name__ == '__main__':
    unittest.main()
