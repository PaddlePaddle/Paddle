import unittest
import numpy as np
from op_test import OpTest


def _reference_training(x, scale, offset, epsilon, data_format):
    if data_format != "NHWC":
        raise ValueError("data_format must be NHWC, got %s." % data_format)
    x_square = x * x
    x_square_sum = np.sum(x_square, (0, 1, 2))
    x_sum = np.sum(x, axis=(0, 1, 2))
    element_count = np.size(x) / int(np.shape(x)[-1])
    mean = x_sum / element_count
    var = x_square_sum / element_count - mean * mean
    normalized = (x - mean) / np.sqrt(var + epsilon)
    return (normalized * scale + offset), mean, var


def _reference_grad(x, grad_y, scale, mean, var, epsilon, data_format):
    # Use the following formulas to calculate gradients:
    # grad_scale =
    #   sum(grad_y * (x - mean)) * rsqrt(var + epsilon)
    #
    # grad_offset = sum(output_y)
    #
    # grad_x =
    #   1/N * scale * rsqrt(var + epsilon) * (N * grad_y - sum(grad_y) -
    #   (x - mean) * sum(grad_y * (x - mean)) / (var + epsilon))
    if data_format != "NHWC":
        raise ValueError("data_format must be NHWC, got %s." % data_format)
    grad_x = scale * (grad_y - np.mean(
        grad_y, axis=(0, 1, 2)) - (x - mean) * np.mean(
            grad_y * (x - mean), axis=(0, 1, 2)) /
                      (var + epsilon)) / np.sqrt(var + epsilon)
    grad_scale = np.sum(grad_y * (x - mean) / np.sqrt(var + epsilon),
                        axis=(0, 1, 2))
    grad_offset = np.sum(grad_y, axis=(0, 1, 2))
    return grad_x, grad_scale, grad_offset


class TestBatchNormOp(OpTest):
    def setUp(self):
        self.op_type = "batch_norm"

        channel_num = 2
        x_shape = [2, 3, 4, channel_num]
        scale_shape = [channel_num]

        # input
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        scale_val = np.random.random_sample(scale_shape).astype(np.float32)
        bias_val = np.random.random_sample(scale_shape).astype(np.float32)

        mean = np.zeros(scale_shape).astype(np.float32)
        variance = np.zeros(scale_shape).astype(np.float32)

        data_format = "NHWC"
        epsilon = 0.00001

        y_ref, mean_ref, var_ref = _reference_training(
            x_val, scale_val, bias_val, epsilon, data_format)

        momentum = 0.9
        mean_out = mean_ref * (1 - momentum)
        variance_out = var_ref * (1 - momentum)
        saved_variance = 1 / np.sqrt(var_ref + epsilon)

        #  for gradient test
        y_grad = np.ones(x_shape).astype(np.float32)
        grad_x_ref, grad_scale_ref, grad_bias_ref = _reference_grad(
            x_val, y_grad, scale_val, mean_ref, var_ref, epsilon, data_format)
        self.grad_x_ref = grad_x_ref
        self.grad_scale_ref = grad_scale_ref
        self.grad_bias_ref = grad_bias_ref

        self.inputs = {
            "X": x_val,
            "Scale": scale_val,
            "Bias": bias_val,
            "Mean": mean,
            "Variance": variance
        }
        self.outputs = {
            "Y": y_ref,
            "MeanOut": mean_out,
            "VarianceOut": variance_out,
            "SavedMean": mean_ref,
            "SavedVariance":
            saved_variance  # SavedVariance have been sqrt and revert to speed up training.
        }
        self.attrs = {
            'is_test': False,
            "tensor_format": data_format,
            "epsilon": epsilon
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        user_defined_grads = [
            self.grad_x_ref, self.grad_scale_ref, self.grad_bias_ref
        ]
        self.check_grad(
            ['X', 'Scale', 'Bias'], 'Y', user_defined_grads=user_defined_grads)


if __name__ == '__main__':
    unittest.main()
