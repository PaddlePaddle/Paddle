import unittest
import numpy as np
from op_test import OpTest, get_backward_op, grad_var_name
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


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


def create_tensor_on_place(scope, var_name, var, place):
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_lod([[]])
        tensor.set_dims(var.shape)
        tensor.set(var, place)
    return tensor


def set_output_grad(scope, outputs, place):
    def __set_tensor__(name):
        out_tensor = scope.find_var(name).get_tensor()
        grad_tensor = scope.var(grad_var_name(name)).get_tensor()
        out_dtype = out_tensor.dtype()
        if out_dtype == core.DataType.FP64:
            data = np.ones(out_tensor.shape(), dtype=np.float64)
        elif out_dtype == core.DataType.FP32:
            data = np.ones(out_tensor.shape(), dtype=np.float32)
        else:
            raise ValueError("Not supported data type " + str(out_dtype))

        grad_tensor.set(data, place)

    for output in outputs:
        __set_tensor__(output)


class TestBatchNromOp1(OpTest):
    def test_forward(self):
        # attr
        data_format = "NHWC"
        epsilon = 0.00001
        momentum = 0.9

        channel_num = 2
        x_shape = [2, 3, 4, channel_num]
        scale_shape = [channel_num]

        # input
        x_val = np.random.random_sample(x_shape).astype(np.float32)
        scale_val = np.random.random_sample(scale_shape).astype(np.float32)
        bias_val = np.random.random_sample(scale_shape).astype(np.float32)

        mean = np.zeros(scale_shape).astype(np.float32)
        variance = np.zeros(scale_shape).astype(np.float32)

        # run forward
        y_out, saved_mean, var_ref = _reference_training(
            x_val, scale_val, bias_val, epsilon, data_format)

        # run backward
        mean_out = saved_mean * (1 - momentum)
        variance_out = var_ref * (1 - momentum)
        saved_variance = 1 / np.sqrt(var_ref + epsilon)

        #  for gradient test
        y_grad = np.ones(x_shape).astype(np.float32)
        grad_x_ref, grad_scale_ref, grad_bias_ref = _reference_grad(
            x_val, y_grad, scale_val, saved_mean, var_ref, epsilon, data_format)

        def test_with_place(place):
            scope = core.Scope()

            # create input
            x_tensor = create_tensor_on_place(scope, "x_val", x_val, place)
            scale_tensor = create_tensor_on_place(scope, "scale_val", scale_val,
                                                  place)
            bias_tensor = create_tensor_on_place(scope, "bias_val", bias_val,
                                                 place)
            mean_tensor = create_tensor_on_place(scope, "mean", mean, place)
            variance_tensor = create_tensor_on_place(scope, "variance",
                                                     variance, place)

            # create output
            y_tensor = create_tensor_on_place(scope, "y_out", None, place)
            saved_mean_tensor = create_tensor_on_place(scope, "saved_mean",
                                                       None, place)
            saved_variance_tensor = create_tensor_on_place(
                scope, "saved_variance", None, place)
            mean_out_tensor = mean_tensor
            variance_out_tensor = variance_tensor

            batch_norm_op = Operator(
                "batch_norm",
                # inputs
                X="x_val",
                Scale="scale_val",
                Bias="bias_val",
                Mean="mean",
                Variance="variance",
                # outputs
                Y="y_out",
                MeanOut="mean",
                VarianceOut="variance",
                SavedMean="saved_mean",
                SavedVariance="saved_variance",
                # attrs
                is_test=False,
                tensor_format=data_format,
                momentum=momentum,
                epsilon=epsilon)

            ctx = core.DeviceContext.create(place)
            batch_norm_op.run(scope, ctx)

            # get result
            self.assertTrue(
                np.allclose(
                    np.array(y_tensor), y_out, atol=1e-4), "y_out")
            self.assertTrue(
                np.allclose(
                    np.array(saved_mean_tensor), saved_mean, atol=1e-4),
                "saved_mean")
            self.assertTrue(
                np.allclose(
                    np.array(saved_variance_tensor), saved_variance, atol=1e-4),
                "saved_variance")
            self.assertTrue(
                np.allclose(
                    np.array(mean_out_tensor), mean_out, atol=1e-4),
                "mean_out")
            self.assertTrue(
                np.allclose(
                    np.array(variance_out_tensor), variance_out, atol=1e-4),
                "variance_out")

            batch_norm_op_grad = get_backward_op(scope, batch_norm_op, set())
            set_output_grad(
                scope,
                ["y_out", "mean", "variance", "saved_mean", "saved_variance"],
                place)
            batch_norm_op_grad.run(scope, ctx)

            x_grad_tensor = create_tensor_on_place(scope,
                                                   grad_var_name("x_val"), None,
                                                   place)
            scale_grad_tensor = create_tensor_on_place(
                scope, grad_var_name("scale_val"), None, place)
            bias_grad_tensor = create_tensor_on_place(scope,
                                                      grad_var_name("bias_val"),
                                                      None, place)

            self.assertTrue(
                np.allclose(
                    np.array(x_grad_tensor), grad_x_ref, atol=1e-4),
                "y_out")
            self.assertTrue(
                np.allclose(
                    np.array(scale_grad_tensor), grad_scale_ref, atol=1e-4),
                "grad_scale_ref")
            self.assertTrue(
                np.allclose(
                    np.array(bias_grad_tensor), grad_bias_ref, atol=1e-4),
                "grad_bias_ref")

        places = [core.CPUPlace()]
        if core.is_compile_gpu():
            places.append(core.GPUPlace(0))
        for place in places:
            test_with_place(place)


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

    def _check_output(self):
        self.check_output()

    def _check_grad(self):
        user_defined_grads = [
            self.grad_x_ref, self.grad_scale_ref, self.grad_bias_ref
        ]
        self.check_grad(
            ['X', 'Scale', 'Bias'], 'Y', user_defined_grads=user_defined_grads)


if __name__ == '__main__':
    unittest.main()
