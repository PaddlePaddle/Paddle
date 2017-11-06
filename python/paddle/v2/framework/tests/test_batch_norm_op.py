import unittest
import numpy as np
from op_test import OpTest
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


def grad_var_name(var_name):
    return var_name + "@GRAD"


def get_backward_op(scope, op, no_grad_set):
    backward_op = core.Operator.backward(op, no_grad_set)
    for input in backward_op.input_vars():
        var = scope.var(input)
        var.get_tensor()
    for output in backward_op.output_vars():
        var = scope.var(output)
        var.get_tensor()
    return backward_op


def _reference_training(x, scale, offset, epsilon, data_format):
    if data_format == "NCHW":
        n, c, h, w = x.shape
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 2, 3))
        x_sum = np.sum(x, axis=(0, 2, 3))
        element_count = np.size(x) / int(np.shape(x)[1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        mean_tile = np.reshape(mean, (1, c, 1, 1))
        mean_tile = np.tile(mean_tile, (n, 1, h, w))
        var_tile = np.reshape(var, (1, c, 1, 1))
        var_tile = np.tile(var_tile, (n, 1, h, w))
        normalized = (x - mean_tile) / np.sqrt(var_tile + epsilon)
        scale_tile = np.reshape(scale, (1, c, 1, 1))
        scale_tile = np.tile(scale_tile, (n, 1, h, w))
        offset_tile = np.reshape(offset, (1, c, 1, 1))
        offset_tile = np.reshape(offset_tile, (1, c, 1, 1))
        y = normalized * scale_tile + offset_tile
        return y, mean, var
    elif data_format == "NHWC":
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 1, 2))
        x_sum = np.sum(x, axis=(0, 1, 2))
        element_count = np.size(x) / int(np.shape(x)[-1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        normalized = (x - mean) / np.sqrt(var + epsilon)
        return (normalized * scale + offset), mean, var
    else:
        raise ValueError("Unknown data order.")


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

    # transfer from (N, C, H, W) to (N, H, W, C) to simplify computation
    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
        grad_y = np.transpose(grad_y, (0, 2, 3, 1))

        # raise ValueError("data_format must be NHWC, got %s." % data_format)
    grad_x = scale * (grad_y - np.mean(
        grad_y, axis=(0, 1, 2)) - (x - mean) * np.mean(
            grad_y * (x - mean), axis=(0, 1, 2)) /
                      (var + epsilon)) / np.sqrt(var + epsilon)
    grad_scale = np.sum(grad_y * (x - mean) / np.sqrt(var + epsilon),
                        axis=(0, 1, 2))
    grad_offset = np.sum(grad_y, axis=(0, 1, 2))

    # transfer back to N, C, H, W
    if data_format == "NCHW":
        grad_x = np.transpose(grad_x, (0, 3, 1, 2))
        x = np.transpose(x, (0, 3, 1, 2))
        grad_y = np.transpose(grad_y, (0, 3, 1, 2))
    return grad_x, grad_scale, grad_offset


def create_or_get_tensor(scope, var_name, var, place):
    tensor = scope.var(var_name).get_tensor()
    if var is not None:
        assert isinstance(var, np.ndarray)
        tensor.set_lod([[]])
        tensor.set_dims(var.shape)
        tensor.set(var, place)
    return tensor


def set_output_grad(scope, outputs, place, feed_dict=None):
    def __set_tensor__(name, data=None):
        out_tensor = scope.find_var(name).get_tensor()
        grad_tensor = scope.var(grad_var_name(name)).get_tensor()
        out_dtype = out_tensor.dtype()
        if data is None:
            if out_dtype == core.DataType.FP64:
                data = np.ones(out_tensor.shape(), dtype=np.float64)
            elif out_dtype == core.DataType.FP32:
                data = np.ones(out_tensor.shape(), dtype=np.float32)
            else:
                raise ValueError("Not supported data type " + str(out_dtype))
        grad_tensor.set(data, place)

    for output in outputs:
        data = None
        if output in feed_dict:
            data = feed_dict[output]
        __set_tensor__(output, data)


class TestBatchNormOp(OpTest):
    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def test_python(self):
        data_format = "NHWC"
        epsilon = 0.00001
        momentum = 0.9

        # N, H, W, C: 2, 3, 4, 2
        n, h, w, c = 2, 3, 4, 2
        x_shape = [n, h, w, c]
        scale_shape = [c]

        x_val = np.random.random_sample(x_shape).astype(np.float32)
        scale_val = np.random.random_sample(scale_shape).astype(np.float32)
        bias_val = np.random.random_sample(scale_shape).astype(np.float32)

        mean = np.zeros(scale_shape).astype(np.float32)
        variance = np.ones(scale_shape).astype(np.float32)

        # run forward
        y_out, saved_mean, var_ref = _reference_training(
            x_val, scale_val, bias_val, epsilon, "NHWC")

        #
        mean_out = saved_mean * (1. - momentum) + momentum * mean
        variance_out = var_ref * (1. - momentum) + momentum * variance
        saved_variance = 1. / np.sqrt(var_ref + epsilon)

        # running N, C, H, W case
        # should produce the same results
        x_shape2 = [n, c, h, w]
        x_val2 = np.transpose(x_val, (0, 3, 1, 2))
        y_out2, saved_mean2, var_ref2 = _reference_training(
            x_val2, scale_val, bias_val, epsilon, "NCHW")

        self.__assert_close(saved_mean, saved_mean2, "batch mean")
        self.__assert_close(var_ref, var_ref2, "batch variance")

        # transfer (N, C, H, W) back to (N, H, W, C)
        y_out2_trans = np.transpose(y_out2, (0, 2, 3, 1))
        self.__assert_close(y_out, y_out2_trans, "batch variance")
        print 'python: NHWC, NCHW, forward checking passed'

        # test backward now
        # NHWC
        self.y_grad = np.random.random_sample(x_shape).astype(np.float32)
        y_grad = self.y_grad
        # y_grad = np.ones(x_shape).astype(np.float32)
        x_grad_ref, scale_grad_ref, bias_grad_ref = _reference_grad(
            x_val, y_grad, scale_val, saved_mean, var_ref, epsilon, "NHWC")

        # NCHW
        y_grad2 = np.transpose(y_grad, (0, 3, 1, 2))
        # y_grad2 = np.ones(x_shape2).astype(np.float32)
        x_grad_ref2, scale_grad_ref2, bias_grad_ref2 = _reference_grad(
            x_val2, y_grad2, scale_val, saved_mean2, var_ref2, epsilon, "NCHW")

        self.__assert_close(scale_grad_ref, scale_grad_ref2, "scale gradient")
        self.__assert_close(bias_grad_ref, bias_grad_ref2, "bias gradient")

        x_grad_transpose = np.transpose(x_grad_ref2, (0, 2, 3, 1))
        self.__assert_close(x_grad_ref, x_grad_transpose, "x gradient")
        print 'python: NHWC, NCHW, backward checking passed'

    def test_forward_backward(self):
        def test_with_place(place, tensor_format):
            # attr
            epsilon = 0.00001
            momentum = 0.9

            # N, H, W, C: 12, 3, 4, 2
            n, h, w, c = 2, 3, 4, 2

            if data_format == "NHWC":
                x_shape = [n, h, w, c]
            elif data_format == "NCHW":
                x_shape = [n, c, h, w]
            else:
                raise ValueError("Unknown data type.")
            scale_shape = [c]

            x_val = np.random.random_sample(x_shape).astype(np.float32)
            scale_val = np.random.random_sample(scale_shape).astype(np.float32)
            bias_val = np.random.random_sample(scale_shape).astype(np.float32)

            mean = np.zeros(scale_shape).astype(np.float32)
            variance = np.ones(scale_shape).astype(np.float32)

            # run forward
            y_out, saved_mean, var_ref = _reference_training(
                x_val, scale_val, bias_val, epsilon, data_format)

            # update moving mean and variance
            mean_out = saved_mean * (1. - momentum) + momentum * mean
            variance_out = var_ref * (1. - momentum) + momentum * variance
            saved_variance = 1. / np.sqrt(var_ref + epsilon)

            #  for gradient test
            # y_grad = np.ones(x_shape).astype(np.float32)
            y_grad = np.zeros(x_shape).astype(np.float32)
            y_grad[0, 0, 0, 0] = 1.
            # y_grad = np.random.random_sample(x_shape).astype(np.float32)
            x_grad_ref, scale_grad_ref, bias_grad_ref = _reference_grad(
                x_val, y_grad, scale_val, saved_mean, var_ref, epsilon,
                data_format)

            scope = core.Scope()

            # create input
            x_tensor = create_or_get_tensor(scope, "x_val", x_val, place)
            scale_tensor = create_or_get_tensor(scope, "scale_val", scale_val,
                                                place)
            bias_tensor = create_or_get_tensor(scope, "bias_val", bias_val,
                                               place)
            mean_tensor = create_or_get_tensor(scope, "mean", mean, place)
            variance_tensor = create_or_get_tensor(scope, "variance", variance,
                                                   place)

            # create output
            y_tensor = create_or_get_tensor(scope, "y_out", None, place)
            saved_mean_tensor = create_or_get_tensor(scope, "saved_mean", None,
                                                     place)
            saved_variance_tensor = create_or_get_tensor(
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
                tensor_format=tensor_format,
                momentum=momentum,
                epsilon=epsilon)

            ctx = core.DeviceContext.create(place)
            batch_norm_op.run(scope, ctx)

            # check forward result
            self.__assert_close(y_tensor, y_out, "y_out")
            self.__assert_close(saved_mean_tensor, saved_mean, "saved_mean")
            self.__assert_close(saved_variance_tensor, saved_variance,
                                "saved_variance")
            self.__assert_close(mean_out_tensor, mean_out, "mean_out")
            if isinstance(place, core.GPUPlace):
                atol = 5e-2
            else:
                atol = 1e-4
            self.__assert_close(variance_out_tensor, variance_out,
                                "variance_out", atol)
            print "op test forward passed: ", str(place), tensor_format

            # run backward
            batch_norm_op_grad = get_backward_op(scope, batch_norm_op, set())
            set_output_grad(
                scope,
                ["y_out", "mean", "variance", "saved_mean", "saved_variance"],
                place,
                feed_dict={"y_out": y_grad})
            batch_norm_op_grad.run(scope, ctx)

            x_grad_tensor = create_or_get_tensor(scope,
                                                 grad_var_name("x_val"), None,
                                                 place)
            scale_grad_tensor = create_or_get_tensor(scope,
                                                     grad_var_name("scale_val"),
                                                     None, place)
            bias_grad_tensor = create_or_get_tensor(scope,
                                                    grad_var_name("bias_val"),
                                                    None, place)

            # check gradient output
            self.__assert_close(x_grad_tensor, x_grad_ref, "x_grad")
            self.__assert_close(scale_grad_tensor, scale_grad_ref, "scale_grad")
            self.__assert_close(bias_grad_tensor, bias_grad_ref, "bias_grad")
            print "op test backward passed: ", str(place), tensor_format

        places = [core.CPUPlace()]
        if core.is_compile_gpu() and core.op_support_gpu("batch_norm"):
            places.append(core.GPUPlace(0))
        for place in places:
            for data_format in ["NCHW", "NHWC"]:
                test_with_place(place, data_format)


if __name__ == '__main__':
    unittest.main()
