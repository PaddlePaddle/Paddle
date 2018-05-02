#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.framework import grad_var_name


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
    x_shape = x.shape
    if len(x_shape) == 2:
        if data_format == "NCHW":
            x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
        else:
            x = np.reshape(x, (x.shape[0], 1, 1, x.shape[1]))

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
        if len(x_shape) == 2:
            y = np.reshape(y, (y.shape[0], y.shape[1]))
        return y, mean, var
    elif data_format == "NHWC":
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 1, 2))
        x_sum = np.sum(x, axis=(0, 1, 2))
        element_count = np.size(x) / int(np.shape(x)[-1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        normalized = (x - mean) / np.sqrt(var + epsilon)
        y = normalized * scale + offset
        if len(x_shape) == 2:
            y = np.reshape(y, x_shape)
        return y, mean, var
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
    x_shape = x.shape

    if len(x_shape) == 2:
        if data_format == "NCHW":
            x = np.reshape(x, (x.shape[0], x.shape[1], 1, 1))
            grad_y = np.reshape(grad_y,
                                (grad_y.shape[0], grad_y.shape[1], 1, 1))
        else:
            x = np.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
            grad_y = np.reshape(grad_y,
                                (grad_y.shape[0], 1, 1, grad_y.shape[1]))

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

    if len(x_shape) == 2:
        grad_x = np.reshape(grad_x, x_shape)
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
            if out_dtype == core.VarDesc.VarType.FP64:
                data = np.ones(out_tensor.shape(), dtype=np.float64)
            elif out_dtype == core.VarDesc.VarType.FP32:
                data = np.ones(out_tensor.shape(), dtype=np.float32)
            else:
                raise ValueError("Not supported data type " + str(out_dtype))
        grad_tensor.set(data, place)

    for output in outputs:
        data = None
        if output in feed_dict:
            data = feed_dict[output]
        __set_tensor__(output, data)


class TestMKLDNNBatchNormOp(OpTest):
    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def test_forward_backward(self):
        def test_with_place(place, data_layout, shape):
            # attr
            epsilon = 0.00001
            momentum = 0.9

            # n, h, w, c = 2, 3, 4, 2
            n, c, h, w = shape[0], shape[1], shape[2], shape[3]
            # data_format is "NCHW":
            x_shape = [n, c, h, w]

            scale_shape = [c]

            x_val = np.random.random_sample(x_shape).astype(np.float32)
            scale_val = np.random.random_sample(scale_shape).astype(np.float32)
            bias_val = np.random.random_sample(scale_shape).astype(np.float32)

            mean = np.zeros(scale_shape).astype(np.float32)
            variance = np.ones(scale_shape).astype(np.float32)

            # run forward
            y_out, saved_mean, var_ref = _reference_training(
                x_val, scale_val, bias_val, epsilon, data_layout)

            # update moving mean and variance
            mean_out = saved_mean * (1. - momentum) + momentum * mean
            variance_out = var_ref * (1. - momentum) + momentum * variance
            saved_variance = 1. / np.sqrt(var_ref + epsilon)

            #  for gradient test
            y_grad = np.ones(x_shape).astype(np.float32)
            y_grad = np.zeros(x_shape).astype(np.float32)
            if len(y_grad.shape) == 2:
                y_grad[0, 0] = 1.
            else:
                y_grad[0, 0, 0, 0] = 1.
            y_grad = np.random.random_sample(x_shape).astype(np.float32)
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
                momentum=momentum,
                epsilon=epsilon,
                use_mkldnn=True)

            batch_norm_op.run(scope, place)

            # check forward result
            self.__assert_close(y_tensor, y_out, "y_out")
            self.__assert_close(saved_mean_tensor, saved_mean, "saved_mean")
            self.__assert_close(saved_variance_tensor, saved_variance,
                                "saved_variance")
            self.__assert_close(mean_out_tensor, mean_out, "mean_out")
            if isinstance(place, core.CUDAPlace):
                atol = 5e-2
            else:
                atol = 1e-4
            self.__assert_close(variance_out_tensor, variance_out,
                                "variance_out", atol)
            print "op test forward passed: ", str(place), data_layout

            batch_norm_op_grad = get_backward_op(scope, batch_norm_op, set())
            set_output_grad(
                scope,
                ["y_out", "mean", "variance", "saved_mean", "saved_variance"],
                place,
                feed_dict={"y_out": y_grad})

            batch_norm_op_grad.run(scope, place)

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
            print "op test backward passed: ", str(place), data_layout

#

        place = core.CPUPlace()
        data_format = "NCHW"
        test_with_place(place, data_format, [2, 3, 4, 5])

if __name__ == '__main__':
    unittest.main()
