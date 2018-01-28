#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from operator import mul
from op_test import OpTest
import paddle.v2.fluid.core as core
from paddle.v2.fluid.op import Operator
from paddle.v2.fluid.framework import grad_var_name


def get_backward_op(scope, op, no_grad_set):
    backward_op = core.Operator.backward(op, no_grad_set)
    for input in backward_op.input_vars():
        var = scope.var(input)
        var.get_tensor()
    for output in backward_op.output_vars():
        var = scope.var(output)
        var.get_tensor()
    return backward_op


def _reference_layer_norm_naive(x, scale, beta, epsilon, begin_norm_axis=1):
    old_shape = x.shape
    N = reduce(mul, old_shape[0:begin_norm_axis], 1)
    D = reduce(mul, old_shape[begin_norm_axis:len(old_shape)], 1)
    x.shape = [N, D]
    mean = np.mean(x, axis=1)
    var = np.var(x, axis=1) + epsilon
    output = scale * np.divide((x - mean.reshape([N, 1])),
                               (np.sqrt(var)).reshape([N, 1])) + beta
    output.shape = old_shape
    x.shape = old_shape
    return output, mean, var


def _reference_layer_norm_grad(x, grad_y, scale, mean, var, begin_norm_axis=1):
    x_shape = x.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis:len(x_shape)], 1)
    grad_y.shape = [N, D]
    x.shape = [N, D]
    mean.shape = [N, 1]
    var.shape = [N, 1]

    d_scale = np.sum(grad_y).reshape([1, ])
    d_bias = np.sum(((x - mean) * np.sqrt(1 / var)) * grad_y).reshape([1, ])

    dx_end = np.sqrt(1.0 / var) * grad_y

    d_mean_0 = np.sum(-np.sqrt(1.0 / var) * grad_y, axis=1).reshape([N, 1])
    # d_mean_1 = np.sum(-1.0 / var * (x - mean) * grad_y, axis=1).reshape(
    #     [N, 1]) * (-1.0 / D * np.sqrt(1.0 / var) *
    #                np.sum(x - mean, axis=1).reshape([N, 1])).reshape([N, 1])
    d_mean = 1.0 / D * (d_mean_0)

    d_std = np.sum(-1.0 / var * (x - mean) * grad_y, axis=1).reshape([N, 1]) * (
        1.0 / D * np.sqrt(1.0 / var).reshape([N, 1]) * (x - mean))

    grad_x = scale * (dx_end + d_mean + d_std)

    grad_y.shape = x_shape
    x.shape = x_shape

    return grad_x, d_bias, d_scale


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


class TestLayerNormdOp(OpTest):
    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(
            np.allclose(
                np.array(tensor).reshape(np_array.shape), np_array, atol=atol),
            msg)

    def __assert_grad_close(self,
                            tensor,
                            np_array,
                            name,
                            place,
                            max_relative_error=0.02):
        a = np.array(tensor).reshape(np_array.shape)
        b = np_array
        abs_a = np.abs(a)
        abs_a[abs_a < 1e-5] = 1

        diff_mat = np.abs(a - b) / abs_a
        max_diff = np.max(diff_mat)

        def err_msg():
            offset = np.argmax(diff_mat > max_relative_error)
            return ("%s Variable %s max gradient diff %f over limit %f, "
                    "the first error element is %d, %f, %f") % (
                        "Gradient Check On %s" % str(place), name, max_diff,
                        max_relative_error, offset, a.flatten()[offset],
                        b.flatten()[offset])

        self.assertLessEqual(max_diff, max_relative_error, err_msg())

    def test_forward_backward(self):
        def test_with_place(place, shape, begin_norm_axis=1):
            assert begin_norm_axis > 0 and begin_norm_axis < len(
                shape), 'begin_norm_axis must be between 0 and len(shape)-1.'
            # attr
            epsilon = 0.00001
            x_shape = shape
            scale_shape = [1]
            np.random.random(123)
            x_val = np.random.random_sample(x_shape).astype(np.float32)
            scale_val = np.random.random_sample(scale_shape).astype(np.float32)
            bias_val = np.random.random_sample(scale_shape).astype(np.float32)

            # run forward
            y_out, saved_mean, var_ref = _reference_layer_norm_naive(
                x_val, scale_val, bias_val, epsilon, begin_norm_axis)

            #  for gradient test
            y_grad = np.random.random_sample(x_shape).astype(np.float32)

            x_grad_ref, scale_grad_ref, bias_grad_ref = _reference_layer_norm_grad(
                x_val, y_grad, scale_val, saved_mean, var_ref, begin_norm_axis)

            scope = core.Scope()

            # create input
            x_tensor = create_or_get_tensor(scope, "X", x_val, place)
            scale_tensor = create_or_get_tensor(scope, "Scale", scale_val,
                                                place)
            bias_tensor = create_or_get_tensor(scope, "Bias", bias_val, place)

            # create output
            y_tensor = create_or_get_tensor(scope, "Y", None, place)
            mean_tensor = create_or_get_tensor(scope, "Mean", None, place)
            variance_tensor = create_or_get_tensor(scope, "Variance", None,
                                                   place)

            layer_norm_op = Operator(
                "layer_norm",
                # inputs
                X="X",
                Scale="Scale",
                Bias="Bias",
                # outputs
                Y="Y",
                Mean="Mean",
                Variance="Variance",
                # attrs
                epsilon=epsilon,
                begin_norm_axis=begin_norm_axis)

            layer_norm_op.run(scope, place)

            # check forward result
            if isinstance(place, core.CUDAPlace):
                atol = 5e-2
            else:
                atol = 1e-4
            self.__assert_close(y_tensor, y_out, "Y", atol)
            self.__assert_close(mean_tensor, saved_mean, "Mean", atol)
            self.__assert_close(variance_tensor, var_ref, "Variance", atol)

            # run backward
            layer_norm_op_grad = get_backward_op(scope, layer_norm_op, set())
            set_output_grad(
                scope, ["Y", "Mean", "Variance"],
                place,
                feed_dict={"Y": y_grad})
            layer_norm_op_grad.run(scope, place)

            x_grad_tensor = create_or_get_tensor(scope,
                                                 grad_var_name("X"), None,
                                                 place)
            scale_grad_tensor = create_or_get_tensor(scope,
                                                     grad_var_name("Scale"),
                                                     None, place)
            bias_grad_tensor = create_or_get_tensor(scope,
                                                    grad_var_name("Bias"), None,
                                                    place)

            # check gradient output
            self.__assert_grad_close(x_grad_tensor, x_grad_ref, "x_grad", place)
            self.__assert_grad_close(scale_grad_tensor, scale_grad_ref,
                                     "scale_grad", place)
            self.__assert_grad_close(bias_grad_tensor, bias_grad_ref,
                                     "bias_grad", place)

        places = [core.CPUPlace()]
        if core.is_compile_gpu() and core.op_support_gpu("layer_norm"):
            places.append(core.CUDAPlace(0))

        for place in places:
            test_with_place(place, [2, 3, 4, 5], begin_norm_axis=1)
            test_with_place(place, [2, 3, 4, 5], begin_norm_axis=3)


if __name__ == '__main__':
    unittest.main()
