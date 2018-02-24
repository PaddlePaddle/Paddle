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

from operator import mul
from op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.framework import grad_var_name

np.random.random(123)


def _reference_layer_norm_naive(x, scale, beta, epsilon, begin_norm_axis=1):
    x_shape = x.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis:len(x_shape)], 1)
    x.shape = [N, D]

    mean = np.mean(x, axis=1)
    var = np.var(x, axis=1) + epsilon
    output = scale.reshape([1, D]) * np.divide(
        (x - mean.reshape([N, 1])),
        (np.sqrt(var)).reshape([N, 1])) + beta.reshape([1, D])

    x.shape, output.shape = x_shape, x_shape
    return output, mean, var


def _reference_layer_norm_grad(x, grad_y, scale, mean, var, begin_norm_axis=1):
    x_shape = x.shape
    scale_shape = scale.shape
    N = reduce(mul, x_shape[0:begin_norm_axis], 1)
    D = reduce(mul, x_shape[begin_norm_axis:len(x_shape)], 1)
    x.shape, grad_y.shape = [N, D], [N, D]
    var.shape, mean.shape = [N, 1], [N, 1]
    scale.shape = [1, D]

    # d_bias
    d_bias = np.sum(grad_y, axis=0).reshape([1, D])
    # d_scale
    d_scale = np.sum(((x - mean) * np.sqrt(1 / var)) * grad_y,
                     axis=0).reshape([1, D])
    # dx
    dx_end = scale * np.sqrt(1.0 / var) * grad_y
    d_mean_0 = np.sum(-np.sqrt(1.0 / var) * grad_y * scale, axis=1).reshape(
        [N, 1])  # the second part equals to zero.
    d_mean = 1.0 / D * d_mean_0
    d_std = np.sum(
        -(1.0 / var) * (x - mean) * grad_y * scale, axis=1).reshape([N, 1]) * (
            1.0 / D * np.sqrt(1.0 / var).reshape([N, 1]) * (x - mean))

    grad_x = dx_end + d_mean + d_std

    grad_x.shape, x.shape, grad_y.shape = x_shape, x_shape, x_shape
    scale.shape = scale_shape
    var.shape, mean.shape = [N, ], [N, ]
    return grad_x, d_scale, d_bias


def get_backward_op(scope, op, no_grad_set):
    backward_op = core.Operator.backward(op, no_grad_set)
    for input in backward_op.input_vars():
        var = scope.var(input)
        var.get_tensor()
    for output in backward_op.output_vars():
        var = scope.var(output)
        var.get_tensor()
    return backward_op


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


class TestLayerNormdOp(OpTest):
    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def __assert_grad_close(self,
                            tensor,
                            np_array,
                            name,
                            place,
                            max_relative_error=0.02):
        a = np.array(tensor)
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

    def check_forward_backward(self, shape, begin_norm_axis):
        def test_with_place(place, shape, begin_norm_axis=1):
            # setUp
            assert begin_norm_axis > 0 and begin_norm_axis < len(
                shape), 'begin_norm_axis must be between 0 and len(shape)-1.'
            # attr
            epsilon = 0.00001
            x_shape = shape
            D = reduce(mul, x_shape[begin_norm_axis:len(x_shape)], 1)
            scale_shape = [D]

            x_val = np.random.random_sample(x_shape).astype(np.float32)
            scale_val = np.random.random_sample(scale_shape).astype(np.float32)
            bias_val = np.random.random_sample(scale_shape).astype(np.float32)
            y_grad = np.random.random_sample(x_shape).astype(np.float32)

            # run forward
            y_out, saved_mean, var_ref = _reference_layer_norm_naive(
                x_val, scale_val, bias_val, epsilon, begin_norm_axis)
            naive_fw = {"Y": y_out, "Mean": saved_mean, "Variance": var_ref}

            # get gradient
            x_grad_ref, scale_grad_ref, bias_grad_ref = _reference_layer_norm_grad(
                x_val, y_grad, scale_val, saved_mean, var_ref, begin_norm_axis)
            naive_grad = {
                "X": x_grad_ref,
                "Scale": scale_grad_ref,
                "Bias": bias_grad_ref
            }

            scope = core.Scope()

            # create input
            input_map = {"X": x_val, "Scale": scale_val, "Bias": bias_val}
            for i_name in input_map:
                create_or_get_tensor(scope, i_name, input_map[i_name], place)

            # create output
            output_map = {"Y": None, "Mean": None, "Variance": None}
            output_tensor = {}
            for o_name in output_map:
                output_tensor[o_name] = create_or_get_tensor(
                    scope, o_name, output_map[o_name], place)

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
            atol = 5e-2 if isinstance(place, core.CUDAPlace) else 1e-4
            for o_tensor in output_tensor:
                self.__assert_close(output_tensor[o_tensor], naive_fw[o_tensor],
                                    o_tensor, atol)

            # run backward
            layer_norm_op_grad = get_backward_op(scope, layer_norm_op, set())
            set_output_grad(
                scope, ["Y", "Mean", "Variance"],
                place,
                feed_dict={"Y": y_grad})
            layer_norm_op_grad.run(scope, place)

            # get output
            grad_tensor = {}
            for o_name in naive_grad:
                grad_tensor[o_name] = x_ = create_or_get_tensor(
                    scope, grad_var_name(o_name), None, place)

            # check gradient output
            for o_grad in naive_grad:
                self.__assert_grad_close(grad_tensor[o_grad],
                                         naive_grad[o_grad], o_grad + "@GRAD",
                                         place)

        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm"):
            places.append(core.CUDAPlace(0))

        for place in places:
            test_with_place(place, shape, begin_norm_axis)

    def test_check_forward_backward_with_scale_and_bias(self):
        self.check_forward_backward(shape=[2, 3, 4, 5], begin_norm_axis=1)
        self.check_forward_backward(shape=[2, 3, 4, 5], begin_norm_axis=3)

    def test_check_forward_backward_with_scale(self):
        pass  # TODO(zcd)

    def test_check_forward_backward_with_bias(self):
        pass  # TODO(zcd)

    def test_check_forward_backward(self):
        pass  # TODO(zcd)


if __name__ == '__main__':
    unittest.main()
