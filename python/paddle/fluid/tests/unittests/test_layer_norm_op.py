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
import paddle.fluid.core as core
import paddle.fluid as fluid

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


class TestLayerNormdOp(unittest.TestCase):
    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def check_forward_backward(self, shape, begin_norm_axis):
        def test_with_place(place, shape, begin_norm_axis):
            # attr
            epsilon = 0.00001
            x_shape = shape
            D = reduce(mul, x_shape[begin_norm_axis:len(x_shape)], 1)
            scale_shape = [D]

            np.random.seed(123)
            x = np.random.random_sample(x_shape).astype(np.float32)
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
            y_grad = np.random.random_sample(x_shape).astype(np.float32)

            # reference forward & backward
            y, mean, variance = _reference_layer_norm_naive(
                x, scale, bias, epsilon, begin_norm_axis)
            x_grad, scale_grad, bias_grad = _reference_layer_norm_grad(
                x, y_grad, scale, mean, variance, begin_norm_axis)

            var_dict = locals()
            var_dict['y@GRAD'] = y_grad
            var_names = [
                'x', 'scale', 'bias', 'mean', 'variance', 'y', 'y@GRAD'
            ]
            ground_truth = {name: var_dict[name] for name in var_names}

            program = fluid.Program()
            with fluid.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(
                        name=name,
                        dtype='float32',
                        shape=ground_truth[name].shape)
                layer_norm_op = block.append_op(
                    type="layer_norm",
                    inputs={
                        "X": block.var('x'),
                        "Scale": block.var('scale'),
                        "Bias": block.var('bias'),
                    },
                    outputs={
                        "Y": block.var('y'),
                        "Mean": block.var('mean'),  # share the same memory
                        "Variance":
                        block.var('variance'),  # share the same memory
                    },
                    attrs={
                        "epsilon": epsilon,
                        "begin_norm_axis": begin_norm_axis
                    })

                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    layer_norm_op.desc, set(), [])
                grad_op_desc = grad_op_desc_list[0]
                new_op_desc = block.desc.append_op()
                new_op_desc.copy_from(grad_op_desc)
                for var_name in grad_op_desc.output_arg_names():
                    block.desc.var(var_name.encode("ascii"))
                grad_op_desc.infer_var_type(block.desc)
                grad_op_desc.infer_shape(block.desc)
                for arg in grad_op_desc.output_arg_names():
                    grad_var = block.desc.find_var(arg.encode("ascii"))
                    grad_var.set_dtype(core.VarDesc.VarType.FP32)

                exe = fluid.Executor(place)
                out = exe.run(program,
                              feed={
                                  name: var_dict[name]
                                  for name in ['x', 'scale', 'bias', 'y@GRAD']
                              },
                              fetch_list=[
                                  'y', 'mean', 'variance', 'x@GRAD',
                                  'scale@GRAD', 'bias@GRAD'
                              ])
                self.__assert_close(y, out[0], "y")
                self.__assert_close(mean, out[1], "mean")
                self.__assert_close(variance, out[2], "variance", 1e-3)
                self.__assert_close(x_grad, out[3], "x_grad")
                self.__assert_close(scale_grad, out[4], "scale_grad", 1e-3)
                self.__assert_close(bias_grad, out[5], "bias_grad")

        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("layer_norm"):
            places.append(core.CUDAPlace(0))

        for place in places:
            test_with_place(place, shape, begin_norm_axis)

    def test_check_forward_backward_with_scale_and_bias(self):
        self.check_forward_backward(shape=[2, 3, 4, 5], begin_norm_axis=1)
        self.check_forward_backward(shape=[2, 3, 4, 5], begin_norm_axis=3)


if __name__ == '__main__':
    unittest.main()
