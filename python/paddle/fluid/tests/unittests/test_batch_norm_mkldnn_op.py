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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from op_test import OpTest
from paddle.fluid.framework import grad_var_name
from test_batch_norm_op import TestBatchNormOpInference, TestBatchNormOpTraining, _reference_training, _reference_grad


class TestMKLDNNBatchNormOpTraining(TestBatchNormOpTraining):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        if not np.allclose(np.array(tensor), np_array, atol=atol):
            import pdb
            pdb.set_trace()
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def test_forward_backward(self):
        def test_with_place(place, data_layout, shape):
            # attr
            epsilon = 0.00001
            momentum = 0.9
            if data_layout == "NCHW":
                n, c, h, w = shape[0], shape[1], shape[2], shape[3]
            else:
                n, h, w, c = shape[0], shape[1], shape[2], shape[3]
            scale_shape = [c]

            np.random.seed(123)
            x = np.random.random_sample(shape).astype(np.float32)
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
            mean = np.zeros(scale_shape).astype(np.float32)
            variance = np.ones(scale_shape).astype(np.float32)

            # run forward
            y, saved_mean, saved_variance = _reference_training(
                x, scale, bias, epsilon, data_layout)
            mean_out = saved_mean * (1. - momentum) + momentum * mean
            variance_out = saved_variance * (1. - momentum
                                             ) + momentum * variance
            # run backward
            y_grad = np.random.random_sample(shape).astype(np.float32)
            x_grad, scale_grad, bias_grad = _reference_grad(
                x, y_grad, scale, saved_mean, saved_variance, epsilon,
                data_layout)

            var_dict = locals()
            var_dict['y@GRAD'] = y_grad

            var_names = [
                'x', 'scale', 'bias', 'mean', 'variance', 'y', 'saved_mean',
                'saved_variance'
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
                bn_op = block.append_op(
                    type="batch_norm",
                    inputs={
                        "X": block.var('x'),
                        "Scale": block.var('scale'),
                        "Bias": block.var('bias'),
                        "Mean": block.var('mean'),
                        "Variance": block.var('variance')
                    },
                    outputs={
                        "Y": block.var('y'),
                        "MeanOut": block.var('mean'),  # share the same memory
                        "VarianceOut":
                        block.var('variance'),  # share the same memory
                        "SavedMean": block.var('saved_mean'),
                        "SavedVariance": block.var('saved_variance')
                    },
                    attrs={
                        "momentum": momentum,
                        "epsilon": epsilon,
                        "is_test": False,
                        "data_layout": data_layout,
                        "use_mkldnn": self.use_mkldnn
                    })
                block.create_var(name='y@GRAD', dtype='float32', shape=y.shape)

                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    bn_op.desc, set(), [])
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
                out = exe.run(
                    program,
                    feed={
                        name: var_dict[name]
                        for name in
                        ['x', 'scale', 'bias', 'mean', 'variance', 'y@GRAD']
                    },
                    fetch_list=[
                        'y', 'mean', 'variance', 'saved_mean', 'saved_variance',
                        'x@GRAD', 'scale@GRAD', 'bias@GRAD'
                    ])

            self.__assert_close(y, out[0], "y")
            self.__assert_close(mean_out, out[1], "mean")
            self.__assert_close(variance_out, out[2], "variance", 1e-3)
            self.__assert_close(saved_mean, out[3], "saved_mean")
            self.__assert_close(saved_variance, out[4], "saved_variance", 1e-3)
            self.__assert_close(x_grad, out[5], "x_grad")
            self.__assert_close(scale_grad, out[6], "scale_grad")
            self.__assert_close(bias_grad, out[7], "bias_grad")

            print "op test forward backward passed: ", str(place), data_layout

        place = core.CPUPlace()
        data_format = "NCHW"
        test_with_place(place, data_format, [2, 3, 4, 5])


class TestMKLDNNBatchNormOpInference(TestBatchNormOpInference):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NCHW"

        self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
