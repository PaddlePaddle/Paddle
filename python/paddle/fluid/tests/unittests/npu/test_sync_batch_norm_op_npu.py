#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
import sys
import random
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.op_test import OpTest, _set_use_system_allocator
from paddle.fluid import Program, program_guard

from paddle.fluid.tests.unittests.test_batch_norm_op import _reference_testing, _cal_mean_variance, _reference_training, _reference_grad

_set_use_system_allocator(False)
paddle.enable_static()

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
paddle.seed(SEED)


@unittest.skipIf(False, "skip for tmp")
class TestBatchNormOpInference(unittest.TestCase):
    def setUp(self):
        self.dtype = np.float32
        self.init_kernel_type()
        self.data_formats = ["NCHW", "NHWC"]

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol), msg)

    def check_with_place(self, place, data_layout, dtype, shape):
        epsilon = epsilon = 0.00001
        if len(shape) == 2:
            x_shape = shape
            c = x_shape[1]
        else:
            n, h, w, c = shape[0], shape[1], shape[2], shape[3]
            if data_layout == "NHWC":
                x_shape = [n, h, w, c]
            elif data_layout == "NCHW":
                x_shape = [n, c, h, w]
            else:
                raise ValueError("Unknown data layout.")
        scale_shape = [c]

        x = np.random.random_sample(x_shape).astype(dtype)
        x = x - 0.5
        scale = np.random.random_sample(scale_shape).astype(np.float32)
        bias = np.random.random_sample(scale_shape).astype(np.float32)
        mean = np.zeros(scale_shape).astype(np.float32)
        variance = np.ones(scale_shape).astype(np.float32)
        y = _reference_testing(x, scale, bias, mean, variance, epsilon,
                               data_layout).astype(dtype)
        var_dict = locals()
        var_names = ["x", "scale", "bias", "mean", "variance", "y"]
        ground_truth = {name: var_dict[name] for name in var_names}
        ground_truth["saved_mean"] = mean
        ground_truth["saved_variance"] = variance

        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in ground_truth:
                block.create_var(
                    name=name, dtype="float32", shape=ground_truth[name].shape)
            inputs = {
                "X": block.var("x"),
                "Scale": block.var("scale"),
                "Bias": block.var("bias"),
                "Mean": block.var("mean"),
                "Variance": block.var("variance")
            }
            attrs = {
                "epsilon": epsilon,
                "is_test": True,
                "data_layout": data_layout,
                "use_mkldnn": False,
                "fuse_with_relu": False,
            }
            outputs = {
                "Y": block.var("y"),
                "MeanOut": block.var("mean"),  # share memory
                "VarianceOut": block.var("variance"),  # share memory
                "SavedMean": block.var("saved_mean"),
                "SavedVariance": block.var("saved_variance")
            }
            block.create_var(name="reserve_space", dtype='float32')
            outputs["ReserveSpace"] = block.var('reserve_space')
            bn_op = block.append_op(
                type="sync_batch_norm",
                inputs=inputs,
                outputs=outputs,
                attrs=attrs)

            program._sync_with_cpp()

            exe = fluid.Executor(place)
            out = exe.run(
                program,
                feed={
                    name: ground_truth[name]
                    for name in ["x", "scale", "bias", "mean", "variance"]
                },
                fetch_list=["y"])

            print('expect: ', var_dict["y"])
            print('actual: ', out[0])
            self.__assert_close(var_dict["y"], out[0], "y", atol=1e-3)

    def test_check_output(self):
        place = core.NPUPlace(0)
        for data_format in self.data_formats:
            self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])

    def init_kernel_type(self):
        pass


@unittest.skipIf(False, "skip for tmp")
class TestFP16BatchNormOpInference(TestBatchNormOpInference):
    def setUp(self):
        self.dtype = np.float16
        self.init_kernel_type()
        self.data_formats = ["NCHW", "NHWC"]


@unittest.skipIf(False, "skip for tmp")
class TestBatchNormOpTraining(unittest.TestCase):
    def setUp(self):
        self.set_npu()
        self.set_dtype()
        self.use_mkldnn = False
        self.fuse_with_relu = False
        self.data_formats = ["NCHW", "NHWC"]
        # self.data_formats = ["NCHW"]
        # self.data_formats = ["NHWC"]
        self.momentum = 0.9
        self.use_momentum_variable = False
        self.epsilon = 0.00001
        self.init_kernel_type()
        self.init_test_case()

    def set_npu(self):
        self.__class__.use_npu = True

    def set_dtype(self):
        self.dtype = np.float32

    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = set()
        self.fetch_list = [
            "y", 'mean', 'variance', 'saved_mean', 'saved_variance', 'x@GRAD',
            'scale@GRAD', 'bias@GRAD'
        ]

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        self.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol))

    def ref_forward_backward(self, x, y_grad, scale, bias, mean, variance,
                             epsilon, momentum, shape, data_layout):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ref_forward_backward')

        # run forward
        y, saved_mean, var_ref = _reference_training(x, scale, bias, epsilon,
                                                     data_layout)
        mean_out = saved_mean * (1. - momentum) + momentum * mean
        variance_out = var_ref * (1. - momentum) + momentum * variance
        saved_variance = 1. / np.sqrt(var_ref + epsilon)
        # run backward
        x_grad, scale_grad, bias_grad = _reference_grad(
            x, y_grad, scale, saved_mean, var_ref, epsilon, data_layout)

        print('var_ref: ', var_ref)
        print('mean_out: ', mean_out)
        print('variance_out: ', variance_out)
        print('saved_mean: ', saved_mean)
        print('saved_variance: ', saved_variance)
        print('y: ', y)
        print('x_grad: ', x_grad)
        print('scale_grad: ', scale_grad)
        print('bias_grad: ', bias_grad)

        return y, mean_out, variance_out, saved_mean, saved_variance, x_grad, scale_grad, bias_grad

    def set_mean_variance(self, scale_shape, x, data_layout):
        mean, variance = _cal_mean_variance(x, self.epsilon, data_layout)
        mean_pre = np.zeros(scale_shape).astype(np.float32)
        variance_pre = np.ones(scale_shape).astype(np.float32)
        # computing global mean/variance for one step
        if self.use_global_stats:
            mom = self.momentum
            mean = mean * (1. - mom) + mom * mean_pre
            variance = variance * (1. - mom) + mom * variance_pre
        return mean, variance

    def test_forward_backward(self):
        def test_with_place(place, data_layout, shape):
            # attr
            epsilon = self.epsilon
            momentum = self.momentum
            if data_layout == "NCHW":
                n, c, h, w = shape[0], shape[1], shape[2], shape[3]
            else:
                n, h, w, c = shape[0], shape[1], shape[2], shape[3]
            scale_shape = [c]

            np.random.seed(123)
            x = np.random.random_sample(shape).astype(self.dtype)
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
            mean, variance = self.set_mean_variance(scale_shape, x, data_layout)
            mean = mean.astype(np.float32)
            variance = variance.astype(np.float32)
            y_grad = np.random.random_sample(shape).astype(np.float32)
            momentum_var = np.array([momentum]).astype(np.float32)

            y, mean_out, variance_out, saved_mean, saved_variance, x_grad, scale_grad, bias_grad = self.ref_forward_backward(
                x, y_grad, scale, bias, mean, variance, epsilon, momentum,
                shape, data_layout)

            var_dict = locals()
            var_dict['y@GRAD'] = y_grad
            var_dict['x@GRAD'] = x_grad
            var_dict['scale@GRAD'] = scale_grad
            var_dict['bias@GRAD'] = bias_grad

            # print('var_dict: ', var_dict)

            var_names = [
                'x', 'scale', 'bias', 'mean', 'variance', "y", 'saved_mean',
                'saved_variance', 'momentum_var'
            ]
            ground_truth = {name: var_dict[name] for name in var_names}

            # print('ground_truth: ', ground_truth)

            program = fluid.Program()
            with fluid.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(
                        name=name,
                        dtype='float32',
                        shape=ground_truth[name].shape)
                inputs = {
                    "X": block.var('x'),
                    "Scale": block.var('scale'),
                    "Bias": block.var('bias'),
                    "Mean": block.var('mean'),
                    "Variance": block.var('variance')
                }
                attrs = {
                    "epsilon": epsilon,
                    "is_test": False,
                    "data_layout": data_layout,
                    "use_mkldnn": self.use_mkldnn,
                    "fuse_with_relu": self.fuse_with_relu,
                    "use_global_stats": self.use_global_stats
                }
                if self.use_momentum_variable:
                    print('11111111')
                    inputs['MomentumTensor'] = block.var('momentum_var')
                else:
                    attrs['momentum'] = momentum
                    print('22222222')
                print('attrs: ', attrs)

                outputs = {
                    "Y": block.var("y"),
                    "MeanOut": block.var('mean'),  # share memory
                    "VarianceOut": block.var('variance'),  # share memory
                    "SavedMean": block.var('saved_mean'),
                    "SavedVariance": block.var('saved_variance')
                }
                block.create_var(name="reserve_space", dtype='float32')
                outputs["ReserveSpace"] = block.var('reserve_space')
                bn_op = block.append_op(
                    type="sync_batch_norm",
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs)
                block.create_var(name='y@GRAD', dtype='float32', shape=y.shape)

                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    bn_op.desc, self.no_grad_set, [])
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

                program._sync_with_cpp()

                exe = fluid.Executor(place)
                out = exe.run(program,
                              feed={
                                  name: var_dict[name]
                                  for name in [
                                      'x', 'scale', 'bias', 'mean', 'variance',
                                      'y@GRAD', 'momentum_var'
                                  ]
                              },
                              fetch_list=self.fetch_list)

            for id, name in enumerate(self.fetch_list):
                print('id: ', id)
                print('name: ', name)
                # print('var_dict[name]: ', var_dict[name])
                # print('out[id]: ', out[id])

                #if name == 'x@GRAD':
                #    print('var_dict[name]: ', var_dict[name])
                #    print('out[id]: ', out[id])
                #    continue

                #if name == 'scale@GRAD':
                #    print('var_dict[name]: ', var_dict[name])
                #    print('out[id]: ', out[id])
                #    continue

                if name == 'variance':
                    self.__assert_close(
                        var_dict[name], out[id], name, atol=1e-3)
                    continue

                # if name == 'x@GRAD':
                #     self.__assert_close(
                #         var_dict[name], out[id], name, atol=1e-5)
                #     continue

                if self.dtype == np.float16:
                    if name == 'y':
                        self.__assert_close(
                            var_dict[name], out[id], name, atol=1e-2)
                    if name == 'saved_mean':
                        self.__assert_close(
                            var_dict[name], out[id], name, atol=1e-2)
                else:
                    self.__assert_close(var_dict[name], out[id], name)

            print("op test forward and backward passed: ",
                  str(place), data_layout)

        for data_format in self.data_formats:
            test_with_place(core.NPUPlace(0), data_format, [2, 3, 4, 5])

    def init_kernel_type(self):
        pass


@unittest.skipIf(False, "skip for tmp")
class TestFP16BatchNormOpTraining(TestBatchNormOpTraining):
    def set_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(False, "skip for tmp")
class TestBatchNormOpTrainingCase1(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = set(['scale@GRAD', 'bias@GRAD'])
        self.fetch_list = ['y', 'mean', 'variance', 'x@GRAD']


@unittest.skipIf(False, "skip for tmp")
class TestBatchNormOpTrainingCase3(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = set(['x@GRAD'])
        self.fetch_list = ['y', 'mean', 'variance', 'scale@GRAD', 'bias@GRAD']


@unittest.skipIf(False, "skip for tmp")
class TestBatchNormOpTrainingMomentumVariable(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_momentum_variable = True
        self.use_global_stats = False
        self.no_grad_set = set()
        self.fetch_list = [
            'y', 'mean', 'variance', 'saved_mean', 'saved_variance', 'x@GRAD',
            'scale@GRAD', 'bias@GRAD'
        ]


if __name__ == "__main__":
    unittest.main()
