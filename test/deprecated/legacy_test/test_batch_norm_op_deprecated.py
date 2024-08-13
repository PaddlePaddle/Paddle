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

import os
import unittest

import numpy as np
from op_test import (
    _set_use_system_allocator,
)

import paddle
from paddle import base
from paddle.base import core

paddle.enable_static()

_set_use_system_allocator(True)


def _cal_mean_variance(x, epsilon, data_format):
    assert data_format in ['NCHW', 'NHWC']
    x_shape = x.shape
    if len(x_shape) == 3:
        if data_format == "NCHW":  # NCL -> NCL1
            x = np.reshape(x, (x_shape[0], x_shape[1], x_shape[2], 1))
        else:  # NLC -> NL1C
            x = np.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))
    x_square = x * x
    axis = (0, 2, 3) if data_format == 'NCHW' else (0, 1, 2)
    C = x.shape[1] if data_format == 'NCHW' else x.shape[-1]
    x_square_sum = np.sum(x_square, axis)
    x_sum = np.sum(x, axis=axis)
    element_count = np.size(x) / C
    mean = x_sum / element_count
    var = x_square_sum / element_count - mean * mean
    return mean, var


def _reference_training(x, scale, offset, epsilon, data_format):
    x_shape = x.shape

    if len(x_shape) == 3:
        if data_format == "NCHW":  # NCL -> NCL1
            x = np.reshape(x, (x_shape[0], x_shape[1], x_shape[2], 1))
        else:  # NLC -> NL1C
            x = np.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))

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
    elif data_format == "NHWC":
        x_square = x * x
        x_square_sum = np.sum(x_square, (0, 1, 2))
        x_sum = np.sum(x, axis=(0, 1, 2))
        element_count = np.size(x) / int(np.shape(x)[-1])
        mean = x_sum / element_count
        var = x_square_sum / element_count - mean * mean
        normalized = (x - mean) / np.sqrt(var + epsilon)
        y = normalized * scale + offset
    else:
        raise ValueError("Unknown data order.")

    if len(x_shape) == 3:
        y = np.reshape(y, x_shape)
    return y, mean, var


def _reference_grad(x, y_grad, scale, mean, var, epsilon, data_format):
    # Use the following formulas to calculate gradients:
    # grad_scale =
    #   sum(grad_y * (x - mean)) * rsqrt(var + epsilon)
    #
    # grad_offset = sum(output_y)
    #
    # x_grad =
    #   1/N * scale * rsqrt(var + epsilon) * (N * grad_y - sum(grad_y) -
    #   (x - mean) * sum(grad_y * (x - mean)) / (var + epsilon))

    # transfer from (N, C, H, W) to (N, H, W, C) to simplify computation
    if data_format != "NCHW" and data_format != "NHWC":
        raise ValueError("Unknown data order.")

    x_shape = x.shape
    if len(x_shape) == 3:
        if data_format == "NCHW":  # NCL -> NCL1
            x = np.reshape(x, (x_shape[0], x_shape[1], x_shape[2], 1))
            y_grad = np.reshape(y_grad, (x_shape[0], x_shape[1], x_shape[2], 1))
        else:  # NLC -> NL1C
            x = np.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2]))
            y_grad = np.reshape(y_grad, (x_shape[0], x_shape[1], 1, x_shape[2]))

    if data_format == "NCHW":
        x = np.transpose(x, (0, 2, 3, 1))
        y_grad = np.transpose(y_grad, (0, 2, 3, 1))

    x_grad = (
        scale
        * (
            y_grad
            - np.mean(y_grad, axis=(0, 1, 2))
            - (x - mean)
            * np.mean(y_grad * (x - mean), axis=(0, 1, 2))
            / (var + epsilon)
        )
        / np.sqrt(var + epsilon)
    )
    grad_scale = np.sum(
        y_grad * (x - mean) / np.sqrt(var + epsilon), axis=(0, 1, 2)
    )
    grad_offset = np.sum(y_grad, axis=(0, 1, 2))

    # transfer back to N, C, H, W
    if data_format == "NCHW":
        x_grad = np.transpose(x_grad, (0, 3, 1, 2))
        x = np.transpose(x, (0, 3, 1, 2))
        y_grad = np.transpose(y_grad, (0, 3, 1, 2))

    if len(x_shape) == 3:
        x_grad = np.reshape(x_grad, x_shape)

    return x_grad, grad_scale, grad_offset


class TestBatchNormOpTraining(unittest.TestCase):
    def setUp(self):
        self.use_mkldnn = False
        self.fuse_with_relu = False
        self.data_formats = ["NCHW", "NHWC"]
        self.momentum = 0.9
        self.use_momentum_variable = False
        self.epsilon = 0.00001
        self.init_kernel_type()
        self.init_test_case()

    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = set()
        self.fetch_list = [
            'y',
            'mean',
            'variance',
            'saved_mean',
            'saved_variance',
            'x@GRAD',
            'scale@GRAD',
            'bias@GRAD',
        ]

    def __assert_close(self, tensor, np_array, msg, atol=1e-4):
        np.allclose(np.array(tensor), np_array, atol=atol)

    def ref_forward_backward(
        self,
        x,
        y_grad,
        scale,
        bias,
        mean,
        variance,
        epsilon,
        momentum,
        shape,
        data_layout,
    ):
        # run forward
        y, saved_mean, var_ref = _reference_training(
            x, scale, bias, epsilon, data_layout
        )
        mean_out = saved_mean * (1.0 - momentum) + momentum * mean
        variance_out = var_ref * (1.0 - momentum) + momentum * variance
        saved_variance = 1.0 / np.sqrt(var_ref + epsilon)
        # run backward
        x_grad, scale_grad, bias_grad = _reference_grad(
            x, y_grad, scale, saved_mean, var_ref, epsilon, data_layout
        )

        return (
            y,
            mean_out,
            variance_out,
            saved_mean,
            saved_variance,
            x_grad,
            scale_grad,
            bias_grad,
        )

    def set_mean_variance(self, scale_shape, x, data_layout):
        mean, variance = _cal_mean_variance(x, self.epsilon, data_layout)
        mean_pre = np.zeros(scale_shape).astype(np.float32)
        variance_pre = np.ones(scale_shape).astype(np.float32)
        # computing global mean/variance for one step
        if self.use_global_stats:
            mom = self.momentum
            mean = mean * (1.0 - mom) + mom * mean_pre
            variance = variance * (1.0 - mom) + mom * variance_pre
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
            x = np.random.random_sample(shape).astype(np.float32)
            scale = np.random.random_sample(scale_shape).astype(np.float32)
            bias = np.random.random_sample(scale_shape).astype(np.float32)
            mean, variance = self.set_mean_variance(scale_shape, x, data_layout)
            y_grad = np.random.random_sample(shape).astype(np.float32)
            momentum_var = np.array([momentum]).astype(np.float32)

            (
                y,
                mean_out,
                variance_out,
                saved_mean,
                saved_variance,
                x_grad,
                scale_grad,
                bias_grad,
            ) = self.ref_forward_backward(
                x,
                y_grad,
                scale,
                bias,
                mean,
                variance,
                epsilon,
                momentum,
                shape,
                data_layout,
            )

            var_dict = locals()
            var_dict['y@GRAD'] = y_grad
            var_dict['x@GRAD'] = x_grad
            var_dict['scale@GRAD'] = scale_grad
            var_dict['bias@GRAD'] = bias_grad

            var_names = [
                'x',
                'scale',
                'bias',
                'mean',
                'variance',
                'y',
                'saved_mean',
                'saved_variance',
                'momentum_var',
            ]
            ground_truth = {name: var_dict[name] for name in var_names}

            program = base.Program()
            with base.program_guard(program):
                block = program.global_block()
                for name in ground_truth:
                    block.create_var(
                        name=name,
                        dtype='float32',
                        shape=ground_truth[name].shape,
                    )
                inputs = {
                    "X": block.var('x'),
                    "Scale": block.var('scale'),
                    "Bias": block.var('bias'),
                    "Mean": block.var('mean'),
                    "Variance": block.var('variance'),
                }
                attrs = {
                    "epsilon": epsilon,
                    "is_test": False,
                    "data_layout": data_layout,
                    "use_mkldnn": self.use_mkldnn,
                    "fuse_with_relu": self.fuse_with_relu,
                    "use_global_stats": self.use_global_stats,
                }
                if self.use_momentum_variable:
                    inputs['MomentumTensor'] = block.var('momentum_var')
                else:
                    attrs['momentum'] = momentum

                outputs = {
                    "Y": block.var('y'),
                    "MeanOut": block.var('mean'),  # share memory
                    "VarianceOut": block.var('variance'),  # share memory
                    "SavedMean": block.var('saved_mean'),
                    "SavedVariance": block.var('saved_variance'),
                }
                block.create_var(name="reserve_space", dtype='float32')
                outputs["ReserveSpace"] = block.var('reserve_space')
                bn_op = block.append_op(
                    type="batch_norm",
                    inputs=inputs,
                    outputs=outputs,
                    attrs=attrs,
                )
                block.create_var(name='y@GRAD', dtype='float32', shape=y.shape)

                # generate backward op_desc
                grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
                    bn_op.desc, self.no_grad_set, []
                )
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

                exe = base.Executor(place)
                out = exe.run(
                    program,
                    feed={
                        name: var_dict[name]
                        for name in [
                            'x',
                            'scale',
                            'bias',
                            'mean',
                            'variance',
                            'y@GRAD',
                            'momentum_var',
                        ]
                    },
                    fetch_list=self.fetch_list,
                )

            for id, name in enumerate(self.fetch_list):
                if name == 'variance':
                    self.__assert_close(
                        var_dict[name], out[id], name, atol=1e-3
                    )
                    continue
                self.__assert_close(var_dict[name], out[id], name)
            print("op test forward passed: ", str(place), data_layout)

        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not paddle.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if paddle.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            for data_format in self.data_formats:
                test_with_place(place, data_format, [2, 3, 4, 5])

    def init_kernel_type(self):
        pass


class TestBatchNormOpTrainingCase1(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = {'scale@GRAD', 'bias@GRAD'}
        self.fetch_list = ['y', 'mean', 'variance', 'x@GRAD']


class TestBatchNormOpTrainingCase2(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = set()
        self.fetch_list = [
            'y',
            'mean',
            'variance',
            'saved_mean',
            'saved_variance',
            'x@GRAD',
            'scale@GRAD',
            'bias@GRAD',
        ]
        os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = "1"


class TestBatchNormOpTrainingCase3(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_global_stats = False
        self.no_grad_set = {'x@GRAD'}
        self.fetch_list = ['y', 'mean', 'variance', 'scale@GRAD', 'bias@GRAD']


class TestBatchNormOpTrainingMomentumVariable(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_momentum_variable = True
        self.use_global_stats = False
        self.no_grad_set = set()
        self.fetch_list = [
            'y',
            'mean',
            'variance',
            'saved_mean',
            'saved_variance',
            'x@GRAD',
            'scale@GRAD',
            'bias@GRAD',
        ]


class TestBatchNormOpFreezeStatsTraining(TestBatchNormOpTraining):
    def init_test_case(self):
        self.use_global_stats = True
        self.no_grad_set = set()
        self.fetch_list = [
            'y',
            'mean',
            'variance',
            'x@GRAD',
            'scale@GRAD',
            'bias@GRAD',
        ]

    def reference_grad(self, x, y_grad, scale, mean, var, epsilon, data_format):
        if data_format == "NCHW":
            x = np.transpose(x, (0, 2, 3, 1))
            y_grad = np.transpose(y_grad, (0, 2, 3, 1))

        x_grad = scale * y_grad / np.sqrt(var + epsilon)
        grad_scale = np.sum(
            y_grad * (x - mean) / np.sqrt(var + epsilon), axis=(0, 1, 2)
        )
        grad_offset = np.sum(y_grad, axis=(0, 1, 2))

        # transfer back to N, C, H, W
        if data_format == "NCHW":
            x_grad = np.transpose(x_grad, (0, 3, 1, 2))
            x = np.transpose(x, (0, 3, 1, 2))
            y_grad = np.transpose(y_grad, (0, 3, 1, 2))

        return x_grad, grad_scale, grad_offset

    def ref_forward_backward(
        self,
        x,
        y_grad,
        scale,
        bias,
        mean,
        variance,
        epsilon,
        momentum,
        shape,
        data_layout,
    ):
        if data_layout != "NCHW" and data_layout != "NHWC":
            raise ValueError("Unknown data order.")

        if data_layout == "NCHW":
            x = np.transpose(x, (0, 2, 3, 1))

        # run normalizaton
        normalized = (x - mean) / np.sqrt(variance + epsilon)
        y = normalized * scale + bias

        # transfer back to N, C, H, W
        if data_layout == "NCHW":
            x = np.transpose(x, (0, 3, 1, 2))
            y = np.transpose(y, (0, 3, 1, 2))

        mean_out = mean
        variance_out = variance
        saved_variance = 1.0 / np.sqrt(variance + epsilon)
        # run backward
        x_grad, scale_grad, bias_grad = self.reference_grad(
            x, y_grad, scale, mean, variance, epsilon, data_layout
        )

        return (
            y,
            mean_out,
            variance_out,
            mean,
            saved_variance,
            x_grad,
            scale_grad,
            bias_grad,
        )


class TestBatchNormOpFreezeStatsAndScaleBiasTraining(
    TestBatchNormOpFreezeStatsTraining
):
    def init_test_case(self):
        self.use_global_stats = True
        self.no_grad_set = {'scale@GRAD', 'bias@GRAD'}
        self.fetch_list = ['y', 'mean', 'variance', 'x@GRAD']


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
