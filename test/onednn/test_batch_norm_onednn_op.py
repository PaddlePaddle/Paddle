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

import sys
import unittest

import numpy as np
from onednn_op_test import check_if_onednn_batchnorm_primitives_exist_in_bwd
from op_test import _set_use_system_allocator, pir_executor_guard

sys.path.append("../legacy_test")
from test_batch_norm_op import TestBatchNormOpInference

from paddle.base import core

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


class TestONEDNNBatchNormOpTraining(unittest.TestCase):
    def setUp(self):
        self.use_onednn = False
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

    def init_kernel_type(self):
        self.use_onednn = True
        self.data_formats = ["NCHW"]

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

        # run forward
        y, saved_mean, saved_variance = _reference_training(
            x, scale, bias, epsilon, data_layout
        )
        mean_out = saved_mean * (1.0 - momentum) + momentum * mean
        variance_out = saved_variance * (1.0 - momentum) + momentum * variance
        # run backward
        x_grad, scale_grad, bias_grad = _reference_grad(
            x, y_grad, scale, saved_mean, saved_variance, epsilon, data_layout
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

    def test_forward_backward(self):
        super().test_forward_backward()
        with pir_executor_guard():
            super().test_forward_backward()


class TestONEDNNBatchNormOpTraining_NHWC(TestONEDNNBatchNormOpTraining):
    def init_kernel_type(self):
        self.use_onednn = True
        self.data_formats = ["NHWC"]


class TestONEDNNBatchNormOpExistedPrimitives(TestONEDNNBatchNormOpTraining):
    def init_test_case(self):
        TestONEDNNBatchNormOpTraining.init_test_case(self)
        self.fetch_list = ['y', 'x@GRAD']

    def test_forward_backward(self):
        place = core.CPUPlace()
        shape = [2, 3, 4, 5]
        scale_shape = [3]
        data_layout = "NCHW"
        # initialize the ground-truth
        np.random.seed(123)
        x = np.random.random_sample(shape).astype(np.float32)
        scale = np.random.random_sample(scale_shape).astype(np.float32)
        bias = np.random.random_sample(scale_shape).astype(np.float32)
        mean, variance = self.set_mean_variance(scale_shape, x, data_layout)
        y_grad = np.random.random_sample(shape).astype(np.float32)

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
            self.epsilon,
            self.momentum,
            shape,
            data_layout,
        )
        var_dict = locals()
        var_dict['y@GRAD'] = y_grad
        var_dict['x@GRAD'] = x_grad
        var_dict['scale@GRAD'] = scale_grad
        var_dict['bias@GRAD'] = bias_grad
        check_if_onednn_batchnorm_primitives_exist_in_bwd(
            self, var_dict, place, shape, data_layout
        )


class TestONEDNNBatchNormOpInference(TestBatchNormOpInference):
    def init_kernel_type(self):
        self.use_onednn = True

    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NCHW"
        self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])
        self.check_with_place_without_scale_and_bias(
            place, data_format, self.dtype, [2, 3, 4, 5]
        )
        with pir_executor_guard():
            self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])
            self.check_with_place_without_scale_and_bias(
                place, data_format, self.dtype, [2, 3, 4, 5]
            )


class TestONEDNNBatchNormOpInference_NHWC(TestONEDNNBatchNormOpInference):
    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NHWC"
        self.check_with_place(place, data_format, self.dtype, [2, 4, 5, 3])
        self.check_with_place_without_scale_and_bias(
            place, data_format, self.dtype, [2, 4, 5, 3]
        )


class TestONEDNNBatchNormOpWithReluInference(TestBatchNormOpInference):
    def init_kernel_type(self):
        self.use_onednn = True
        self.fuse_with_relu = True

    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NCHW"
        self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])
        with pir_executor_guard():
            self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])


if __name__ == '__main__':
    from paddle import enable_static

    enable_static()
    unittest.main()
