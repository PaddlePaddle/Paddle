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
from mkldnn_op_test import check_if_mkldnn_batchnorm_primitives_exist_in_bwd
from op_test import _set_use_system_allocator, pir_executor_guard

sys.path.append("../deprecated/legacy_test")
from test_batch_norm_op import TestBatchNormOpInference
from test_batch_norm_op_deprecated import (
    TestBatchNormOpTraining,
    _reference_grad,
    _reference_training,
)

from paddle.base import core

_set_use_system_allocator(True)


class TestMKLDNNBatchNormOpTraining(TestBatchNormOpTraining):
    def init_kernel_type(self):
        self.use_mkldnn = True
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


class TestMKLDNNBatchNormOpTraining_NHWC(TestMKLDNNBatchNormOpTraining):
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_formats = ["NHWC"]


class TestMKLDNNBatchNormOpExistedPrimitives(TestMKLDNNBatchNormOpTraining):
    def init_test_case(self):
        TestMKLDNNBatchNormOpTraining.init_test_case(self)
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
        check_if_mkldnn_batchnorm_primitives_exist_in_bwd(
            self, var_dict, place, shape, data_layout
        )


class TestMKLDNNBatchNormOpInference(TestBatchNormOpInference):
    def init_kernel_type(self):
        self.use_mkldnn = True

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


class TestMKLDNNBatchNormOpInference_NHWC(TestMKLDNNBatchNormOpInference):
    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NHWC"
        self.check_with_place(place, data_format, self.dtype, [2, 4, 5, 3])
        self.check_with_place_without_scale_and_bias(
            place, data_format, self.dtype, [2, 4, 5, 3]
        )


class TestMKLDNNBatchNormOpWithReluInference(TestBatchNormOpInference):
    def init_kernel_type(self):
        self.use_mkldnn = True
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
