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

<<<<<<< HEAD
import unittest

import numpy as np
from mkldnn_op_test import check_if_mkldnn_batchnorm_primitives_exist_in_bwd

import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import _set_use_system_allocator
from paddle.fluid.tests.unittests.test_batch_norm_op import (
    TestBatchNormOpInference,
    TestBatchNormOpTraining,
    _reference_grad,
    _reference_training,
)
=======
from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.op_test import OpTest, _set_use_system_allocator
from paddle.fluid.framework import grad_var_name
from paddle.fluid.tests.unittests.test_batch_norm_op import TestBatchNormOpInference, TestBatchNormOpTraining, _reference_training, _reference_grad
from mkldnn_op_test import check_if_mkldnn_batchnorm_primitives_exist_in_bwd
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

_set_use_system_allocator(True)


class TestMKLDNNBatchNormOpTraining(TestBatchNormOpTraining):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_formats = ["NCHW"]

<<<<<<< HEAD
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
=======
    def ref_forward_backward(self, x, y_grad, scale, bias, mean, variance,
                             epsilon, momentum, shape, data_layout):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if data_layout != "NCHW" and data_layout != "NHWC":
            raise ValueError("Unknown data order.")

        # run forward
        y, saved_mean, saved_variance = _reference_training(
<<<<<<< HEAD
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


class TestMKLDNNBatchNormOpTraining_NHWC(TestMKLDNNBatchNormOpTraining):
=======
            x, scale, bias, epsilon, data_layout)
        mean_out = saved_mean * (1. - momentum) + momentum * mean
        variance_out = saved_variance * (1. - momentum) + momentum * variance
        # run backward
        x_grad, scale_grad, bias_grad = _reference_grad(x, y_grad, scale,
                                                        saved_mean,
                                                        saved_variance, epsilon,
                                                        data_layout)

        return y, mean_out, variance_out, saved_mean, saved_variance, x_grad, scale_grad, bias_grad


class TestMKLDNNBatchNormOpTraining_NHWC(TestMKLDNNBatchNormOpTraining):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.data_formats = ["NHWC"]


class TestMKLDNNBatchNormOpExistedPrimitives(TestMKLDNNBatchNormOpTraining):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

<<<<<<< HEAD
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
=======
        y, mean_out, variance_out, saved_mean, saved_variance, x_grad, scale_grad, bias_grad = self.ref_forward_backward(
            x, y_grad, scale, bias, mean, variance, self.epsilon, self.momentum,
            shape, data_layout)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        var_dict = locals()
        var_dict['y@GRAD'] = y_grad
        var_dict['x@GRAD'] = x_grad
        var_dict['scale@GRAD'] = scale_grad
        var_dict['bias@GRAD'] = bias_grad
        check_if_mkldnn_batchnorm_primitives_exist_in_bwd(
<<<<<<< HEAD
            self, var_dict, place, shape, data_layout
        )


class TestMKLDNNBatchNormOpInference(TestBatchNormOpInference):
=======
            self, var_dict, place, shape, data_layout)


class TestMKLDNNBatchNormOpInference(TestBatchNormOpInference):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_kernel_type(self):
        self.use_mkldnn = True

    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NCHW"
        self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])


class TestMKLDNNBatchNormOpInference_NHWC(TestMKLDNNBatchNormOpInference):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NHWC"
        self.check_with_place(place, data_format, self.dtype, [2, 4, 5, 3])


class TestMKLDNNBatchNormOpWithReluInference(TestBatchNormOpInference):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.fuse_with_relu = True

    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NCHW"
        self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])


if __name__ == '__main__':
    from paddle import enable_static
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    enable_static()
    unittest.main()
