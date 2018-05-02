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
        self.data_formats = ["NCHW"]

    def ref_forward_backward(self, x, y_grad, scale, bias, mean, variance,
                             epsilon, momentum, shape, data_layout):
        # run forward
        y, saved_mean, saved_variance = _reference_training(
            x, scale, bias, epsilon, data_layout)
        mean_out = saved_mean * (1. - momentum) + momentum * mean
        variance_out = saved_variance * (1. - momentum) + momentum * variance
        # run backward
        x_grad, scale_grad, bias_grad = _reference_grad(
            x, y_grad, scale, saved_mean, saved_variance, epsilon, data_layout)

        return y, mean_out, variance_out, saved_mean, saved_variance, x_grad, scale_grad, bias_grad


class TestMKLDNNBatchNormOpInference(TestBatchNormOpInference):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def test_check_output(self):
        place = core.CPUPlace()
        data_format = "NCHW"

        self.check_with_place(place, data_format, self.dtype, [2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
