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

from __future__ import print_function
import unittest
import numpy as np

from operator import mul
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest


def group_norm_naive(x, scale, bias, epsilon, groups):
    N, C, H, W = x.shape
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    output = (x - mean) / np.sqrt(var + epsilon)
    output = output.reshape((N, C, H, W)) * scale.reshape(
        (-1, 1, 1)) + bias.reshape((-1, 1, 1))
    return output, mean.reshape((N, G)), var.reshape((N, G))


class TestGroupNormOp(OpTest):
    def setUp(self):
        self.op_type = "group_norm"
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.shape = (2, 4, 3, 3)
        self.attrs = {'epsilon': 1e-5, 'groups': 2}
        self.init_test_case()

        input = np.random.random(self.shape).astype(self.dtype)
        scale = np.random.random([self.shape[1]]).astype(self.dtype)
        bias = np.random.random([self.shape[1]]).astype(self.dtype)
        output, mean, var = group_norm_naive(
            input, scale, bias, self.attrs['epsilon'], self.attrs['groups'])

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(input),
            'Scale': OpTest.np_dtype_to_fluid_dtype(scale),
            'Bias': OpTest.np_dtype_to_fluid_dtype(bias)
        }
        self.outputs = {'Y': output, 'Mean': mean, 'Variance': var}

    def test_check_output(self):
        place = core.CPUPlace()
        self.check_output_with_place(place, atol=1e-5)
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)

    def test_check_grad(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place, set(['X', 'Scale', 'Bias']), 'Y', max_relative_error=0.02)
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                set(['X', 'Scale', 'Bias']),
                'Y',
                max_relative_error=0.005)

    def init_test_case(self):
        pass


class TestGroupNormOp1(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 1


class TestGroupNormOp2(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 4


class TestGroupNormOpBigEps1(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 1
        self.attrs['epsilon'] = 0.5


class TestGroupNormOpBigEps2(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['groups'] = 4
        self.attrs['epsilon'] = 0.5


class TestGroupNormOpBigEps3(TestGroupNormOp):
    def init_test_case(self):
        self.attrs['epsilon'] = 0.5


if __name__ == '__main__':
    unittest.main()
