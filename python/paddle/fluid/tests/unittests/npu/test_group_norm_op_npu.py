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

import unittest
import numpy as np

import sys

sys.path.append("..")

from operator import mul
from op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid

paddle.enable_static()


def group_norm_naive(x, scale, bias, epsilon, groups, data_layout):
    if data_layout == "NHWC":
        x = np.transpose(x, (0, 3, 1, 2))  # NHWC => NCHW
    N, C, H, W = x.shape
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    xnorm = (x - mean) / np.sqrt(var + epsilon)
    xnorm = xnorm.reshape((N, C, H, W))
    output = xnorm * scale.reshape((-1, 1, 1)) + bias.reshape((-1, 1, 1))
    if data_layout == "NHWC":
        output = np.transpose(output, (0, 2, 3, 1))  # NCHW => NHWC
        xnorm = np.transpose(xnorm, (0, 2, 3, 1))
    return output, mean.reshape((N, G)), var.reshape((N, G))


class TestGroupNormOpError(unittest.TestCase):

    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):

            def test_x_type():
                input = np.random.random(2, 100, 3, 5).astype('float32')
                groups = 2
                fluid.layers.group_norm(input, groups)

            self.assertRaises(TypeError, test_x_type)

            def test_x_dtype():
                x2 = fluid.layers.data(name='x2',
                                       shape=[2, 100, 3, 5],
                                       dtype='int32')
                groups = 2
                fluid.layers.group_norm(x2, groups)

            self.assertRaises(TypeError, test_x_dtype)


class TestGroupNormOp(OpTest):

    def setUp(self):
        self.set_npu()
        self.op_type = 'group_norm'
        self.place = paddle.NPUPlace(0)

        self.init_dtype()

        self.data_format = "NCHW"
        self.atol = 1e-6
        self.max_relative_error = 0.005
        self.shape = (2, 100, 3, 5)
        self.attrs = {'epsilon': 1e-5, 'groups': 2, 'data_layout': "NCHW"}
        self.compare_between_place = False
        self.init_test_case()

        input = np.random.random(self.shape).astype(self.dtype)
        if self.data_format == "NHWC":
            input = np.transpose(input, (0, 2, 3, 1))
        scale = np.random.random([self.shape[1]]).astype(self.dtype)
        bias = np.random.random([self.shape[1]]).astype(self.dtype)
        output, mean, var = group_norm_naive(input, scale, bias,
                                             self.attrs['epsilon'],
                                             self.attrs['groups'],
                                             self.data_format)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(input),
            'Scale': OpTest.np_dtype_to_fluid_dtype(scale),
            'Bias': OpTest.np_dtype_to_fluid_dtype(bias)
        }
        self.outputs = {'Y': output, 'Mean': mean, 'Variance': var}
        self.attrs['data_layout'] = self.data_format

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=self.atol)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        self.__class__.exist_check_grad = True
        inputs_to_check = ['X', 'Scale', 'Bias']
        output_names = 'Y'
        no_grad_set = set()
        cpu_place = fluid.CPUPlace()
        cpu_grads = self._get_gradient(inputs_to_check, cpu_place, output_names,
                                       no_grad_set)
        npu_grads = self._get_gradient(inputs_to_check, self.place,
                                       output_names, no_grad_set)

        self._assert_is_close(cpu_grads, npu_grads, inputs_to_check,
                              self.max_relative_error,
                              "Gradient Check between places")

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


class TestGroupNormOp1_With_NHWC(TestGroupNormOp):

    def init_test_case(self):
        self.attrs['groups'] = 1
        self.data_format = "NHWC"


class TestGroupNormOp2_With_NHWC(TestGroupNormOp):

    def init_test_case(self):
        self.attrs['groups'] = 4
        self.data_format = "NHWC"


class TestGroupNormOpBigEps1_With_NHWC(TestGroupNormOp):

    def init_test_case(self):
        self.attrs['groups'] = 1
        self.attrs['epsilon'] = 0.5
        self.data_format = "NHWC"


class TestGroupNormOpBigEps2_With_NHWC(TestGroupNormOp):

    def init_test_case(self):
        self.attrs['groups'] = 4
        self.attrs['epsilon'] = 0.5
        self.data_format = "NHWC"


class TestGroupNormOpBigEps3_With_NHWC(TestGroupNormOp):

    def init_test_case(self):
        self.attrs['epsilon'] = 0.5
        self.data_format = "NHWC"


class TestGroupNormOpFP16(TestGroupNormOp):

    def init_dtype(self):
        self.dtype = np.float16


class TestGroupNormOpFP16_With_NHWC(TestGroupNormOp):

    def init_dtype(self):
        self.dtype = np.float16

    def init_test_case(self):
        self.data_format = "NHWC"


class TestGroupNormException(unittest.TestCase):
    # data_layout is not NHWC or NCHW
    def test_exception(self):
        data = fluid.data(name='data', shape=[None, 3, 3, 4], dtype="float64")

        def attr_data_format():
            out = fluid.layers.group_norm(input=data,
                                          groups=2,
                                          data_layout="NDHW")

        self.assertRaises(ValueError, attr_data_format)


if __name__ == '__main__':
    unittest.main()
