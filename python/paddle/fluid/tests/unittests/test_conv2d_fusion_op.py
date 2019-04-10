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

import paddle.fluid.core as core
from op_test import OpTest

from test_conv2d_op import conv2d_forward_naive


class TestConv2dFusionOp(OpTest):
    def setUp(self):
        self.op_type = "conv2d_fusion"
        self.exhaustive_search = False
        self.data_format = "AnyLayout"
        self.dtype = np.float32
        self.activation = 'relu'
        self.add_bias = True
        self.add_residual_data = True
        self.channels = None
        self.outputs = None

        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_bias_residual()
        self.init_activation()
        self.set_search_method()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)

        self.output, _, _, _, _ = conv2d_forward_naive(
            input, filter, self.groups, conv2d_param)
        self.output = self.output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }

        if self.add_residual_data:
            residual_data = np.random.random(self.output.shape).astype(
                self.dtype)
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                residual_data)
            self.output += residual_data

        if self.add_bias:
            bias = np.random.random(self.filter_size[0]).astype(self.dtype)
            self.inputs['Bias'] = OpTest.np_dtype_to_fluid_dtype(bias)
            self.output = self.output + bias.reshape((1, bias.size, 1, 1))

        assert self.activation in ['relu', 'identity']
        if self.activation == 'relu':
            self.output = np.maximum(self.output, 0)

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'data_format': self.data_format,
            'exhaustive_search': self.exhaustive_search,
            'activation': self.activation,
            'split_channels': self.channels
        }
        self.outputs = {'Output': self.output}

        self.set_outputs()

    def testcuda(self):
        return core.is_compiled_with_cuda()

    def test_check_output(self):
        if self.testcuda():
            place = core.CUDAPlace(0)
            # TODO(minqiyang): support fusion op in dygraph mode
            self.check_output_with_place(place, atol=1e-5, check_dygraph=False)
        else:
            pass

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def init_bias_residual(self):
        self.add_bias = True
        self.add_residual_data = True

    def init_activation(self):
        self.activation = 'relu'

    def set_search_method(self):
        self.exhaustive_search = False

    def set_outputs(self):
        pass


class TestWithoutResidual(TestConv2dFusionOp):
    def init_bias_residual(self):
        self.add_residual_data = False


class TestIdentityActivation(TestConv2dFusionOp):
    def init_activation(self):
        self.activation = 'identity'


class TestIdentityActivation(TestConv2dFusionOp):
    def init_activation(self):
        self.activation = 'identity'
        self.add_residual_data = False


class TestWithGroup(TestConv2dFusionOp):
    def init_group(self):
        self.groups = 3


class TestWithDilation(TestConv2dFusionOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 3


class TestCUDNNExhaustiveSearch(TestConv2dFusionOp):
    def set_search_method(self):
        self.exhaustive_search = True


class TestMultipleOutputs(TestConv2dFusionOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [1, 32, 17, 17]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [126, f_c, 3, 3]
        self.channels = [84, 42]

    def set_outputs(self):
        out1 = self.output[:, 0:84, :, :]
        out2 = self.output[:, 84:126, :, :]
        self.outputs['Outputs'] = [('out1', out1), ('out2', out2)]


if __name__ == '__main__':
    unittest.main()
