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
from op_test import OpTest

from test_conv2d_op import conv2d_forward_naive


def create_test_padding_SAME_class(parent):

    class TestPaddingSAMECase(parent):

        def init_paddings(self):
            self.pad = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSAMECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSAMECase


def create_test_padding_VALID_class(parent):

    class TestPaddingVALIDCase(parent):

        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


class TestConv2DFusionOp(OpTest):

    def setUp(self):
        self.op_type = "conv2d_fusion"
        self.exhaustive_search = False
        self.data_format = "NCHW"
        self.dtype = np.float32
        self.activation = 'relu'
        self.add_residual_data = True
        self.split_channels = None
        self.outputs = None
        self.padding_algorithm = "EXIPLICIT"

        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_residual()
        self.init_activation()
        self.init_paddings()
        self.set_search_method()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)
        bias = np.random.random(self.filter_size[0]).astype(self.dtype)

        self.output, _, _, _, _ = conv2d_forward_naive(input, filter,
                                                       self.groups,
                                                       conv2d_param,
                                                       self.padding_algorithm,
                                                       self.data_format)

        self.output = self.output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter),
            'Bias': OpTest.np_dtype_to_fluid_dtype(bias)
        }

        if self.add_residual_data:
            residual_data = np.random.random(self.output.shape).astype(
                self.dtype)
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                residual_data)
            self.output += residual_data

        # Add bias
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
            'padding_algorithm': self.padding_algorithm
        }
        if self.split_channels is not None:
            self.attrs['split_channels'] = self.split_channels

        self.outputs = {'Output': self.output}

        self.set_outputs()

    def has_cuda(self):
        return core.is_compiled_with_cuda()

    def test_check_output(self):
        if self.has_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)

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

    def init_residual(self):
        self.add_residual_data = True

    def init_activation(self):
        self.activation = 'relu'

    def set_search_method(self):
        self.exhaustive_search = False

    def set_outputs(self):
        pass

    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "EXPLICIT"


class TestWithoutResidual(TestConv2DFusionOp):

    def init_residual(self):
        self.add_residual_data = False


class TestIdentityActivation(TestConv2DFusionOp):

    def init_activation(self):
        self.activation = 'identity'


class TestIdentityActivation1(TestConv2DFusionOp):

    def init_activation(self):
        self.activation = 'identity'
        self.add_residual_data = False


class TestWithGroup(TestConv2DFusionOp):

    def init_group(self):
        self.groups = 3


class TestWithDilation(TestConv2DFusionOp):

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


class TestCUDNNExhaustiveSearch(TestConv2DFusionOp):

    def set_search_method(self):
        self.exhaustive_search = True


class TestMultipleOutputs(TestConv2DFusionOp):

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [1, 32, 17, 17]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [126, f_c, 3, 3]
        self.split_channels = [84, 42]

    def set_outputs(self):
        out1 = self.output[:, 0:84, :, :]
        out2 = self.output[:, 84:126, :, :]
        self.outputs['Outputs'] = [('out1', out1), ('out2', out2)]


class TestAsyPadding(TestConv2DFusionOp):

    def init_paddings(self):
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithPad_AsyPadding(TestConv2DFusionOp):

    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 3, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithStride_AsyPadding(TestConv2DFusionOp):

    def init_test_case(self):
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 3, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWith1x1_AsyPadding(TestConv2DFusionOp):

    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [2, 2, 4, 0]
        self.padding_algorithm = "EXPLICIT"


class TestWithGroup_AsyPadding(TestConv2DFusionOp):

    def init_group(self):
        self.groups = 3


class TestWithDepthWise3x3_AsyPadding(TestConv2DFusionOp):

    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [3, 4, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [8, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 4

    def init_paddings(self):
        self.pad = [1, 3, 2, 1]
        self.padding_algorithm = "EXPLICIT"


class TestWithDepthWise5x5_AsyPadding(TestConv2DFusionOp):

    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 4, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [8, f_c, 5, 5]

    def init_group(self):
        self.groups = 4

    def init_paddings(self):
        self.pad = [0, 1, 1, 0]
        self.padding_algorithm = "EXPLICIT"


class TestWithDepthWise7x7_AsyPadding(TestConv2DFusionOp):

    def init_test_case(self):
        self.stride = [2, 2]
        self.input_size = [2, 8, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [16, f_c, 7, 7]

    def init_group(self):
        self.groups = 8

    def init_paddings(self):
        self.pad = [1, 3, 4, 1]
        self.padding_algorithm = "EXPLICIT"


class TestWithDilation_AsyPadding(TestConv2DFusionOp):

    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [0, 1, 3, 0]
        self.padding_algorithm = "EXPLICIT"


class TestWithInput1x1Filter1x1_AsyPadding(TestConv2DFusionOp):

    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [0, 3, 4, 0]
        self.padding_algorithm = "EXPLICIT"


create_test_padding_SAME_class(TestAsyPadding)
create_test_padding_SAME_class(TestWithPad_AsyPadding)
create_test_padding_SAME_class(TestWithStride_AsyPadding)
create_test_padding_SAME_class(TestWithGroup_AsyPadding)
create_test_padding_SAME_class(TestWithInput1x1Filter1x1_AsyPadding)

create_test_padding_VALID_class(TestAsyPadding)
create_test_padding_VALID_class(TestWithPad_AsyPadding)
create_test_padding_VALID_class(TestWithStride_AsyPadding)
create_test_padding_VALID_class(TestWithGroup_AsyPadding)
create_test_padding_VALID_class(TestWithInput1x1Filter1x1_AsyPadding)

if __name__ == '__main__':
    unittest.main()
