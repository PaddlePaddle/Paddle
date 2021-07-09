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
sys.path.append("..")
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest

from test_conv2d_op import conv2d_forward_naive

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestConv2DOp(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_data_format(self):
        self.data_format = "NCHW"

    def setUp(self):
        self.set_npu()
        self.op_type = "conv2d"
        self.init_data_format()
        self.init_dtype()
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

        output, _, _, _, _ = conv2d_forward_naive(
            input,
            filter,
            self.groups,
            conv2d_param,
            data_format=self.data_format)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'data_format': self.data_format,
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output_with_place(fluid.NPUPlace(0), atol=1e-2)

    def test_check_grad(self):
        self.check_grad_with_place(
            fluid.NPUPlace(0), {'Input', 'Filter'},
            'Output',
            max_relative_error=0.03)

    def test_check_grad_no_filter(self):
        self.check_grad_with_place(
            fluid.NPUPlace(0), ['Input'],
            'Output',
            max_relative_error=0.03,
            no_grad_set=set(['Filter']))

    def test_check_grad_no_input(self):
        self.check_grad_with_place(
            fluid.NPUPlace(0), ['Filter'],
            'Output',
            max_relative_error=0.03,
            no_grad_set=set(['Input']))

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


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestConv2DOpCase2(TestConv2DOp):
    def init_test_case(self):
        self.pad = [2, 2]
        self.stride = [2, 2]
        self.input_size = [4, 5, 16, 16]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestConv2DOpCase3(TestConv2DOp):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 5, 5, 6]
        assert np.mod(self.input_size[3], self.groups) == 0
        f_c = self.input_size[3] // self.groups
        self.filter_size = [6, f_c, 3, 3]


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestConv2DOpCase4(TestConv2DOpCase3):
    def init_group(self):
        self.groups = 3


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestConv2DOpCase5(TestConv2DOpCase4):
    def init_dilation(self):
        self.dilations = [2, 2]


if __name__ == "__main__":
    unittest.main()
