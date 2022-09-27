#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid

import sys

sys.path.append('..')
from op_test import OpTest
from test_deformable_conv_op import dconv_im2col_gemm, deform_conv2d_wrapper

paddle.enable_static()


class TestModulatedDeformableConvOp(OpTest):

    def setUp(self):
        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True
        self.python_api = deform_conv2d_wrapper
        self.op_type = "deformable_conv"
        self.init_type()
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        offset = 10 * np.random.random(self.offset_size).astype(self.dtype)
        mask = 10 * np.random.random(self.mask_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)

        output = dconv_im2col_gemm(input, offset, mask, filter, self.groups,
                                   conv_param)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Offset': OpTest.np_dtype_to_fluid_dtype(offset),
            'Mask': OpTest.np_dtype_to_fluid_dtype(mask),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'deformable_groups': self.deformable_groups,
            'im2col_step': self.im2col_step,
            'dilations': self.dilations,
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output_with_place(self.place, check_eager=False)

    def test_check_grad(self):
        self.check_grad_with_place(self.place,
                                   {'Input', 'Offset', 'Mask', 'Filter'},
                                   'Output',
                                   max_relative_error=0.05,
                                   check_eager=False)

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 8, 4, 4]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 3, 3]
        self.im2col_step = 1
        self.deformable_groups = 1
        offset_c = 2 * self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        mask_c = self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        self.offset_size = [
            self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
        ]
        self.mask_size = [
            self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
        ]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def init_type(self):
        self.dtype = np.float32


class TestWithStride(TestModulatedDeformableConvOp):

    def init_test_case(self):
        self.pad = [3, 3]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.im2col_step = 1
        self.deformable_groups = 1
        offset_c = 2 * self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        mask_c = self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        self.offset_size = [
            self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
        ]
        self.mask_size = [
            self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
        ]


class TestWithDilation(TestModulatedDeformableConvOp):

    def init_test_case(self):
        self.pad = [2, 2]
        self.stride = [1, 1]
        self.input_size = [4, 3, 4, 4]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.im2col_step = 1
        self.deformable_groups = 1
        offset_c = 2 * self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        mask_c = self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        self.offset_size = [
            self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
        ]
        self.mask_size = [
            self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
        ]

    def init_dilation(self):
        self.dilations = [2, 2]


class TestWith3x3(TestModulatedDeformableConvOp):

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.im2col_step = 1
        self.deformable_groups = 1
        offset_c = 2 * self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        mask_c = self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        self.offset_size = [
            self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
        ]
        self.mask_size = [
            self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
        ]


if __name__ == '__main__':
    unittest.main()
