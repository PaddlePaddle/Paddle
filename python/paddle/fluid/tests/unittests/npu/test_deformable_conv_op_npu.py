#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
import sys
sys.path.append("..")
from op_test import OpTest

from test_deformable_conv_op import dmc_bilinear, dconv_im2col_gemm

paddle.enable_static()

class TestModulatedDeformableConvOp(OpTest):
    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def setUp(self):
        self.set_npu()
        self.op_type = "deformable_conv"
        self.dtype = np.float32
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
        self.check_output_with_place(self.place, atol=1e-2)

    def test_check_grad(self):
        pass
        # self.check_grad(
        #     {'Input', 'Offset', 'Mask', 'Filter'},
        #     'Output',
        #     max_relative_error=0.05)

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


# class TestWithStride(TestModulatedDeformableConvOp):
#     def init_test_case(self):
#         self.pad = [3, 3]
#         self.stride = [2, 2]
#         self.input_size = [2, 3, 5, 5]  # NCHW
#         assert np.mod(self.input_size[1], self.groups) == 0
#         f_c = self.input_size[1] // self.groups
#         self.filter_size = [6, f_c, 3, 3]
#         self.im2col_step = 1
#         self.deformable_groups = 1
#         offset_c = 2 * self.deformable_groups * self.filter_size[
#             2] * self.filter_size[3]
#         mask_c = self.deformable_groups * self.filter_size[
#             2] * self.filter_size[3]
#         self.offset_size = [
#             self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
#         ]
#         self.mask_size = [
#             self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
#         ]


# class TestWithDilation(TestModulatedDeformableConvOp):
#     def init_test_case(self):
#         self.pad = [2, 2]
#         self.stride = [1, 1]
#         self.input_size = [4, 3, 4, 4]  # NCHW
#         assert np.mod(self.input_size[1], self.groups) == 0
#         f_c = self.input_size[1] // self.groups
#         self.filter_size = [6, f_c, 3, 3]
#         self.im2col_step = 1
#         self.deformable_groups = 1
#         offset_c = 2 * self.deformable_groups * self.filter_size[
#             2] * self.filter_size[3]
#         mask_c = self.deformable_groups * self.filter_size[
#             2] * self.filter_size[3]
#         self.offset_size = [
#             self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
#         ]
#         self.mask_size = [
#             self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
#         ]

#     def init_dilation(self):
#         self.dilations = [2, 2]


# class TestWith3x3(TestModulatedDeformableConvOp):
#     def init_test_case(self):
#         self.pad = [1, 1]
#         self.stride = [1, 1]
#         self.input_size = [2, 3, 5, 5]  # NCHW
#         assert np.mod(self.input_size[1], self.groups) == 0
#         f_c = self.input_size[1] // self.groups
#         self.filter_size = [6, f_c, 3, 3]
#         self.im2col_step = 1
#         self.deformable_groups = 1
#         offset_c = 2 * self.deformable_groups * self.filter_size[
#             2] * self.filter_size[3]
#         mask_c = self.deformable_groups * self.filter_size[
#             2] * self.filter_size[3]
#         self.offset_size = [
#             self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
#         ]
#         self.mask_size = [
#             self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
#         ]


# class TestWithGroup(TestModulatedDeformableConvOp):
#     def init_group(self):
#         self.groups = 2


# class TestDeformConv2DAPI(unittest.TestCase):
#     def test_api(self):
#         def test_deform_conv2d_v2():
#             paddle.enable_static()
#             input = paddle.static.data(
#                 name='input_v2', shape=[None, 3, 32, 32], dtype='float32')
#             offset = paddle.static.data(
#                 name='offset_v2', shape=[None, 4, 32, 32], dtype='float32')
#             mask = paddle.static.data(
#                 name='mask_v2', shape=[None, 2, 32, 32], dtype='float32')
#             out = paddle.static.nn.deform_conv2d(
#                 input, offset, mask, num_filters=4, filter_size=1)

#             assert (out.shape == (-1, 4, 32, 32))

#         test_deform_conv2d_v2()


if __name__ == '__main__':
    unittest.main()
