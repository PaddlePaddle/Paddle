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
from test_conv2d_op import conv2d_forward_naive, TestConv2dOp


def conv2d_forward_refer(input, filter, group, conv_param):
    out, in_n, out_h, out_w, out_c = conv2d_forward_naive(input, filter, group,
                                                          conv_param)
    out_tmp = np.zeros((in_n, out_h, out_w, out_c))
    for n in range(in_n):
        for i in range(out_h):
            for j in range(out_w):
                for m in range(out_c):
                    out_tmp[n, i, j, m] = out[n, m, i, j]
    return out_tmp.reshape(in_n, out_c, out_h, out_w)


class TestConv2dInt8Op(TestConv2dOp):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.data_format = "AnyLayout"
        self.weighttype = np.float32
        self.use_mkldnn = True
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_dtype()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        filter = np.random.random(self.filter_size).astype(self.weighttype)
        if self.srctype == np.uint8:
            input = np.random.randint(0, 10,
                                      self.input_size).astype(self.srctype)
        else:
            input = np.random.randint(-5, 5,
                                      self.input_size).astype(self.srctype)
            input_shift = (np.ones(self.input_size) * 128).astype(np.uint8)

        if self.srctype == np.int8:
            filter_int = np.round(filter * self.scale_weights[0] *
                                  0.5).astype(np.int32)
            scale_output_shift = self.scale_out / (self.scale_in *
                                                   self.scale_weights[0] * 0.5)
            output1 = conv2d_forward_refer(
                np.round((input.astype(np.int32) + input_shift) *
                         self.scale_in).astype(np.int32), filter_int,
                self.groups,
                conv2d_param).astype(np.float32) * scale_output_shift
            output2 = conv2d_forward_refer(
                np.round((input_shift) * self.scale_in).astype(np.int32),
                filter_int, self.groups,
                conv2d_param).astype(np.float32) * scale_output_shift
            output = np.round(output1 - output2).astype(self.dsttype)
        else:
            filter_int = np.round(filter *
                                  self.scale_weights[0]).astype(np.int32)
            scale_output_shift = self.scale_out / (self.scale_in *
                                                   self.scale_weights[0])
            output1 = conv2d_forward_refer(
                input.astype(np.int32), filter_int, self.groups,
                conv2d_param).astype(np.float32)
            output = np.round(output1 * scale_output_shift).astype(self.dsttype)

        self.inputs = {
            'Input':
            OpTest.np_dtype_to_fluid_dtype(input.astype(self.srctype)),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
            'exhaustive_search': self.exhaustive_search,
            'Scale_in': self.scale_in,
            'Scale_out': self.scale_out,
            'Scale_weights': self.scale_weights,
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), atol=0)

    def test_check_grad(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def test_check_grad_no_input(self):
        pass

    def init_test_case(self):
        TestConv2dOp.init_test_case(self)
        f_c = self.input_size[1] // self.groups
        self.filter_size = [1, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]

    def init_dtype(self):
        self.srctype = np.uint8
        self.dsttype = np.int8


#--------------------test conv2d u8 in and s8 out--------------------


class TestConv2d(TestConv2dInt8Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]


class TestWithPad(TestConv2d):
    def init_test_case(self):
        TestConv2d.init_test_case(self)
        self.pad = [1, 1]


class TestWithGroup(TestConv2d):
    def init_group(self):
        self.groups = 3


class TestWithStride(TestConv2dInt8Op):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.8
        self.scale_weights = [10.0]


class TestWith1x1(TestConv2dInt8Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [1, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [12.0]


class TestWithInput1x1Filter1x1(TestConv2dInt8Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]

    def init_group(self):
        self.groups = 3


#--------------------test conv2d s8 in and s8 out--------------------


def create_test_int8_class(parent):
    class TestInt8Case(parent):
        def init_dtype(self):
            self.srctype = np.int8
            self.dsttype = np.int8

    cls_name = "{0}_{1}".format(parent.__name__, "s8s8")
    TestInt8Case.__name__ = cls_name
    globals()[cls_name] = TestInt8Case


create_test_int8_class(TestConv2dInt8Op)
create_test_int8_class(TestWithPad)
create_test_int8_class(TestWithStride)
create_test_int8_class(TestWithGroup)
create_test_int8_class(TestWith1x1)
create_test_int8_class(TestWithInput1x1Filter1x1)

if __name__ == '__main__':
    unittest.main()
