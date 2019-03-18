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
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_conv2d_op import conv2d_forward_naive, TestConv2dOp
from mkldnn_op_test import format_reorder


def conv2d_forward_refer(input, filter, group, conv_param):
    out, in_n, out_h, out_w, out_c = conv2d_forward_naive(input, filter, group,
                                                          conv_param)
    size = [in_n, out_c, out_h, out_w]
    return format_reorder(out, size)


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
        self.init_fuse_relu()
        self.init_fuse_residual()
        self.init_data_type()

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
            if self.fuse_residual:
                input_residual = np.random.randint(
                    -5, 5, self.input_residual_size).astype(self.srctype)
                output_tmp = np.round(output1 - output2 + format_reorder(
                    input_residual, self.input_residual_size).astype(
                        self.srctype) * (self.scale_out / self.scale_in_eltwise
                                         ))
                if self.fuse_relu:
                    output = np.maximum(output_tmp, 0).astype(self.dsttype)
                else:
                    output = output_tmp.astype(self.dsttype)
            else:
                if self.fuse_relu:
                    output = np.maximum(np.round(output1 - output2),
                                        0).astype(self.dsttype)
                else:
                    output = np.round(output1 - output2).astype(self.dsttype)

        else:
            filter_int = np.round(filter *
                                  self.scale_weights[0]).astype(np.int32)
            scale_output_shift = self.scale_out / (self.scale_in *
                                                   self.scale_weights[0])
            output1 = conv2d_forward_refer(
                input.astype(np.int32), filter_int, self.groups,
                conv2d_param).astype(np.float32)
            if self.fuse_residual:
                input_residual = np.random.randint(
                    0, 10, self.input_residual_size).astype(self.srctype)
                output_tmp = np.round(output1 * (self.scale_out / (
                    self.scale_in * self.scale_weights[0])) + format_reorder(
                        input_residual, self.input_residual_size).astype(
                            np.int32) * (self.scale_out / self.scale_in_eltwise
                                         ))
                output_tmp2 = np.round(output1 * (
                    self.scale_out / (self.scale_in * self.scale_weights[0])))
                if self.fuse_relu:
                    output = np.maximum(output_tmp, 0).astype(self.dsttype)
                else:
                    output = output_tmp.astype(self.dsttype)
            else:
                if self.fuse_relu:
                    output = np.maximum(output_tmp2, 0).astype(self.dsttype)
                else:
                    output = output_tmp2.astype(self.dsttype)

        self.inputs = {
            'Input':
            OpTest.np_dtype_to_fluid_dtype(input.astype(self.srctype)),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        if self.fuse_residual:
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                input_residual)

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
            'Scale_in_eltwise': self.scale_in_eltwise,
            'fuse_relu': self.fuse_relu,
            'fuse_residual_connection': self.fuse_residual
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
        self.input_size = [1, 1, 5, 5]  # NCHW
        f_c = self.input_size[1] // self.groups
        self.input_residual_size = [1, 2, 3, 3]
        self.filter_size = [2, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]
        self.scale_in_eltwise = 0.6

    def init_data_type(self):
        self.srctype = np.uint8
        self.dsttype = np.int8

    def init_fuse_relu(self):
        self.fuse_relu = True

    def init_fuse_residual(self):
        self.fuse_residual = True


#--------------------test conv2d u8 in and u8 out with residual fuse--------------------


class TestConv2d(TestConv2dInt8Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]
        self.scale_in_eltwise = 0.6


class TestWithPad(TestConv2d):
    def init_test_case(self):
        TestConv2d.init_test_case(self)
        self.pad = [1, 1]
        self.input_residual_size = [2, 6, 5, 5]


class TestWithGroup(TestConv2d):
    def init_group(self):
        self.groups = 3


class TestWithStride(TestConv2dInt8Op):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.scale_in = 1.0
        self.scale_out = 0.8
        self.scale_weights = [10.0]
        self.scale_in_eltwise = 0.5


class TestWith1x1(TestConv2dInt8Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [1, 3, 5, 5]
        self.input_residual_size = [1, 6, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [12.0]
        self.scale_in_eltwise = 0.5


class TestWithInput1x1Filter1x1(TestConv2dInt8Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]
        self.input_residual_size = [2, 6, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]
        self.scale_in = 1.0
        self.scale_out = 0.5
        self.scale_weights = [10.0]
        self.scale_in_eltwise = 0.8

    def init_group(self):
        self.groups = 3


def init_data_type_with_fusion(self, input_dt, fuse_relu, fuse_residual):
    self.srctype = input_dt
    self.dsttype = np.uint8 if fuse_relu else np.int8

    def init_fuse_relu(self):
        self.fuse_relu = fuse_relu

    def init_fuse_residual(self):
        self.fuse_residual = fuse_residual


def create_test_int8_class(parent):

    #--------------------test conv2d s8 in and u8 out--------------------

    class TestS8U8Case(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.int8, True, False)

    #--------------------test conv2d s8 in and s8 out--------------------

    class TestS8S8Case(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.int8, False, False)

    #--------------------test conv2d u8 in and s8 out--------------------

    class TestU8S8Case(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.uint8, False, False)

    #--------------------test conv2d u8 in and u8 out without residual fuse--------------------

    class TestU8U8Case(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.uint8, True, False)

    #--------------------test conv2d s8 in and u8 out with residual fuse--------------------

    class TestS8U8ResCase(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.int8, True, True)

    #--------------------test conv2d s8 in and s8 out with residual fuse--------------------

    class TestS8S8ResCase(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.int8, False, True)

    #--------------------test conv2d u8 in and s8 out with residual fuse--------------------

    class TestU8S8ResCase(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.uint8, False, True)

    cls_name_s8u8 = "{0}_relu_{1}_residual_0".format(parent.__name__, "1")
    cls_name_s8s8 = "{0}_relu_{1}_residual_0".format(parent.__name__, "0")
    cls_name_u8s8 = "{0}_relu_{1}_residual_0".format(parent.__name__, "0")
    cls_name_u8u8 = "{0}_relu_{1}_residual_0".format(parent.__name__, "1")
    cls_name_s8u8_re_1 = "{0}_relu_{1}_residual_{2}".format(parent.__name__,
                                                            "1", "1")
    cls_name_s8s8_re_1 = "{0}_relu_{1}_residual_{2}".format(parent.__name__,
                                                            "0", "1")
    cls_name_u8s8_re_1 = "{0}_relu_{1}_residual_{2}".format(parent.__name__,
                                                            "0", "1")
    TestS8U8Case.__name__ = cls_name_s8u8
    TestS8S8Case.__name__ = cls_name_s8s8
    TestU8S8Case.__name__ = cls_name_u8s8
    TestU8U8Case.__name__ = cls_name_u8u8
    TestS8U8ResCase.__name__ = cls_name_s8u8_re_1
    TestS8S8ResCase.__name__ = cls_name_s8s8_re_1
    TestU8S8ResCase.__name__ = cls_name_u8s8_re_1
    globals()[cls_name_s8u8] = TestS8U8Case
    globals()[cls_name_s8s8] = TestS8S8Case
    globals()[cls_name_u8s8] = TestU8S8Case
    globals()[cls_name_u8u8] = TestU8U8Case
    globals()[cls_name_s8u8_re_1] = TestS8U8ResCase
    globals()[cls_name_s8s8_re_1] = TestS8S8ResCase
    globals()[cls_name_u8s8_re_1] = TestU8S8ResCase


create_test_int8_class(TestConv2dInt8Op)
create_test_int8_class(TestWithPad)
create_test_int8_class(TestWithStride)
create_test_int8_class(TestWithGroup)
create_test_int8_class(TestWith1x1)
create_test_int8_class(TestWithInput1x1Filter1x1)

if __name__ == '__main__':
    unittest.main()
