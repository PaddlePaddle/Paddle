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


def conv3d_forward_naive(input, filter, group, conv_param):
    in_n, in_c, in_d, in_h, in_w = input.shape
    out_c, f_c, f_d, f_h, f_w = filter.shape
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c / group

    stride, pad, dilation = conv_param['stride'], conv_param['pad'], conv_param[
        'dilations']

    out_d = 1 + (in_d + 2 * pad[0] - (dilation[0] * (f_d - 1) + 1)) / stride[0]
    out_h = 1 + (in_h + 2 * pad[1] - (dilation[1] * (f_h - 1) + 1)) / stride[1]
    out_w = 1 + (in_w + 2 * pad[2] - (dilation[2] * (f_w - 1) + 1)) / stride[2]

    out = np.zeros((in_n, out_c, out_d, out_h, out_w))

    d_bolck_d = (dilation[0] * (f_d - 1) + 1)
    d_bolck_h = (dilation[1] * (f_h - 1) + 1)
    d_bolck_w = (dilation[2] * (f_w - 1) + 1)

    input_pad = np.pad(input, ((0, ), (0, ), (pad[0], ), (pad[1], ),
                               (pad[2], )),
                       mode='constant',
                       constant_values=0)

    filter_dilation = np.zeros((out_c, f_c, d_bolck_d, d_bolck_h, d_bolck_w))
    filter_dilation[:, :, 0:d_bolck_d:dilation[0], 0:d_bolck_h:dilation[1], 0:
                    d_bolck_w:dilation[2]] = filter

    for d in range(out_d):
        for i in range(out_h):
            for j in range(out_w):
                for g in range(group):
                    input_pad_masked = \
                        input_pad[:, g * f_c:(g + 1) * f_c,
                        d * stride[0]:d * stride[0] + d_bolck_d,
                        i * stride[1]:i * stride[1] + d_bolck_h,
                        j * stride[2]:j * stride[2] + d_bolck_w]

                    f_sub = filter_dilation[g * sub_out_c:(g + 1) *
                                            sub_out_c, :, :, :, :]
                    for k in range(sub_out_c):
                        out[:, g * sub_out_c + k, d, i, j] = \
                            np.sum(input_pad_masked * f_sub[k, :, :, :, :],
                                   axis=(1, 2, 3, 4))

    return out


class TestConv3dOp(OpTest):
    def setUp(self):
        self.use_cudnn = False
        self.init_group()
        self.init_op_type()
        self.init_dilation()
        self.init_test_case()

        conv3d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'data_format': 'AnyLayout'  # TODO(dzhwinter) : should be fix latter
        }
        input = np.random.random(self.input_size).astype("float32")
        filter = np.random.random(self.filter_size).astype("float32")
        output = conv3d_forward_naive(input, filter, self.groups,
                                      conv3d_param).astype("float32")

        self.inputs = {'Input': input, 'Filter': filter}
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations
        }
        self.outputs = {'Output': output}

    def testcudnn(self):
        return core.is_compiled_with_cuda() and self.use_cudnn

    def test_check_output(self):
        if self.testcudnn():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
        else:
            self.check_output()

    def test_check_grad(self):
        if self.testcudnn():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                set(['Input', 'Filter']),
                'Output',
                max_relative_error=0.03)
        else:
            self.check_grad(
                set(['Input', 'Filter']), 'Output', max_relative_error=0.03)

    def test_check_grad_no_filter(self):
        if self.testcudnn():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['Input'],
                'Output',
                max_relative_error=0.03,
                no_grad_set=set(['Filter']))
        else:
            self.check_grad(
                ['Input'],
                'Output',
                max_relative_error=0.03,
                no_grad_set=set(['Filter']))

    def test_check_grad_no_input(self):
        if self.testcudnn():
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['Filter'],
                'Output',
                max_relative_error=0.03,
                no_grad_set=set(['Input']))
        else:
            self.check_grad(
                ['Filter'],
                'Output',
                max_relative_error=0.03,
                no_grad_set=set(['Input']))

    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 3, 3, 3]

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 1

    def init_op_type(self):
        self.op_type = "conv3d"


class TestCase1(TestConv3dOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 3, 3, 3]


class TestWithGroup1(TestConv3dOp):
    def init_group(self):
        self.groups = 3


class TestWithGroup2(TestCase1):
    def init_group(self):
        self.groups = 3


class TestWith1x1(TestConv3dOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 4, 4, 4]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 1, 1, 1]

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithInput1x1Filter1x1(TestConv3dOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 1, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 1, 1, 1]

    def init_dilation(self):
        self.dilations = [1, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithDilation(TestConv3dOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.input_size = [2, 3, 6, 6, 6]  # NCDHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] / self.groups
        self.filter_size = [6, f_c, 2, 2, 2]

    def init_dilation(self):
        self.dilations = [2, 2, 2]

    def init_group(self):
        self.groups = 3


class TestCUDNN(TestConv3dOp):
    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d"


class TestWithGroup1CUDNN(TestWithGroup1):
    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d"


class TestWithGroup2CUDNN(TestWithGroup2):
    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d"


class TestWith1x1CUDNN(TestWith1x1):
    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d"


class TestWithInput1x1Filter1x1CUDNN(TestWithInput1x1Filter1x1):
    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d"


# FIXME(typhoonzero): find a way to determine if
# using cudnn > 6 in python
# class TestWithDilationCUDNN(TestWithDilation):
#     def init_op_type(self):
#         self.op_type = "conv3d"

if __name__ == '__main__':
    exit(0)
    unittest.main()
