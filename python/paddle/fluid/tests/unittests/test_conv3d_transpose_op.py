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


def conv3dtranspose_forward_naive(input_, filter_, attrs):
    in_n, in_c, in_d, in_h, in_w = input_.shape
    f_c, f_out_c, f_d, f_h, f_w = filter_.shape
    groups = attrs['groups']
    assert in_c == f_c
    out_c = f_out_c * groups
    sub_in_c = in_c / groups

    stride, pad, dilations = attrs['strides'], attrs['paddings'], attrs[
        'dilations']

    d_bolck_d = dilations[0] * (f_d - 1) + 1
    d_bolck_h = dilations[1] * (f_h - 1) + 1
    d_bolck_w = dilations[2] * (f_w - 1) + 1
    out_d = (in_d - 1) * stride[0] + d_bolck_d
    out_h = (in_h - 1) * stride[1] + d_bolck_h
    out_w = (in_w - 1) * stride[2] + d_bolck_w
    out = np.zeros((in_n, out_c, out_d, out_h, out_w))

    for n in range(in_n):
        for d in range(in_d):
            for i in range(in_h):
                for j in range(in_w):
                    for g in range(groups):
                        input_masked = input_[n, g * sub_in_c:(g + 1
                                                               ) * sub_in_c, d,
                                              i, j]  # (c)
                        input_masked = np.reshape(input_masked,
                                                  (sub_in_c, 1, 1, 1))
                        input_masked = np.tile(input_masked, (1, f_d, f_h, f_w))

                        for k in range(f_out_c):
                            tmp_out = np.sum(input_masked * filter_[
                                g * sub_in_c:(g + 1) * sub_in_c, k, :, :, :],
                                             axis=0)
                            d1, d2 = d * stride[0], d * stride[0] + d_bolck_d
                            i1, i2 = i * stride[1], i * stride[1] + d_bolck_h
                            j1, j2 = j * stride[2], j * stride[2] + d_bolck_w
                            out[n, g * f_out_c + k, d1:d2:dilations[0], i1:i2:
                                dilations[1], j1:j2:dilations[2]] += tmp_out

    out = out[:, :, pad[0]:out_d - pad[0], pad[1]:out_h - pad[1], pad[2]:out_w -
              pad[2]]
    return out


class TestConv3dTransposeOp(OpTest):
    def setUp(self):
        # init as conv transpose
        self.use_cudnn = False
        self.init_op_type()
        self.init_test_case()

        input_ = np.random.random(self.input_size).astype("float32")
        filter_ = np.random.random(self.filter_size).astype("float32")

        self.inputs = {'Input': input_, 'Filter': filter_}
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'dilations': self.dilations,
            'groups': self.groups,
            'use_cudnn': self.use_cudnn,
            'data_format': 'AnyLayout'  # TODO(dzhwinter) : should be fix latter
        }

        output = conv3dtranspose_forward_naive(input_, filter_,
                                               self.attrs).astype("float32")

        self.outputs = {'Output': output}

    def test_check_output(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
        else:
            self.check_output()

    def test_check_grad(self):
        if self.use_cudnn:
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
        if self.use_cudnn:
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
        if self.use_cudnn:
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
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.op_type = "conv3d_transpose"


class TestWithPad(TestConv3dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithGroups(TestConv3dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 2
        self.input_size = [2, 4, 5, 5, 5]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 3, 3, 3, 3]


class TestWithStride(TestConv3dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithDilation(TestConv3dTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [2, 2, 2]
        self.groups = 1
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


# ------------ test_cudnn ------------
class TestCUDNN(TestConv3dTransposeOp):
    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"


class TestCUDNNWithPad(TestWithPad):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"


class TestCUDNNWithStride(TestWithStride):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [2, 3, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"


class TestCUDNNWithGroups(TestWithGroups):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 2
        self.input_size = [2, 4, 5, 5, 5]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 3, 3, 3, 3]

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"


# Please Don't remove the following code.
# Currently, CI use cudnn V5.0 which not support dilation conv.
# class TestCUDNNWithDilation(TestWithDilation):
#     def init_test_case(self):
#         self.pad = [1, 1, 1]
#         self.stride = [2, 2, 2]
#         self.dilations = [2, 2, 2]
#         self.input_size = [2, 3, 5, 5, 5]  # NCDHW
#         f_c = self.input_size[1]
#         self.filter_size = [f_c, 6, 3, 3, 3]
#
#     def init_op_type(self):
#         self.op_type = "conv3d_transpose"

if __name__ == '__main__':
    unittest.main()
