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

import unittest

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def conv2dtranspose_forward_naive(input_, filter_, attrs):
    padding_algorithm = attrs['padding_algorithm']
    if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
        raise ValueError(
            f"Unknown Attr(padding_algorithm): '{padding_algorithm}'. "
            "It can only be 'SAME' or 'VALID'."
        )

    if attrs['data_format'] == 'NHWC':
        input_ = np.transpose(input_, [0, 3, 1, 2])
    in_n, in_c, in_h, in_w = input_.shape
    f_c, f_out_c, f_h, f_w = filter_.shape
    groups = attrs['groups']
    assert in_c == f_c
    out_c = f_out_c * groups
    sub_in_c = in_c // groups

    stride, pad, dilations = (
        attrs['strides'],
        attrs['paddings'],
        attrs['dilations'],
    )

    # update pad and dilation
    def _get_padding_with_SAME(input_shape, kernel_size, kernel_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(
            input_shape, kernel_size, kernel_stride
        ):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max(
                ((out_size - 1) * stride_size + filter_size - input_size, 0)
            )
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    ksize = filter_.shape[2:4]
    if padding_algorithm == "VALID":
        pad = [0, 0, 0, 0]
    elif padding_algorithm == "SAME":
        dilations = [1, 1]
        input_data_shape = input_.shape[2:4]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)

    pad_h_0, pad_h_1 = pad[0], pad[0]
    pad_w_0, pad_w_1 = pad[1], pad[1]
    if len(pad) == 4:
        pad_h_0, pad_h_1 = pad[0], pad[1]
        pad_w_0, pad_w_1 = pad[2], pad[3]

    d_block_h = dilations[0] * (f_h - 1) + 1
    d_block_w = dilations[1] * (f_w - 1) + 1
    out_h = (in_h - 1) * stride[0] + d_block_h
    out_w = (in_w - 1) * stride[1] + d_block_w
    if 'output_size' in attrs:
        output_size = attrs['output_size']
        out_h = output_size[0] + pad_h_0 + pad_h_1
        out_w = output_size[1] + pad_w_0 + pad_w_1
    out_pad_h = 0
    out_pad_w = 0
    if 'output_padding' in attrs:
        out_pad_h = attrs['output_padding'][0]
        out_pad_w = attrs['output_padding'][1]
    out = np.zeros(
        (in_n, out_c, out_h + out_pad_h, out_w + out_pad_w), dtype=input_.dtype
    )

    for n in range(in_n):
        for i in range(in_h):
            for j in range(in_w):
                for g in range(groups):
                    input_masked = input_[
                        n, g * sub_in_c : (g + 1) * sub_in_c, i, j
                    ]  # (c)
                    input_masked = np.reshape(input_masked, (sub_in_c, 1, 1))
                    input_masked = np.tile(input_masked, (1, f_h, f_w))

                    for k in range(f_out_c):
                        tmp_out = np.sum(
                            input_masked
                            * filter_[
                                g * sub_in_c : (g + 1) * sub_in_c, k, :, :
                            ],
                            axis=0,
                        )
                        i1, i2 = i * stride[0], i * stride[0] + d_block_h
                        j1, j2 = j * stride[1], j * stride[1] + d_block_w
                        out[
                            n,
                            g * f_out_c + k,
                            i1 : i2 : dilations[0],
                            j1 : j2 : dilations[1],
                        ] += tmp_out

    out = out[
        :,
        :,
        pad_h_0 : out_h - pad_h_1 + out_pad_h,
        pad_w_0 : out_w - pad_w_1 + out_pad_w,
    ]
    if attrs['data_format'] == 'NHWC':
        out = np.transpose(out, [0, 2, 3, 1])
    return out


class XPUTestConv2DTransposeOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'conv2d_transpose'
        self.use_dynamic_create_class = False

    class TestConv2DTransposeOp(XPUOpTest):
        def setUp(self):
            # init as conv transpose
            self.need_check_grad = True
            self.is_test = False
            self.use_cudnn = False
            self.use_mkldnn = False
            self.output_size = None
            self.output_padding = []
            self.data_format = "NCHW"
            self.pad = [0, 0]
            self.padding_algorithm = "EXPLICIT"
            self.init_op_type()
            self.init_test_case()
            self.__class__.op_type = "conv2d_transpose"

            input_ = np.random.random(self.input_size).astype(self.dtype)
            filter_ = np.random.random(self.filter_size).astype(self.dtype)

            self.inputs = {'Input': input_, 'Filter': filter_}
            self.attrs = {
                'strides': self.stride,
                'paddings': self.pad,
                'padding_algorithm': self.padding_algorithm,
                'groups': self.groups,
                'dilations': self.dilations,
                'use_cudnn': self.use_cudnn,
                'is_test': self.is_test,
                'use_mkldnn': self.use_mkldnn,
                'data_format': self.data_format,
            }
            if self.output_size is not None:
                self.attrs['output_size'] = self.output_size

            if len(self.output_padding) > 0:
                self.attrs['output_padding'] = self.output_padding

            output = conv2dtranspose_forward_naive(
                input_, filter_, self.attrs
            ).astype(self.dtype)

            self.outputs = {'Output': output}

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad_no_input(self):
            if self.need_check_grad:
                self.check_grad_with_place(
                    self.place, ['Filter'], 'Output', no_grad_set={'Input'}
                )

        def test_check_grad_no_filter(self):
            if self.need_check_grad:
                self.check_grad_with_place(
                    self.place, ['Input'], 'Output', no_grad_set={'Filter'}
                )

        def test_check_grad(self):
            if self.need_check_grad:
                self.check_grad_with_place(
                    self.place, {'Input', 'Filter'}, 'Output'
                )

        def init_test_case(self):
            self.pad = [0, 0]
            self.stride = [1, 1]
            self.dilations = [1, 1]
            self.groups = 1
            self.input_size = [2, 3, 5, 5]  # NCHW
            f_c = self.input_size[1]
            self.filter_size = [f_c, 6, 3, 3]

        def init_op_type(self):
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.op_type = "conv2d_transpose"

    class TestWithSymmetricPad(TestConv2DTransposeOp):
        def init_test_case(self):
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.dilations = [1, 1]
            self.groups = 1
            self.input_size = [2, 3, 5, 5]  # NCHW
            f_c = self.input_size[1]
            self.filter_size = [f_c, 6, 3, 3]

    class TestWithAsymmetricPad(TestConv2DTransposeOp):
        def init_test_case(self):
            self.pad = [1, 0, 1, 2]
            self.stride = [1, 1]
            self.dilations = [1, 1]
            self.groups = 1
            self.input_size = [2, 3, 5, 5]  # NCHW
            f_c = self.input_size[1]
            self.filter_size = [f_c, 6, 3, 3]

    class TestWithSAMEPad(TestConv2DTransposeOp):
        def init_test_case(self):
            self.stride = [2, 1]
            self.dilations = [1, 2]
            self.groups = 1
            self.input_size = [2, 3, 6, 5]  # NCHW
            f_c = self.input_size[1]
            self.filter_size = [f_c, 6, 4, 3]
            self.padding_algorithm = 'SAME'

    class TestWithVALIDPad(TestConv2DTransposeOp):
        def init_test_case(self):
            self.stride = [1, 1]
            self.dilations = [1, 1]
            self.groups = 1
            self.input_size = [2, 3, 5, 5]  # NCHW
            f_c = self.input_size[1]
            self.filter_size = [f_c, 6, 3, 3]
            self.padding_algorithm = 'VALID'

    class TestWithGroups(TestConv2DTransposeOp):
        def init_test_case(self):
            self.pad = [1, 1]
            self.stride = [1, 1]
            self.dilations = [1, 1]
            self.groups = 2
            self.input_size = [2, 4, 5, 5]  # NCHW
            f_c = self.input_size[1]
            self.filter_size = [f_c, 3, 3, 3]

    class TestWithStride(TestConv2DTransposeOp):
        def init_test_case(self):
            self.pad = [1, 1]
            self.stride = [2, 2]
            self.dilations = [1, 1]
            self.groups = 1
            self.input_size = [2, 3, 5, 5]  # NCHW
            f_c = self.input_size[1]
            self.filter_size = [f_c, 6, 3, 3]

    class TestWithEvenUpsample(TestConv2DTransposeOp):
        def init_test_case(self):
            self.pad = [2, 2]
            self.stride = [2, 2]
            self.groups = 1
            self.dilations = [1, 1]
            self.output_size = [14, 14]
            self.input_size = [2, 3, 7, 7]  # NCHW
            f_c = self.input_size[1]
            self.filter_size = [f_c, 6, 5, 5]


support_types = get_xpu_op_support_types('conv2d_transpose')
for stype in support_types:
    create_test_class(globals(), XPUTestConv2DTransposeOp, stype)

if __name__ == '__main__':
    unittest.main()
