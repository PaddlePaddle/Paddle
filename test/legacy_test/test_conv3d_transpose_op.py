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

import paddle

paddle.enable_static()
from op_test import OpTest, copy_bits_from_float_to_uint16

from paddle.base import core


def convert_float_to_uint16(float_list, data_format="NCHW"):
    if data_format == "NHWC":
        float_list = np.transpose(float_list, [0, 4, 1, 2, 3])

    new_output = []
    for x in np.nditer(float_list):
        new_output.append(np.uint16(copy_bits_from_float_to_uint16(x)))
    new_output = np.reshape(new_output, float_list.shape).view(np.uint16)

    if data_format == "NHWC":
        new_output = np.transpose(new_output, [0, 2, 3, 4, 1])
    return new_output


def conv3dtranspose_forward_naive(input_, filter_, attrs):
    padding_algorithm = attrs['padding_algorithm']
    if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
        raise ValueError(
            f"Unknown Attr(padding_algorithm): '{padding_algorithm}'. "
            "It can only be 'SAME' or 'VALID'."
        )

    if attrs['data_format'] == 'NHWC':
        input_ = np.transpose(input_, [0, 4, 1, 2, 3])
    in_n, in_c, in_d, in_h, in_w = input_.shape
    f_c, f_out_c, f_d, f_h, f_w = filter_.shape
    groups = attrs['groups']
    assert in_c == f_c
    out_c = f_out_c * groups
    sub_in_c = in_c // groups

    stride, pad, dilations = (
        attrs['strides'],
        attrs['paddings'],
        attrs['dilations'],
    )

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

    ksize = filter_.shape[2:5]
    if padding_algorithm == "VALID":
        pad = [0, 0, 0, 0, 0, 0]
    elif padding_algorithm == "SAME":
        dilations = [1, 1, 1]
        input_data_shape = input_.shape[2:5]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)

    pad_d_0, pad_d_1 = pad[0], pad[0]
    pad_h_0, pad_h_1 = pad[1], pad[1]
    pad_w_0, pad_w_1 = pad[2], pad[2]
    if len(pad) == 6:
        pad_d_0, pad_d_1 = pad[0], pad[1]
        pad_h_0, pad_h_1 = pad[2], pad[3]
        pad_w_0, pad_w_1 = pad[4], pad[5]

    d_block_d = dilations[0] * (f_d - 1) + 1
    d_block_h = dilations[1] * (f_h - 1) + 1
    d_block_w = dilations[2] * (f_w - 1) + 1
    out_d = (in_d - 1) * stride[0] + d_block_d
    out_h = (in_h - 1) * stride[1] + d_block_h
    out_w = (in_w - 1) * stride[2] + d_block_w
    out = np.zeros((in_n, out_c, out_d, out_h, out_w))

    for n in range(in_n):
        for d in range(in_d):
            for i in range(in_h):
                for j in range(in_w):
                    for g in range(groups):
                        input_masked = input_[
                            n, g * sub_in_c : (g + 1) * sub_in_c, d, i, j
                        ]  # (c)
                        input_masked = np.reshape(
                            input_masked, (sub_in_c, 1, 1, 1)
                        )
                        input_masked = np.tile(input_masked, (1, f_d, f_h, f_w))

                        for k in range(f_out_c):
                            tmp_out = np.sum(
                                input_masked
                                * filter_[
                                    g * sub_in_c : (g + 1) * sub_in_c,
                                    k,
                                    :,
                                    :,
                                    :,
                                ],
                                axis=0,
                            )
                            d1, d2 = d * stride[0], d * stride[0] + d_block_d
                            i1, i2 = i * stride[1], i * stride[1] + d_block_h
                            j1, j2 = j * stride[2], j * stride[2] + d_block_w
                            out[
                                n,
                                g * f_out_c + k,
                                d1 : d2 : dilations[0],
                                i1 : i2 : dilations[1],
                                j1 : j2 : dilations[2],
                            ] += tmp_out

    out = out[
        :,
        :,
        pad_d_0 : out_d - pad_d_1,
        pad_h_0 : out_h - pad_h_1,
        pad_w_0 : out_w - pad_w_1,
    ]
    if attrs['data_format'] == 'NHWC':
        out = np.transpose(out, [0, 2, 3, 4, 1])
    return out


def create_test_cudnn_fp16_class(parent, grad_check=True):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestConv3DTransposeCUDNNFP16(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = np.float16

        def test_check_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place, atol=2e-2)

        def test_check_grad_no_filter(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(
                    place, ['Input'], 'Output', no_grad_set={'Filter'}
                )

        def test_check_grad_no_input(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(
                    place, ['Filter'], 'Output', no_grad_set={'Input'}
                )

    cls_name = "{}_{}".format(parent.__name__, "CUDNNFP16OP")
    TestConv3DTransposeCUDNNFP16.__name__ = cls_name
    globals()[cls_name] = TestConv3DTransposeCUDNNFP16


def create_test_cudnn_bf16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support bfloat16",
    )
    class TestConv3DTransposeCUDNNBF16(parent):
        def init_kernel_type(self):
            self.use_cudnn = True
            self.dtype = np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                {'Input', 'Filter'},
                'Output',
            )

        def test_check_grad_no_filter(self):
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['Input'],
                'Output',
                no_grad_set={'Filter'},
            )

        def test_check_grad_no_input(self):
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['Filter'],
                'Output',
                no_grad_set={'Input'},
            )

    cls_name = "{}_{}".format(parent.__name__, "CUDNNBF16OP")
    TestConv3DTransposeCUDNNBF16.__name__ = cls_name
    globals()[cls_name] = TestConv3DTransposeCUDNNBF16


def conv3d_transpose_wrapper(
    x,
    weight,
    stride=1,
    padding=0,
    output_padding=[],
    output_size=[],
    padding_algorithm="EXPLICIT",
    groups=1,
    dilation=1,
    data_format="NCDHW",
):
    if data_format == "AnyLayout":
        data_format = "NCDHW"
    return paddle._C_ops.conv3d_transpose(
        x,
        weight,
        stride,
        padding,
        output_padding,
        output_size,
        padding_algorithm,
        groups,
        dilation,
        data_format,
    )


class TestConv3DTransposeOp(OpTest):
    def setUp(self):
        # init as conv transpose
        self.use_cudnn = False
        self.check_no_input = False
        self.check_no_filter = False
        self.data_format = 'NCHW'
        self.pad = [0, 0, 0]
        self.padding_algorithm = "EXPLICIT"
        self.init_op_type()
        self.init_kernel_type()
        self.init_test_case()

        if self.is_bfloat16_op():
            input = np.random.random(self.input_size).astype(np.float32)
            filter = np.random.random(self.filter_size).astype(np.float32)
        else:
            input = np.random.random(self.input_size).astype(self.dtype)
            filter = np.random.random(self.filter_size).astype(self.dtype)

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'padding_algorithm': self.padding_algorithm,
            'dilations': self.dilations,
            'groups': self.groups,
            'use_cudnn': self.use_cudnn,
            'data_format': self.data_format,
        }

        output = conv3dtranspose_forward_naive(
            input, filter, self.attrs
        ).astype("float32")

        if self.is_bfloat16_op():
            self.inputs = {
                'Input': convert_float_to_uint16(input),
                'Filter': convert_float_to_uint16(filter),
            }
        else:
            self.inputs = {
                'Input': input,
                'Filter': filter,
            }
            output = output.astype(self.dtype)

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
                {'Input', 'Filter'},
                'Output',
                max_relative_error=0.03,
            )
        else:
            self.check_grad(
                {'Input', 'Filter'}, 'Output', max_relative_error=0.03
            )

    def test_check_grad_no_filter(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['Input'],
                'Output',
                max_relative_error=0.03,
                no_grad_set={'Filter'},
            )
        elif self.check_no_filter:
            self.check_grad(
                ['Input'],
                'Output',
                max_relative_error=0.03,
                no_grad_set={'Filter'},
            )

    def test_check_grad_no_input(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['Filter'],
                'Output',
                max_relative_error=0.03,
                no_grad_set={'Input'},
            )
        elif self.check_no_input:
            self.check_grad(
                ['Filter'],
                'Output',
                max_relative_error=0.03,
                no_grad_set={'Input'},
            )

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
        self.python_api = conv3d_transpose_wrapper

    def init_kernel_type(self):
        self.dtype = np.float32


class TestWithSymmetricPad(TestConv3DTransposeOp):
    def init_test_case(self):
        self.check_no_input = True
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithAsymmetricPad(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 0, 1, 0, 1, 2]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithSAMEPad(TestConv3DTransposeOp):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.dilations = [1, 2, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 6]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 4]
        self.padding_algorithm = 'SAME'


class TestWithVALIDPad(TestConv3DTransposeOp):
    def init_test_case(self):
        self.stride = [2, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 4, 3]
        self.padding_algorithm = 'VALID'


class TestWithStride(TestConv3DTransposeOp):
    def init_test_case(self):
        self.check_no_filter = True
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class TestWithGroups(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 2
        self.input_size = [1, 2, 5, 5, 5]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 3, 3, 3, 3]


class TestWithDilation(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [2, 2, 2]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]


class Test_NHWC(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 5, 5, 5, 2]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'


# ------------ test_cudnn ------------
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNN(TestConv3DTransposeOp):
    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithSymmetricPad(TestWithSymmetricPad):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithAsymmetricPad(TestWithAsymmetricPad):
    def init_test_case(self):
        self.pad = [1, 1, 1, 0, 0, 2]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 4, 4, 4]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithSAMEPad(TestWithSAMEPad):
    def init_test_case(self):
        self.stride = [1, 1, 2]
        self.dilations = [1, 2, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 4, 3]
        self.padding_algorithm = 'SAME'

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithVALIDPad(TestWithVALIDPad):
    def init_test_case(self):
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.padding_algorithm = 'VALID'

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithStride(TestWithStride):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 2, 5, 5, 5]  # NCDHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3, 3]

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithGroups(TestWithGroups):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 2
        self.input_size = [1, 2, 5, 5, 5]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 3, 3, 3, 3]

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper

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
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNN_NHWC(TestConv3DTransposeOp):
    def init_test_case(self):
        self.pad = [0, 0, 0]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 5, 5, 5, 2]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithSymmetricPad_NHWC(TestWithSymmetricPad):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 5, 5, 5, 2]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithAsymmetricPad_NHWC(TestWithAsymmetricPad):
    def init_test_case(self):
        self.pad = [1, 0, 1, 0, 0, 2]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 5, 5, 5, 2]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithStride_NHWC(TestWithStride):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [2, 2, 2]
        self.dilations = [1, 1, 1]
        self.groups = 1
        self.input_size = [1, 5, 5, 5, 2]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 6, 3, 3, 3]
        self.data_format = 'NHWC'

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestCUDNNWithGroups_NHWC(TestWithGroups):
    def init_test_case(self):
        self.pad = [1, 1, 1]
        self.stride = [1, 1, 1]
        self.dilations = [1, 1, 1]
        self.groups = 2
        self.input_size = [1, 5, 5, 5, 2]  # NDHWC
        f_c = self.input_size[-1]
        self.filter_size = [f_c, 3, 3, 3, 3]
        self.data_format = 'NHWC'

    def init_op_type(self):
        self.use_cudnn = True
        self.op_type = "conv3d_transpose"
        self.python_api = conv3d_transpose_wrapper


# ----------------Conv3DTransposeCUDNN fp16----------------
create_test_cudnn_fp16_class(TestConv3DTransposeOp)
create_test_cudnn_fp16_class(TestWithSymmetricPad)
create_test_cudnn_fp16_class(TestWithAsymmetricPad)
create_test_cudnn_fp16_class(TestWithSAMEPad)
create_test_cudnn_fp16_class(TestWithVALIDPad)
create_test_cudnn_fp16_class(TestWithStride)
create_test_cudnn_fp16_class(TestWithGroups)
create_test_cudnn_fp16_class(TestWithDilation)
create_test_cudnn_fp16_class(Test_NHWC)


# ----------------Conv3DTransposeCUDNN bf16----------------
create_test_cudnn_bf16_class(TestConv3DTransposeOp)
create_test_cudnn_bf16_class(TestWithSymmetricPad)
create_test_cudnn_bf16_class(TestWithAsymmetricPad)
create_test_cudnn_bf16_class(TestWithSAMEPad)
create_test_cudnn_bf16_class(TestWithVALIDPad)
create_test_cudnn_bf16_class(TestWithStride)
create_test_cudnn_bf16_class(TestWithGroups)
create_test_cudnn_bf16_class(TestWithDilation)
create_test_cudnn_bf16_class(Test_NHWC)


class TestConv3dTranspose(unittest.TestCase):
    def error_weight_input(self):
        array = np.array([1], dtype=np.float32)
        x = paddle.to_tensor(
            np.reshape(array, [1, 1, 1, 1, 1]), dtype='float32'
        )
        weight = paddle.to_tensor(np.reshape(array, [1]), dtype='float32')
        paddle.nn.functional.conv3d_transpose(x, weight, bias=0)

    def test_type_error(self):
        self.assertRaises(ValueError, self.error_weight_input)


if __name__ == '__main__':
    unittest.main()
