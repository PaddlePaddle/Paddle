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

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest
from paddle.fluid import Program, program_guard


def conv2d_forward_naive(input,
                         filter,
                         group,
                         conv_param,
                         padding_algorithm='EXPLICIT',
                         data_format='NCHW'):
    if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
        raise ValueError("Unknown Attr(padding_algorithm): '%s'. "
                         "It can only be 'SAME' or 'VALID'." %
                         str(padding_algorithm))

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Unknown Attr(data_format): '%s' ."
                         "It can only be 'NCHW' or 'NHWC'." % str(data_format))

    channel_last = (data_format == "NHWC")
    if channel_last:
        input = np.transpose(input, [0, 3, 1, 2])

    in_n, in_c, in_h, in_w = input.shape
    f_n, f_c, f_h, f_w = filter.shape
    out_n = in_n
    out_c = f_n
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group
    sub_f_n = f_n // group

    stride, pad, dilation = conv_param['stride'], conv_param['pad'], conv_param[
        'dilation']

    # update pad and dilation
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(input_shape, pool_size,
                                                        pool_stride):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max((
                (out_size - 1) * stride_size + filter_size - input_size, 0))
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    ksize = filter.shape[2:4]
    if padding_algorithm == "VALID":
        pad = [0, 0, 0, 0]
    elif padding_algorithm == "SAME":
        dilation = [1, 1]
        input_data_shape = input.shape[2:4]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)

    pad_h_0, pad_h_1 = pad[0], pad[0]
    pad_w_0, pad_w_1 = pad[1], pad[1]
    if len(pad) == 4:
        pad_h_0, pad_h_1 = pad[0], pad[1]
        pad_w_0, pad_w_1 = pad[2], pad[3]
    out_h = 1 + (in_h + pad_h_0 + pad_h_1 - (dilation[0] *
                                             (f_h - 1) + 1)) // stride[0]
    out_w = 1 + (in_w + pad_w_0 + pad_w_1 - (dilation[1] *
                                             (f_w - 1) + 1)) // stride[1]
    out = np.zeros((out_n, out_c, out_h, out_w))

    d_bolck_h = (dilation[0] * (f_h - 1) + 1)
    d_bolck_w = (dilation[1] * (f_w - 1) + 1)

    input_pad = np.pad(input, ((0, 0), (0, 0), (pad_h_0, pad_h_1),
                               (pad_w_0, pad_w_1)),
                       mode='constant',
                       constant_values=0)

    filter_dilation = np.zeros((f_n, f_c, d_bolck_h, d_bolck_w))
    filter_dilation[:, :, 0:d_bolck_h:dilation[0], 0:d_bolck_w:dilation[
        1]] = filter

    for i in range(out_h):
        for j in range(out_w):
            for g in range(group):
                input_pad_masked = \
                    input_pad[:, g * f_c:(g + 1) * f_c,
                    i * stride[0]:i * stride[0] + d_bolck_h,
                    j * stride[1]:j * stride[1] + d_bolck_w]

                f_sub = filter_dilation[g * sub_f_n:(g + 1) * sub_f_n, :, :, :]
                # sub_f_n == sub_out_c
                for k in range(sub_out_c):
                    # Multiplication of Corresponding Elements, then sum all
                    out[:, g * sub_out_c + k, i, j] = \
                        np.sum(input_pad_masked * f_sub[k, :, :, :],
                               axis=(1, 2, 3))

    if channel_last:
        out = np.transpose(out, [0, 2, 3, 1])

    return out, in_n, out_h, out_w, out_c


def create_test_cudnn_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCUDNNCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

    cls_name = "{0}_{1}".format(parent.__name__, "CUDNN")
    TestCUDNNCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNCase


def create_test_cudnn_fp16_class(parent, grad_check=True):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestConv2DCUDNNFp16(parent):
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
                    place, ['Input'], 'Output', no_grad_set=set(['Filter']))

        def test_check_grad_no_input(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(
                    place, ['Filter'], 'Output', no_grad_set=set(['Input']))

    cls_name = "{0}_{1}".format(parent.__name__, "CUDNNFp16")
    TestConv2DCUDNNFp16.__name__ = cls_name
    globals()[cls_name] = TestConv2DCUDNNFp16


def create_test_channel_last_class(parent):
    class TestChannelLastCase(parent):
        def init_data_format(self):
            self.data_format = "NHWC"

        def init_test_case_2(self):
            N, C, H, W = self.input_size
            self.input_size = [N, H, W, C]

    cls_name = "{0}_{1}".format(parent.__name__, "ChannelLast")
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase


def create_test_cudnn_channel_last_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCudnnChannelLastCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_data_format(self):
            self.data_format = "NHWC"

        def init_test_case_2(self):
            N, C, H, W = self.input_size
            self.input_size = [N, H, W, C]

    cls_name = "{0}_{1}".format(parent.__name__, "CudnnChannelLast")
    TestCudnnChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestCudnnChannelLastCase


def create_test_cudnn_channel_last_fp16_class(parent, grad_check=True):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCudnnChannelLastFp16(parent):
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
                    place, ['Input'], 'Output', no_grad_set=set(['Filter']))

        def test_check_grad_no_input(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place) and grad_check:
                self.check_grad_with_place(
                    place, ['Filter'], 'Output', no_grad_set=set(['Input']))

        def init_data_format(self):
            self.data_format = "NHWC"

        def init_test_case_2(self):
            N, C, H, W = self.input_size
            self.input_size = [N, H, W, C]

    cls_name = "{0}_{1}".format(parent.__name__, "CudnnChannelLastFp16")
    TestCudnnChannelLastFp16.__name__ = cls_name
    globals()[cls_name] = TestCudnnChannelLastFp16


def create_test_padding_SAME_class(parent):
    class TestPaddingSMAECase(parent):
        def init_paddings(self):
            self.pad = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


def create_test_cudnn_padding_SAME_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCUDNNPaddingSMAECase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "CudnnPaddingSAMEOp")
    TestCUDNNPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingSMAECase


def create_test_cudnn_padding_VALID_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestCUDNNPaddingVALIDCase(parent):
        def init_kernel_type(self):
            self.use_cudnn = True

        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "CudnnPaddingVALIDOp")
    TestCUDNNPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestCUDNNPaddingVALIDCase


class TestConv2DOp(OpTest):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.fuse_relu_before_depthwise_conv = False
        self.data_format = "AnyLayout"
        # explicilty use float32 for ROCm, as MIOpen does not yet support float64
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        if not self.has_cuda():
            self.fuse_relu_before_depthwise_conv = False
        if self.fuse_relu_before_depthwise_conv:
            input = input - 0.5
            input -= (input < 0) * 0.1
            input += (input >= 0) * 0.1
            input2 = np.maximum(input, 0.0)
        else:
            input2 = input
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

        output, _, _, _, _ = conv2d_forward_naive(input2, filter, self.groups,
                                                  conv2d_param)
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
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
            'fuse_relu_before_depthwise_conv':
            self.fuse_relu_before_depthwise_conv,
            'exhaustive_search': self.exhaustive_search
        }
        self.outputs = {'Output': output}

    def has_cuda(self):
        return core.is_compiled_with_cuda() and (self.use_cudnn or
                                                 self.use_cuda)

    def test_check_output(self):
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output_with_place(
            place, atol=1e-5, check_dygraph=(self.use_mkldnn == False))

    def test_check_grad(self):
        if self.dtype == np.float16 or (hasattr(self, "no_need_check_grad") and
                                        self.no_need_check_grad == True):
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad_with_place(
            place, {'Input', 'Filter'},
            'Output',
            max_relative_error=0.02,
            check_dygraph=(self.use_mkldnn == False))

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16 or (hasattr(self, "no_need_check_grad") and
                                        self.no_need_check_grad == True):
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad_with_place(
            place, ['Input'],
            'Output',
            max_relative_error=0.02,
            no_grad_set=set(['Filter']),
            check_dygraph=(self.use_mkldnn == False))

    def test_check_grad_no_input(self):
        if self.dtype == np.float16 or (hasattr(self, "no_need_check_grad") and
                                        self.no_need_check_grad == True):
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad_with_place(
            place, ['Filter'],
            'Output',
            no_grad_set=set(['Input']),
            check_dygraph=(self.use_mkldnn == False))

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_test_case_2(self):
        pass

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        pass


class TestWithPad(TestConv2DOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


class TestWithStride(TestConv2DOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


class TestWithGroup(TestConv2DOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.group = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [18, f_c, 3, 3]


class TestWith1x1(TestConv2DOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithDepthWise3x3(TestConv2DOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [3, 4, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 4


class TestWithDepthWise5x5(TestConv2DOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 4, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [8, f_c, 5, 5]

    def init_group(self):
        self.groups = 4


class TestWithDepthWise7x7(TestConv2DOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 8, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [16, f_c, 7, 7]

    def init_group(self):
        self.groups = 8


class TestWithDilation(TestConv2DOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 3


class TestWithInput1x1Filter1x1(TestConv2DOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [100, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


#----------------Conv2DCUDNN----------------

create_test_cudnn_class(TestConv2DOp)
create_test_cudnn_class(TestWithPad)
create_test_cudnn_class(TestWithStride)
create_test_cudnn_class(TestWithGroup)
create_test_cudnn_class(TestWith1x1)
create_test_cudnn_class(TestWithInput1x1Filter1x1)

#----------------Conv2DCUDNN fp16----------------

create_test_cudnn_fp16_class(TestConv2DOp, grad_check=False)
create_test_cudnn_fp16_class(TestWithPad, grad_check=False)
create_test_cudnn_fp16_class(TestWithStride, grad_check=False)
create_test_cudnn_fp16_class(TestWithGroup, grad_check=False)
create_test_cudnn_fp16_class(TestWith1x1, grad_check=False)
create_test_cudnn_fp16_class(TestWithInput1x1Filter1x1, grad_check=False)

#----------------TestDepthwiseConv -----


class TestDepthwiseConv(TestConv2DOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv2(TestConv2DOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv3(TestConv2DOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvWithDilation(TestConv2DOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvWithDilation2(TestConv2DOp):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvandFuse(TestConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv2andFuse(TestConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConv3andFuse(TestConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvWithDilationandFuse(TestConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestDepthwiseConvWithDilation2andFuse(TestConv2DOp):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"


class TestCUDNNExhaustiveSearch(TestConv2DOp):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.exhaustive_search = True


class TestConv2DOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                # the input of conv2d must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                fluid.layers.conv2d(x1, 1, 1)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                # the input dtype of conv2d must be float16 or float32 or float64
                # float16 only can be set on GPU place
                x2 = fluid.layers.data(
                    name='x2', shape=[3, 4, 5, 6], dtype="int32")
                fluid.layers.conv2d(x2, 1, 1)

            self.assertRaises(TypeError, test_dtype)


# Please Don't remove the following code.
# Currently, CI use cudnn V5.0 which not support dilation conv.
# class TestCUDNNWithDilation(TestWithDilation):
#     def init_op_type(self):
#         self.op_type = "conv_cudnn"

# ---- test asymmetric padding ----


class TestConv2DOp_v2(OpTest):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.fuse_relu_before_depthwise_conv = False
        # explicilty use float32 for ROCm, as MIOpen does not yet support float64
        self.dtype = np.float32 if core.is_compiled_with_rocm() else np.float64
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_data_format()
        self.init_test_case()
        self.init_paddings()
        self.init_test_case_2()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        if not self.has_cuda():
            self.fuse_relu_before_depthwise_conv = False
        if self.fuse_relu_before_depthwise_conv:
            input = input - 0.5
            input -= (input < 0) * 0.1
            input += (input >= 0) * 0.1
            input2 = np.maximum(input, 0.0)
        else:
            input2 = input
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)
        output, _, _, _, _ = conv2d_forward_naive(
            input2, filter, self.groups, conv2d_param, self.padding_algorithm,
            self.data_format)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'padding_algorithm': self.padding_algorithm,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
            'fuse_relu_before_depthwise_conv':
            self.fuse_relu_before_depthwise_conv,
            'exhaustive_search': self.exhaustive_search
        }
        self.outputs = {'Output': output}

    def has_cuda(self):
        return core.is_compiled_with_cuda() and (self.use_cudnn or
                                                 self.use_cuda)

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_output_with_place(
            place, atol=1e-5, check_dygraph=(self.use_mkldnn == False))

    def test_check_grad(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(
            place, {'Input', 'Filter'},
            'Output',
            max_relative_error=0.02,
            check_dygraph=(self.use_mkldnn == False))

    def test_check_grad_no_filter(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(
            place, ['Input'],
            'Output',
            max_relative_error=0.02,
            no_grad_set=set(['Filter']),
            check_dygraph=(self.use_mkldnn == False))

    def test_check_grad_no_input(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        place = core.CUDAPlace(0) if self.has_cuda() else core.CPUPlace()
        self.check_grad_with_place(
            place, ['Filter'],
            'Output',
            no_grad_set=set(['Input']),
            check_dygraph=(self.use_mkldnn == False))

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 4, 3]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        pass

    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_test_case_2(self):
        pass


class TestConv2DOp_AsyPadding(TestConv2DOp_v2):
    def init_paddings(self):
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithPad_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 3, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithStride_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 3, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithGroup_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.group = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 4, 3]


class TestWith1x1_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [2, 2, 4, 0]
        self.padding_algorithm = "EXPLICIT"


class TestWithDepthWise3x3_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [3, 4, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [16, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 4

    def init_paddings(self):
        self.pad = [1, 3, 2, 1]
        self.padding_algorithm = "EXPLICIT"


class TestWithDepthWise5x5_AsyPadding(TestConv2DOp_v2):
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


class TestWithDepthWise7x7_AsyPadding(TestConv2DOp_v2):
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


class TestWithDilation_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [0, 1, 3, 0]
        self.padding_algorithm = "EXPLICIT"


class TestWithInput1x1Filter1x1_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [40, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [0, 3, 4, 0]
        self.padding_algorithm = "EXPLICIT"


create_test_cudnn_class(TestConv2DOp_AsyPadding)
create_test_cudnn_class(TestWithPad_AsyPadding)
create_test_cudnn_class(TestWithStride_AsyPadding)
create_test_cudnn_class(TestWithGroup_AsyPadding)
create_test_cudnn_class(TestWith1x1_AsyPadding)
create_test_cudnn_class(TestWithInput1x1Filter1x1_AsyPadding)


class TestDepthwiseConv_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.use_cuda = True
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 1, 0, 1]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv2_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.use_cuda = True
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [0, 1, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv3_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.use_cuda = True
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 1, 0, 0]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvWithDilation_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 1, 2, 1]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvWithDilation2_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [0, 1, 1, 0]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvandFuse_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [2, 1, 2, 3]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv2andFuse_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 1, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConv3andFuse_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 2, 0, 2]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvWithDilationandFuse_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [2, 1, 1, 0]
        self.padding_algorithm = "EXPLICIT"


class TestDepthwiseConvWithDilation2andFuse_AsyPadding(TestConv2DOp_v2):
    def init_test_case(self):
        self.fuse_relu_before_depthwise_conv = True
        self.use_cuda = True
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.groups = 3
        self.dilations = [2, 2]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]
        self.op_type = "depthwise_conv2d"

    def init_paddings(self):
        self.pad = [1, 3, 1, 3]
        self.padding_algorithm = "EXPLICIT"


#---------- test SAME VALID -----------
create_test_padding_SAME_class(TestConv2DOp_AsyPadding)
create_test_padding_SAME_class(TestWithPad_AsyPadding)
create_test_padding_SAME_class(TestWithStride_AsyPadding)
create_test_padding_SAME_class(TestWithGroup_AsyPadding)
create_test_padding_SAME_class(TestWithInput1x1Filter1x1_AsyPadding)

create_test_padding_VALID_class(TestConv2DOp_AsyPadding)
create_test_padding_VALID_class(TestWithPad_AsyPadding)
create_test_padding_VALID_class(TestWithStride_AsyPadding)
create_test_padding_VALID_class(TestWithGroup_AsyPadding)
create_test_padding_VALID_class(TestWithInput1x1Filter1x1_AsyPadding)

create_test_cudnn_padding_SAME_class(TestConv2DOp_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithPad_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithStride_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithGroup_AsyPadding)
create_test_cudnn_padding_SAME_class(TestWithInput1x1Filter1x1_AsyPadding)

create_test_cudnn_padding_VALID_class(TestConv2DOp_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithPad_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithStride_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithGroup_AsyPadding)
create_test_cudnn_padding_VALID_class(TestWithInput1x1Filter1x1_AsyPadding)

# depthwise conv2d

create_test_padding_SAME_class(TestDepthwiseConv_AsyPadding)
create_test_padding_SAME_class(TestDepthwiseConvWithDilation_AsyPadding)
create_test_padding_SAME_class(TestDepthwiseConvandFuse_AsyPadding)
create_test_padding_SAME_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)

create_test_padding_VALID_class(TestDepthwiseConv_AsyPadding)
create_test_padding_VALID_class(TestDepthwiseConvWithDilation_AsyPadding)
create_test_padding_VALID_class(TestDepthwiseConvandFuse_AsyPadding)
create_test_padding_VALID_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)

# ------------ test channel last ---------
create_test_channel_last_class(TestConv2DOp_AsyPadding)
create_test_channel_last_class(TestWithPad_AsyPadding)
create_test_channel_last_class(TestWithGroup_AsyPadding)
create_test_channel_last_class(TestWith1x1_AsyPadding)
create_test_channel_last_class(TestWithInput1x1Filter1x1_AsyPadding)

create_test_channel_last_class(TestDepthwiseConv_AsyPadding)
create_test_channel_last_class(TestDepthwiseConvWithDilation2_AsyPadding)
create_test_channel_last_class(TestDepthwiseConvandFuse_AsyPadding)
create_test_channel_last_class(TestDepthwiseConvWithDilationandFuse_AsyPadding)

create_test_cudnn_channel_last_class(TestConv2DOp_AsyPadding)
create_test_cudnn_channel_last_class(TestWithPad_AsyPadding)
create_test_cudnn_channel_last_class(TestWithStride_AsyPadding)
create_test_cudnn_channel_last_class(TestWithGroup_AsyPadding)
create_test_cudnn_channel_last_class(TestWithDilation_AsyPadding)

create_test_cudnn_channel_last_fp16_class(
    TestConv2DOp_AsyPadding, grad_check=False)
create_test_cudnn_channel_last_fp16_class(
    TestWithPad_AsyPadding, grad_check=False)
create_test_cudnn_channel_last_fp16_class(
    TestWithStride_AsyPadding, grad_check=False)
create_test_cudnn_channel_last_fp16_class(
    TestWithGroup_AsyPadding, grad_check=False)
create_test_cudnn_channel_last_fp16_class(
    TestWithDilation_AsyPadding, grad_check=False)


# --------- test python API ---------------
class TestConv2DAPI(unittest.TestCase):
    def test_api(self):

        input_NHWC = fluid.layers.data(
            name="input_NHWC",
            shape=[2, 5, 5, 3],
            append_batch_size=False,
            dtype="float32")

        input_NCHW = fluid.layers.data(
            name="input_NCHW",
            shape=[2, 3, 5, 5],
            append_batch_size=False,
            dtype="float32")

        fluid.layers.conv2d(
            input=input_NHWC,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=0,
            dilation=[1, 1],
            groups=1,
            data_format="NCHW")

        fluid.layers.conv2d(
            input=input_NCHW,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=[1, 2, 1, 0],
            dilation=[1, 1],
            groups=1,
            data_format="NCHW")

        fluid.layers.conv2d(
            input=input_NCHW,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
            dilation=[1, 1],
            groups=1,
            data_format="NCHW")

        fluid.layers.conv2d(
            input=input_NHWC,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
            dilation=[1, 1],
            groups=1,
            data_format="NHWC")

        fluid.layers.conv2d(
            input=input_NCHW,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding="SAME",
            dilation=[1, 1],
            groups=1,
            data_format="NCHW")

        fluid.layers.conv2d(
            input=input_NCHW,
            num_filters=3,
            filter_size=[3, 3],
            stride=[1, 1],
            padding="VALID",
            dilation=[1, 1],
            groups=1,
            data_format="NCHW")

    def test_depthwise_conv2d(self):
        x_var = paddle.uniform((2, 8, 8, 4), dtype='float32', min=-1., max=1.)
        conv = paddle.nn.Conv2D(
            in_channels=4,
            out_channels=4,
            kernel_size=(3, 3),
            groups=4,
            data_format='NHWC')
        y_var = conv(x_var)


class TestConv2DAPI_Error(unittest.TestCase):
    def test_api(self):
        input = fluid.layers.data(
            name="input",
            shape=[2, 5, 5, 5],
            append_batch_size=False,
            dtype="float32")

        # ValueError: cudnn
        def run_1():
            fluid.layers.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=1,
                use_cudnn=[0],
                data_format="NCHW")

        self.assertRaises(ValueError, run_1)

        # ValueError: data_format
        def run_2():
            fluid.layers.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NCHWC")

        self.assertRaises(ValueError, run_2)

        # ValueError: padding
        def run_3():
            fluid.layers.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding="SAMEE",
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NCHW")

        self.assertRaises(ValueError, run_3)

        def run_4():
            fluid.layers.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=[[0, 1], [0, 1], [0, 1], [0, 1]],
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NCHW")

        self.assertRaises(ValueError, run_4)

        def run_5():
            fluid.layers.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=[[0, 1], [0, 1], [0, 1], [0, 1]],
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NHWC")

        self.assertRaises(ValueError, run_5)

        # ValueError: channel dimmention
        x = fluid.layers.data(
            name="x",
            shape=[2, 5, 5, -1],
            append_batch_size=False,
            dtype="float32")

        def run_6():
            fluid.layers.conv2d(
                input=x,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=1,
                use_cudnn=False,
                data_format="NHWC")

        self.assertRaises(ValueError, run_6)

        # ValueError: groups
        def run_7():
            fluid.layers.conv2d(
                input=input,
                num_filters=3,
                filter_size=[3, 3],
                stride=[1, 1],
                padding=0,
                dilation=[1, 1],
                groups=3,
                use_cudnn=False,
                data_format="NHWC")

        self.assertRaises(ValueError, run_7)


if __name__ == '__main__':
    unittest.main()
