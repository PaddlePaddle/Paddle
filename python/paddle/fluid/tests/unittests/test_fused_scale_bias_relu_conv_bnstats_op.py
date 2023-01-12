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

from __future__ import print_function

import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci
from test_conv2d_op import conv2d_forward_naive

import paddle
import paddle.fluid.core as core


def skip_unit_test():
    return (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.get_device_capability()[0] < 8
    )


skip_msg = "only support with cuda and Ampere or later devices"


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedScaleBiasReluConvBnstatsOp(OpTest):
    def setUp(self):
        self.op_type = "fused_scale_bias_relu_conv_bnstats"
        self.exhaustive_search = False
        self.data_format = "NHWC"
        self.dtype = np.float16
        self.outputs = None
        self.padding_algorithm = "EXIPLICIT"
        self.init_attr()
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_paddings()
        self.set_search_method()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations,
        }

        c_dim = self.input_size[-1]
        input = np.random.random(self.input_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)
        bias = np.random.random(c_dim).astype(self.dtype)
        scale = np.random.random(c_dim).astype(self.dtype)

        # calculate reference
        input_ref = input.astype(np.float32)
        if self.fuse_prologue:
            input_ref *= scale.reshape((1, 1, 1, c_dim)).astype(
                np.float32
            )  # scale
            input_ref += bias.reshape((1, 1, 1, c_dim)).astype(
                np.float32
            )  # bias
            input_ref = np.maximum(input_ref, 0)  # relu

        self.output, _, _, _, _ = conv2d_forward_naive(
            input_ref,
            filter,
            self.groups,
            conv2d_param,
            self.padding_algorithm,
            self.data_format,
        )

        self.output = self.output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter),
        }

        if self.fuse_prologue:
            extra_inputs = {
                'Bias': OpTest.np_dtype_to_fluid_dtype(bias),
                'Scale': OpTest.np_dtype_to_fluid_dtype(scale),
            }
            self.inputs.update(extra_inputs)

        self.attrs = {
            'fuse_prologue': self.fuse_prologue,
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'data_format': self.data_format,
            'padding_algorithm': self.padding_algorithm,
        }

        k_dim = self.filter_size[0]
        empty_sum_output = np.zeros(k_dim).astype(np.float32)
        empty_sqsum_output = np.zeros(k_dim).astype(np.float32)

        self.outputs = {
            'Output': self.output,
            'SumOutput': empty_sum_output,
            'SqSumOutput': empty_sqsum_output,
        }
        # SumOutput, SqSumOutput will be checked in test_fused_bn_finalize_op
        self.skip_list = ['SumOutput', 'SqSumOutput']

    def has_cuda(self):
        return core.is_compiled_with_cuda()

    def test_check_output(self):
        if self.has_cuda():
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place, no_check_set=self.skip_list, atol=2e-2
            )

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 5, 5, 8]  # NHWC
        assert np.mod(self.input_size[-1], self.groups) == 0
        f_c = self.input_size[-1] // self.groups
        self.filter_size = [16, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def set_search_method(self):
        self.exhaustive_search = False

    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_attr(self):
        self.fuse_prologue = True


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(skip_unit_test(), skip_msg)
class TestFusedScaleBiasReluConvBnstatsOpNoPrologue(
    TestFusedScaleBiasReluConvBnstatsOp
):
    def init_attr(self):
        self.fuse_prologue = False


def create_test_padding_SAME_class(parent):
    @skip_check_grad_ci(reason="no grap op")
    @unittest.skipIf(skip_unit_test(), skip_msg)
    class TestPaddingSMAECase(parent):
        def init_paddings(self):
            self.pad = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_padding_VALID_class(parent):
    @skip_check_grad_ci(reason="no grap op")
    @unittest.skipIf(skip_unit_test(), skip_msg)
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


create_test_padding_SAME_class(TestFusedScaleBiasReluConvBnstatsOp)
create_test_padding_VALID_class(TestFusedScaleBiasReluConvBnstatsOp)

if __name__ == '__main__':
    unittest.main()
