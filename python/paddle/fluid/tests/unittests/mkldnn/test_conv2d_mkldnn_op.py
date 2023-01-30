# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import unittest

import numpy as np

from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
from paddle.fluid.tests.unittests.test_conv2d_op import (
    TestConv2DOp,
    TestConv2DOp_v2,
)
=======
from __future__ import print_function

import os
import unittest
import numpy as np

import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
from paddle.fluid.tests.unittests.test_conv2d_op import TestConv2DOp, TestConv2DOp_v2
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def conv2d_bias_naive(out, bias):
    _, out_c, _, _ = out.shape

    for l in range(out_c):
        out[:, l, :, :] = out[:, l, :, :] + bias[l]
    return out


def conv2d_residual_naive(out, residual):
    assert out.shape == residual.shape
    out = np.add(out, residual)
    return out


class TestConv2DMKLDNNOp(TestConv2DOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        self.data_format = "NCHW"
        self.use_mkldnn = True
        self._cpu_only = True
        self.dtype = np.float32

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def setUp(self):
        self.fuse_bias = False
        self.bias_size = None
        self.fuse_activation = ""
        self.fuse_alpha = 0
        self.fuse_beta = 0
        self.fuse_residual_connection = False
        self.input_residual_size = None

        TestConv2DOp.setUp(self)

        output = self.outputs['Output']

<<<<<<< HEAD
        # mkldnn only support either conv-sum-relu, or conv-relu.
=======
        #mkldnn only support either conv-sum-relu, or conv-relu.
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        if self.fuse_bias and self.bias_size is not None:
            bias = np.random.random(self.bias_size).astype(self.dtype)
            output = conv2d_bias_naive(output, bias)
            output = output.astype(self.dtype)
            self.attrs['fuse_bias'] = self.fuse_bias
            self.inputs['Bias'] = OpTest.np_dtype_to_fluid_dtype(bias)

<<<<<<< HEAD
        if (
            self.fuse_residual_connection
            and self.input_residual_size is not None
        ):
            input_residual = np.random.random(self.input_residual_size).astype(
                self.dtype
            )
            output = conv2d_residual_naive(output, input_residual)

            self.attrs[
                'fuse_residual_connection'
            ] = self.fuse_residual_connection
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                input_residual
            )
=======
        if self.fuse_residual_connection and self.input_residual_size is not None:
            input_residual = np.random.random(self.input_residual_size).astype(
                self.dtype)
            output = conv2d_residual_naive(output, input_residual)

            self.attrs[
                'fuse_residual_connection'] = self.fuse_residual_connection
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                input_residual)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if self.fuse_activation == "relu":
            output = np.maximum(output, 0).astype(self.dsttype)

        if self.fuse_activation == "relu6":
<<<<<<< HEAD
            output = np.minimum(np.maximum(output, 0), self.fuse_alpha).astype(
                self.dsttype
            )
=======
            output = np.minimum(np.maximum(output, 0),
                                self.fuse_alpha).astype(self.dsttype)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        output = output.astype(self.dtype)

        self.attrs['fuse_bias'] = self.fuse_bias
        self.attrs['fuse_activation'] = self.fuse_activation
        self.attrs['fuse_alpha'] = self.fuse_alpha
        self.attrs['fuse_beta'] = self.fuse_beta
        self.attrs['fuse_residual_connection'] = self.fuse_residual_connection

        self.outputs['Output'] = output


@skip_check_grad_ci(
<<<<<<< HEAD
    reason="Fusion is for inference only, check_grad is not required."
)
class TestWithbreluFusion(TestConv2DMKLDNNOp):
=======
    reason="Fusion is for inference only, check_grad is not required.")
class TestWithbreluFusion(TestConv2DMKLDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        TestConv2DMKLDNNOp.init_test_case(self)
        self.fuse_activation = "relu6"
        self.fuse_alpha = 6.0
        self.dsttype = np.float32


@skip_check_grad_ci(
<<<<<<< HEAD
    reason="Fusion is for inference only, check_grad is not required."
)
class TestWithFuse(TestConv2DMKLDNNOp):
=======
    reason="Fusion is for inference only, check_grad is not required.")
class TestWithFuse(TestConv2DMKLDNNOp):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        TestConv2DMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.fuse_bias = True
        self.bias_size = [6]
        self.fuse_residual_connection = True
        self.input_residual_size = [2, 6, 5, 5]


class TestWithPadWithBias(TestConv2DMKLDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        TestConv2DMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.input_size = [2, 3, 6, 6]


class TestWithStride(TestConv2DMKLDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        TestConv2DMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]


class TestWithGroup(TestConv2DMKLDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 6, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_group(self):
        self.groups = 3


class TestWith1x1(TestConv2DMKLDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        TestConv2DMKLDNNOp.init_test_case(self)
        self.filter_size = [40, 3, 1, 1]


class TestWithInput1x1Filter1x1(TestConv2DMKLDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        TestConv2DMKLDNNOp.init_test_case(self)
        self.input_size = [2, 60, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


class TestConv2DOp_AsyPadding_MKLDNN(TestConv2DOp_v2):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_kernel_type(self):
        self.use_mkldnn = True
        self.dtype = np.float32

    def init_paddings(self):
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestConv2DOp_Same_MKLDNN(TestConv2DOp_AsyPadding_MKLDNN):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "SAME"


class TestConv2DOp_Valid_MKLDNN(TestConv2DOp_AsyPadding_MKLDNN):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_paddings(self):
        self.pad = [1, 1]
        self.padding_algorithm = "VALID"


class TestConv2DOp_Valid_NHWC_MKLDNN(TestConv2DOp_Valid_MKLDNN):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_test_case_2(self):
        N, C, H, W = self.input_size
        self.input_size = [N, H, W, C]


class TestConv2DOp_Same_NHWC_MKLDNN(TestConv2DOp_Valid_NHWC_MKLDNN):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "SAME"


class TestConv2DOp_AsyPadding_NHWC_MKLDNN(TestConv2DOp_Valid_NHWC_MKLDNN):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_paddings(self):
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestMKLDNNDilations(TestConv2DMKLDNNOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        TestConv2DMKLDNNOp.init_test_case(self)
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


if __name__ == '__main__':
    from paddle import enable_static
<<<<<<< HEAD

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    enable_static()
    unittest.main()
