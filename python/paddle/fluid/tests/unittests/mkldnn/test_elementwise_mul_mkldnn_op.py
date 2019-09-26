#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid.tests.unittests.test_elementwise_mul_op import *
from paddle.fluid.tests.unittests.test_conv2d_op import conv2d_forward_naive
from paddle.fluid.tests.unittests.mkldnn.mkldnn_op_test import __assert_close
import paddle.fluid as fluid


# For UT coverage, integrate conv2d + elementwise-mul so that nchw16C could be automatically chosen when mkldnn-kernel is enabled
class TestElementwiseMulMKLDNNOp_Integrated_With_Convs(ElementwiseMulOp):
    def setUp(self):
        self.dtype = np.float32
        self.init_dtype()
        self.init_kernel_type()
        self.init_axis()
        self._cpu_only = True
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.groups = 1
        self.input_size = [1, 3, 5, 5]  # NCHW
        self.filter_size = [16, 3, 3, 3]
        self.filter_size2 = [1, 16, 2, 2]
        self.dilations = [1, 1]
        self.use_cudnn = False
        self.data_format = "NCHW"
        self.input = np.random.random(self.input_size).astype(self.dtype)
        self.filter = np.random.random(self.filter_size).astype(self.dtype)
        self.filter2 = np.random.random(self.filter_size2).astype(self.dtype)
        self.elt_mul_y_size = [1, 16]
        self.elt_mul_y = np.random.random(self.elt_mul_y_size).astype(
            self.dtype)
        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }
        conv_out, _, _, _, _ = conv2d_forward_naive(
            self.input, self.filter, self.groups, conv2d_param)  #[1, 16, 2, 2]
        self.conv_output = conv_out
        self.elt_mul_output = self.conv_output * self.elt_mul_y.reshape(
            1, 16, 1, 1)  # the result shape is [1, 16, 2, 2]
        conv_output2, _, _, _, _ = conv2d_forward_naive(
            self.elt_mul_output, self.filter2, self.groups, conv2d_param)
        self.conv_output2 = conv_output2
        self.fetch_list = ["conv_output2"]

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_output(self):
        ground_truth = {
            "input": self.input,
            "filter": self.filter,
            "filter2": self.filter2,
            "conv_output": self.conv_output,
            "elt_mul_y": self.elt_mul_y,
            "elt_mul_output": self.elt_mul_output,
            "conv_output2": self.conv_output2,
        }
        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in ground_truth:
                block.create_var(
                    name=name, dtype="float32", shape=ground_truth[name].shape)
            conv2d_op = block.append_op(
                type="conv2d",
                inputs={
                    "Input": block.var('input'),
                    'Filter': block.var('filter')
                },
                outputs={"Output": block.var('conv_output')},
                attrs={
                    'strides': self.stride,
                    'paddings': self.pad,
                    'groups': self.groups,
                    'dilations': self.dilations,
                    'use_cudnn': self.use_cudnn,
                    'use_mkldnn': self.use_mkldnn
                })
            elementwise_mul_op = block.append_op(
                type="elementwise_mul",
                inputs={
                    'X': block.var('conv_output'),
                    'Y': block.var('elt_mul_y'),
                },
                outputs={"Out": block.var('elt_mul_output')},
                attrs={
                    'use_cudnn': self.use_cudnn,
                    'use_mkldnn': self.use_mkldnn,
                    'axis': self.axis
                })
            conv2d_op2 = block.append_op(
                type="conv2d",
                inputs={
                    "Input": block.var('elt_mul_output'),
                    'Filter': block.var('filter2')
                },
                outputs={"Output": block.var('conv_output2')},
                attrs={
                    'strides': self.stride,
                    'paddings': self.pad,
                    'groups': self.groups,
                    'dilations': self.dilations,
                    'use_cudnn': self.use_cudnn,
                    'use_mkldnn': self.use_mkldnn,
                    'data_format': self.data_format
                })
            place = core.CPUPlace()
            exe = fluid.Executor(place)
            out = exe.run(
                program,
                feed={
                    name: ground_truth[name]
                    for name in ["input", "filter", "filter2", "elt_mul_y"]
                },
                fetch_list=self.fetch_list)

            for id, name in enumerate(self.fetch_list):
                self.assertTrue(
                    np.allclose(
                        ground_truth[name], out[id], atol=1e-4), name)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNCHW(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = np.random.rand(1, 16).astype(self.dtype)

        self.out = self.x * self.y.reshape(1, 16, 1, 1)

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNCHW16C(ElementwiseMulOp):
    def init_input_output(self):
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = x * y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNCHW16C, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNoReorders(ElementwiseMulOp):
    def init_input_output(self):
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = x * y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNoReorders, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackWithReorder1(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.y = y.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = self.x * y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackWithReorder1, self).setUp()
        self.attrs["x_data_format"] = "nchw"
        self.attrs["y_data_format"] = "nchw16c"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackWithReorder2(ElementwiseMulOp):
    def init_input_output(self):
        self.y = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        x = np.random.rand(1, 16, 2, 2).astype(self.dtype)
        self.x = x.transpose(0, 2, 3, 1).reshape(1, 16, 2, 2)

        self.out = x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackWithReorder2, self).setUp()
        self.attrs["x_data_format"] = "nchw16c"
        self.attrs["y_data_format"] = "nchw"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


class TestElementwiseMulMKLDNNOp_FallbackNoReorders2(ElementwiseMulOp):
    def init_input_output(self):
        self.x = np.random.rand(1, 16).astype(self.dtype)
        self.y = np.random.rand(1, 16).astype(self.dtype)

        self.out = self.x * self.y

    def setUp(self):
        super(TestElementwiseMulMKLDNNOp_FallbackNoReorders2, self).setUp()
        self.attrs["x_data_format"] = "nc"
        self.attrs["y_data_format"] = "nc"
        self._cpu_only = True

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_axis(self):
        self.axis = 0

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


if __name__ == '__main__':
    unittest.main()
