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
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, skip_check_grad_ci
from paddle.fluid.tests.unittests.test_conv2d_op import conv2d_forward_naive
'''
Some tests differ from the tests defined in test_elementwise_add_op.py
because MKLDNN does not support tensors of number of dimensions 3.
Such dimensions cause exceptions in MKLDNN reorder primitive.
'''


class TestMKLDNNElementwiseAddOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_add"
        self.init_kernel_type()
        self.init_dtype()
        self.init_axis()
        self.init_input_output()
        self.init_data_format()
        self._cpu_only = True
        self.use_cudnn = False

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {
            'axis': self.axis,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format
        }
        self.outputs = {'Out': self.out}

    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1

    def init_data_format(self):
        self.data_format = 'AnyLayout'

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype(self.dtype)
        self.out = np.add(self.x, self.y)

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=(self.use_mkldnn == False))

    def test_check_grad_normal(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X', 'Y'], 'Out', check_dygraph=(self.use_mkldnn == False))

    def test_check_grad_ingore_x(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_dygraph=(self.use_mkldnn == False))

    def test_check_grad_ingore_y(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_dygraph=(self.use_mkldnn == False))


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestMKLDNNElementwiseAddOp_scalar(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1,1) to test broadcast.")
class TestMKLDNNElementwiseAddOp_scalar2(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(1, 1).astype(self.dtype)
        self.out = self.x + self.y


class TestMKLDNNElementwiseAddOp_Vector(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((32, )).astype(self.dtype)
        self.y = np.random.random((32, )).astype(self.dtype)
        self.out = np.add(self.x, self.y)


class TestMKLDNNElementwiseAddOp_broadcast_0(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3, 4).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(100, 1, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestMKLDNNElementwiseAddOp_broadcast_1(TestMKLDNNElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 100, 3, 4).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 100, 1, 1)

    def init_axis(self):
        self.axis = 1


class TestMKLDNNElementwiseAddOp_broadcast_2(TestMKLDNNElementwiseAddOp):
    def print_info(self):
        print("### TestMKLDNNElementwiseAddOp_broadcast2 ###")

    def init_input_output(self):
        self.x = np.random.rand(2, 2, 3, 100).astype(self.dtype)
        self.y = np.random.rand(100).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1, 1, 100)


class TestMKLDNNElementwiseAddOp_broadcast_3(TestMKLDNNElementwiseAddOp):
    def print_info(self):
        print("### TestMKLDNNElementwiseAddOp_bradcast_3 ###")

    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(3, 4).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 3, 4, 1)

    def init_axis(self):
        self.axis = 1


class TestMKLDNNElementwiseAddOp_broadcast_4(TestMKLDNNElementwiseAddOp):
    def print_info(self):
        print("### TestMKLDNNElementwiseAddOp_broadcast_4 ###")

    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
        self.y = np.random.rand(2, 1).astype(self.dtype)
        self.out = self.x + self.y.reshape(2, 1, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestMKLDNNElementwiseAddOp_rowwise_add_0(TestMKLDNNElementwiseAddOp):
    def print_info(self):
        print("### TestMKLDNNElementwiseAddOp_rowwise_add_0 ###")

    def init_input_output(self):
        self.x = np.random.rand(2, 10, 12, 3).astype(self.dtype)
        self.y = np.random.rand(10, 12).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 10, 12, 1)

    def init_axis(self):
        self.axis = 1


class TestMKLDNNElementwiseAddOp_rowwise_add_1(TestMKLDNNElementwiseAddOp):
    def print_info(self):
        print("### TestMKLDNNElementwiseAddOp_rowwise_add_1 ###")

    def init_input_output(self):
        self.x = np.random.rand(2, 1).astype(self.dtype)
        self.y = np.random.rand(1).astype(self.dtype)
        self.out = self.x + self.y.reshape(1, 1)

    def init_axis(self):
        self.axis = 1


class TestMKLDNNElementwiseAddOp_channelwise_add(TestMKLDNNElementwiseAddOp):
    def print_info(self):
        print("### TestMKLDNNElementwiseAddOp_channelwise_add ###")

    def init_input_output(self):
        self.x = np.random.rand(100, 2, 3, 3).astype(self.dtype)
        self.y = np.random.rand(100, 1, 1, 1).astype(self.dtype)
        self.out = self.x + self.y


# For UT coverage, integrate NHWC conv2d + elementwise_add so that NHWC input
# data was used in the elementwise_add
@skip_check_grad_ci(
    reason="This test is to verify the NHWC case for inference only.")
class TestMKLDNNElementwiseAddOp_NHWC(OpTest):
    def setUp(self):
        self.use_mkldnn = True
        self._cpu_only = True
        self.use_cudnn = False
        self.dtype = np.float32
        self.data_format = 'NHWC'
        self.axis = 0
        self.groups = 1
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [1, 5, 5, 3]
        self.filter_size = [16, 3, 3, 3]
        self.elt_add_y_size = [1]
        self.prepare_inputs()
        self.prepare_outputs()

    def prepare_inputs(self):
        self.input = np.random.random(self.input_size).astype(self.dtype)
        self.filter = np.random.random(self.filter_size).astype(self.dtype)
        self.elt_add_y = np.random.random(self.elt_add_y_size).astype(
            self.dtype)
        self.inputs = {
            'input': self.input,
            'filter': self.filter,
            'elt_add_y': self.elt_add_y
        }
        self.var_dims = {name: self.inputs[name].shape for name in self.inputs}

    def prepare_outputs(self):
        self.conv2d_param = {
            'pad': self.pad,
            'stride': self.stride,
            'dilation': self.dilations
        }
        conv_out, _, _, _, _ = conv2d_forward_naive(
            self.input,
            self.filter,
            self.groups,
            self.conv2d_param,
            data_format=self.data_format)
        self.conv_output = conv_out
        y = np.broadcast_to(self.elt_add_y.reshape(1, 1, 1, 1), (1, 3, 3, 16))
        self.elt_add_output = np.add(self.conv_output, y)
        self.elt_add_output_nchw = np.transpose(self.elt_add_output,
                                                [0, 3, 1, 2])
        self.fetch_list = ['elt_add_output']
        self.var_dims.update({
            'conv_output': self.conv_output.shape,
            'elt_add_output': self.elt_add_output.shape
        })

    def test_check_output(self):
        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in self.var_dims:
                block.create_var(
                    name=name, dtype="float32", shape=self.var_dims[name])
            conv2d_op = block.append_op(
                type="conv2d",
                inputs={
                    'Input': block.var('input'),
                    'Filter': block.var('filter')
                },
                outputs={"Output": block.var('conv_output')},
                attrs={
                    'strides': self.stride,
                    'paddings': self.pad,
                    'groups': self.groups,
                    'dilations': self.dilations,
                    'use_cudnn': self.use_cudnn,
                    'use_mkldnn': self.use_mkldnn,
                    'data_format': self.data_format
                })
            elementwise_add_op = block.append_op(
                type="elementwise_add",
                inputs={
                    'X': block.var('conv_output'),
                    'Y': block.var('elt_add_y'),
                },
                outputs={"Out": block.var('elt_add_output')},
                attrs={
                    'use_cudnn': self.use_cudnn,
                    'use_mkldnn': self.use_mkldnn,
                    'axis': self.axis
                })
            place = core.CPUPlace()
            exe = fluid.Executor(place)
            out = exe.run(program, feed=self.inputs, fetch_list=self.fetch_list)

            # Fetch op returns output in NCHW so we need to use NCHW ground truth tensor for comparison as well
            self.assertTrue(
                np.allclose(
                    self.elt_add_output_nchw, out[0], atol=1e-4),
                'elt_add_output')

    def test_check_grad_normal(self):
        pass

    def test_check_grad_ingore_x(self):
        pass

    def test_check_grad_ingore_y(self):
        pass


if __name__ == '__main__':
    unittest.main()
