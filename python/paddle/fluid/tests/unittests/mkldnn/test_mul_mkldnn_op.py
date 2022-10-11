#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from numpy.matrixlib import defmatrix
import paddle
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16, OpTestTool


@OpTestTool.skip_if_not_cpu_bf16()
class TestMulOneDNNOp(OpTest):

    def setUp(self):
        self.op_type = "mul"
        self.attrs = {'use_mkldnn': True}
        self.init_shapes_and_attrs()

        self.x_fp32 = np.random.random(self.x_shape).astype(np.float32)
        self.y_fp32 = np.random.random(self.y_shape).astype(np.float32)

        self.x = self.x_fp32
        self.y = self.y_fp32

        self.init_inputs_dtype()

        self.inputs = {'X': self.x, 'Y': self.y}

        output = np.dot(np.reshape(self.x_fp32, self.np_x_shape),
                        np.reshape(self.y_fp32, self.np_y_shape))
        self.outputs = {'Out': np.reshape(output, self.out_shape)}

    def init_shapes_and_attrs(self):
        self.x_shape = (20, 5)
        self.y_shape = (5, 21)

        self.np_x_shape = (20, 5)
        self.np_y_shape = (5, 21)

        self.out_shape = (20, 21)

    def init_inputs_dtype(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        self.check_grad_with_place(core.CPUPlace(), ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad_with_place(core.CPUPlace(), ['Y'], 'Out', set('X'))

    def test_check_grad_ingore_y(self):
        self.check_grad_with_place(core.CPUPlace(), ['X'], 'Out', set('Y'))


class TestMulXNumColDims2OneDNNOp(TestMulOneDNNOp):

    def init_shapes_and_attrs(self):
        self.x_shape = (6, 7, 5)
        self.y_shape = (5, 21)

        self.np_x_shape = (42, 5)
        self.np_y_shape = (5, 21)

        self.out_shape = (6, 7, 21)

        self.attrs["x_num_col_dims"] = 2


class TestMulYNumColDims2OneDNNOp(TestMulOneDNNOp):

    def init_shapes_and_attrs(self):
        self.x_shape = (20, 6)
        self.y_shape = (2, 3, 21)

        self.np_x_shape = (20, 6)
        self.np_y_shape = (6, 21)

        self.out_shape = (20, 21)

        self.attrs["y_num_col_dims"] = 2


class TestMulYAndXNumColDims2OneDNNOp(TestMulOneDNNOp):

    def init_shapes_and_attrs(self):
        self.x_shape = (10, 5, 6)
        self.y_shape = (2, 3, 21)

        self.np_x_shape = (50, 6)
        self.np_y_shape = (6, 21)

        self.out_shape = (10, 5, 21)

        self.attrs["x_num_col_dims"] = 2
        self.attrs["y_num_col_dims"] = 2


class TestMulBF16OneDNNOp(TestMulOneDNNOp):

    def init_inputs_dtype(self):
        self.x = convert_float_to_uint16(self.x)
        self.y = convert_float_to_uint16(self.y)

    def calculate_grads(self):
        x_np = np.reshape(self.x_fp32, self.np_x_shape)
        y_np = np.reshape(self.y_fp32, self.np_y_shape)

        self.dout = self.outputs['Out']
        self.dout_np = np.reshape(self.dout, (x_np.shape[0], y_np.shape[1]))

        y_np_trans = np.transpose(y_np, (1, 0))
        x_np_trans = np.transpose(x_np, (1, 0))

        self.dx = np.matmul(self.dout_np, y_np_trans)
        self.dy = np.matmul(x_np_trans, self.dout_np)

    def test_check_grad(self):
        self.calculate_grads()
        self.check_grad_with_place(
            core.CPUPlace(), ['X', 'Y'],
            'Out',
            user_defined_grads=[self.dx, self.dy],
            user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

    def test_check_grad_ingore_x(self):
        self.calculate_grads()
        self.check_grad_with_place(
            core.CPUPlace(), ['Y'],
            'Out',
            set('X'),
            user_defined_grads=[self.dy],
            user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

    def test_check_grad_ingore_y(self):
        self.calculate_grads()
        self.check_grad_with_place(
            core.CPUPlace(), ['X'],
            'Out',
            set('Y'),
            user_defined_grads=[self.dx],
            user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
