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
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core


class TestMeshgridOp(OpTest):
    def setUp(self):
        self.op_type = "meshgrid"
        self.dtype = self.get_dtype()
        ins, outs = self.init_test_data()
        self.inputs = {'X': [('x%d' % i, ins[i]) for i in range(len(ins))]}
        self.outputs = {
            'Out': [('out%d' % i, outs[i]) for i in range(len(outs))]
        }
        self.use_cudnn = False

    def get_dtype(self):
        return "float64"

    def init_kernel_type(self):
        pass

    def test_check_output(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_dygraph=True)
        else:
            self.check_output(check_dygraph=True)

    def test_check_grad(self):
        if self.use_cudnn:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, ['x1'], ['out1'], check_dygraph=True, atol=1e-5)
        else:
            self.check_grad(['x0'], ['out0'])
            self.check_grad(['x1'], ['out1'])

    def init_test_data(self):
        self.shape = self.get_x_shape()
        ins = []
        outs = []
        for i in range(len(self.shape)):
            ins.append(np.random.random((self.shape[i], )).astype(self.dtype))

        for i in range(len(self.shape)):
            out_reshape = [1] * len(self.shape)
            out_reshape[i] = self.shape[i]
            out_temp = np.reshape(ins[i], out_reshape)
            outs.append(np.broadcast_to(out_temp, self.shape))
        return ins, outs

    def get_x_shape(self):
        return [4, 200]


class TestMeshgridOp2(TestMeshgridOp):
    def get_x_shape(self):
        return [10, 100]

    def init_kernel_type(self):
        self.use_cudnn = True


@skip_check_grad_ci(
    reason="The function 'check_grad' for large inputs is too slow.")
class TestMeshgridOp3(TestMeshgridOp):
    def get_x_shape(self):
        return [4, 5, 6]


if __name__ == '__main__':
    unittest.main()
