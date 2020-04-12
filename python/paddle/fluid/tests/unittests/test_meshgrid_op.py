#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
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

    def get_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
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
        return [100, 200]


class TestMeshgridOp2(TestMeshgridOp):
    def get_x_shape(self):
        return [100, 300]


class TestMeshgridOp3(unittest.TestCase):
    def test_api(self):
        x = fluid.data(shape=[100], dtype='int32', name='x')
        y = fluid.data(shape=[200], dtype='int32', name='y')

        input_1 = np.random.randint(0, 100, [100, ]).astype('int32')
        input_2 = np.random.randint(0, 100, [200, ]).astype('int32')

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = fluid.Executor(place=fluid.CPUPlace())
        grid_x, grid_y = paddle.tensor.meshgrid([x, y])
        res_1, res_2 = exe.run(fluid.default_main_program(),
                               feed={'x': input_1,
                                     'y': input_2},
                               fetch_list=[grid_x, grid_y])

        assert np.array_equal(res_1, out_1)
        assert np.array_equal(res_2, out_2)


class TestMeshgridOp4(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_input_type():
                x = fluid.data(shape=[200], dtype='float32', name='x2')
                paddle.tensor.meshgrid(x)

        self.assertRaises(TypeError, test_input_type)


class TestMeshgridOp5(unittest.TestCase):
    def test_api_with_dygraph(self):
        input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
        input_4 = np.random.randint(0, 100, [200, ]).astype('int32')

        with fluid.dygraph.guard():
            tensor_3 = fluid.dygraph.to_variable(input_3)
            tensor_4 = fluid.dygraph.to_variable(input_4)
            res_3, res_4 = paddle.tensor.meshgrid([tensor_3, tensor_4])

            assert np.array_equal(res_3.shape, [100, 200])
            assert np.array_equal(res_4.shape, [100, 200])


if __name__ == '__main__':
    unittest.main()
