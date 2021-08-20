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
import sys
sys.path.append("..")
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
import paddle
from paddle.fluid import compiler, Program, program_guard, core

paddle.enable_static()


class TestMeshgridOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "meshgrid"
        self.dtype = self.get_dtype()
        ins, outs = self.init_test_data()
        self.inputs = {'X': [('x%d' % i, ins[i]) for i in range(len(ins))]}
        self.outputs = {
            'Out': [('out%d' % i, outs[i]) for i in range(len(outs))]
        }

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def get_dtype(self):
        return "float32"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        pass

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


@skip_check_grad_ci(
    reason="The backward test is not supported for float16 type on NPU.")
class TestMeshgridOpFP16(TestMeshgridOp):
    def get_dtype(self):
        return "float16"


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

        exe = fluid.Executor(place=fluid.NPUPlace(0))
        grid_x, grid_y = paddle.tensor.meshgrid(x, y)
        res_1, res_2 = exe.run(fluid.default_main_program(),
                               feed={'x': input_1,
                                     'y': input_2},
                               fetch_list=[grid_x, grid_y])

        self.assertTrue(np.allclose(res_1, out_1))
        self.assertTrue(np.allclose(res_2, out_2))


class TestMeshgridOp4(unittest.TestCase):
    def test_list_input(self):
        x = fluid.data(shape=[100], dtype='int32', name='x')
        y = fluid.data(shape=[200], dtype='int32', name='y')

        input_1 = np.random.randint(0, 100, [100, ]).astype('int32')
        input_2 = np.random.randint(0, 100, [200, ]).astype('int32')

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = fluid.Executor(place=fluid.NPUPlace(0))
        grid_x, grid_y = paddle.tensor.meshgrid([x, y])
        res_1, res_2 = exe.run(fluid.default_main_program(),
                               feed={'x': input_1,
                                     'y': input_2},
                               fetch_list=[grid_x, grid_y])

        self.assertTrue(np.allclose(res_1, out_1))
        self.assertTrue(np.allclose(res_2, out_2))


class TestMeshgridOp5(unittest.TestCase):
    def test_tuple_input(self):
        x = fluid.data(shape=[100], dtype='int32', name='x')
        y = fluid.data(shape=[200], dtype='int32', name='y')

        input_1 = np.random.randint(0, 100, [100, ]).astype('int32')
        input_2 = np.random.randint(0, 100, [200, ]).astype('int32')

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = fluid.Executor(place=fluid.NPUPlace(0))
        grid_x, grid_y = paddle.tensor.meshgrid((x, y))
        res_1, res_2 = exe.run(fluid.default_main_program(),
                               feed={'x': input_1,
                                     'y': input_2},
                               fetch_list=[grid_x, grid_y])

        self.assertTrue(np.allclose(res_1, out_1))
        self.assertTrue(np.allclose(res_2, out_2))


class TestMeshgridOp6(unittest.TestCase):
    def test_api_with_dygraph(self):
        paddle.disable_static(paddle.NPUPlace(0))
        input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
        input_4 = np.random.randint(0, 100, [200, ]).astype('int32')

        out_3 = np.reshape(input_3, [100, 1])
        out_3 = np.broadcast_to(out_3, [100, 200])
        out_4 = np.reshape(input_4, [1, 200])
        out_4 = np.broadcast_to(out_4, [100, 200])

        tensor_3 = paddle.to_tensor(input_3)
        tensor_4 = paddle.to_tensor(input_4)
        res_3, res_4 = paddle.tensor.meshgrid(tensor_3, tensor_4)

        self.assertTrue(np.allclose(res_3.numpy(), out_3))
        self.assertTrue(np.allclose(res_4.numpy(), out_4))
        paddle.enable_static()


class TestMeshgridOp7(unittest.TestCase):
    def test_api_with_dygraph_list_input(self):
        paddle.disable_static(paddle.NPUPlace(0))
        input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
        input_4 = np.random.randint(0, 100, [200, ]).astype('int32')

        out_3 = np.reshape(input_3, [100, 1])
        out_3 = np.broadcast_to(out_3, [100, 200])
        out_4 = np.reshape(input_4, [1, 200])
        out_4 = np.broadcast_to(out_4, [100, 200])

        tensor_3 = paddle.to_tensor(input_3)
        tensor_4 = paddle.to_tensor(input_4)
        res_3, res_4 = paddle.meshgrid([tensor_3, tensor_4])

        self.assertTrue(np.allclose(res_3.numpy(), out_3))
        self.assertTrue(np.allclose(res_4.numpy(), out_4))
        paddle.enable_static()


class TestMeshgridOp8(unittest.TestCase):
    def test_api_with_dygraph_tuple_input(self):
        paddle.disable_static(paddle.NPUPlace(0))
        input_3 = np.random.randint(0, 100, [100, ]).astype('int32')
        input_4 = np.random.randint(0, 100, [200, ]).astype('int32')

        out_3 = np.reshape(input_3, [100, 1])
        out_3 = np.broadcast_to(out_3, [100, 200])
        out_4 = np.reshape(input_4, [1, 200])
        out_4 = np.broadcast_to(out_4, [100, 200])

        tensor_3 = paddle.to_tensor(input_3)
        tensor_4 = paddle.to_tensor(input_4)
        res_3, res_4 = paddle.tensor.meshgrid((tensor_3, tensor_4))

        self.assertTrue(np.allclose(res_3.numpy(), out_3))
        self.assertTrue(np.allclose(res_4.numpy(), out_4))
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
