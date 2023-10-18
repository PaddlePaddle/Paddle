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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core


def meshgrid_wrapper(x):
    return paddle.tensor.meshgrid(x[0], x[1])


class TestMeshgridOp(OpTest):
    def setUp(self):
        self.op_type = "meshgrid"
        self.prim_op_type = "comp"
        self.python_api = meshgrid_wrapper
        self.public_python_api = meshgrid_wrapper
        self.init_data_type()
        self.init_inputs_and_outputs()
        self.python_out_sig = ['out0', 'out1']
        self.if_enable_cinn()

    def init_data_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output(check_prim=True)

    def test_check_grad(self):
        self.check_grad(['x0'], ['out0', 'out1'], check_prim=True)

    def init_inputs_and_outputs(self):
        self.shape = self.get_x_shape()
        ins = []
        outs = []
        for i in range(len(self.shape)):
            ins.append(np.random.random((self.shape[i],)).astype(self.dtype))

        for i in range(len(self.shape)):
            out_reshape = [1] * len(self.shape)
            out_reshape[i] = self.shape[i]
            out_temp = np.reshape(ins[i], out_reshape)
            outs.append(np.broadcast_to(out_temp, self.shape))
        self.inputs = {'X': [('x%d' % i, ins[i]) for i in range(len(ins))]}
        self.outputs = {
            'Out': [('out%d' % i, outs[i]) for i in range(len(outs))]
        }

    def get_x_shape(self):
        return [100, 200]

    def if_enable_cinn(self):
        # 拆解tile_grad导致cinn运行超时
        self.enable_cinn = False


class TestMeshgridOp2(TestMeshgridOp):
    def get_x_shape(self):
        return [100, 300]


class TestMeshgridOp2Fp16(TestMeshgridOp):
    def get_x_shape(self):
        return [100, 300]

    def init_data_type(self):
        self.dtype = np.float16


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestMeshgridOpBFP16OP(TestMeshgridOp):
    def init_data_type(self):
        self.data_type = np.uint16

    def init_inputs_and_outputs(self):
        self.shape = self.get_x_shape()
        ins = []
        outs = []
        for i in range(len(self.shape)):
            ins.append(np.random.random((self.shape[i],)).astype(self.dtype))

        for i in range(len(self.shape)):
            out_reshape = [1] * len(self.shape)
            out_reshape[i] = self.shape[i]
            out_temp = np.reshape(ins[i], out_reshape)
            outs.append(np.broadcast_to(out_temp, self.shape))
        self.inputs = {
            'X': [
                ('x%d' % i, convert_float_to_uint16(ins[i]))
                for i in range(len(ins))
            ]
        }
        self.outputs = {
            'Out': [
                ('out%d' % i, convert_float_to_uint16(outs[i]))
                for i in range(len(outs))
            ]
        }

    def if_enable_cinn(self):
        self.enable_cinn = False

    def test_check_output(self):
        self.check_output_with_place(place=paddle.CUDAPlace(0))

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.CUDAPlace(0), ['x0'], ['out0', 'out1'], check_prim=True
        )


class TestMeshgridOp3(unittest.TestCase):
    def test_api(self):
        x = paddle.static.data(shape=[100], dtype='int32', name='x')
        y = paddle.static.data(shape=[200], dtype='int32', name='y')

        input_1 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype('int32')
        input_2 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype('int32')

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = base.Executor(place=base.CPUPlace())
        grid_x, grid_y = paddle.tensor.meshgrid(x, y)
        res_1, res_2 = exe.run(
            base.default_main_program(),
            feed={'x': input_1, 'y': input_2},
            fetch_list=[grid_x, grid_y],
        )
        np.testing.assert_array_equal(res_1, out_1)
        np.testing.assert_array_equal(res_2, out_2)


class TestMeshgridOp4(unittest.TestCase):
    def test_list_input(self):
        x = paddle.static.data(shape=[100], dtype='int32', name='x')
        y = paddle.static.data(shape=[200], dtype='int32', name='y')

        input_1 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype('int32')
        input_2 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype('int32')

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = base.Executor(place=base.CPUPlace())
        grid_x, grid_y = paddle.tensor.meshgrid([x, y])
        res_1, res_2 = exe.run(
            base.default_main_program(),
            feed={'x': input_1, 'y': input_2},
            fetch_list=[grid_x, grid_y],
        )

        np.testing.assert_array_equal(res_1, out_1)
        np.testing.assert_array_equal(res_2, out_2)


class TestMeshgridOp5(unittest.TestCase):
    def test_tuple_input(self):
        x = paddle.static.data(shape=[100], dtype='int32', name='x')
        y = paddle.static.data(shape=[200], dtype='int32', name='y')

        input_1 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype('int32')
        input_2 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype('int32')

        out_1 = np.reshape(input_1, [100, 1])
        out_1 = np.broadcast_to(out_1, [100, 200])
        out_2 = np.reshape(input_2, [1, 200])
        out_2 = np.broadcast_to(out_2, [100, 200])

        exe = base.Executor(place=base.CPUPlace())
        grid_x, grid_y = paddle.tensor.meshgrid((x, y))
        res_1, res_2 = exe.run(
            base.default_main_program(),
            feed={'x': input_1, 'y': input_2},
            fetch_list=[grid_x, grid_y],
        )

        np.testing.assert_array_equal(res_1, out_1)
        np.testing.assert_array_equal(res_2, out_2)


class TestMeshgridOp6(unittest.TestCase):
    def test_api_with_dygraph(self):
        input_3 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype('int32')
        input_4 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype('int32')

        with base.dygraph.guard():
            tensor_3 = base.dygraph.to_variable(input_3)
            tensor_4 = base.dygraph.to_variable(input_4)
            res_3, res_4 = paddle.tensor.meshgrid(tensor_3, tensor_4)

            np.testing.assert_array_equal(res_3.shape, [100, 200])
            np.testing.assert_array_equal(res_4.shape, [100, 200])


class TestMeshgridOp7(unittest.TestCase):
    def test_api_with_dygraph_list_input(self):
        input_3 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype('int32')
        input_4 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype('int32')

        with base.dygraph.guard():
            tensor_3 = base.dygraph.to_variable(input_3)
            tensor_4 = base.dygraph.to_variable(input_4)
            res_3, res_4 = paddle.tensor.meshgrid([tensor_3, tensor_4])

            np.testing.assert_array_equal(res_3.shape, [100, 200])
            np.testing.assert_array_equal(res_4.shape, [100, 200])


class TestMeshgridOp8(unittest.TestCase):
    def test_api_with_dygraph_tuple_input(self):
        input_3 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype('int32')
        input_4 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype('int32')

        with base.dygraph.guard():
            tensor_3 = base.dygraph.to_variable(input_3)
            tensor_4 = base.dygraph.to_variable(input_4)
            res_3, res_4 = paddle.tensor.meshgrid((tensor_3, tensor_4))

            np.testing.assert_array_equal(res_3.shape, [100, 200])
            np.testing.assert_array_equal(res_4.shape, [100, 200])


class TestMeshGrid_ZeroDim(TestMeshgridOp):
    def init_inputs_and_outputs(self):
        self.shape = self.get_x_shape()
        ins = []
        outs = []
        ins.append(np.random.random([]).astype(self.dtype))
        ins.append(np.random.random([2]).astype(self.dtype))
        ins.append(np.random.random([3]).astype(self.dtype))
        for i in range(len(self.shape)):
            out_reshape = [1] * len(self.shape)
            out_reshape[i] = self.shape[i]
            out_temp = np.reshape(ins[i], out_reshape)
            outs.append(np.broadcast_to(out_temp, self.shape))
        self.inputs = {'X': [('x%d' % i, ins[i]) for i in range(len(ins))]}
        self.outputs = {
            'Out': [('out%d' % i, outs[i]) for i in range(len(outs))]
        }

    def get_x_shape(self):
        return [1, 2, 3]

    def if_enable_cinn(self):
        self.enable_cinn = False


class TestMeshgridEager(unittest.TestCase):
    def test_dygraph_api(self):
        input_1 = np.random.randint(
            0,
            100,
            [
                100,
            ],
        ).astype('int32')
        input_2 = np.random.randint(
            0,
            100,
            [
                200,
            ],
        ).astype('int32')

        with base.dygraph.guard():
            tensor_1 = base.dygraph.to_variable(input_1)
            tensor_2 = base.dygraph.to_variable(input_2)
            tensor_1.stop_gradient = False
            tensor_2.stop_gradient = False
            res_1, res_2 = paddle.tensor.meshgrid((tensor_1, tensor_2))
            sum = paddle.add_n([res_1, res_2])
            sum.backward()
            tensor_eager_1 = base.dygraph.to_variable(input_1)
            tensor_eager_2 = base.dygraph.to_variable(input_2)
            tensor_eager_1.stop_gradient = False
            tensor_eager_2.stop_gradient = False
            res_eager_1, res_eager_2 = paddle.tensor.meshgrid(
                (tensor_eager_1, tensor_eager_2)
            )
            sum_eager = paddle.add_n([res_eager_1, res_eager_2])
            sum_eager.backward()
            self.assertEqual(
                (tensor_1.grad.numpy() == tensor_eager_1.grad.numpy()).all(),
                True,
            )
            self.assertEqual(
                (tensor_2.grad.numpy() == tensor_eager_2.grad.numpy()).all(),
                True,
            )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
