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

import unittest

import sys
import numpy as np
from eager_op_test import OpTest

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.inference as paddle_infer

'''
- TestCummaxAPT
 ---- cpu
    ---- dygraph
    ---- static
 ---- gpu
    ---- dygraph
    ---- static
- TestCummaxOP
 ---- axis
 ---- dtype of input x
- TestCummaxAPIFP16(not implemented)
- TestCummaxOPFP16(not implemented)
'''

def cummax_dim2(arr, axis=None):
    if axis is None:
        arr = arr.flatten()
        cummax = np.maximum.accumulate(arr)
        shape = arr.shape
        indices = np.zeros(shape).astype('int32')
        max_val = -sys.maxsize
        max_ind = 0
        for i in range(shape[0]):
            if arr[i] > max_val:
                max_val = max(arr[i], max_val)
                max_ind = i
                indices[i] = i
            else:
                indices[i] = max_ind
    else:
        cummax = np.maximum.accumulate(arr, axis)
        shape = arr.shape
        indices = np.zeros(shape).astype('int32')
        if axis < 0: axis = axis + len(shape)
        if axis == 0:
            for j in range(shape[1]):
                max_ind = 0
                max_val = -sys.maxsize
                for i in range(shape[0]):
                    if arr[i][j] > max_val:
                        max_val = arr[i][j]
                        max_ind = i
                        indices[i][j] = i
                    else:
                        indices[i][j] = max_ind
        elif axis == 1:
            for i in range(shape[0]):
                max_ind = 0
                max_val = -sys.maxsize
                for j in range(shape[1]):
                    if arr[i][j] > max_val:
                        max_val = arr[i][j]
                        max_ind = j
                        indices[i][j] = j
                    else:
                        indices[i][j] = max_ind
        else:
            raise Exception("unfeasible axis")
    return cummax, indices


class TestCummaxOp(OpTest):
    def setUp(self):
        self.op_type = "cummax"
        self.python_api = paddle.cummax
        self.dtype = 'float64'
        self.shape = (10, 10)
        self.axis = -1
        self.indices_type = 3 #{2: "int32", 3: "int64"}
        self.flatten = False
        self.set_attrs()

        self.input_data = np.random.random(self.shape).astype(self.dtype)
        self.np_res, self.np_ind = cummax_dim2(self.input_data, axis=self.axis)
        self.set_input()

        self.enable_cinn = False
        self.attrs = {'axis': self.axis, 'ind_dtype': self.indices_type, 'flatten': self.flatten}
        self.inputs = {'x': self.input_data}
        self.outputs = {'out': self.np_res, 'indices': self.np_ind}
    
    def set_attrs(self):
        pass

    def set_input(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x'], 'out')


# class TestCummaxOpAxis1(TestCummaxOp):
#     def set_attrs(self):
#         self.axis = 0


# class TestCummaxOpAxis2(TestCummaxOp):
#     def set_attrs(self):
#         self.axis = -2


# class TestCummaxOpFlatten(TestCummaxOp):
#     def set_attrs(self):
#         self.shape = (100)
#         self.flatten = True
    
#     def set_input(self):
#         self.np_res, self.np_ind = cummax_dim2(self.input_data)


# class TestCummaxOpIndexType(TestCummaxOp):
#     def set_attrs(self):
#         self.indices_type = 2


class TestCummaxAPI(unittest.TestCase):
    def run_cases(self):
        data_np = np.arange(12).reshape(3, 4).astype(np.int32)
        data = paddle.to_tensor(data_np)

        y, indices = paddle.cummax(data)
        z, ind = cummax_dim2(data_np)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

        y, indices = paddle.cummax(data, axis=0)
        z, ind = cummax_dim2(data_np, axis=0)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

        y, indices = paddle.cummax(data, axis=-1)
        z, ind = cummax_dim2(data_np, axis=-1)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

        y, indices = paddle.cummax(data, dtype=np.int32)
        self.assertTrue(indices.dtype == core.VarDesc.VarType.INT32)

        y, indices = paddle.cummax(data, axis=-2)
        z, ind = cummax_dim2(data_np, axis=-2)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

        data_np = np.arange(12).reshape(3, 4).astype(np.int32)
        data = paddle.to_tensor(data_np)
        y, indices = paddle.cummax(data, axis=0)
        z, ind = cummax_dim2(data_np, axis=0)
        np.testing.assert_array_equal(z, y.numpy())
        np.testing.assert_array_equal(ind, indices.numpy())

    def run_static(self, use_gpu=False):
        with fluid.program_guard(fluid.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data('x', [100, 100])
            y1, indices1 = paddle.cummax(x)
            y2, indices2 = paddle.cummax(x, axis=0)
            y3, indices3 = paddle.cummax(x, axis=-1)
            y4, indices4 = paddle.cummax(x, dtype='int64')
            y5, indices5 = paddle.cummax(x, dtype=np.int32)
            y6, indices6 = paddle.cummax(x, axis=-2)

            place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            print(y1.name)
            out = exe.run(
                feed={'x': data_np},
                fetch_list=[
                    y1.name,
                    y2.name,
                    y3.name,
                    y4.name,
                    y5.name,
                    y6.name,
                ],
            )

            z, ind = cummax_dim2(data_np)
            np.testing.assert_allclose(z, out[0][0], rtol=1e-05)
            z, ind = cummax_dim2(data_np, axis=0)
            np.testing.assert_allclose(z, out[1][0], rtol=1e-05)
            z, ind = cummax_dim2(data_np, axis=-1)
            np.testing.assert_allclose(z, out[2][0], rtol=1e-05)
            self.assertTrue(out[3][1].dtype == np.int64)
            self.assertTrue(out[4][1].dtype == np.int32)
            z, ind = cummax_dim2(data_np, axis=-2)
            np.testing.assert_allclose(z, out[5][0], rtol=1e-05)

    def test_cpu(self):
        paddle.disable_static(paddle.fluid.CPUPlace())
        print("dynamic run start in cpu")
        self.run_cases()
        paddle.enable_static()
        print("static run start in cpu")
        self.run_static()

    def test_gpu(self):
        if not fluid.core.is_compiled_with_cuda():
            return
        paddle.disable_static(paddle.fluid.CUDAPlace(0))
        self.run_cases()
        paddle.enable_static()

        self.run_static(use_gpu=True)


# class TestCummaxFP16(unittest.TestCase):
#     def check_main(self, x_np, dtype):
#         paddle.disable_static()
#         x = paddle.to_tensor(x_np.astype(dtype))
#         x.stop_gradient = False
#         y = paddle.cummax(x, dtype=dtype)
#         x_g = paddle.grad(y, [x])
#         y_np = y.numpy().astype('float32')
#         x_g_np = x_g[0].numpy().astype('float32')
#         paddle.enable_static()
#         return y_np, x_g_np

#     def test_main(self):
#         if not paddle.is_compiled_with_cuda():
#             return

#         np.random.seed(20)
#         x_np = np.random.random([10, 12])
#         y_np_1, x_g_np_1 = self.check_main(x_np, 'float16')
#         y_np_2, x_g_np_2 = self.check_main(x_np, 'float32')

#         np.testing.assert_allclose(y_np_1, y_np_2, rtol=1e-03)
#         np.testing.assert_allclose(x_g_np_1, x_g_np_2, rtol=1e-03)


if __name__ == '__main__':
    unittest.main()
