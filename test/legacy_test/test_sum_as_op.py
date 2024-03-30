# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

np.random.seed(100)
paddle.seed(100)


def get_reduce_dims(x, y):
    diff = len(x.shape) - len(y.shape)
    axis = []
    for i in range(diff):
        axis.append(i)
    for i in range(len(y.shape)):
        if y.shape[i] != x.shape[i + diff]:
            axis.append(i + diff)
    return axis


class TestSumAsOp(unittest.TestCase):
    def init_type(self):
        self.dtype = 'float64'

    def init_shape(self):
        self.shape_x = [300, 200, 600]
        self.shape_y = [200, 600]

    def init_data(self):
        self.x = np.random.random(self.shape_x).astype(self.dtype)
        self.y = np.random.random(self.shape_y).astype(self.dtype)

    def init(self):
        np.random.seed(2023)
        self.init_type()
        self.init_shape()
        self.init_data()

    def test_sum_as_dynamic(self):
        self.init()
        paddle.disable_static()
        x = paddle.to_tensor(self.x, stop_gradient=False)
        y = paddle.to_tensor(self.y)
        p = paddle.to_tensor(self.x, stop_gradient=False)

        out = paddle.sum_as(x, y)
        out.backward()
        x_grad = x.grad.numpy()

        reduce_dims = get_reduce_dims(self.x, self.y)
        ans = paddle.sum(p, axis=reduce_dims)
        ans.backward()
        p_grad = p.grad.numpy()

        # check forward
        np.testing.assert_allclose(out.numpy(), ans.numpy(), rtol=1e-6)
        # check backward
        np.testing.assert_allclose(x_grad, p_grad, rtol=1e-6)

    def test_sum_as_static(self):
        self.init()

        def base_net(flag=None):
            if flag == 'static':
                # static graph
                paddle.enable_static()
                main_program = paddle.static.Program()
                with paddle.static.program_guard(main_program):
                    x = paddle.static.data('x', self.shape_x, dtype=self.dtype)
                    y = paddle.static.data('y', self.shape_y, dtype=self.dtype)
                    x.stop_gradient = False
                    out = paddle.sum_as(x, y)
                    gradients = paddle.static.gradients(out, [x])
                    exe = paddle.static.Executor()

                    [fwd, dx] = exe.run(
                        feed={'x': self.x, 'y': self.y},
                        fetch_list=[out, gradients],
                    )
            else:
                # dynamic graph
                paddle.disable_static()
                x = paddle.to_tensor(self.x, stop_gradient=False)
                y = paddle.to_tensor(self.y)
                # axis = get_reduce_dims(x,y)
                # fwd = paddle.sum(x, axis=axis)
                fwd = paddle.sum_as(x, y)
                fwd.backward()
                dx = x.grad.numpy()
                fwd = fwd.numpy()
            return fwd, dx

        res_ref = base_net()
        res = base_net('static')
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)

    def test_sum_as_dynamic_to_static(self):
        self.init()
        paddle.core._set_prim_all_enabled(True)

        def func(x, y):
            return paddle.sum_as(x, y)

        static_func = paddle.jit.to_static(func, full_graph=True)

        # ==== dygraph computation ====
        x = paddle.to_tensor(self.x, stop_gradient=False)
        y = paddle.to_tensor(self.y)
        ref_out = func(x, y)
        ref_out.backward()
        ref_grad = x.grad.numpy()
        x.clear_gradient()

        # ==== to static compuatation ====
        actual_out = static_func(x, y)
        actual_out.backward()
        actual_grad = x.grad.numpy()

        paddle.core._set_prim_all_enabled(False)

        np.testing.assert_allclose(
            ref_out.numpy(), actual_out.numpy(), atol=1e-6, rtol=1e-6
        )

        np.testing.assert_allclose(ref_grad, actual_grad, atol=1e-6, rtol=1e-6)


class TestSumAsOp2(TestSumAsOp):
    def init_shape(self):
        self.shape_x = [300, 200, 600]
        self.shape_y = [600]


class TestSumAsOp3(TestSumAsOp):
    def init_type(self):
        self.dtype = 'float32'


# class TestSumAsOp4(TestSumAsOp):
#     def init_type(self):
#         self.dtype = 'bool'


class TestSumAsOp5(TestSumAsOp):
    def init_type(self):
        self.dtype = 'float16'


class TestSumAsOp6(TestSumAsOp):
    def init_type(self):
        self.dtype = 'uint16'


class TestSumAsOp7(TestSumAsOp):
    def init_type(self):
        self.dtype = 'int16'


# class TestSumAsOp8(TestSumAsOp):
#     def init_type(self):
#         self.dtype = 'int32'


class TestSumAsOp9(TestSumAsOp):
    def init_type(self):
        self.dtype = 'int64'


'''
# op_test
class TestSumAsOp4(OpTest):

    def setUp(self):
        self.init_dtype()
        self.init_input()
        self.init_attrs()
        self.calc_output()

        self.python_api = paddle.sum_as
        self.public_python_api = paddle.sum_as
        self.op_type = "sum_as"
        self.prim_op_type = "prim"
        self.inputs = {'x': self.x, 'y': self.y}
        self.outputs = {'out': self.out}
        self.if_enable_cinn()

    def init_dtype(self):
        self.dtype = np.float64

    def init_input(self):
        self.x = np.random.randn(300, 200, 600).astype(self.dtype)
        self.y = np.random.randn(200, 600).astype(self.dtype)

    def init_attrs(self):
        self.attrs = {'dim': [0]}

    def if_enable_cinn(self):
        pass

    def calc_output(self):
        self.out = self.x.sum(axis=tuple(self.attrs['dim']))

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ['x','y'],
            'out',
        )
'''

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
