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


def sum_as_net(x, y):
    out_dtype = x.dtype
    diff = len(x.shape) - len(y.shape)
    axis = []
    for i in range(diff):
        axis.append(i)
    for i in range(len(y.shape)):
        if y.shape[i] != x.shape[i + diff]:
            axis.append(i + diff)
    return np.sum(x, axis=tuple(axis), dtype=out_dtype)


# class TestSumAsOp(OpTest):
#     def setUp(self):
#         self.op_type = "sum_as"
#         self.python_api = paddle.sum_as
#         self.init_config()
#         self.inputs = {'x': self.x, 'y': self.y}
#         self.target = sum_as_net(self.inputs['x'], self.inputs['y'])
#         self.outputs = {'out': self.target}

#     def test_check_output(self):
#         self.check_output()

#     def test_check_grad(self):
#         self.check_grad(['x'], ['out'])

#     def init_config(self):
#         self.x = np.random.randn(300, 200, 600).astype('float64')
#         self.y = np.random.randn(200, 600).astype('float64')


class TestSumAsOp2(unittest.TestCase):
    def test_sum_as(self):
        np.random.seed(2023)
        data = np.random.randn(300, 200, 600).astype('float64')
        x = paddle.to_tensor(data, stop_gradient=False)
        y = paddle.to_tensor(np.random.randn(200, 600).astype('float64'))

        # forward
        p = paddle.to_tensor(data, stop_gradient=False)
        reduce_dims = [0]
        out = paddle.sum_as(x, y)
        ans = paddle.sum(p, axis=reduce_dims)
        np.testing.assert_allclose(out.numpy(), ans.numpy(), rtol=1e-5)

        # backward
        out.backward()
        x_grad = x.grad.numpy()
        p.backward()
        p_grad = p.grad.numpy()
        np.testing.assert_allclose(x_grad, p_grad, rtol=1e-5)


class TestSumAsOp3(unittest.TestCase):
    def test_sum_as_static(self):
        np.random.seed(2023)
        paddle.disable_static()

        def forward(x, y):
            return paddle.sum_as(x, y)

        paddle.disable_static()
        data = np.random.randn(300, 200, 600).astype('float64')
        x = paddle.to_tensor(data, stop_gradient=False)
        y = paddle.to_tensor(np.random.randn(200, 600).astype('float64'))

        out_dynamic = forward(x, y)
        out_dynamic_result = out_dynamic.numpy()
        out_dynamic.backward()
        x_dynamic_grad = x.grad.numpy()
        x.stop_gradient = False

        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            x_static = paddle.static.data(
                name='x', shape=[300, 200, 600], dtype='float64'
            )
            y_static = paddle.static.data(
                name='y', shape=[200, 600], dtype='float64'
            )
            out_static = forward(x_static, y_static)
            loss = paddle.mean(out_static)

        exe = paddle.static.Executor()
        x_np = data
        y_np = y.numpy()
        res = exe.run(
            paddle.static.default_main_program(),
            fetch_list=[loss, out_static],
            feed={'x': x_np, 'y': y_np},
        )
        loss_val, out_static_result = res

        np.testing.assert_allclose(
            out_static_result, out_dynamic_result, rtol=1e-5
        )

        exe.run(
            paddle.static.default_main_program(),
            fetch_list=[x_static.grad],
            feed={'x': x_np, 'y': y_np},
        )
        x_static_grad = x_static.grad.numpy()
        np.testing.assert_allclose(x_static_grad, x_dynamic_grad, rtol=1e-5)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
