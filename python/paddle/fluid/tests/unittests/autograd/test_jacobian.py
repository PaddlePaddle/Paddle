# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.compat as cpt
from utils import _compute_numerical_jacobian

# class TestJacobian(unittest.TestCase):
#     @classmethod
#     def setUpClass(self):
#         self.shape = (4, 4)
#         self.dtype = 'float32'
#         self.np_dtype = np.float32
#         self.numerical_delta = 1e-4
#         self.rtol = 1e-3
#         self.atol = 1e-3
#         self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
#         self.y = paddle.rand(shape=self.shape, dtype=self.dtype)

#     def test_single_input_and_single_output(self):
#         def func(x):
#             return paddle.matmul(x, x)

#         numerical_jacobian = _compute_numerical_jacobian(
#             func, self.x, self.numerical_delta, self.np_dtype)
#         self.x.stop_gradient = False
#         jacobian = paddle.autograd.jacobian(func, self.x)
#         assert np.allclose(jacobian.numpy(), numerical_jacobian[0][0],
#                            self.rtol, self.atol)

#     def test_single_input_and_multi_output(self):
#         def func(x):
#             return paddle.matmul(x, x), x * x

#         numerical_jacobian = _compute_numerical_jacobian(
#             func, self.x, self.numerical_delta, self.np_dtype)
#         self.x.stop_gradient = False
#         jacobian = paddle.autograd.jacobian(func, self.x)
#         for i in range(len(jacobian)):
#             assert np.allclose(jacobian[i].numpy(), numerical_jacobian[i][0],
#                                self.rtol, self.atol)

#     def test_multi_input_and_single_output(self):
#         def func(x, y):
#             return paddle.matmul(x, y)

#         numerical_jacobian = _compute_numerical_jacobian(
#             func, [self.x, self.y], self.numerical_delta, self.np_dtype)
#         self.x.stop_gradient = False
#         self.y.stop_gradient = False
#         jacobian = paddle.autograd.jacobian(func, [self.x, self.y])
#         for j in range(len(jacobian)):
#             assert np.allclose(jacobian[j].numpy(), numerical_jacobian[0][j],
#                                self.rtol, self.atol)

#     def test_multi_input_and_multi_output(self):
#         def func(x, y):
#             return paddle.matmul(x, y), x * y

#         numerical_jacobian = _compute_numerical_jacobian(
#             func, [self.x, self.y], self.numerical_delta, self.np_dtype)
#         self.x.stop_gradient = False
#         self.y.stop_gradient = False
#         jacobian = paddle.autograd.jacobian(func, [self.x, self.y])
#         for i in range(len(jacobian)):
#             for j in range(len(jacobian[0])):
#                 assert np.allclose(jacobian[i][j].numpy(),
#                                    numerical_jacobian[i][j], self.rtol,
#                                    self.atol)

#     def test_allow_unused_false(self):
#         def func(x, y):
#             return paddle.matmul(x, x)

#         try:
#             self.x.stop_gradient = False
#             self.y.stop_gradient = False
#             jacobian = paddle.autograd.jacobian(func, [self.x, self.y])
#         except ValueError as e:
#             error_msg = cpt.get_exception_message(e)
#             assert error_msg.find("allow_unused") > 0

#     def test_allow_unused_true(self):
#         def func(x, y):
#             return paddle.matmul(x, x)

#         numerical_jacobian = _compute_numerical_jacobian(
#             func, [self.x, self.y], self.numerical_delta, self.np_dtype)
#         self.x.stop_gradient = False
#         self.y.stop_gradient = False
#         jacobian = paddle.autograd.jacobian(
#             func, [self.x, self.y], allow_unused=True)
#         assert np.allclose(jacobian[0].numpy(), numerical_jacobian[0][0],
#                            self.rtol, self.atol)
#         assert jacobian[1] is None

#     def test_create_graph_false(self):
#         def func(x, y):
#             return paddle.matmul(x, y)

#         numerical_jacobian = _compute_numerical_jacobian(
#             func, [self.x, self.y], self.numerical_delta, self.np_dtype)
#         self.x.stop_gradient = False
#         self.y.stop_gradient = False
#         jacobian = paddle.autograd.jacobian(func, [self.x, self.y])
#         for j in range(len(jacobian)):
#             assert jacobian[j].stop_gradient == True
#             assert np.allclose(jacobian[j].numpy(), numerical_jacobian[0][j],
#                                self.rtol, self.atol)
#         try:
#             paddle.grad(jacobian[0], [self.x, self.y])
#         except RuntimeError as e:
#             error_msg = cpt.get_exception_message(e)
#             assert error_msg.find("has no gradient") > 0

#     def test_create_graph_true(self):
#         def func(x, y):
#             return paddle.matmul(x, y)

#         numerical_jacobian = _compute_numerical_jacobian(
#             func, [self.x, self.y], self.numerical_delta, self.np_dtype)
#         self.x.stop_gradient = False
#         self.y.stop_gradient = False
#         jacobian = paddle.autograd.jacobian(
#             func, [self.x, self.y], create_graph=True)
#         for j in range(len(jacobian)):
#             assert jacobian[j].stop_gradient == False
#             assert np.allclose(jacobian[j].numpy(), numerical_jacobian[0][j],
#                                self.rtol, self.atol)
#         double_grad = paddle.grad(jacobian[0], [self.x, self.y])
#         assert double_grad is not None

# class TestJacobianFloat64(TestJacobian):
#     @classmethod
#     def setUpClass(self):
#         self.shape = (4, 4)
#         self.dtype = 'float64'
#         self.np_dtype = np.float64
#         self.numerical_delta = 1e-7
#         self.rtol = 1e-7
#         self.atol = 1e-7
#         self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
#         self.y = paddle.rand(shape=self.shape, dtype=self.dtype)


class TestJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.shape = (4, 2)
        self.weight_shape = (2, 4)
        self.other_shape = (4, 2)
        self.dtype = 'float32'
        self.np_dtype = np.float32
        self.numerical_delta = 1e-4
        self.rtol = 1e-3
        self.atol = 1e-3
        paddle.seed(123)
        self.x = paddle.rand(shape=self.shape, dtype=self.dtype)
        self.weight = paddle.rand(shape=self.weight_shape, dtype=self.dtype)
        self.other = paddle.rand(shape=self.other_shape, dtype=self.dtype)

    def _stack_tensor_or_return_none(origin_list):
        assert len(origin_list) > 0, "Can't not stack an empty list"
        return paddle.stack(
            origin_list, axis=0) if isinstance(origin_list[0],
                                               paddle.Tensor) else None

    def test_single_input_and_single_output(self):
        def func(x):
            return paddle.matmul(paddle.matmul(x, self.weight), self.other)

        self.x.stop_gradient = False
        jacobian = paddle.autograd.functional.batch_jacobian(
            func,
            self.x, )
        print(jacobian)

        numerical_jacobian = _compute_numerical_jacobian(
            func, [self.x], self.numerical_delta, self.np_dtype)
        print(numerical_jacobian)
        d_values = paddle.reshape(
            paddle.to_tensor(numerical_jacobian), shape=[4, 2, 4, 2])

        jac_i = list([] for _ in range(2))
        for j in range(2):
            for k in range(2):
                row_k = paddle.diag(d_values[:, j, :, k])
                jac_i[j].append(paddle.reshape(row_k, shape=[-1]))
                print("row_k:", j, " ", k, " ", row_k)

        # print("jac_i:", jac_i) 
        # print(numerical_jacobian_tuple)
        # numerical_jacobina_tuple = tuple()
        # jac_i = list([] for _ in range(2))
        # for j in range(2):
        #     for k in range(2):
        #         row_k = paddle.diag(d_values[:, j, :, k])
        #         jac_i[j].append(
        #             paddle.reshape(
        #                 row_k, shape=[-1])
        #             if isinstance(row_k[j], paddle.Tensor) else None)
        #         # jac_i
        #         # print("diag:", paddle.diag(d_values[:, j, :, k])) # 提取对角线元素
        # numerical_jacobina_tuple += (tuple(self._stack_tensor_or_return_none(jac_i_j) for jac_i_j in jac_i), )
        # print(numerical_jacobina_tuple)
        # print(jac_i[0])


if __name__ == "__main__":
    unittest.main()
