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

import numpy as np

import paddle

x_shape = (1, 3)
y_shape = (3, 1)

dtype = "float32"

# x = np.mat([[1.0, 2.0, 3.0],
#             [4.0, 5.0, 6.0]])

# y = np.mat([[1.0, 2.0],
#             [3.0, 4.0],
#             [5.0, 6.0]])

x = np.random.rand(1, 1, 5, 1)
y = np.random.rand(
    5,
)

# 定义矩阵X和Y
X = paddle.to_tensor(x, dtype='float32', stop_gradient=False)
Y = paddle.to_tensor(y, dtype='float32', stop_gradient=False)

# 定义一个简单的计算图：Z = X * Y
print(X)
print(Y)
Z = paddle.matmul(X, Y, transpose_x=True)
print(Z)

# # 创建一个用于保存梯度的tensor，并设置requires_grad=True来开启梯度计算
# X = paddle.to_tensor(X, stop_gradient=False)
# Y = paddle.to_tensor(Y, stop_gradient=False)

# 计算Z相对于X和Y的梯度
Z.backward()

# 获取X和Y的梯度
grad_X = X.grad
grad_Y = Y.grad

print("梯度相对于X:")
print(grad_X)
print("梯度相对于Y:")
print(grad_Y)

x1 = x.copy()
y1 = y.copy()
X1 = paddle.to_tensor(x1, dtype='float32', stop_gradient=False)
Y1 = paddle.to_tensor(y1, dtype='float32', stop_gradient=False)
z1 = np.ones((Z.shape), dtype="float32")
Z1 = paddle.to_tensor(z1, dtype='float32', stop_gradient=False)

if len(Z1.shape) < len(Y1.shape):
    Z1 = paddle.unsqueeze(Z1, axis=len(Z1.shape) - 1)
if len(Z1.shape) < len(X1.shape):
    Z1 = paddle.unsqueeze(Z1, axis=len(Z1.shape))
if len(X1.shape) < 2:
    X1 = paddle.unsqueeze(X1, axis=0)
if len(Y1.shape) < 2:
    Y1 = paddle.unsqueeze(Y1, axis=1)


print(X1)
print(Y1)
print(Z1)

check_grad_x = paddle.matmul(Z1, Y1, transpose_y=True)
# print(check_grad_x)
reduce_dim = [0, 1, 2]
check_grad_x = paddle.sum(check_grad_x, reduce_dim)
# check_grad_x = paddle.transpose(check_grad_x, [0, 1, 3, 2])
print(check_grad_x)


# check_grad_y = paddle.matmul(X1, Z1, transpose_x=False)

# check_grad_y = paddle.transpose(check_grad_y, [0, 1, 3, 2])
# print(check_grad_y)
# Y1_slice = paddle.slice(Y1, axes=[0, 1, 2, 3], starts=[0, 0, 0, 0], ends=[1, 1, 5, 2])
# print(Y1_slice)
# print(check_grad_x)


# z1 = np.mat([1, 1, 1, 1])

# x1 = np.mat([[1.0, 2.0, 3.0]])

# y1 = np.mat([[1.0, 2.0, 3.0, 4.0],
#             [3.0, 4.0, 3.0, 4.0],
#             [5.0, 6.0, 3.0, 4.0]])

# X1 = paddle.to_tensor(x1, dtype='float32', stop_gradient=True)
# Y1 = paddle.to_tensor(y1, dtype='float32', stop_gradient=True)

# X1_sum = paddle.sum(Y1, axis=1, keepdim=False)
# X1_sum_un = paddle.unsqueeze(X1_sum, axis=0)
# X1_expand = paddle.expand(X1_sum_un, [X1.shape[0], X1_sum.shape[0]])
# X1_reshape = paddle.reshape(X1_expand, X1.shape)
# print(X1_reshape)

# Y1_sum = paddle.sum(X1, axis=0, keepdim=False)
# Y1_sum_un = paddle.unsqueeze(Y1_sum, axis=1)
# Y1_expand = paddle.expand(Y1_sum_un, [Y1_sum.shape[0], Y1.shape[1]])
# Y1_reshape = paddle.reshape(Y1_expand, Y1.shape)
# print(Y1_reshape)
