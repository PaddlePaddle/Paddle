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

x = np.random.random(x_shape).astype(dtype)
y = np.random.random(y_shape).astype(dtype)

# 定义矩阵X和Y
X = paddle.to_tensor(x, dtype='float32', stop_gradient=False)
Y = paddle.to_tensor(y, dtype='float32', stop_gradient=False)

# 定义一个简单的计算图：Z = X * Y
print(X)
print(Y)
Z = paddle.matmul(X, Y)
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

# import paddle

# data = paddle.to_tensor([1, 2, 3], dtype='int32')
# print(data)
# out = paddle.expand(data, shape=[2, 3])
# print(out)
