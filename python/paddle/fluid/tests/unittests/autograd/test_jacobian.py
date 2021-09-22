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

import paddle


def func_1(x):
    return paddle.matmul(x, x)


def func_2(x):
    return paddle.matmul(x, x), x * x


def func_3(x, y):
    return paddle.matmul(x, y)


def func_4(x, y):
    return paddle.matmul(x, y), x * y


def func_5(x, y):
    return paddle.matmul(x, y), x * x


x = paddle.ones(shape=(2, 2))
y = paddle.ones(shape=(2, 2))
x.stop_gradient = False
y.stop_gradient = False

z = paddle.autograd.jacobian(func_1, x)

print("z: ", z)
print("x.grad: ", x.grad)

z = paddle.autograd.jacobian(func_2, x)

print("z: ", z)
print("x.grad: ", x.grad)

z = paddle.autograd.jacobian(func_3, inputs=[x, y])

print("z: ", z)
print("x.grad: ", x.grad)

z = paddle.autograd.jacobian(func_4, inputs=[x, y], create_graph=True)

print("z: ", z)
print("x.grad: ", x.grad)

z = paddle.autograd.jacobian(func_5, inputs=[x, y], allow_unused=True)

print("z: ", z)
print("x.grad: ", x.grad)
