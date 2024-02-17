# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F

np.random.seed(10)


# 0D Tensor
def test0():
    x = paddle.to_tensor(0.2)
    out = F.softmax(x)


def test1():
    shape = [2, 3, 4]
    data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
    x = paddle.to_tensor(data)
    out = F.softmax(x, axis=1)  # axis 不能指向最后一维


def test2():
    shape = [2, 3, 1025]
    data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
    x = paddle.to_tensor(data)
    out = F.softmax(x, axis=-1)  # dim > 1024 || sizeof(T) > 4


def test3():
    shape = [2, 3, 120000]
    data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
    x = paddle.to_tensor(data)
    out = F.softmax(
        x, axis=-1, dtype='bfloat16'
    )  # 不支持bf16的设备 dim > 100000  dtype = 'bfloat16'


def test4():
    for i in range(11):
        shape = [2, 3, 2**i]
        data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
        x = paddle.to_tensor(data)
        out = F.softmax(x, axis=-1, dtype='float32')  # dtype = 'float32'


def test5():
    for i in range(2, 11):
        shape = [2, 3, 2**i]
        data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
        x = paddle.to_tensor(data)
        out = F.softmax(x, axis=-1, dtype='float16')  # dtype != 'float32'


def test6():
    shape = [2, 3, 2]
    data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
    x = paddle.to_tensor(data)
    out = F.softmax(x, axis=-1, dtype='float16')  # dtype != 'float32'
    for i in range(2, 11):
        shape = [2, 3, (2**i + 2)]
        data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
        x = paddle.to_tensor(data)
        out = F.softmax(x, axis=-1, dtype='float16')  # dtype != 'float32'


def test7():
    shape = [2, 3, 1]
    data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
    x = paddle.to_tensor(data)
    out = F.softmax(x, axis=-1, dtype='float16')  # dtype != 'float32'
    for i in range(1, 11):
        shape = [2, 3, (2**i + 1)]
        data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
        x = paddle.to_tensor(data)
        out = F.softmax(x, axis=-1, dtype='float16')  # dtype != 'float32'


def test8():
    shape = [180000000, 3, 4]
    data = np.random.uniform(-1.0, 1.0, shape).astype('float32')
    x = paddle.to_tensor(data)
    out = F.softmax(x, axis=1)  # axis 不能指向最后一维


test8()
