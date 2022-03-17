# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


def reduce(x):
    print(x)
    return paddle.sum(x)


def reduce_dim(x):
    return paddle.sum(x, axis=0)


def matmul(x, y):
    return paddle.matmul(x, y)


def mul(x, y):
    return x * y


def pow(x, y):
    return paddle.pow(x, y)


def o2(x, y):
    return paddle.multiply(x, y), paddle.matmul(x, y.t())


def unuse(x, y):
    return paddle.sum(x)


def nested(x):
    def inner(y):
        return x * y

    return inner


def square(x):
    return x * x
