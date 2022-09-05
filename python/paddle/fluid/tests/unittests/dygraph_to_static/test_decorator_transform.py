#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy
import paddle
import unittest
import os
import tempfile
import paddle.inference as paddle_infer
from paddle.fluid.framework import program_guard, Program
import numpy as np
from paddle.fluid import core
from functools import wraps


def deco1(fun):

    @wraps(fun)
    def inner(*args, **kwargs):
        return fun(*args, **kwargs)

    return inner


@deco1
def deco2(fun):

    @wraps(fun)
    def inner(*args, **kwargs):
        _x = 2
        if (_x < 1):
            print('small')
        else:
            print('large')
        _t = paddle.to_tensor([1])
        _tt = fun(*args, **kwargs)
        return paddle.add(_t, _tt)

    return inner


@deco2
def fun1(x, y=0):
    a = paddle.to_tensor([y])
    print('in fun1, x=%s' % (x))
    return a


@deco1
@deco2
def fun2(x, y=0):
    a = paddle.to_tensor([5])
    b = fun1(x, y)
    return paddle.add(a, b)


@paddle.jit.to_static
def case1(x, y=0):
    fun1(x, y)


@paddle.jit.to_static
def case2(x, y=0):
    fun2(x, y)


class TestDecoratorTransform(unittest.TestCase):

    def test_deco_transform(self):
        case1('case1', 5)

    def test_multi_deco_transform(self):
        case2('case2', 8)


if __name__ == '__main__':
    unittest.main()
