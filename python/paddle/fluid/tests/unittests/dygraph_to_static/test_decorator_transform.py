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

import paddle
import unittest
import numpy as np
import decos
from functools import wraps


def deco1(func):

    @wraps(func)
    def inner(*args, **kwargs):
        print('in deco1, added 1')
        _x = 2
        if (_x < 1):
            _x += 1
        else:
            _x -= 1
        _t = paddle.to_tensor([1])
        _tt = func(*args, **kwargs)
        return paddle.add(_t, _tt)

    return inner


def deco2(fun):

    @wraps(fun)
    def inner(*args, **kwargs):
        print('in deco2, added 2')
        _t = paddle.to_tensor([2])
        _tt = fun(*args, **kwargs)
        return paddle.add(_t, _tt)

    return inner


def deco3(x=3):

    def inner_deco(func):

        @wraps(func)
        def inner(*args, **kwargs):
            print('in deco3, added {}'.format(x))
            _t = paddle.to_tensor(x)
            _tt = func(*args, **kwargs)
            return paddle.add(_t, _tt)

        return inner

    return inner_deco


def deco4(func=None, x=0):

    def decorated(pyfunc):

        @wraps(pyfunc)
        def inner_deco(*args, **kwargs):
            print('in deco4, added {}'.format(x))
            _t = paddle.to_tensor(x)
            _tt = pyfunc(*args, **kwargs)
            return paddle.add(_t, _tt)

        return inner_deco

    if func == None:
        return decorated
    return decorated(func)


@deco2
def fun1(x, y=0):
    a = paddle.to_tensor(y)
    print('in fun1, x=%d' % (x))
    return a


@deco1
@deco2
def fun2(x, y=0):
    a = paddle.to_tensor(y)
    print('in fun2, x=%d' % (x))
    return a


@deco3(3)
def fun3(x, y=0):
    a = paddle.to_tensor(y)
    print('in fun3, x=%d' % (x))
    return a


@deco4(x=4)
def fun4(x, y=0):
    a = paddle.to_tensor(y)
    print('in fun4, x=%d' % (x))
    return a


@deco2
@deco4(x=5)
def fun5(x, y=0):
    a = paddle.to_tensor(y)
    print('in fun5, x=%d' % (x))
    return a


@decos.deco1
@decos.deco2(2)
def fun6(x, y=0):
    a = paddle.to_tensor(y)
    print('in fun6, x=%d' % (x))
    return a


@paddle.jit.to_static
def forward():
    funcs = [fun1, fun2, fun3, fun4, fun5, fun6]
    out = []
    for idx, fun in enumerate(funcs):
        out.append(fun(idx + 1, idx + 1))
    return out


class TestDecoratorTransform(unittest.TestCase):

    def test_deco_transform(self):
        outs = forward()
        np.testing.assert_allclose(outs[0], np.array(3), rtol=1e-05)
        np.testing.assert_allclose(outs[1], np.array(5), rtol=1e-05)
        np.testing.assert_allclose(outs[2], np.array(6), rtol=1e-05)
        np.testing.assert_allclose(outs[3], np.array(8), rtol=1e-05)
        np.testing.assert_allclose(outs[4], np.array(12), rtol=1e-05)
        np.testing.assert_allclose(outs[5], np.array(9), rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
