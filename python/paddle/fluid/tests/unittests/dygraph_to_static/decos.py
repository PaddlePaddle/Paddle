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

import numpy
import paddle

from functools import wraps


def deco1(fun):

    @wraps(fun)
    def inner(*args, **kwargs):
        print('in decos.deco1, added 1')
        _t = paddle.to_tensor([1])
        _tt = fun(*args, **kwargs)
        return paddle.add(_t, _tt)

    return inner


def deco2(x=0):

    def inner_deco(func):

        @wraps(func)
        def inner(*args, **kwargs):
            print('in decos.deco2, added {}'.format(x))
            _t = paddle.to_tensor(x)
            _tt = func(*args, **kwargs)
            return paddle.add(_t, _tt)

        return inner

    return inner_deco
