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

import contextlib
import sys

import numpy as np
import paddle

DEVICES = [paddle.CPUPlace()]
if paddle.is_compiled_with_cuda():
    DEVICES.append(paddle.CUDAPlace(0))

DEFAULT_DTYPE = 'float64'

TEST_CASE_NAME = 'suffix'
# All test case will use float64 for compare percision, refs:
# https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64
RTOL = {
    'float32': 1e-03,
    'complex64': 1e-3,
    'float64': 1e-5,
    'complex128': 1e-5
}
ATOL = {'float32': 0.0, 'complex64': 0, 'float64': 0.0, 'complex128': 0}


def xrand(shape=(10, 10, 10), dtype=DEFAULT_DTYPE, min=1.0, max=10.0):
    return ((np.random.rand(*shape).astype(dtype)) * (max - min) + min)


def place(devices, key='place'):
    def decorate(cls):
        module = sys.modules[cls.__module__].__dict__
        raw_classes = {
            k: v
            for k, v in module.items() if k.startswith(cls.__name__)
        }

        for raw_name, raw_cls in raw_classes.items():
            for d in devices:
                test_cls = dict(raw_cls.__dict__)
                test_cls.update({key: d})
                new_name = raw_name + '.' + d.__class__.__name__
                module[new_name] = type(new_name, (raw_cls, ), test_cls)
            del module[raw_name]
        return cls

    return decorate


def parameterize(fields, values=None):

    fields = [fields] if isinstance(fields, str) else fields
    params = [dict(zip(fields, vals)) for vals in values]

    def decorate(cls):
        test_cls_module = sys.modules[cls.__module__].__dict__
        for k, v in enumerate(params):
            test_cls = dict(cls.__dict__)
            test_cls.update(v)
            name = cls.__name__ + str(k)
            name = name + '.' + v.get('suffix') if v.get('suffix') else name

            test_cls_module[name] = type(name, (cls, ), test_cls)

        for m in list(cls.__dict__):
            if m.startswith("test"):
                delattr(cls, m)
        return cls

    return decorate


@contextlib.contextmanager
def stgraph(func, *args):
    """static graph exec context"""
    paddle.enable_static()
    mp, sp = paddle.static.Program(), paddle.static.Program()
    with paddle.static.program_guard(mp, sp):
        input = paddle.static.data('input', x.shape, dtype=x.dtype)
        output = func(input, n, axes, norm)

    exe = paddle.static.Executor(place)
    exe.run(sp)
    [output] = exe.run(mp, feed={'input': x}, fetch_list=[output])
    yield output
    paddle.disable_static()
