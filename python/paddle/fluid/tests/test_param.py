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

import unittest
import contextlib
import re
import sys
import unittest

import numpy as np
import paddle

DEVICES = [paddle.CPUPlace()]
if paddle.is_compiled_with_cuda():
    DEVICES.append(paddle.CUDAPlace(0))

TEST_CASE_NAME = 'suffix'
# All test case will use float64 for compare percision, refs:
# https://github.com/PaddlePaddle/Paddle/wiki/Upgrade-OP-Precision-to-Float64
RTOL = {
    'float32': 1e-03,
    'complex64': 1e-3,
    'float64': 1e-7,
    'complex128': 1e-7
}
ATOL = {'float32': 0.0, 'complex64': 0, 'float64': 0.0, 'complex128': 0}


def rand_x(dims=1,
           dtype='float64',
           min_dim_len=1,
           max_dim_len=3,
           complex=False):
    shape = [np.random.randint(min_dim_len, max_dim_len) for i in range(dims)]
    if complex:
        return np.random.randn(*shape).astype(
            dtype) + 1.j * np.random.randn(*shape).astype(dtype)
    else:
        return np.random.randn(*shape).astype(dtype)


def place(devices, key='place'):
    print('place', devices)
    print('place', key)

    def decorate(cls):
        print('place', cls.__name__)
        print('place', cls.__module__)
        print('place', sys.modules[cls.__module__])

        module = sys.modules[cls.__module__].__dict__
        print('place', module)

        raw_classes = {
            k: v
            for k, v in module.items() if k.startswith(cls.__name__)
        }
        print('place', raw_classes)

        for raw_name, raw_cls in raw_classes.items():
            for d in devices:
                print('place raw_cls', raw_cls)
                test_cls = dict(raw_cls.__dict__)
                print('place test_cls', d, test_cls)
                test_cls.update({key: d})
                new_name = raw_name + '.' + d.__class__.__name__
                print('place new_name', new_name)
                module[new_name] = type(new_name, (raw_cls, ), test_cls)
            del module[raw_name]
        return cls

    return decorate


def parameterize(fields, values=None):

    fields = [fields] if isinstance(fields, str) else fields
    params = [dict(zip(fields, vals)) for vals in values]

    print('param', fields)
    print('param', len(values))

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


# @place(DEVICES)
# @parameterize(
#     (TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
#     [('test_x_float64', rand_x(5, np.float64), None, -1, 'backward'),
#      ('test_x_complex', rand_x(5, complex=True), None, -1, 'backward'),
#      ('test_n_grater_input_length', rand_x(5,
#                                            max_dim_len=5), 11, -1, 'backward'),
#      ('test_n_smaller_than_input_length', rand_x(
#          5, min_dim_len=5, complex=True), 3, -1, 'backward'),
#      ('test_axis_not_last', rand_x(5), None, 3, 'backward'),
#      ('test_norm_forward', rand_x(5), None, 3, 'forward'),
#      ('test_norm_ortho', rand_x(5), None, 3, 'ortho')])
@place(DEVICES)
@parameterize((TEST_CASE_NAME, 'x', 'n', 'axis', 'norm'),
              [('test_x_float64', rand_x(2, np.float64), None, -1, 'backward')])
class TestFft(unittest.TestCase):

    def test_fft(self):
        """Test fft with norm condition
        """
        print(self.suffix, self.x.shape, self.n, self.axis, self.norm)


if __name__ == '__main__':
    unittest.main()
