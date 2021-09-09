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

import re
import sys
import unittest
from math import prod

import numpy as np
import paddle


def setUpModule():
    global rtol
    global atol
    rtol = {'float32': 1e-6, 'float64': 1e-6}
    atol = {'float32': 0.0, 'float64': 0.0}


def tearDownModule():
    pass


def rand_x(dims=1, dtype='float32', min_dim_len=1, max_dim_len=10):
    """generate random input"""
    shape = {np.random.randint(min_dim_len, max_dim_len) for i in range(dims)}
    return np.random.randn(*shape).astype(dtype)


def parameterize(attrs, input_values=None):
    """ Parameterizes a test class by setting attributes on the class.
    """
    if isinstance(attrs, str):
        attrs = [attrs]
    input_dicts = (attrs if input_values is None else
                   [dict(zip(attrs, vals)) for vals in input_values])

    def decorator(base_class):
        """class decorator"""
        test_class_module = sys.modules[base_class.__module__].__dict__
        for idx, input_dict in enumerate(input_dicts):
            test_class_dict = dict(base_class.__dict__)
            test_class_dict.update(input_dict)

            name = class_name(base_class, idx, input_dict)

            test_class_module[name] = type(name, (base_class, ),
                                           test_class_dict)

        for method_name in list(base_class.__dict__):
            if method_name.startswith("test"):
                delattr(base_class, method_name)
        return base_class

    return decorator


def class_name(cls, num, params_dict):
    suffix = to_safe_name(
        next((v for v in params_dict.values() if isinstance(v, str)), ""))
    if "test_case" in params_dict:
        suffix = to_safe_name(params_dict["test_case"])
    return "{}_{}{}".format(cls.__name__, num, suffix and "_" + suffix)


def to_safe_name(s):
    return str(re.sub("[^a-zA-Z0-9_]+", "_", s))


@parameterize(('x', 'n', 'axis', 'norm'), [
    (rand_x(1), None, -1, 'backward'),
    (rand_x(3, np.float32), None, -1, 'backward'),
    (rand_x(3, np.float64), None, -1, 'backward'),
])
class TestRfft(unittest.TestCase):
    def test_rfft(self):
        self.assertTrue(
            np.allclose(
                np.fft.rfft(self.x, self.n, self.axis, self.norm),
                paddle.tensor.fft.rfft(
                    paddle.to_tensor(self.x), self.n, self.axis, self.norm),
                rtol=rtol.get(str(self.x.dtype)),
                atol=atol.get(str(self.x.dtype))))


if __name__ == '__main__':
    unittest.main()
