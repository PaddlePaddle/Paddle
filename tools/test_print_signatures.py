#! /usr/bin/env python

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
"""
TestCases for print_signatures.py

sample lines from API_DEV.spec:
    paddle.autograd.backward (ArgSpec(args=['tensors', 'grad_tensors', 'retain_graph'], varargs=None, keywords=None, defaults=(None, False)), ('document', '33a4434c9d123331499334fbe0274870'))
    paddle.autograd.PyLayer (paddle.autograd.py_layer.PyLayer, ('document', 'c26adbbf5f1eb43d16d4a399242c979e'))
    paddle.autograd.PyLayer.apply (ArgSpec(args=['cls'], varargs=args, keywords=kwargs, defaults=None), ('document', 'cb78696dc032fb8af2cba8504153154d'))
"""
import functools
import hashlib
import unittest

from print_signatures import md5


def func_example(param_a, param_b):
    """
    example function
    """
    pass


def func_example_2(func=functools.partial(func_example, 1)):
    """
    example function 2
    """
    pass


class ClassExample:
    """
    example Class
    """

    def example_method(self):
        """
        class method
        """
        pass


class Test_all_in_print_signatures(unittest.TestCase):
    def test_md5(self):
        algo = hashlib.md5()
        algo.update(func_example.__doc__.encode('utf-8'))
        digest = algo.hexdigest()
        self.assertEqual(digest, md5(func_example.__doc__))


if __name__ == '__main__':
    unittest.main()
