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

import unittest

import logging
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import ProgramTranslator
from paddle.fluid.dygraph.dygraph_to_static.convert_call_func import CONVERSION_OPTIONS
from test_program_translator import get_source_code
from paddle.jit import to_static


def dyfunc_generator():
    for i in range(100):
        yield paddle.fluid.dygraph.to_variable([i] * 10)


def main_func():
    """ Error will raise, but we only report a warning not intercept
     """
    for i in dyfunc_generator():
        print(i)


class TestConvertGenerator(unittest.TestCase):

    def test_raise_error(self):
        with self.assertRaises(Exception):
            to_static(main_func)()


if __name__ == '__main__':
    unittest.main()
