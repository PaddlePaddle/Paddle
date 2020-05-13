#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import unittest

import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import declarative


def dyfunc_assert_variable(x):
    x_v = fluid.dygraph.to_variable(x)
    assert x_v


class TestAssertVariable(unittest.TestCase):
    def test(self):
        with fluid.dygraph.guard():
            dyfunc_assert_variable(numpy.asarray([1]))


if __name__ == '__main__':
    unittest.main()
