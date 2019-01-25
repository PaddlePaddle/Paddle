# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import paddle.fluid as fluid
from paddle.fluid.imperative.base import to_variable
import numpy as np


class Split_test(fluid.imperative.Layer):
    def __init__(self):
        super(Split_test, self).__init__()

    def _build_once(self, input):
        pass

    def forward(self, input):
        out = fluid.layers.split(input, num_or_sections=4, dim=-1)
        return out


class TestImperativePtbRnn(unittest.TestCase):
    def test_spilt(self):
        with fluid.imperative.guard():
            inp = to_variable(np.arange(160).reshape(4, 40).astype('float32'))
            st = Split_test()
            out = st(inp)


if __name__ == '__main__':
    unittest.main()
