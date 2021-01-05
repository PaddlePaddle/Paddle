# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn

INPUT_SIZE = 10
OUTPUT_SIZE = 1


class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        return self._linear(x)


def print_hook(grad):
    print(grad)


def mul_hook(grad):
    return grad * 2


class TestDynamicBackwardHook(unittest.TestCase):
    def setUp(self):
        pass

    def test_register_hook_for_leaf_var(self):
        x = paddle.randn([1, IMAGE_SIZE], 'float32')
        x.stop_gradient = False
        x.register_hook(print_hook)

        model = LinearNet()
        out = model(x)
        out.backward()

        print(x.gradient())


if __name__ == '__main__':
    unittest.main()
