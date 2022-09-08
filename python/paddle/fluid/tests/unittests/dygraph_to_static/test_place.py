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

import numpy as np
import paddle
import unittest


class TestPlace(unittest.TestCase):

    def test_place(self):

        def func(x):
            x = paddle.to_tensor([1, 2, 3, 4])
            p = x.place()
            print(p)
            return x

        x = paddle.to_tensor([3])
        print(paddle.jit.to_static(func).code)
        print(paddle.jit.to_static(func)(x))


if __name__ == '__main__':
    unittest.main()
