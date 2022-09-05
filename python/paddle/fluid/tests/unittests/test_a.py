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
import paddle
import numpy as np


class TestMoveAxis(unittest.TestCase):

    def test_a(self):
        x = paddle.rand((1, 5800, 5800, 64))
        y = paddle.transpose(x, perm=[0, 3, 1, 2])
        xy = paddle.transpose(y, perm=[0, 2, 3, 1])
        assert np.array_equal(x.numpy(), xy.numpy())


if __name__ == '__main__':
    unittest.main()
