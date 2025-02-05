# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import math
import unittest

import numpy as np

import paddle


class TestConstant(unittest.TestCase):
    def test_inf(self):
        x = np.array([paddle.inf])
        np.testing.assert_equal(repr(x), 'array([inf])')

    def test_none_index(self):
        # `None` index adds newaxis
        a = np.array([1, 2, 3])
        np.testing.assert_equal(a[None], a[paddle.newaxis])
        np.testing.assert_equal(a[None].ndim, a.ndim + 1)

    def test_nan(self):
        x = np.array([paddle.nan])
        np.testing.assert_equal(repr(x), 'array([nan])')

    def test_pi(self):
        np.testing.assert_equal(paddle.pi, math.pi)

    def test_e(self):
        np.testing.assert_almost_equal(paddle.e, 2.718281828459045, decimal=15)


if __name__ == '__main__':
    unittest.main()
