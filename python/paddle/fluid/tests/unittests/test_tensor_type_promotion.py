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

from __future__ import print_function, division

import unittest
import numpy as np
import warnings
import paddle
from paddle.fluid.framework import _test_eager_guard


class TestTensorTypePromotion(unittest.TestCase):

    def setUp(self):
        self.x = paddle.to_tensor([2, 3])
        self.y = paddle.to_tensor([1.0, 2.0])

    def add_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x + self.y
            self.assertTrue(
                "The dtype of left and right variables are not the same" in str(
                    context[-1].message))

    def sub_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x - self.y
            self.assertTrue(
                "The dtype of left and right variables are not the same" in str(
                    context[-1].message))

    def mul_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x * self.y
            self.assertTrue(
                "The dtype of left and right variables are not the same" in str(
                    context[-1].message))

    def div_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x / self.y
            self.assertTrue(
                "The dtype of left and right variables are not the same" in str(
                    context[-1].message))

    def test_operator(self):
        with _test_eager_guard():
            self.setUp()
            # add and sub has been sunk to cpp level, there is no warnings to catch by this test.
            self.mul_operator()
            self.div_operator()
        self.setUp()
        self.add_operator()
        self.sub_operator()
        self.mul_operator()
        self.div_operator()


if __name__ == '__main__':
    unittest.main()
