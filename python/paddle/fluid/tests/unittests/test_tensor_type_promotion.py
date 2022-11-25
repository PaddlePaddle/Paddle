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

import unittest
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
<<<<<<< HEAD
                "The dtype of left and right variables are not the same"
                in str(context[-1].message)
            )
=======
                "The dtype of left and right variables are not the same" in str(
                    context[-1].message))
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

    def sub_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x - self.y
            self.assertTrue(
<<<<<<< HEAD
                "The dtype of left and right variables are not the same"
                in str(context[-1].message)
            )
=======
                "The dtype of left and right variables are not the same" in str(
                    context[-1].message))
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

    def mul_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x * self.y
            self.assertTrue(
<<<<<<< HEAD
                "The dtype of left and right variables are not the same"
                in str(context[-1].message)
            )
=======
                "The dtype of left and right variables are not the same" in str(
                    context[-1].message))
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e

    def div_operator(self):
        with warnings.catch_warnings(record=True) as context:
            warnings.simplefilter("always")
            self.x / self.y
            self.assertTrue(
<<<<<<< HEAD
                "The dtype of left and right variables are not the same"
                in str(context[-1].message)
            )

    def test_operator(self):
        with _test_eager_guard():
            pass
            # add / sub / mul / div has been sunk to cpp level, there is no warnings to catch by this test.
        self.setUp()
        self.add_operator()
        self.sub_operator()
        self.mul_operator()
        self.div_operator()
=======
                "The dtype of left and right variables are not the same" in str(
                    context[-1].message))
>>>>>>> e170b253fc2cfc81aeb39c17a0fffc8e08311f1e


if __name__ == '__main__':
    unittest.main()
