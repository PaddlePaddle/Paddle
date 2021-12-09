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

import paddle.fluid as fluid
import paddle
from paddle.fluid.wrapped_decorator import wrap_decorator
import unittest
from unittest import TestCase
import numpy as np


def _dygraph_guard_(func):
    def __impl__(*args, **kwargs):
        if fluid.in_dygraph_mode():
            return func(*args, **kwargs)
        else:
            with fluid.dygraph.guard():
                return func(*args, **kwargs)

    return __impl__


dygraph_guard = wrap_decorator(_dygraph_guard_)


class TestDygraphClearGradient(TestCase):
    def setUp(self):
        self.input_shape = [10, 2]

    @dygraph_guard
    def test_tensor_method_clear_gradient_case1(self):
        input = paddle.uniform(self.input_shape)
        linear = paddle.nn.Linear(2, 3)
        out = linear(input)
        out.backward()
        linear.weight.clear_gradient()

        # actual result
        gradient_actual = linear.weight.grad
        # expected result
        gradient_expected = np.zeros([2, 3]).astype('float64')
        self.assertTrue(np.allclose(gradient_actual.numpy(), gradient_expected))

    @dygraph_guard
    def test_tensor_method_clear_gradient_case2(self):
        input = paddle.uniform(self.input_shape)
        linear = paddle.nn.Linear(2, 3)
        out = linear(input)
        out.backward()
        # default arg set_to_zero is true
        # so, False means real clear gradient
        linear.weight.clear_gradient(False)

        # before ._gradient_set_empty(False), 
        # the return of ._is_gradient_set_empty() should be True
        self.assertTrue(linear.weight._is_gradient_set_empty())

        # reset, because ClearGradient will call SetIsEmpty(True), but this is not our expectation.
        linear.weight._gradient_set_empty(False)
        # after ._gradient_set_empty(False), 
        # the return of ._is_gradient_set_empty() should be False
        self.assertFalse(linear.weight._is_gradient_set_empty())

        # actual result
        gradient_actual = linear.weight.grad
        # expected result
        self.assertTrue(np.empty(gradient_actual))


if __name__ == '__main__':
    unittest.main()
