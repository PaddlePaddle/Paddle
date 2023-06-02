#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

from cinn import framework
from test_utils import SingleOpTester


class OpTest_add_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        return X + Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 0)
        self.to_test_op(
            [[100, 32], [100, 32]], [[100, 32]], "elementwise_add", attrs
        )


class OpTest_add_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        return X + Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 1)
        self.to_test_op([[3, 2], [2]], [[3, 2]], "elementwise_add", attrs)


class OpTest_mul_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        return X * Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 0)
        self.to_test_op(
            [[100, 32], [100, 32]], [[100, 32]], "elementwise_mul", attrs
        )


class OpTest_mul_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X, Y] = inputs_data
        return X * Y

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("axis", 1)
        self.to_test_op([[3, 2], [2]], [[3, 2]], "elementwise_mul", attrs)


class OpTest_scale_0(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        return X * attrs.attr_store["scale"] + attrs.attr_store["bias"]

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("scale", 0.7)
        attrs.set_attr("bias", 0.3)
        self.to_test_op([[100, 32]], [[100, 32]], "scale", attrs)


class OpTest_scale_1(SingleOpTester):
    def create_target_data(self, inputs_data, attrs):
        [X] = inputs_data
        return (X + attrs.attr_store["bias"]) * attrs.attr_store["scale"]

    def test_op(self):
        attrs = framework.NodeAttr()
        attrs.set_attr("scale", 0.6)
        attrs.set_attr("bias", 0.4)
        attrs.set_attr("bias_after_scale", False)
        self.to_test_op([[100, 32]], [[100, 32]], "scale", attrs)


if __name__ == "__main__":
    unittest.main()
