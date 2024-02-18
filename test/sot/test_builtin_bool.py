# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from test_case_base import TestCaseBase

import paddle


class TestObject:
    pass


class TestObjectWithBool:
    def __bool__(self):
        return False


def object_bool(obj):
    if obj:
        return True
    else:
        return False


class TestBuiltinBool(TestCaseBase):
    def test_object(self):
        object = TestObject()
        self.assert_results(object_bool, object)

    def test_object_with_bool(self):
        object = TestObjectWithBool()
        self.assert_results(object_bool, object)

    def test_layer(self):
        layer = paddle.nn.Linear(10, 1)
        self.assert_results(object_bool, layer)


if __name__ == "__main__":
    unittest.main()
