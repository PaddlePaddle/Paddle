# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from utils import dygraph_guard

import paddle


class TestTypeAsBase(unittest.TestCase):
    def test__index__with_0size_tensor(self):
        with dygraph_guard():
            x = paddle.randn([0])
            l = [1, 2, 3]
            with self.assertRaisesRegex(
                AssertionError,
                "only one element variable can be converted to python index.",
            ):
                l[x]

    def test__index__with_non_scalar_tensor(self):
        with dygraph_guard():
            l = [1, 2, 3]
            x = paddle.to_tensor([1]).reshape(1, 1, 1)
            self.assertEqual(l[x], l[x.item()])


if __name__ == "__main__":
    unittest.main()
