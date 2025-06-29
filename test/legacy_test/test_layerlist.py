#   Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


class TestLayerListEmptyInsert(unittest.TestCase):
    def test_insert_empty_list(self):

        # Test successful case - insert at index 0
        layers = paddle.nn.LayerList()
        linear = paddle.nn.Linear(10, 10)
        try:
            layers.insert(0, linear)
            self.assertEqual(len(layers), 1)
            self.assertTrue(layers[0] is linear)
        except Exception as e:
            self.fail(f"Insert at index 0 raised unexpected exception: {e}")

        # Test failure case - insert at index 1
        layers = paddle.nn.LayerList()
        with self.assertRaises(AssertionError):
            layers.insert(1, paddle.nn.Linear(10, 10))

        # Test successful case - insert at index 1 of non-empty list
        linear1 = paddle.nn.Linear(15, 15)
        linear2 = paddle.nn.Linear(20, 20)
        layers = paddle.nn.LayerList([linear1])
        try:
            layers.insert(1, linear2)
            self.assertEqual(len(layers), 2)
            self.assertTrue(layers[1] is linear2)
        except Exception as e:
            self.fail(f"Insert at index 1 raised unexpected exception: {e}")


if __name__ == "__main__":
    unittest.main()
