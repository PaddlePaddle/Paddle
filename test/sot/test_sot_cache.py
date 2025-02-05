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

import unittest

from paddle.jit.sot.utils import Cache


class CustomCache(Cache):
    def key_fn(self, x):
        return x

    def value_fn(self, x):
        return x


class TestCache(unittest.TestCase):
    def test_basic(self):
        cache = CustomCache()
        self.assertEqual(cache(1), 1)
        self.assertEqual(cache(1), 1)

    def test_int_collision(self):
        cache = CustomCache()
        # -1 and -2 hash to the same value
        self.assertEqual(cache(-1), -1)
        self.assertEqual(cache(-2), -2)

    def test_bool_and_int_collision(self):
        cache = CustomCache()
        # 1 and True hash to the same value
        self.assertEqual(cache(True), True)
        self.assertEqual(cache(1), 1)
        # 0 and False hash to the same value
        self.assertEqual(cache(False), False)
        self.assertEqual(cache(0), 0)


if __name__ == "__main__":
    unittest.main()
