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

from paddle.jit.utils import OrderedSet


class TestOrderedSet(unittest.TestCase):
    def test_iter(self):
        s = OrderedSet([1, 2, 3])
        self.assertEqual(list(s), [1, 2, 3])

    def test_or(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([2, 3, 4])
        self.assertEqual(s1 | s2, OrderedSet([1, 2, 3, 4]))

    def test_ior(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([2, 3, 4])
        s1 |= s2
        self.assertEqual(s1, OrderedSet([1, 2, 3, 4]))

    def test_and(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([2, 3, 4])
        self.assertEqual(s1 & s2, OrderedSet([2, 3]))

    def test_iand(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([2, 3, 4])
        s1 &= s2
        self.assertEqual(s1, OrderedSet([2, 3]))

    def test_sub(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([2, 3, 4])
        self.assertEqual(s1 - s2, OrderedSet([1]))

    def test_isub(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([2, 3, 4])
        s1 -= s2
        self.assertEqual(s1, OrderedSet([1]))

    def test_xor(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([2, 3, 4])
        self.assertEqual(s1 ^ s2, OrderedSet([1, 4]))

    def test_ixor(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([2, 3, 4])
        s1 ^= s2
        self.assertEqual(s1, OrderedSet([1, 4]))

    def test_add(self):
        s = OrderedSet([1, 2, 3])
        s.add(4)
        self.assertEqual(s, OrderedSet([1, 2, 3, 4]))

    def test_remove(self):
        s = OrderedSet([1, 2, 3])
        s.remove(2)
        self.assertEqual(s, OrderedSet([1, 3]))

    def test_contains(self):
        s = OrderedSet([1, 2, 3])
        self.assertTrue(2 in s)
        self.assertFalse(4 in s)

    def test_len(self):
        s = OrderedSet([1, 2, 3])
        self.assertEqual(len(s), 3)

    def test_bool(self):
        s = OrderedSet([1, 2, 3])
        self.assertTrue(bool(s))
        s = OrderedSet()
        self.assertFalse(bool(s))

    def test_eq(self):
        s1 = OrderedSet([1, 2, 3])
        s2 = OrderedSet([1, 2, 3])
        self.assertEqual(s1, s2)
        s3 = OrderedSet([3, 2, 1])
        self.assertNotEqual(s1, s3)

    def test_repr(self):
        s = OrderedSet([1, 2, 3])
        self.assertEqual(repr(s), "OrderedSet(1, 2, 3)")


if __name__ == '__main__':
    unittest.main()
