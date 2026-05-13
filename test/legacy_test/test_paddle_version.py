# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from packaging.version import Version

import paddle
from paddle.paddle_version import PaddleVersion


class TestPaddleVersion(unittest.TestCase):
    def setUp(self):
        self.v = PaddleVersion('2.6.0')

    def test_module_attributes(self):
        self.assertTrue(hasattr(paddle, 'paddle_version'))
        self.assertTrue(hasattr(paddle.paddle_version, 'PaddleVersion'))
        self.assertTrue(hasattr(paddle.paddle_version, '__version__'))

    def test_paddle_version_alias(self):
        self.assertIs(paddle.__version__, paddle.paddle_version.__version__)
        self.assertIsInstance(paddle.__version__, PaddleVersion)

    def test_is_str_subclass(self):
        self.assertIsInstance(self.v, str)
        self.assertEqual(str(self.v), '2.6.0')
        self.assertEqual(self.v.split('.'), ['2', '6', '0'])
        self.assertTrue(self.v.startswith('2.6'))

    def test_compare_to_version(self):
        self.assertTrue(self.v > Version('2.5.0'))
        self.assertTrue(self.v < Version('2.6.1'))
        self.assertTrue(self.v >= Version('2.6.0'))
        self.assertTrue(self.v <= Version('2.6.0'))
        self.assertTrue(self.v == Version('2.6.0'))

    def test_compare_to_tuple(self):
        self.assertTrue(self.v > (2, 5))
        self.assertTrue(self.v > (2, 5, 9))
        self.assertTrue(self.v < (2, 6, 1))
        self.assertTrue(self.v == (2, 6, 0))

    def test_compare_to_string(self):
        self.assertTrue(self.v > '2.5')
        self.assertTrue(self.v > '2.5.1')
        self.assertTrue(self.v < '2.6.1')
        self.assertTrue(self.v == '2.6.0')

    def test_fallback_on_invalid_version(self):
        self.assertFalse(self.v == 'parrot')
        self.assertNotEqual(self.v, 'parrot')

    def test_hashable(self):
        d = {self.v: 'paddle'}
        self.assertEqual(d[self.v], 'paddle')
        self.assertEqual(d[PaddleVersion('2.6.0')], 'paddle')

    def test_prerelease(self):
        v = PaddleVersion('2.6.0a')
        self.assertTrue(v < '2.6.0')
        self.assertTrue(v > '2.5.99')


if __name__ == '__main__':
    unittest.main()
