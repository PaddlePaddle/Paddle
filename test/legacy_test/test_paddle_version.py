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


class TestPaddleVersionPEP440(unittest.TestCase):
    """PEP 440 coverage: ``[N!]N(.N)*[{a|b|rc}N][.postN][.devN][+local]``.

    See https://peps.python.org/pep-0440/ for the full grammar.
    """

    def test_release_no_suffix(self):
        self.assertTrue(PaddleVersion('3.4.0') == '3.4.0')
        self.assertTrue(PaddleVersion('3.4.0') < '3.4.1')
        self.assertTrue(PaddleVersion('3.4.0') > '3.3.99')

    def test_alpha(self):
        # a / alpha are equivalent spellings; ordering: dev < a < b < rc < release
        self.assertTrue(PaddleVersion('3.4.0a1') == Version('3.4.0alpha1'))
        self.assertTrue(PaddleVersion('3.4.0a1') < '3.4.0b1')
        self.assertTrue(PaddleVersion('3.4.0a1') < '3.4.0')

    def test_beta(self):
        self.assertTrue(PaddleVersion('3.4.0b1') == Version('3.4.0beta1'))
        self.assertTrue(PaddleVersion('3.4.0b1') > '3.4.0a9')
        self.assertTrue(PaddleVersion('3.4.0b1') < '3.4.0rc1')

    def test_rc(self):
        self.assertTrue(PaddleVersion('3.4.0rc1') > '3.4.0b9')
        self.assertTrue(PaddleVersion('3.4.0rc1') < '3.4.0')
        self.assertTrue(PaddleVersion('3.4.0rc1') < '3.4.0rc2')

    def test_post(self):
        # post releases sort AFTER their base release
        self.assertTrue(PaddleVersion('3.4.0.post1') > '3.4.0')
        self.assertTrue(PaddleVersion('3.4.0.post1') < '3.4.1')
        self.assertTrue(PaddleVersion('3.4.0.post20260430') > '3.4.0.post1')

    def test_dev(self):
        # dev releases sort BEFORE pre-releases of the same base
        self.assertTrue(PaddleVersion('3.4.0.dev1') < '3.4.0a1')
        self.assertTrue(PaddleVersion('3.4.0.dev1') < '3.4.0')
        self.assertTrue(PaddleVersion('3.4.0.dev2') > '3.4.0.dev1')

    def test_local_version(self):
        # Local identifier ``+xxx`` sorts AFTER the plain release, per PEP 440
        self.assertTrue(PaddleVersion('3.4.0+local') > '3.4.0')
        self.assertTrue(PaddleVersion('3.4.0+local') < '3.4.1')

    def test_local_version_post_commit_hash(self):
        # Realistic Paddle nightly-style version with date-post + git hash local id
        v = PaddleVersion('3.4.0.post20260430+a6ff7cd15ad')
        self.assertTrue(v > '3.4.0')
        self.assertTrue(v > '3.4.0.post20260429')
        self.assertTrue(v == Version('3.4.0.post20260430+a6ff7cd15ad'))
        # local identifier is preserved as a str
        self.assertIn('+a6ff7cd15ad', str(v))

    def test_epoch(self):
        # Epoch ``N!`` sorts above everything without an epoch
        self.assertTrue(PaddleVersion('1!1.0') > '999.0')
        self.assertTrue(PaddleVersion('2!1.0') > PaddleVersion('1!9.9'))

    def test_full_ordering_chain(self):
        # Canonical PEP 440 ordering: dev < a < b < rc < release < post
        chain = [
            PaddleVersion('3.4.0.dev1'),
            PaddleVersion('3.4.0a1'),
            PaddleVersion('3.4.0b1'),
            PaddleVersion('3.4.0rc1'),
            PaddleVersion('3.4.0'),
            PaddleVersion('3.4.0.post1'),
            PaddleVersion('3.4.0.post1+local'),
        ]
        for earlier, later in zip(chain, chain[1:]):
            self.assertTrue(earlier < later, f"expected {earlier} < {later}")


if __name__ == '__main__':
    unittest.main()
