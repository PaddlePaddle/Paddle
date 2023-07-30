#! python

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

import os
import unittest

from sampcd_processor_utils import get_test_results
from sampcd_processor_xdoctest import Xdoctester


def _clear_environ():
    for k in {'CPU', 'GPU', 'XPU', 'DISTRIBUTED'}:
        if k in os.environ:
            del os.environ[k]


class TestXdoctester(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_init(self):
        doctester = Xdoctester()
        self.assertEqual(doctester.debug, False)
        self.assertEqual(doctester.style, 'freeform')
        self.assertEqual(doctester.target, 'codeblock')
        self.assertEqual(doctester.mode, 'native')
        self.assertEqual(doctester.config['analysis'], 'auto')

        doctester = Xdoctester(analysis='static')
        self.assertEqual(doctester.config['analysis'], 'static')

    def test_convert_directive(self):
        doctester = Xdoctester()
        docstring_input = "# doctest: -SKIP\n"
        docstring_output = doctester.convert_directive(docstring_input)
        docstring_target = "# xdoctest: -SKIP\n"
        self.assertEqual(docstring_output, docstring_target)

        docstring_input = '# doctest: +SKIP("skip this test...")\n'
        docstring_output = doctester.convert_directive(docstring_input)
        docstring_target = '# xdoctest: +SKIP("skip this test...")\n'
        self.assertEqual(docstring_output, docstring_target)

        docstring_input = """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: -REQUIRES(env:GPU)
                    >>> print(1-1)
                    0

                .. code-block:: python
                    :name: code-example-2

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)
                    >>> print(1-1)
                    0
            """
        docstring_output = doctester.convert_directive(docstring_input)
        docstring_target = """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # xdoctest: +SKIP
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # xdoctest: -REQUIRES(env:GPU)
                    >>> print(1-1)
                    0

                .. code-block:: python
                    :name: code-example-2

                    this is some blabla...

                    >>> # xdoctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)
                    >>> print(1-1)
                    0
            """
        self.assertEqual(docstring_output, docstring_target)

    def test_prepare(self):
        doctester = Xdoctester()

        _clear_environ()
        test_capacity = {'cpu'}
        doctester.prepare(test_capacity)
        self.assertTrue(os.environ['CPU'])
        self.assertFalse(os.environ.get('GPU'))

        _clear_environ()
        test_capacity = {'cpu', 'gpu'}
        doctester.prepare(test_capacity)
        self.assertTrue(os.environ['CPU'])
        self.assertTrue(os.environ['GPU'])
        self.assertFalse(os.environ.get('cpu'))
        self.assertFalse(os.environ.get('gpu'))
        self.assertFalse(os.environ.get('XPU'))

        _clear_environ()


class TestGetTestResults(unittest.TestCase):
    def test_run_cpu(self):
        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:CPU)
                    >>> print(1-1)
                    0
            """,
            'one_minus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:GPU)
                    >>> print(1-1)
                    0
            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)

        tr_0, tr_1, tr_2, tr_3 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-0', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-1', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)

        self.assertIn('one_minus_one', tr_2.name)
        self.assertIn('code-example-0', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertTrue(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertFalse(tr_2.failed)

        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-1', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertFalse(tr_3.passed)
        self.assertTrue(tr_3.skipped)
        self.assertFalse(tr_3.failed)

    def test_run_gpu(self):
        _clear_environ()

        test_capacity = {'cpu', 'gpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +REQUIRES(env: GPU)
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:XPU)
                    >>> print(1-1)
                    0
            """,
            'one_minus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> print(1-1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:GPU, env: XPU)
                    >>> print(1-1)
                    0
            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)

        tr_0, tr_1, tr_2, tr_3 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-0', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-1', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertFalse(tr_1.passed)
        self.assertTrue(tr_1.skipped)
        self.assertFalse(tr_1.failed)

        self.assertIn('one_minus_one', tr_2.name)
        self.assertIn('code-example-0', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertFalse(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertTrue(tr_2.failed)

        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-1', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertFalse(tr_3.passed)
        self.assertTrue(tr_3.skipped)
        self.assertFalse(tr_3.failed)

    def test_run_xpu_distributed(self):
        _clear_environ()

        test_capacity = {'cpu', 'xpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +REQUIRES(env: GPU)
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:XPU)
                    >>> print(1-1)
                    0
            """,
            'one_minus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> print(1-1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:CPU, env: XPU)
                    >>> print(1-1)
                    0
            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)

        tr_0, tr_1, tr_2, tr_3 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-0', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-1', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)

        self.assertIn('one_minus_one', tr_2.name)
        self.assertIn('code-example-0', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertFalse(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertTrue(tr_2.failed)

        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-1', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertTrue(tr_3.passed)
        self.assertFalse(tr_3.skipped)
        self.assertFalse(tr_3.failed)

    def test_style_google(self):
        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='google', target='docstring')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:CPU)
                    >>> print(1-1)
                    0

            Examples:

                .. code-block:: python
                    :name: code-example-2

                    >>> print(1+2)
                    3
            """,
            'one_minus_one': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:GPU)
                    >>> print(1-1)
                    0

            Examples:

                .. code-block:: python
                    :name: code-example-2

                    >>> print(1+1)
                    3
            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)

        tr_0, tr_1, tr_2, tr_3 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertNotIn('code-example-1', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_plus_one', tr_1.name)
        self.assertNotIn('code-example-2', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)

        self.assertIn('one_minus_one', tr_2.name)
        self.assertNotIn('code-example-1', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertFalse(tr_2.passed)
        self.assertTrue(tr_2.skipped)
        self.assertFalse(tr_2.failed)

        self.assertIn('one_minus_one', tr_3.name)
        self.assertNotIn('code-example-2', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertFalse(tr_3.passed)
        self.assertFalse(tr_3.skipped)
        self.assertTrue(tr_3.failed)

        _clear_environ()

        test_capacity = {'cpu', 'gpu'}
        doctester = Xdoctester(style='google', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:CPU)
                    >>> print(1-1)
                    0

            Examples:

                .. code-block:: python
                    :name: code-example-2

                    >>> print(1+2)
                    3
            """,
            'one_minus_one': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:GPU)
                    >>> print(1-1)
                    0

            Examples:

                .. code-block:: python
                    :name: code-example-2

                    >>> print(1+1)
                    3
            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 4)

        tr_0, tr_1, tr_2, tr_3 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-1', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-2', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)

        self.assertIn('one_minus_one', tr_2.name)
        self.assertIn('code-example-1', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertTrue(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertFalse(tr_2.failed)

        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-2', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertFalse(tr_3.passed)
        self.assertFalse(tr_3.skipped)
        self.assertTrue(tr_3.failed)

    def test_style_freeform(self):
        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='docstring')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:CPU)
                    >>> print(1-1)
                    0

            Examples:

                .. code-block:: python
                    :name: code-example-2

                    >>> print(1+2)
                    3
            """,
            'one_minus_one': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:CPU)
                    >>> print(1-1)
                    0

            Examples:

                .. code-block:: python
                    :name: code-example-2

                    >>> print(1+1)
                    3
            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)

        tr_0, tr_1 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertNotIn('code-example', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_minus_one', tr_1.name)
        self.assertNotIn('code-example', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertFalse(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertTrue(tr_1.failed)

        _clear_environ()

        test_capacity = {'cpu', 'gpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:CPU)
                    >>> for i in range(2):
                    ...     print(i)
                    0
                    1

            Examples:

                .. code-block:: python
                    :name: code-example-2

                    >>> print(1+2)
                    3
            """,
            'one_minus_one': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:CPU)
                    >>> print(1-1)
                    0

            Examples:

                .. code-block:: python
                    :name: code-example-2

                    >>> print(1+1)
                    3
            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 5)

        tr_0, tr_1, tr_2, tr_3, tr_4 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example-0', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_plus_one', tr_1.name)
        self.assertIn('code-example-1', tr_1.name)
        self.assertFalse(tr_1.nocode)
        self.assertTrue(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)

        self.assertIn('one_plus_one', tr_2.name)
        self.assertIn('code-example-2', tr_2.name)
        self.assertFalse(tr_2.nocode)
        self.assertTrue(tr_2.passed)
        self.assertFalse(tr_2.skipped)
        self.assertFalse(tr_2.failed)

        self.assertIn('one_minus_one', tr_3.name)
        self.assertIn('code-example-1', tr_3.name)
        self.assertFalse(tr_3.nocode)
        self.assertTrue(tr_3.passed)
        self.assertFalse(tr_3.skipped)
        self.assertFalse(tr_3.failed)

        self.assertIn('one_minus_one', tr_4.name)
        self.assertIn('code-example-2', tr_4.name)
        self.assertFalse(tr_4.nocode)
        self.assertFalse(tr_4.passed)
        self.assertFalse(tr_4.skipped)
        self.assertTrue(tr_4.failed)

    def test_no_code(self):
        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='google', target='docstring')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2
            """,
            'one_minus_one': """
            placeholder

            Examples:

            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)

        tr_0, tr_1 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertNotIn('code-example', tr_0.name)
        self.assertTrue(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertFalse(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_minus_one', tr_1.name)
        self.assertNotIn('code-example', tr_1.name)
        self.assertTrue(tr_1.nocode)
        self.assertFalse(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='google', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2
            """,
            'one_minus_one': """
            placeholder

            Examples:

            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 0)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='docstring')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2
            """,
            'one_minus_one': """
            placeholder

            Examples:

            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)

        tr_0, tr_1 = test_results

        self.assertIn('one_plus_one', tr_0.name)
        self.assertNotIn('code-example', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)

        self.assertIn('one_minus_one', tr_1.name)
        self.assertNotIn('code-example', tr_1.name)
        self.assertTrue(tr_1.nocode)
        self.assertFalse(tr_1.passed)
        self.assertFalse(tr_1.skipped)
        self.assertFalse(tr_1.failed)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'one_plus_one': """
            placeholder

            .. code-block:: python
                :name: code-example-0

                this is some blabla...

                >>> # doctest: +SKIP
                >>> print(1+1)
                2
            """,
            'one_minus_one': """
            placeholder

            Examples:

            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 1)

        tr_0 = test_results[0]

        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)


if __name__ == '__main__':
    unittest.main()
