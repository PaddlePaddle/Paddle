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

import importlib
import os
import sys
import tempfile
import unittest

import xdoctest
from sampcd_processor import Xdoctester
from sampcd_processor_utils import (
    TestResult as _TestResult,  # for pytest
    get_api_md5,
    get_incrementapi,
    get_test_results,
)

from paddle.base import core


def _clear_environ():
    for k in {'CPU', 'GPU', 'XPU', 'DISTRIBUTED'}:
        if k in os.environ:
            del os.environ[k]


class TestTestResult(unittest.TestCase):
    def test_good_result(self):
        r = _TestResult(name='good', passed=True)
        self.assertTrue(r.passed)
        self.assertFalse(r.failed)

        r = _TestResult(name='good', passed=True, failed=False)
        self.assertTrue(r.passed)
        self.assertFalse(r.failed)

        r = _TestResult(name='good', passed=False, failed=True)
        self.assertFalse(r.passed)
        self.assertTrue(r.failed)

        r = _TestResult(name='good', passed=True, nocode=False, time=10)
        self.assertTrue(r.passed)
        self.assertFalse(r.nocode)

        r = _TestResult(
            name='good',
            passed=True,
            timeout=False,
            time=10,
            test_msg='ok',
            extra_info=None,
        )
        self.assertTrue(r.passed)
        self.assertFalse(r.timeout)

    def test_bad_result(self):
        # more than one True result
        r = _TestResult(name='bad', passed=True, failed=True)
        self.assertTrue(r.passed)
        self.assertTrue(r.failed)

        # default result is Fail for True
        r = _TestResult(name='bad')
        self.assertFalse(r.passed)
        self.assertTrue(r.failed)

        # bad arg
        with self.assertRaises(KeyError):
            r = _TestResult(name='good', passed=True, bad=True)


class TestSpecFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.api_dev_spec_filename = os.path.join(
            self.temp_dir.name, 'API_DEV.spec'
        )
        self.api_pr_spec_filename = os.path.join(
            self.temp_dir.name, 'API_PR.spec'
        )
        self.api_diff_spec_filename = os.path.join(
            self.temp_dir.name, 'dev_pr_diff_api.spec'
        )
        self.init_file()

    def tearDown(self):
        self.temp_dir.cleanup()

    def init_file(self):
        raise NotImplementedError


class TestGetApiMd5(TestSpecFile):
    def init_file(self):
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write(
                "\n".join(
                    [
                        """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
                        """paddle.two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55two'))""",
                        """paddle.three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6cthree'))""",
                        """paddle.four_plus_four (paddle.four_plus_four, ('document', 'ff0f188c95030158cc6398d2a6c5four'))""",
                        """paddle.five_plus_five (ArgSpec(), ('document', 'ff0f188c95030158cc6398d2a6c5five'))""",
                    ]
                )
            )

    def test_get_api_md5(self):
        res = get_api_md5(self.api_pr_spec_filename)
        self.assertEqual(
            "ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ff0f188c95030158cc6398d2a6c55one",
            res['paddle.one_plus_one'],
        )
        self.assertEqual(
            "ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ff0f188c95030158cc6398d2a6c55two",
            res['paddle.two_plus_two'],
        )
        self.assertEqual(
            "ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ff0f188c95030158cc6398d2a6cthree",
            res['paddle.three_plus_three'],
        )
        self.assertEqual(
            "ff0f188c95030158cc6398d2a6c5four", res['paddle.four_plus_four']
        )
        self.assertEqual(
            "ArgSpec(), ff0f188c95030158cc6398d2a6c5five",
            res['paddle.five_plus_five'],
        )


class TestGetIncrementApi(TestSpecFile):
    def init_file(self):
        with open(self.api_pr_spec_filename, 'w') as f:
            f.write(
                "\n".join(
                    [
                        """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
                        """paddle.two_plus_two (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55two'))""",
                        """paddle.three_plus_three (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6cthree'))""",
                        """paddle.four_plus_four (paddle.four_plus_four, ('document', 'ff0f188c95030158cc6398d2a6c5four'))""",
                    ]
                )
            )

        with open(self.api_dev_spec_filename, 'w') as f:
            f.write(
                "\n".join(
                    [
                        """paddle.one_plus_one (ArgSpec(args=[], varargs=None, keywords=None, defaults=(,)), ('document', 'ff0f188c95030158cc6398d2a6c55one'))""",
                    ]
                )
            )

    def test_get_incrementapi(self):
        get_incrementapi(
            api_dev_spec_fn=self.api_dev_spec_filename,
            api_pr_spec_fn=self.api_pr_spec_filename,
            api_diff_spec_fn=self.api_diff_spec_filename,
        )
        with open(self.api_diff_spec_filename, 'r') as f:
            lines = f.readlines()
            self.assertCountEqual(
                [
                    "paddle.two_plus_two\n",
                    "paddle.three_plus_three\n",
                    "paddle.four_plus_four\n",
                ],
                lines,
            )


class TestXdoctester(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_init(self):
        doctester = Xdoctester()
        self.assertEqual(doctester.debug, False)
        self.assertEqual(doctester.style, 'freeform')
        self.assertEqual(doctester.target, 'codeblock')
        self.assertEqual(doctester.mode, 'native')

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

                >>> # doctest: +SKIP('skip')
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

                >>> # xdoctest: +SKIP('skip')
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
        self.assertFalse(os.environ.get('XPU'))

        _clear_environ()


class TestGetTestResults(unittest.TestCase):
    def test_global_exec(self):
        _clear_environ()

        # test set_default_dtype
        docstrings_to_test = {
            'before_set_default': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> a = paddle.to_tensor(.2)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.20000000)
            """,
            'set_default': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.set_default_dtype('float64')
                    >>> a = paddle.to_tensor(.2)
                    >>> print(a)
                    Tensor(shape=[], dtype=float64, place=Place(cpu), stop_gradient=True,
                    0.20000000)
            """,
            'after_set_default': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> a = paddle.to_tensor(.2)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.20000000)
            """,
        }

        # test old global_exec
        test_capacity = {'cpu'}
        doctester = Xdoctester(
            style='freeform',
            target='codeblock',
            global_exec=r"\n".join(
                [
                    "import paddle",
                    "paddle.device.set_device('cpu')",
                ]
            ),
        )
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 3)

        tr_0, tr_1, tr_2 = test_results

        self.assertIn('before_set_default', tr_0.name)
        self.assertTrue(tr_0.passed)

        self.assertIn('set_default', tr_1.name)
        self.assertTrue(tr_1.passed)

        # tr_2 is passed, because of multiprocessing
        self.assertIn('after_set_default', tr_2.name)
        self.assertTrue(tr_2.passed)

        # test new default global_exec
        doctester = Xdoctester(
            style='freeform',
            target='codeblock',
        )
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 3)

        tr_0, tr_1, tr_2 = test_results

        self.assertIn('before_set_default', tr_0.name)
        self.assertTrue(tr_0.passed)

        self.assertIn('set_default', tr_1.name)
        self.assertTrue(tr_1.passed)

        self.assertIn('after_set_default', tr_2.name)
        self.assertTrue(tr_2.passed)

        # test disable static
        docstrings_to_test = {
            'before_enable_static': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> print(paddle.in_dynamic_mode())
                    True
            """,
            'enable_static': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.enable_static()
                    >>> print(paddle.in_dynamic_mode())
                    False
            """,
            'after_enable_static': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> print(paddle.in_dynamic_mode())
                    True
            """,
        }

        # test old global_exec
        test_capacity = {'cpu'}
        doctester = Xdoctester(
            style='freeform',
            target='codeblock',
            global_exec=r"\n".join(
                [
                    "import paddle",
                    "paddle.device.set_device('cpu')",
                ]
            ),
        )
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 3)

        tr_0, tr_1, tr_2 = test_results

        self.assertIn('before_enable_static', tr_0.name)
        self.assertTrue(tr_0.passed)

        self.assertIn('enable_static', tr_1.name)
        self.assertTrue(tr_1.passed)

        # tr_2 is passed, because of multiprocessing
        self.assertIn('after_enable_static', tr_2.name)
        self.assertTrue(tr_2.passed)

        # test new default global_exec
        doctester = Xdoctester(
            style='freeform',
            target='codeblock',
        )
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 3)

        tr_0, tr_1, tr_2 = test_results

        self.assertIn('before_enable_static', tr_0.name)
        self.assertTrue(tr_0.passed)

        self.assertIn('enable_static', tr_1.name)
        self.assertTrue(tr_1.passed)

        self.assertIn('after_enable_static', tr_2.name)
        self.assertTrue(tr_2.passed)

    @unittest.skipIf(
        not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
        "core is not compiled with CUDA",
    )
    def test_patch_xdoctest_place(self):
        # test patch tensor place
        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'gpu_to_gpu': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('gpu')
                    >>> a = paddle.to_tensor(.2)
                    >>> # Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True, 0.20000000)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    0.20000000)

            """,
            'cpu_to_cpu': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('cpu')
                    >>> a = paddle.to_tensor(.2)
                    >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True, 0.20000000)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.20000000)

            """,
            'gpu_to_cpu': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('gpu')
                    >>> a = paddle.to_tensor(.2)
                    >>> # Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True, 0.20000000)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.20000000)

            """,
            'cpu_to_gpu': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('cpu')
                    >>> a = paddle.to_tensor(.2)
                    >>> # Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True, 0.20000000)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    0.20000000)
            """,
            'gpu_to_cpu_array': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('gpu')
                    >>> a = paddle.to_tensor([[1,2,3], [2,3,4], [3,4,5]])
                    >>> # Tensor(shape=[3, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                    >>> # [[1, 2, 3],
                    >>> # [2, 3, 4],
                    >>> # [3, 4, 5]])
                    >>> print(a)
                    Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
                    [[1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5]])
            """,
            'cpu_to_gpu_array': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('cpu')
                    >>> a = paddle.to_tensor([[1,2,3], [2,3,4], [3,4,5]])
                    >>> # Tensor(shape=[3, 3], dtype=int64, place=Place(cpu), stop_gradient=True,
                    >>> # [[1, 2, 3],
                    >>> # [2, 3, 4],
                    >>> # [3, 4, 5]])
                    >>> print(a)
                    Tensor(shape=[3, 3], dtype=int64, place=Place(gpu:0), stop_gradient=True,
                    [[1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5]])
            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 6)

        tr_0, tr_1, tr_2, tr_3, tr_4, tr_5 = test_results

        self.assertIn('gpu_to_gpu', tr_0.name)
        self.assertTrue(tr_0.passed)

        self.assertIn('cpu_to_cpu', tr_1.name)
        self.assertTrue(tr_1.passed)

        self.assertIn('gpu_to_cpu', tr_2.name)
        self.assertTrue(tr_2.passed)

        self.assertIn('cpu_to_gpu', tr_3.name)
        self.assertTrue(tr_3.passed)

        self.assertIn('gpu_to_cpu_array', tr_4.name)
        self.assertTrue(tr_4.passed)

        self.assertIn('cpu_to_gpu_array', tr_5.name)
        self.assertTrue(tr_5.passed)

        # reload xdoctest.checker
        importlib.reload(xdoctest.checker)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(
            style='freeform', target='codeblock', patch_tensor_place=False
        )
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 6)

        tr_0, tr_1, tr_2, tr_3, tr_4, tr_5 = test_results

        self.assertIn('gpu_to_gpu', tr_0.name)
        self.assertTrue(tr_0.passed)

        self.assertIn('cpu_to_cpu', tr_1.name)
        self.assertTrue(tr_1.passed)

        self.assertIn('gpu_to_cpu', tr_2.name)
        self.assertFalse(tr_2.passed)

        self.assertIn('cpu_to_gpu', tr_3.name)
        self.assertFalse(tr_3.passed)

        self.assertIn('gpu_to_cpu_array', tr_4.name)
        self.assertFalse(tr_4.passed)

        self.assertIn('cpu_to_gpu_array', tr_5.name)
        self.assertFalse(tr_5.passed)

    @unittest.skipIf(
        not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
        "core is not compiled with CUDA",
    )
    def test_patch_xdoctest_float(self):
        # test patch float precision
        # reload xdoctest.checker
        importlib.reload(xdoctest.checker)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='codeblock')
        doctester.prepare(test_capacity)

        docstrings_to_test = {
            'gpu_to_gpu': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('gpu')
                    >>> a = paddle.to_tensor(.123456789)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    0.123456780)

            """,
            'cpu_to_cpu': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('cpu')
                    >>> a = paddle.to_tensor(.123456789)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.123456780)

            """,
            'gpu_to_cpu': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('gpu')
                    >>> a = paddle.to_tensor(.123456789)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                    0.123456780)

            """,
            'cpu_to_gpu': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('cpu')
                    >>> a = paddle.to_tensor(.123456789)
                    >>> print(a)
                    Tensor(shape=[], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    0.123456780)
            """,
            'gpu_to_cpu_array': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('gpu')
                    >>> a = paddle.to_tensor([[1.123456789 ,2,3], [2,3,4], [3,4,5]])
                    >>> print(a)
                    Tensor(shape=[3, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[1.123456780, 2., 3.],
                    [2., 3., 4.],
                    [3., 4., 5.]])
            """,
            'cpu_to_gpu_array': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('cpu')
                    >>> a = paddle.to_tensor([[1.123456789,2,3], [2,3,4], [3,4,5]])
                    >>> print(a)
                    Tensor(shape=[3, 3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [[1.123456780, 2., 3.],
                    [2., 3., 4.],
                    [3., 4., 5.]])
            """,
            'mass_array': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('gpu')
                    >>> a = paddle.to_tensor(
                    ... [[1.123456780, 2., -3, .3],
                    ... [2, 3, +4., 1.2+10.34e-5j],
                    ... [3, 5.e-3, 1e2, 3e-8]]
                    ... )
                    >>> # Tensor(shape=[3, 4], dtype=complex64, place=Place(gpu:0), stop_gradient=True,
                    >>> #       [[ (1.1234568357467651+0j)                    ,
                    >>> #          (2+0j)                                     ,
                    >>> #         (-3+0j)                                     ,
                    >>> #          (0.30000001192092896+0j)                   ],
                    >>> #        [ (2+0j)                                     ,
                    >>> #          (3+0j)                                     ,
                    >>> #          (4+0j)                                     ,
                    >>> #         (1.2000000476837158+0.00010340000153519213j)],
                    >>> #        [ (3+0j)                                     ,
                    >>> #          (0.004999999888241291+0j)                  ,
                    >>> #          (100+0j)                                   ,
                    >>> #          (2.999999892949745e-08+0j)                 ]])
                    >>> print(a)
                    Tensor(shape=[3, 4], dtype=complex64, place=Place(AAA), stop_gradient=True,
                        [[ (1.123456+0j),
                            (2+0j),
                            (-3+0j),
                            (0.3+0j)],
                            [ (2+0j),
                            (3+0j),
                            (4+0j),
                            (1.2+0.00010340j)],
                            [ (3+0j),
                            (0.00499999+0j),
                            (100+0j),
                            (2.999999e-08+0j)]])
            """,
            'float_array': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('cpu')
                    >>> x = [[2, 3, 4], [7, 8, 9]]
                    >>> x = paddle.to_tensor(x, dtype='float32')
                    >>> print(paddle.log(x))
                    Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [[0.69314718, 1.09861231, 1.38629436],
                        [1.94591010, 2.07944155, 2.19722462]])

            """,
            'float_array_diff': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> paddle.device.set_device('cpu')
                    >>> x = [[2, 3, 4], [7, 8, 9]]
                    >>> x = paddle.to_tensor(x, dtype='float32')
                    >>> print(paddle.log(x))
                    Tensor(shape=[2, 3], dtype=float32, place=Place(cpu), stop_gradient=True,
                        [[0.69314712, 1.09861221, 1.386294],
                        [1.94591032, 2.07944156, 2.1972246]])

            """,
            'float_begin': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> print(7.0)
                    7.

            """,
            'float_begin_long': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> print(7.0000023)
                    7.0000024

            """,
            'float_begin_more': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> print(7.0, 5., 6.123456)
                    7.0 5.0 6.123457

            """,
            'float_begin_more_diff': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> print(7.0, 5., 6.123456)
                    7.0 5.0 6.123457

            """,
            'float_begin_more_brief': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> print(7.0, 5., 6.123456)
                    7. 5. 6.123457

            """,
            'float_begin_fail': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> print(7.0100023)
                    7.0000024

            """,
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 15)

        (
            tr_0,
            tr_1,
            tr_2,
            tr_3,
            tr_4,
            tr_5,
            tr_6,
            tr_7,
            tr_8,
            tr_9,
            tr_10,
            tr_11,
            tr_12,
            tr_13,
            tr_14,
        ) = test_results

        self.assertIn('gpu_to_gpu', tr_0.name)
        self.assertTrue(tr_0.passed)

        self.assertIn('cpu_to_cpu', tr_1.name)
        self.assertTrue(tr_1.passed)

        self.assertIn('gpu_to_cpu', tr_2.name)
        self.assertTrue(tr_2.passed)

        self.assertIn('cpu_to_gpu', tr_3.name)
        self.assertTrue(tr_3.passed)

        self.assertIn('gpu_to_cpu_array', tr_4.name)
        self.assertTrue(tr_4.passed)

        self.assertIn('cpu_to_gpu_array', tr_5.name)
        self.assertTrue(tr_5.passed)

        self.assertIn('mass_array', tr_6.name)
        self.assertTrue(tr_6.passed)

        self.assertIn('float_array', tr_7.name)
        self.assertTrue(tr_7.passed)

        self.assertIn('float_array_diff', tr_8.name)
        self.assertTrue(tr_8.passed)

        self.assertIn('float_begin', tr_9.name)
        self.assertTrue(tr_9.passed)

        self.assertIn('float_begin_long', tr_10.name)
        self.assertTrue(tr_10.passed)

        self.assertIn('float_begin_more', tr_11.name)
        self.assertTrue(tr_11.passed)

        self.assertIn('float_begin_more_diff', tr_12.name)
        self.assertTrue(tr_12.passed)

        self.assertIn('float_begin_more_brief', tr_13.name)
        self.assertTrue(tr_13.passed)

        self.assertIn('float_begin_fail', tr_14.name)
        self.assertFalse(tr_14.passed)

        # reload xdoctest.checker
        importlib.reload(xdoctest.checker)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(
            style='freeform', target='codeblock', patch_float_precision=None
        )
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 15)

        (
            tr_0,
            tr_1,
            tr_2,
            tr_3,
            tr_4,
            tr_5,
            tr_6,
            tr_7,
            tr_8,
            tr_9,
            tr_10,
            tr_11,
            tr_12,
            tr_13,
            tr_14,
        ) = test_results

        self.assertIn('gpu_to_gpu', tr_0.name)
        self.assertFalse(tr_0.passed)

        self.assertIn('cpu_to_cpu', tr_1.name)
        self.assertFalse(tr_1.passed)

        self.assertIn('gpu_to_cpu', tr_2.name)
        self.assertFalse(tr_2.passed)

        self.assertIn('cpu_to_gpu', tr_3.name)
        self.assertFalse(tr_3.passed)

        self.assertIn('gpu_to_cpu_array', tr_4.name)
        self.assertFalse(tr_4.passed)

        self.assertIn('cpu_to_gpu_array', tr_5.name)
        self.assertFalse(tr_5.passed)

        self.assertIn('mass_array', tr_6.name)
        self.assertFalse(tr_6.passed)

        self.assertIn('float_array', tr_7.name)
        self.assertTrue(tr_7.passed)

        self.assertIn('float_array_diff', tr_8.name)
        self.assertFalse(tr_8.passed)

        self.assertIn('float_begin', tr_9.name)
        self.assertFalse(tr_9.passed)

        self.assertIn('float_begin_long', tr_10.name)
        self.assertFalse(tr_10.passed)

        self.assertIn('float_begin_more', tr_11.name)
        self.assertFalse(tr_11.passed)

        self.assertIn('float_begin_more_diff', tr_12.name)
        self.assertFalse(tr_12.passed)

        self.assertIn('float_begin_more_brief', tr_13.name)
        self.assertFalse(tr_13.passed)

        self.assertIn('float_begin_fail', tr_14.name)
        self.assertFalse(tr_14.passed)

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

                >>> # doctest: +SKIP('skip')
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

                >>> # doctest: +SKIP('skip')
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

                >>> # doctest: +SKIP('skip')
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

                >>> # doctest: +SKIP('skip')
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

                >>> # doctest: +SKIP('skip')
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

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 0)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(style='freeform', target='docstring')
        doctester.prepare(test_capacity)

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

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 1)

        tr_0 = test_results[0]

        self.assertIn('one_plus_one', tr_0.name)
        self.assertIn('code-example', tr_0.name)
        self.assertFalse(tr_0.nocode)
        self.assertFalse(tr_0.passed)
        self.assertTrue(tr_0.skipped)
        self.assertFalse(tr_0.failed)

    def test_multiprocessing_xdoctester(self):
        docstrings_to_test = {
            'static_0': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import numpy as np
                    >>> import paddle
                    >>> paddle.enable_static()
                    >>> data = paddle.static.data(name='X', shape=[None, 2, 28, 28], dtype='float32')
            """,
            'static_1': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import numpy as np
                    >>> import paddle
                    >>> paddle.enable_static()
                    >>> data = paddle.static.data(name='X', shape=[None, 1, 28, 28], dtype='float32')

            """,
        }

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)

        tr_0, tr_1 = test_results

        self.assertIn('static_0', tr_0.name)
        self.assertTrue(tr_0.passed)

        self.assertIn('static_1', tr_1.name)
        self.assertTrue(tr_1.passed)

    def test_timeout(self):
        docstrings_to_test = {
            'timeout_false': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +TIMEOUT(10)
                    >>> import time
                    >>> time.sleep(0.1)
            """,
            'timeout_true': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +TIMEOUT(10)
                    >>> import time
                    >>> time.sleep(15)
            """,
            'timeout_false_with_skip_0': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +TIMEOUT(10)
                    >>> # doctest: +SKIP('skip')
                    >>> import time
                    >>> time.sleep(0.1)
            """,
            'timeout_false_with_skip_1': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +SKIP('skip')
                    >>> # doctest: +TIMEOUT(10)
                    >>> import time
                    >>> time.sleep(0.1)
            """,
            'timeout_true_with_skip_0': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +TIMEOUT(10)
                    >>> # doctest: +SKIP('skip')
                    >>> import time
                    >>> time.sleep(15)
            """,
            'timeout_true_with_skip_1': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +SKIP('skip')
                    >>> # doctest: +TIMEOUT(10)
                    >>> import time
                    >>> time.sleep(15)
            """,
            'timeout_more_codes': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +TIMEOUT(10)
                    >>> import time
                    >>> time.sleep(0.1)

                .. code-block:: python

                    >>> # doctest: +TIMEOUT(10)
                    >>> import time
                    >>> time.sleep(15)

            """,
        }

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 8)

        tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7 = test_results

        self.assertIn('timeout_false', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.timeout)

        self.assertIn('timeout_true', tr_1.name)
        self.assertFalse(tr_1.passed)
        self.assertTrue(tr_1.timeout)

        self.assertIn('timeout_false_with_skip_0', tr_2.name)
        self.assertFalse(tr_2.passed)
        self.assertFalse(tr_2.timeout)
        self.assertTrue(tr_2.skipped)

        self.assertIn('timeout_false_with_skip_1', tr_3.name)
        self.assertFalse(tr_3.passed)
        self.assertFalse(tr_3.timeout)
        self.assertTrue(tr_3.skipped)

        self.assertIn('timeout_true_with_skip_0', tr_4.name)
        self.assertFalse(tr_4.passed)
        self.assertFalse(tr_4.timeout)
        self.assertTrue(tr_4.skipped)

        self.assertIn('timeout_true_with_skip_1', tr_5.name)
        self.assertFalse(tr_5.passed)
        self.assertFalse(tr_5.timeout)
        self.assertTrue(tr_5.skipped)

        self.assertIn('timeout_more_codes', tr_6.name)
        self.assertTrue(tr_6.passed)
        self.assertFalse(tr_6.timeout)

        self.assertIn('timeout_more_codes', tr_7.name)
        self.assertFalse(tr_7.passed)
        self.assertTrue(tr_7.timeout)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(use_multiprocessing=False)
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 8)

        tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7 = test_results

        self.assertIn('timeout_false', tr_0.name)
        self.assertTrue(tr_0.passed)
        self.assertFalse(tr_0.timeout)

        self.assertIn('timeout_true', tr_1.name)
        self.assertFalse(tr_1.passed)
        self.assertTrue(tr_1.timeout)

        self.assertIn('timeout_false_with_skip_0', tr_2.name)
        self.assertFalse(tr_2.passed)
        self.assertFalse(tr_2.timeout)
        self.assertTrue(tr_2.skipped)

        self.assertIn('timeout_false_with_skip_1', tr_3.name)
        self.assertFalse(tr_3.passed)
        self.assertFalse(tr_3.timeout)
        self.assertTrue(tr_3.skipped)

        self.assertIn('timeout_true_with_skip_0', tr_4.name)
        self.assertFalse(tr_4.passed)
        self.assertFalse(tr_4.timeout)
        self.assertTrue(tr_4.skipped)

        self.assertIn('timeout_true_with_skip_1', tr_5.name)
        self.assertFalse(tr_5.passed)
        self.assertFalse(tr_5.timeout)
        self.assertTrue(tr_5.skipped)

        self.assertIn('timeout_more_codes', tr_6.name)
        self.assertTrue(tr_6.passed)
        self.assertFalse(tr_6.timeout)

        self.assertIn('timeout_more_codes', tr_7.name)
        self.assertFalse(tr_7.passed)
        self.assertTrue(tr_7.timeout)

    def test_bad_statements(self):
        docstrings_to_test = {
            'good_fluid': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import paddle.base
            """,
            'bad_fluid_from': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import paddle
                    >>> from paddle import fluid
            """,
            'no_bad': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +SKIP('reason')
                    >>> import os
            """,
            'bad_fluid_good_skip': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +SKIP('reason')
                    >>> import os
                    >>> from paddle import fluid
            """,
            'bad_fluid_bad_skip': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +SKIP('reason')
                    >>> import os
                    >>> from paddle import fluid
                    >>> # doctest: +SKIP
                    >>> import sys
            """,
            'bad_skip_mix': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +SKIP('reason')
                    >>> import os
                    >>> # doctest: +SKIP
                    >>> import sys
            """,
            'bad_skip': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # doctest: +SKIP
                    >>> import os

            """,
            'bad_skip_empty': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import os
                    >>> # doctest: +SKIP()
                    >>> import sys
            """,
            'good_skip': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import os
                    >>> # doctest: +SKIP('reason')
                    >>> import sys
                    >>> # doctest: -SKIP
                    >>> import math
            """,
            'comment_fluid': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> # import paddle.base
                    >>> import os
            """,
            'oneline_skip': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import os # doctest: +SKIP
                    >>> import sys
            """,
        }

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 11)

        (
            tr_0,
            tr_1,
            tr_2,
            tr_3,
            tr_4,
            tr_5,
            tr_6,
            tr_7,
            tr_8,
            tr_9,
            tr_10,
        ) = test_results

        self.assertIn('good_fluid', tr_0.name)
        self.assertFalse(tr_0.badstatement)
        self.assertTrue(tr_0.passed)

        self.assertIn('bad_fluid_from', tr_1.name)
        self.assertTrue(tr_1.badstatement)
        self.assertFalse(tr_1.passed)

        self.assertIn('no_bad', tr_2.name)
        self.assertFalse(tr_2.badstatement)
        self.assertFalse(tr_2.passed)
        self.assertTrue(tr_2.skipped)

        self.assertIn('bad_fluid_good_skip', tr_3.name)
        self.assertTrue(tr_3.badstatement)
        self.assertFalse(tr_3.passed)

        self.assertIn('bad_fluid_bad_skip', tr_4.name)
        self.assertTrue(tr_4.badstatement)
        self.assertFalse(tr_4.passed)

        self.assertIn('bad_skip_mix', tr_5.name)
        self.assertTrue(tr_5.badstatement)
        self.assertFalse(tr_5.passed)

        self.assertIn('bad_skip', tr_6.name)
        self.assertTrue(tr_6.badstatement)
        self.assertFalse(tr_6.passed)

        self.assertIn('bad_skip_empty', tr_7.name)
        self.assertTrue(tr_7.badstatement)
        self.assertFalse(tr_7.passed)

        self.assertIn('good_skip', tr_8.name)
        self.assertFalse(tr_8.badstatement)
        self.assertTrue(tr_8.passed)

        self.assertIn('comment_fluid', tr_9.name)
        self.assertFalse(tr_9.badstatement)
        self.assertTrue(tr_9.passed)

        self.assertIn('oneline_skip', tr_10.name)
        self.assertTrue(tr_10.badstatement)
        self.assertFalse(tr_10.passed)

    def test_bad_statements_req(self):
        docstrings_to_test = {
            'bad_required': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import sys
                    >>> # required: GPU
                    >>> import os
            """,
            'bad_requires': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import sys
                    >>> # requires: GPU
                    >>> import os
            """,
            'bad_require': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import sys
                    >>> # require   :   GPU
                    >>> import os
            """,
            'bad_require_2': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import sys
                    >>> # require: GPU, xpu
                    >>> import os
            """,
            'bad_req': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import sys
                    >>> #require:gpu
                    >>> import os
            """,
            'ignore_req': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import sys
                    >>> #require:
                    >>> import os
            """,
            'ignore_req_bad_req': """
            this is docstring...

            Examples:

                .. code-block:: python

                    >>> import sys
                    >>> #require: xpu
                    >>> import os
                    >>> #require:
                    >>> import os
            """,
        }

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 7)

        (
            tr_0,
            tr_1,
            tr_2,
            tr_3,
            tr_4,
            tr_5,
            tr_6,
        ) = test_results

        self.assertIn('bad_required', tr_0.name)
        self.assertTrue(tr_0.badstatement)
        self.assertFalse(tr_0.passed)

        self.assertIn('bad_requires', tr_1.name)
        self.assertTrue(tr_1.badstatement)
        self.assertFalse(tr_1.passed)

        self.assertIn('bad_require', tr_2.name)
        self.assertTrue(tr_1.badstatement)
        self.assertFalse(tr_1.passed)

        self.assertIn('bad_require_2', tr_3.name)
        self.assertTrue(tr_3.badstatement)
        self.assertFalse(tr_3.passed)

        self.assertIn('bad_req', tr_4.name)
        self.assertTrue(tr_4.badstatement)
        self.assertFalse(tr_4.passed)

        self.assertIn('ignore_req', tr_5.name)
        self.assertFalse(tr_5.badstatement)
        self.assertTrue(tr_5.passed)

        self.assertIn('ignore_req_bad_req', tr_6.name)
        self.assertTrue(tr_6.badstatement)
        self.assertFalse(tr_6.passed)

    @unittest.skipIf(
        not sys.platform.startswith('linux'),
        "CI checks on linux, we only care this situation about multiprocessing!"
        "Or pickle may fail.",
    )
    def test_single_process_directive(self):
        _clear_environ()

        # test set_default_dtype
        docstrings_to_test = {
            'no_solo': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import multiprocessing
                    >>> p = multiprocessing.Process(
                    ...     target=lambda a, b: a + b,
                    ...     args=(
                    ...     1,
                    ...     2,
                    ...     ),
                    ... )
                    >>> p.start()
                    >>> p.join()
            """,
            'has_solo': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # doctest: +SOLO('can not use add in multiprocess')
                    >>> import multiprocessing
                    >>> p = multiprocessing.Process(
                    ...     target=lambda a, b: a + b,
                    ...     args=(
                    ...     1,
                    ...     2,
                    ...     ),
                    ... )
                    >>> p.start()
                    >>> p.join()
            """,
        }

        # test old global_exec
        test_capacity = {'cpu'}
        doctester = Xdoctester()
        doctester.prepare(test_capacity)

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 2)

        (tr_0, tr_1) = test_results

        self.assertIn('no_solo', tr_0.name)
        self.assertFalse(tr_0.passed)

        self.assertIn('has_solo', tr_1.name)
        self.assertTrue(tr_1.passed)


if __name__ == '__main__':
    unittest.main()
