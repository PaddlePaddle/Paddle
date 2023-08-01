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
import unittest

import xdoctest
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
    def test_patch_xdoctest(self):
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
                    >>> # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [0.20000000])
                    >>> print(a)
                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [0.20000000])

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
                    >>> # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [0.20000000])
                    >>> print(a)
                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.20000000])

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
                    >>> # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True, [0.20000000])
                    >>> print(a)
                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.20000000])

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
                    >>> # Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True, [0.20000000])
                    >>> print(a)
                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [0.20000000])
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
                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [0.123456780])

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
                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.123456780])

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
                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.123456780])

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
                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [0.123456780])
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
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 9)

        tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8 = test_results

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

        # reload xdoctest.checker
        importlib.reload(xdoctest.checker)

        _clear_environ()

        test_capacity = {'cpu'}
        doctester = Xdoctester(
            style='freeform', target='codeblock', patch_float_precision=False
        )
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
                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [0.123456780])

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
                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.123456780])

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
                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.123456780])

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
                    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
                    [0.123456780])
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
        }

        test_results = get_test_results(doctester, docstrings_to_test)
        self.assertEqual(len(test_results), 9)

        tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7, tr_8 = test_results

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
