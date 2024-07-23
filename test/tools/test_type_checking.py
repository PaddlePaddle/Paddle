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

import pathlib
import unittest

# set `PYTHONPATH=<Paddle/tools>` from `CMakeLists.txt`
from type_checking import MypyChecker, get_test_results

FILE_PATH = pathlib.Path(__file__).resolve().parent

# test from `Paddle/test`
BASE_PATH_IN_TEST = FILE_PATH.parent.parent
CONFIG_FILE_IN_TEST = BASE_PATH_IN_TEST / 'pyproject.toml'
CACHE_DIR_IN_TEST = BASE_PATH_IN_TEST / '.mypy_cache'

# test from `Paddle/build/test`
BASE_PATH_IN_BUILD = FILE_PATH.parent.parent.parent
CONFIG_FILE_IN_BUILD = BASE_PATH_IN_BUILD / 'pyproject.toml'
CACHE_DIR_IN_BUILD = BASE_PATH_IN_BUILD / '.mypy_cache'

if CONFIG_FILE_IN_TEST.exists():
    CONFIG_FILE = CONFIG_FILE_IN_TEST
    CACHE_DIR = CACHE_DIR_IN_TEST
elif CONFIG_FILE_IN_BUILD.exists():
    CONFIG_FILE = CONFIG_FILE_IN_BUILD
    CACHE_DIR = CACHE_DIR_IN_BUILD
else:
    raise FileNotFoundError('Can NOT found mypy config file `pyproject.toml`')


class TestMypyChecker(unittest.TestCase):
    def test_mypy_pass(self):
        docstrings_pass = {
            'simple': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import abc
                    >>> print(1)
                    1
            """,
            'multi': """
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
            """,
        }
        docstrings_from_sampcd = {
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
        doctester = MypyChecker(CONFIG_FILE, CACHE_DIR)

        test_results = get_test_results(doctester, docstrings_pass)
        self.assertEqual(len(test_results), 3)

        for tr in test_results:
            self.assertFalse(tr.fail)

        test_results = get_test_results(doctester, docstrings_from_sampcd)
        self.assertEqual(len(test_results), 15)

        for tr in test_results:
            print(tr.msg)
            self.assertFalse(tr.fail)

    def test_mypy_fail(self):
        docstrings_fail = {
            'fail_simple': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import blabla
            """,
            'multi': """
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
                    >>> blabla
                    >>> print(1-1)
                    0

                .. code-block:: python
                    :name: code-example-2

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)
                    >>> blabla
                    >>> print(1-1)
                    0
            """,
        }

        doctester = MypyChecker(CONFIG_FILE, CACHE_DIR)

        test_results = get_test_results(doctester, docstrings_fail)
        self.assertEqual(len(test_results), 3)

        for tr in test_results:
            self.assertTrue(tr.fail)

    def test_mypy_partial_fail(self):
        docstrings_fail = {
            'multi': """
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
                    >>> blabla
                    >>> print(1-1)
                    0

                .. code-block:: python
                    :name: code-example-2

                    this is some blabla...

                    >>> # doctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)
                    >>> print(1-1)
                    0
            """
        }

        doctester = MypyChecker(CONFIG_FILE, CACHE_DIR)

        test_results = get_test_results(doctester, docstrings_fail)
        self.assertEqual(len(test_results), 2)

        tr_0, tr_1 = test_results
        self.assertTrue(tr_0.fail)
        self.assertFalse(tr_1.fail)

    def test_mypy_ignore(self):
        docstrings_ignore = {
            'fail_simple': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> # type: ignore
                    >>> import blabla
            """,
            'multi': """
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

                    >>> # type: ignore
                    >>> # doctest: -REQUIRES(env:GPU)
                    >>> blabla
                    >>> print(1-1)
                    0

                .. code-block:: python
                    :name: code-example-2

                    this is some blabla...

                    >>> # type: ignore
                    >>> # doctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)
                    >>> blabla
                    >>> print(1-1)
                    0
            """,
        }

        doctester = MypyChecker(CONFIG_FILE, CACHE_DIR)

        test_results = get_test_results(doctester, docstrings_ignore)
        self.assertEqual(len(test_results), 3)

        for tr in test_results:
            print(tr.msg)
            self.assertFalse(tr.fail)

        docstrings_pass = {
            'pass': """
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

                    >>> a = 1
                    >>> # type: ignore
                    >>> # doctest: -REQUIRES(env:GPU)
                    >>> blabla
                    >>> print(1-1)
                    0

                .. code-block:: python
                    :name: code-example-2

                    this is some blabla...

                    >>> b = 2
                    >>> # type: ignore
                    >>> # doctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)
                    >>> blabla
                    >>> print(1-1)
                    0
            """,
        }

        doctester = MypyChecker(CONFIG_FILE, CACHE_DIR)

        test_results = get_test_results(doctester, docstrings_pass)
        self.assertEqual(len(test_results), 2)

        for tr in test_results:
            print(tr.msg)
            self.assertFalse(tr.fail)

        docstrings_fail = {
            'fail': """
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

                    >>> import blabla
                    >>> a = 1
                    >>> # type: ignore
                    >>> # doctest: -REQUIRES(env:GPU)
                    >>> blabla
                    >>> print(1-1)
                    0

                .. code-block:: python
                    :name: code-example-2

                    this is some blabla...

                    >>> import blabla
                    >>> # type: ignore
                    >>> # doctest: +REQUIRES(env:GPU, env:XPU, env: DISTRIBUTED)
                    >>> blabla
                    >>> print(1-1)
                    0
            """,
        }

        doctester = MypyChecker(CONFIG_FILE, CACHE_DIR)

        test_results = get_test_results(doctester, docstrings_fail)
        self.assertEqual(len(test_results), 2)

        for tr in test_results:
            print(tr.msg)
            self.assertTrue(tr.fail)


if __name__ == '__main__':
    unittest.main()
