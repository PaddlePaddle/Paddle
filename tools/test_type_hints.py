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

from type_hints import MypyChecker, get_test_results


class TestMypyChecker(unittest.TestCase):
    def test_single_code(self):
        docstrings = {
            'pass': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import paddle
                    >>> a = paddle.to_tensor(.2)
                    >>> print(a)
                    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                    [0.20000000])
            """,
            'fail': """
            placeholder

            Examples:

                .. code-block:: python
                    :name: code-example-1

                    this is some blabla...

                    >>> import blabla
            """,
        }
        doctester = MypyChecker('./mypy.ini')

        test_results = get_test_results(doctester, docstrings)
        self.assertEqual(len(test_results), 2)

        tr_0, tr_1 = test_results

        self.assertEqual(tr_0.exit_status, 0)
        self.assertEqual(tr_1.exit_status, 1)
