# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import unittest
import warnings

import numpy as np

import paddle
from paddle import _legacy_C_ops
from paddle.utils import deprecated

LOWEST_WARNING_POSTION = 3
ERROR_WARNING_POSTION = sys.maxsize

# custom paddle version
paddle.version.major = '1'
paddle.version.minor = '8'
paddle.version.patch = '0'
paddle.version.rc = '0'
paddle.__version__ = '1.8.0'
paddle.version.full_version = '1.8.0'
print("current paddle version: ", paddle.__version__)

paddle.disable_static()


def get_warning_index(api):
    """
    Given an paddle API, return the index of the Warinng information in its doc string if exists;
    If Warinng information doesn't exist, return the default ERROR_WARNING_POSTION, sys.maxsize.

    Args:
        API (python object)

    Returns:
        index (int): the index of the Warinng information in its doc string if exists.
    """

    doc_lst = api.__doc__.splitlines()
    for idx, val in enumerate(doc_lst):
        if (
            val.startswith("Warning: ")
            and val.endswith(" instead.")
            and "and will be removed in future versions." in val
        ):
            return idx
    return ERROR_WARNING_POSTION


class TestDeprecatedDocorator(unittest.TestCase):
    """
    tests for paddle's Deprecated Docorator.
    test_new_multiply: test for new api, which should not insert warning information.
    test_ops_elementwise_mul: test for C++ elementwise_mul op, which should not insert warning information.
    """

    def test_new_multiply(self):
        """
        Test for new multiply api, expected result should be False.
        """

        a = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        b = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        x = paddle.to_tensor(a)
        y = paddle.to_tensor(b)
        res = paddle.multiply(x, y)

        # expected
        expected = LOWEST_WARNING_POSTION

        # captured
        captured = get_warning_index(paddle.multiply)

        # testting
        self.assertLess(expected, captured)

    def test_ops_elementwise_mul(self):
        """
        Test for new C++ elementwise_op, expected result should be True,
        because not matter what base.layers.elementwise_mul is deprecated.
        """

        a = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        b = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        x = paddle.to_tensor(a)
        y = paddle.to_tensor(b)
        res = _legacy_C_ops.elementwise_mul(x, y)

        # expected
        expected = LOWEST_WARNING_POSTION

        # captured
        captured = get_warning_index(paddle.multiply)

        # testting
        self.assertGreater(expected, captured)

    def test_tensor_gradient(self):
        paddle.__version__ = '2.1.0'

        x = paddle.to_tensor([5.0], stop_gradient=False)
        y = paddle.pow(x, 4.0)
        y.backward()

        with warnings.catch_warnings(record=True) as w:
            grad = x.gradient()
            assert (
                'API "paddle.base.dygraph.tensor_patch_methods.gradient" is '
                'deprecated since 2.1.0'
            ) in str(w[-1].message)

    def test_softmax_with_cross_entropy(self):
        paddle.__version__ = '2.0.0'

        data = np.random.rand(128).astype("float32")
        label = np.random.rand(1).astype("int64")
        data = paddle.to_tensor(data)
        label = paddle.to_tensor(label)
        linear = paddle.nn.Linear(128, 100)
        x = linear(data)

        with warnings.catch_warnings(record=True) as w:
            out = paddle.nn.functional.softmax_with_cross_entropy(
                logits=x, label=label
            )
            assert (
                'API "paddle.nn.functional.loss.softmax_with_cross_entropy" is '
                'deprecated since 2.0.0'
            ) in str(w[-1].message)

    def test_deprecated_error(self):
        paddle.__version__ = '2.1.0'

        @deprecated(since="2.1.0", level=2)
        def deprecated_error_func():
            pass

        self.assertRaises(RuntimeError, deprecated_error_func)


if __name__ == '__main__':
    unittest.main()
