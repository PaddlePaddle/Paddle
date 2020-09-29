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

from __future__ import print_function
import paddle
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.static import Program, program_guard
import unittest
import paddle.fluid.core as core
import sys

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
        if val.startswith("Warning: ") and val.endswith(
                " instead."
        ) and "and will be removed in future versions." in val:
            return idx
    return ERROR_WARNING_POSTION


class TestDeprecatedDocorator(unittest.TestCase):
    """
    tests for paddle's Deprecated Docorator.
    test_fluid_data: test for old fluid.data API.
    test_fluid_elementwise_mul: test for old fluid.layers.elementwise_xxx APIs.
    test_new_multiply: test for new api, which should not insert warning information.
    test_ops_elementwise_mul: test for C++ elementwise_mul op, which should not insert warning information.
    """

    def test_fluid_data(self):
        """
        test old fluid elementwise_mul api, it should fire Warinng function, 
        which insert the Warinng info on top of API's doc string.
        """
        paddle.enable_static()
        # Initialization
        x = fluid.data(name='x', shape=[3, 2, 1], dtype='float32')

        # expected
        expected = LOWEST_WARNING_POSTION

        # captured        
        captured = get_warning_index(fluid.data)
        paddle.disable_static()

        # testting
        self.assertGreater(expected, captured)

    def test_fluid_elementwise_mul(self):
        """
        test old fluid elementwise_mul api, it should trigger Warinng function, 
        which insert the Warinng info on top of API's doc string.
        """

        # Initialization
        a = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        b = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        x = paddle.to_tensor(a)
        y = paddle.to_tensor(b)
        res = fluid.layers.elementwise_mul(x, y)

        # expected
        expected = LOWEST_WARNING_POSTION

        # captured   
        captured = get_warning_index(fluid.layers.elementwise_mul)

        # testting
        self.assertGreater(expected, captured)

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
        because not matter what fluid.layers.elementwise_mul is deprecated.
        """

        a = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        b = np.random.uniform(0.1, 1, [51, 76]).astype(np.float32)
        x = paddle.to_tensor(a)
        y = paddle.to_tensor(b)
        res = core.ops.elementwise_mul(x, y)

        # expected
        expected = LOWEST_WARNING_POSTION

        # captured        
        captured = get_warning_index(fluid.layers.elementwise_mul)

        # testting
        self.assertGreater(expected, captured)


if __name__ == '__main__':
    unittest.main()
