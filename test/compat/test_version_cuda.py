# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
import unittest

from paddle.version import cuda


class TestCudaVariable(unittest.TestCase):
    def test_has_signature(self):
        self.assertTrue(hasattr(cuda, '__signature__'))
        self.assertIsInstance(cuda.__signature__, inspect.Signature)
        self.assertEqual(len(cuda.__signature__.parameters), 0)

    def test_has_doc(self):
        self.assertTrue(hasattr(cuda, '__doc__'))
        self.assertIsInstance(cuda.__doc__, str)
        self.assertTrue(len(cuda.__doc__.strip()) > 0)

    def test_inspect_recognizes(self):
        self.assertTrue(inspect.getdoc(cuda))
        self.assertIsInstance(inspect.signature(cuda), inspect.Signature)

    def test_cuda_functionality(self):
        self.assertIsInstance(cuda, str)
        self.assertTrue(len(cuda) > 0)
        self.assertEqual(str(cuda), cuda)
        self.assertTrue(callable(cuda))
        self.assertTrue(
            hasattr(cuda, 'startswith'),
            "Return value of cuda does not have 'startswith' attribute",
        )
        result = cuda()
        self.assertIsInstance(result, str)
        self.assertEqual(result, cuda)
        self.assertTrue(
            hasattr(result, 'startswith'),
            "Return value of cuda() does not have 'startswith' attribute",
        )


if __name__ == "__main__":
    unittest.main()
