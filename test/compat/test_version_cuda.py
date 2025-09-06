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

import unittest

from paddle.version import cuda


class TestCudaVariable(unittest.TestCase):
    def test_cuda_functionality(self):
        if cuda:
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

        else:
            print("no CUDA ")


if __name__ == "__main__":
    unittest.main()
