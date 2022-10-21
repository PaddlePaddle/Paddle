# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.utils.lazy_import import try_import


class TestUtilsLazyImport(unittest.TestCase):

    def setup(self):
        pass

    def func_test_lazy_import(self):
        paddle = try_import('paddle')
        self.assertTrue(paddle.__version__ is not None)

        with self.assertRaises(ImportError) as context:
            paddle2 = try_import('paddle2')

        self.assertTrue('require additional dependencies that have to be' in
                        str(context.exception))

        with self.assertRaises(ImportError) as context:
            paddle2 = try_import('paddle2', 'paddle2 is not installed')

        self.assertTrue('paddle2 is not installed' in str(context.exception))

    def test_lazy_import(self):
        self.func_test_lazy_import()


if __name__ == "__main__":
    unittest.main()
