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

import paddle


class TestCAPI(unittest.TestCase):
    def test_glibcxx_use_cxx11_abi(self):
        val = paddle._C._GLIBCXX_USE_CXX11_ABI
        self.assertIsInstance(
            val, bool, "_GLIBCXX_USE_CXX11_ABI should return a bool"
        )

    def test_get_custom_class_python_wrapper_not_found(self):
        with self.assertRaises(Exception) as cm:
            paddle._C._get_custom_class_python_wrapper("fake_ns", "FakeClass")
        self.assertIn("not found", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
