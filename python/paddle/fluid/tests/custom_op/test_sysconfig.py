# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import os
import unittest
import paddle


class SysConfigTest(unittest.TestCase):
    def test_include(self):
        inc_dir = paddle.sysconfig.get_include()
        inc_dirs = inc_dir.split(os.sep)
        self.assertEqual(inc_dirs[-1], 'include')
        self.assertEqual(inc_dirs[-2], 'paddle')

    def test_libs(self):
        lib_dir = paddle.sysconfig.get_lib()
        lib_dirs = lib_dir.split(os.sep)
        self.assertEqual(lib_dirs[-1], 'libs')
        self.assertEqual(lib_dirs[-2], 'paddle')


if __name__ == '__main__':
    unittest.main()
