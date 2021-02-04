# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os

import paddle.utils.cpp_extension.extension_utils as utils


class TestABIBase(unittest.TestCase):
    def test_environ(self):
        compiler = 'gcc'
        for flag in ['1', 'True', 'true']:
            os.environ['PADDLE_SKIP_CHECK_ABI'] = flag
            self.assertTrue(utils.check_abi_compatibility(compiler))

    def del_environ(self):
        key = 'PADDLE_SKIP_CHECK_ABI'
        if key in os.environ:
            del os.environ[key]


class TestCheckLinux(TestABIBase):
    def test_expected_compiler(self):
        if utils.OS_NAME.startswith('linux'):
            gt = ['gcc', 'g++', 'gnu-c++', 'gnu-cc']
            self.assertListEqual(utils._expected_compiler_current_platform(),
                                 gt)

    def test_gcc_version(self):
        if utils.OS_NAME.startswith('linux'):
            # all CI gcc version > 5.4.0
            self.assertTrue(utils.check_abi_compatibility())


class TestCheckMacOs(TestABIBase):
    def test_expected_compiler(self):
        if utils.OS_NAME.startswith('darwin'):
            gt = ['clang', 'clang++']
            self.assertListEqual(utils._expected_compiler_current_platform(),
                                 gt)

    def test_gcc_version(self):
        # clear environ
        self.del_environ()

        if utils.OS_NAME.startswith('darwin'):
            # clang has no version limitation.
            self.assertTrue(utils.check_abi_compatibility())


class TestCheckWindows(TestABIBase):
    def test_gcc_version(self):
        # clear environ
        self.del_environ()
        if utils.IS_WINDOWS:
            # we skip windows now
            self.assertTrue(utils.check_abi_compatibility())


if __name__ == '__main__':
    unittest.main()
