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
import warnings

import paddle.utils.cpp_extension.extension_utils as utils


class TestABIBase(unittest.TestCase):

    def test_environ(self):
        compiler_list = ['gcc', 'cl']
        for compiler in compiler_list:
            for flag in ['1', 'True', 'true']:
                os.environ['PADDLE_SKIP_CHECK_ABI'] = flag
                self.assertTrue(utils.check_abi_compatibility(compiler))

    def del_environ(self):
        key = 'PADDLE_SKIP_CHECK_ABI'
        if key in os.environ:
            del os.environ[key]


class TestCheckCompiler(TestABIBase):

    def test_expected_compiler(self):
        if utils.OS_NAME.startswith('linux'):
            gt = ['gcc', 'g++', 'gnu-c++', 'gnu-cc']
        elif utils.IS_WINDOWS:
            gt = ['cl']
        elif utils.OS_NAME.startswith('darwin'):
            gt = ['clang', 'clang++']

        self.assertListEqual(utils._expected_compiler_current_platform(), gt)

    def test_compiler_version(self):
        # clear environ
        self.del_environ()
        if utils.OS_NAME.startswith('linux'):
            compiler = 'g++'
        elif utils.IS_WINDOWS:
            compiler = 'cl'
        else:
            compiler = 'clang'

        # Linux: all CI gcc version > 5.4.0
        # Windows: all CI MSVC version > 19.00.24215
        # Mac: clang has no version limitation, always return true
        self.assertTrue(utils.check_abi_compatibility(compiler, verbose=True))

    def test_wrong_compiler_warning(self):
        # clear environ
        self.del_environ()
        compiler = 'python'  # fake wrong compiler
        if not utils.IS_WINDOWS:
            with warnings.catch_warnings(record=True) as error:
                flag = utils.check_abi_compatibility(compiler, verbose=True)
                # check return False
                self.assertFalse(flag)
                # check Compiler Compatibility WARNING
                self.assertTrue(len(error) == 1)
                self.assertTrue(
                    "Compiler Compatibility WARNING" in str(error[0].message))

    def test_exception_windows(self):
        # clear environ
        self.del_environ()
        compiler = 'fake compiler'  # fake command
        if utils.IS_WINDOWS:
            with warnings.catch_warnings(record=True) as error:
                flag = utils.check_abi_compatibility(compiler, verbose=True)
                # check return False
                self.assertFalse(flag)
                # check ABI Compatibility WARNING
                self.assertTrue(len(error) == 1)
                self.assertTrue("Failed to check compiler version for" in str(
                    error[0].message))

    def test_exception_linux(self):
        # clear environ
        self.del_environ()
        compiler = 'python'  # fake command
        if utils.OS_NAME.startswith('linux'):

            def fake():
                return [compiler]

            # mock a fake function
            raw_func = utils._expected_compiler_current_platform
            utils._expected_compiler_current_platform = fake
            with warnings.catch_warnings(record=True) as error:
                flag = utils.check_abi_compatibility(compiler, verbose=True)
                # check return False
                self.assertFalse(flag)
                # check ABI Compatibility WARNING
                self.assertTrue(len(error) == 1)
                self.assertTrue("Failed to check compiler version for" in str(
                    error[0].message))

            # restore
            utils._expected_compiler_current_platform = raw_func

    def test_exception_mac(self):
        # clear environ
        self.del_environ()
        compiler = 'python'  # fake command
        if utils.OS_NAME.startswith('darwin'):

            def fake():
                return [compiler]

            # mock a fake function
            raw_func = utils._expected_compiler_current_platform
            utils._expected_compiler_current_platform = fake
            with warnings.catch_warnings(record=True) as error:
                flag = utils.check_abi_compatibility(compiler, verbose=True)
                # check return True
                self.assertTrue(flag)
                # check ABI Compatibility without WARNING
                self.assertTrue(len(error) == 0)

            # restore
            utils._expected_compiler_current_platform = raw_func


class TestRunCMDException(unittest.TestCase):

    def test_exception(self):
        for verbose in [True, False]:
            with self.assertRaisesRegexp(RuntimeError, "Failed to run command"):
                cmd = "fake cmd"
                utils.run_cmd(cmd, verbose)


if __name__ == '__main__':
    unittest.main()
