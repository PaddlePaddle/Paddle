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

import unittest
import warnings

import paddle
import paddle.version as base_version
from paddle import base


class VersionTest(unittest.TestCase):
    def test_check_output(self):
        warnings.warn(
            f"paddle.__version__: {paddle.__version__}, base_version.full_version: {base_version.full_version}, base_version.major: {base_version.major}, base_version.minor: {base_version.minor}, base_version.patch: {base_version.patch}, base_version.rc: {base_version.rc}."
        )
        ori_full_version = base_version.full_version
        ori_sep_version = [
            base_version.major,
            base_version.minor,
            base_version.patch,
            base_version.rc,
        ]
        [
            base_version.major,
            base_version.minor,
            base_version.patch,
            base_version.rc,
        ] = ['1', '4', '1', '0']

        base.require_version('1')
        base.require_version('1.4')
        base.require_version('1.4.1.0')

        # any version >= 1.4.1 is acceptable.
        base.require_version('1.4.1')

        # if 1.4.1 <= version <= 1.6.0, it is acceptable.
        base.require_version(min_version='1.4.1', max_version='1.6.0')

        # only version 1.4.1 is acceptable.
        base.require_version(min_version='1.4.1', max_version='1.4.1')

        # if installed version is 0.0.0.0, throw warning and skip the checking.
        [
            base_version.major,
            base_version.minor,
            base_version.patch,
            base_version.rc,
        ] = ['0', '0', '0', '0']
        base.require_version('0.0.0')

        base_version.full_version = ori_full_version
        [
            base_version.major,
            base_version.minor,
            base_version.patch,
            base_version.rc,
        ] = ori_sep_version


# Test Errors
class TestErrors(unittest.TestCase):
    def test_errors(self):
        # The type of params must be str.
        def test_input_type():
            base.require_version(100)

        self.assertRaises(TypeError, test_input_type)

        def test_input_type_1():
            base.require_version('0', 200)

        self.assertRaises(TypeError, test_input_type_1)

        # The value of params must be in format r'\d+(\.\d+){0,3}', like '1.5.2.0', '1.6' ...
        def test_input_value_1():
            base.require_version('string')

        self.assertRaises(ValueError, test_input_value_1)

        def test_input_value_1_1():
            base.require_version('1.5', 'string')

        self.assertRaises(ValueError, test_input_value_1_1)

        def test_input_value_2():
            base.require_version('1.5.2.0.0')

        self.assertRaises(ValueError, test_input_value_2)

        def test_input_value_2_1():
            base.require_version('1.5', '1.5.2.0.0')

        self.assertRaises(ValueError, test_input_value_2_1)

        def test_input_value_3():
            base.require_version('1.5.2a.0')

        self.assertRaises(ValueError, test_input_value_3)

        # The installed version must be equal or greater than the required version.
        def test_version():
            base.require_version('100')

        # The installed version must be in [min_version, max_version]
        def test_version_1():
            base.require_version('0.0.0', '1.4')

        def test_version_2():
            base.require_version('1.4.0', '1.2')

        ori_full_version = base_version.full_version
        ori_sep_version = [
            base_version.major,
            base_version.minor,
            base_version.patch,
            base_version.rc,
        ]
        [
            base_version.major,
            base_version.minor,
            base_version.patch,
            base_version.rc,
        ] = ['1', '4', '1', '0']

        self.assertRaises(Exception, test_version)
        self.assertRaises(Exception, test_version_1)
        self.assertRaises(Exception, test_version_2)

        base_version.full_version = ori_full_version
        [
            base_version.major,
            base_version.minor,
            base_version.patch,
            base_version.rc,
        ] = ori_sep_version


if __name__ == "__main__":
    unittest.main()
