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
import paddle.fluid as fluid
import paddle.version as fluid_version
import warnings
import paddle


class VersionTest(unittest.TestCase):

    def test_check_output(self):
        warnings.warn(
            "paddle.__version__: %s, fluid_version.full_version: %s, fluid_version.major: %s, fluid_version.minor: %s, fluid_version.patch: %s, fluid_version.rc: %s."
            % (paddle.__version__, fluid_version.full_version,
               fluid_version.major, fluid_version.minor, fluid_version.patch,
               fluid_version.rc))
        ori_full_version = fluid_version.full_version
        ori_sep_version = [
            fluid_version.major, fluid_version.minor, fluid_version.patch,
            fluid_version.rc
        ]
        [
            fluid_version.major, fluid_version.minor, fluid_version.patch,
            fluid_version.rc
        ] = ['1', '4', '1', '0']

        fluid.require_version('1')
        fluid.require_version('1.4')
        fluid.require_version('1.4.1.0')

        # any version >= 1.4.1 is acceptable.
        fluid.require_version('1.4.1')

        # if 1.4.1 <= version <= 1.6.0, it is acceptable.
        fluid.require_version(min_version='1.4.1', max_version='1.6.0')

        # only version 1.4.1 is acceptable.
        fluid.require_version(min_version='1.4.1', max_version='1.4.1')

        # if installed version is 0.0.0.0, throw warning and skip the checking.
        [
            fluid_version.major, fluid_version.minor, fluid_version.patch,
            fluid_version.rc
        ] = ['0', '0', '0', '0']
        fluid.require_version('0.0.0')

        fluid_version.full_version = ori_full_version
        [
            fluid_version.major, fluid_version.minor, fluid_version.patch,
            fluid_version.rc
        ] = ori_sep_version


# Test Errors
class TestErrors(unittest.TestCase):

    def test_errors(self):
        # The type of params must be str.
        def test_input_type():
            fluid.require_version(100)

        self.assertRaises(TypeError, test_input_type)

        def test_input_type_1():
            fluid.require_version('0', 200)

        self.assertRaises(TypeError, test_input_type_1)

        # The value of params must be in format r'\d+(\.\d+){0,3}', like '1.5.2.0', '1.6' ...
        def test_input_value_1():
            fluid.require_version('string')

        self.assertRaises(ValueError, test_input_value_1)

        def test_input_value_1_1():
            fluid.require_version('1.5', 'string')

        self.assertRaises(ValueError, test_input_value_1_1)

        def test_input_value_2():
            fluid.require_version('1.5.2.0.0')

        self.assertRaises(ValueError, test_input_value_2)

        def test_input_value_2_1():
            fluid.require_version('1.5', '1.5.2.0.0')

        self.assertRaises(ValueError, test_input_value_2_1)

        def test_input_value_3():
            fluid.require_version('1.5.2a.0')

        self.assertRaises(ValueError, test_input_value_3)

        # The installed version must be equal or greater than the required version.
        def test_version():
            fluid.require_version('100')

        # The installed version must be in [min_version, max_version]
        def test_version_1():
            fluid.require_version('0.0.0', '1.4')

        def test_version_2():
            fluid.require_version('1.4.0', '1.2')

        ori_full_version = fluid_version.full_version
        ori_sep_version = [
            fluid_version.major, fluid_version.minor, fluid_version.patch,
            fluid_version.rc
        ]
        [
            fluid_version.major, fluid_version.minor, fluid_version.patch,
            fluid_version.rc
        ] = ['1', '4', '1', '0']

        self.assertRaises(Exception, test_version)
        self.assertRaises(Exception, test_version_1)
        self.assertRaises(Exception, test_version_2)

        fluid_version.full_version = ori_full_version
        [
            fluid_version.major, fluid_version.minor, fluid_version.patch,
            fluid_version.rc
        ] = ori_sep_version


if __name__ == "__main__":
    unittest.main()
