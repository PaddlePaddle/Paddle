# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import re

import paddle.version as fluid_version


class VersionTest(unittest.TestCase):

    def setUp(self):
        self._major_regex = "[0-9]+"
        self._minor_regex = "[0-9]+"
        self._patch_regex = "[0-9]+(\\.(a|b|rc)\\.[0-9]+)?"
        self._rc_regex = "[0-9]+"
        self._version_regex = "[0-9]+\\.[0-9]+\\.[0-9]+(\\.(a|b|rc)\\.[0-9]+)?"
        self._commit_regex = "[0-9a-f]{5,49}"

    def test_check_output(self):
        # check commit format
        self.assertTrue(re.match(self._commit_regex, fluid_version.commit))
        self.assertTrue(isinstance(fluid_version.istaged, bool))

        # check version format
        if fluid_version.istaged:
            self.assertEqual(fluid_version.major, 0)
            self.assertEqual(fluid_version.minor, 0)
            self.assertEqual(fluid_version.patch, "0")
            self.assertEqual(fluid_version.rc, 0)
            self.assertEqual(fluid_version.full_version, "0.0.0")
        else:
            self.assertTrue(re.match(self._major_regex, fluid_version.major))
            self.assertTrue(re.match(self._minor_regex, fluid_version.minor))
            self.assertTrue(re.match(self._patch_regex, fluid_version.patch))
            self.assertTrue(re.match(self._rc_regex, fluid_version.rc))
            self.assertTrue(
                re.match(self._version_regex, fluid_version.full_version))
