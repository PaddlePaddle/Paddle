# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest

from test_case_base import TestCaseBase


# copy from python library _distutils_hack/__init__.py
def find_spec(self, fullname, path, target=None):
    method_name = 'spec_for_{fullname}'.format(
        **{'self': self, 'fullname': fullname}
    )
    method = getattr(self, method_name, lambda: None)
    return method()


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(find_spec, "self", "fullname", "path", None)


if __name__ == "__main__":
    unittest.main()
