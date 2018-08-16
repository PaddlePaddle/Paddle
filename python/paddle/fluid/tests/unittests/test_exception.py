#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import paddle.compat as cpt
import paddle.fluid.core as core
import unittest


class TestException(unittest.TestCase):
    def test_exception(self):
        exception = None
        try:
            core.__unittest_throw_exception__()
        except core.EnforceNotMet as ex:
            self.assertIn("test exception", cpt.get_exception_message(ex))
            exception = ex

        self.assertIsNotNone(exception)


if __name__ == "__main__":
    unittest.main()
