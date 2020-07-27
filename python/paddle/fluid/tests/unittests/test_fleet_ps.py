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

import unittest
from paddle.fluid.framework import default_main_program
main_program = default_main_program()


class TestFleetPS(unittest.TestCase):
    def test_version(self):
        from paddle.fluid.incubate.fleet.parameter_server import version
        transpiler = version.is_transpiler()
        self.assertEqual(transpiler, True)


if __name__ == '__main__':
    unittest.main()
