#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import unittest

=======
from __future__ import print_function

import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import paddle


class TestDygraphModeOfUnittest(unittest.TestCase):
<<<<<<< HEAD
    def test_dygraph_mode(self):
        self.assertTrue(
            paddle.in_dynamic_mode(),
            'Default Mode of Unittest should be dygraph mode, but get static graph mode.',
=======

    def test_dygraph_mode(self):
        self.assertTrue(
            paddle.in_dynamic_mode(),
            'Default Mode of Unittest should be dygraph mode, but get static mode.'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        )


if __name__ == '__main__':
    unittest.main()
