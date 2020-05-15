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

from __future__ import print_function

import unittest

from paddle.fluid.dygraph.dygraph_to_static.utils import index_in_list


class TestIndexInList(unittest.TestCase):
    def test_index_in_list(self):
        list_to_test = [1, 2, 3, 4, 5]
        self.assertEqual(index_in_list(list_to_test, 4), 3)
        self.assertEqual(index_in_list(list_to_test, 1), 0)
        self.assertEqual(index_in_list(list_to_test, 5), 4)
        self.assertEqual(index_in_list(list_to_test, 0), -1)
        self.assertEqual(index_in_list(list_to_test, 6), -1)


if __name__ == '__main__':
    unittest.main()
