# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.v2.plot import Ploter


class TestCommon(unittest.TestCase):
    def test_append(self):
        title1 = "title1"
        title2 = "title2"
        plot_test = Ploter(title1, title2)
        plot_test.append(title1, 1, 2)
        plot_test.append(title1, 2, 5)
        plot_test.append(title2, 3, 4)
        self.assertEqual(plot_test.__plot_data__[title1].step, [1, 2])
        self.assertEqual(plot_test.__plot_data__[title1].value, [2, 5])
        self.assertEqual(plot_test.__plot_data__[title2].step, [3])
        self.assertEqual(plot_test.__plot_data__[title2].value, [4])
        plot_test.reset()
        self.assertEqual(plot_test.__plot_data__[title1].step, [])
        self.assertEqual(plot_test.__plot_data__[title1].value, [])
        self.assertEqual(plot_test.__plot_data__[title2].step, [])
        self.assertEqual(plot_test.__plot_data__[title2].value, [])


if __name__ == '__main__':
    unittest.main()
