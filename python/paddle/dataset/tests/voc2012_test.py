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

<<<<<<< HEAD
import unittest

import paddle.dataset.voc2012
=======
from __future__ import print_function

import paddle.dataset.voc2012
import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

__all__ = []


class TestVOC(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def check_reader(self, reader):
        sum = 0
        label = 0
        for l in reader():
            self.assertEqual(l[0].size, 3 * l[1].size)
            sum += 1
        return sum

    def test_train(self):
        count = self.check_reader(paddle.dataset.voc_seg.train())
        self.assertEqual(count, 2913)

    def test_test(self):
        count = self.check_reader(paddle.dataset.voc_seg.test())
        self.assertEqual(count, 1464)

    def test_val(self):
        count = self.check_reader(paddle.dataset.voc_seg.val())
        self.assertEqual(count, 1449)


if __name__ == '__main__':
    unittest.main()
