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

import paddle.dataset.cifar
=======
from __future__ import print_function

import paddle.dataset.cifar
import unittest
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

__all__ = []


class TestCIFAR(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def check_reader(self, reader):
        sum = 0
        label = 0
        for l in reader():
            self.assertEqual(l[0].size, 3072)
            if l[1] > label:
                label = l[1]
            sum += 1
        return sum, label

    def test_test10(self):
        instances, max_label_value = self.check_reader(
<<<<<<< HEAD
            paddle.dataset.cifar.test10()
        )
=======
            paddle.dataset.cifar.test10())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual(instances, 10000)
        self.assertEqual(max_label_value, 9)

    def test_train10(self):
        instances, max_label_value = self.check_reader(
<<<<<<< HEAD
            paddle.dataset.cifar.train10()
        )
=======
            paddle.dataset.cifar.train10())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual(instances, 50000)
        self.assertEqual(max_label_value, 9)

    def test_test100(self):
        instances, max_label_value = self.check_reader(
<<<<<<< HEAD
            paddle.dataset.cifar.test100()
        )
=======
            paddle.dataset.cifar.test100())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual(instances, 10000)
        self.assertEqual(max_label_value, 99)

    def test_train100(self):
        instances, max_label_value = self.check_reader(
<<<<<<< HEAD
            paddle.dataset.cifar.train100()
        )
=======
            paddle.dataset.cifar.train100())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.assertEqual(instances, 50000)
        self.assertEqual(max_label_value, 99)


if __name__ == '__main__':
    unittest.main()
