#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle


class TestDataFeeder(unittest.TestCase):

    def test_lod_level_1_converter(self):
        sequential = paddle.nn.Sequential()

        for i in range(10):
            sequential.add_sublayer(str(i), paddle.nn.Linear(i + 1, i + 1))

        for item in sequential:
            tmp = item

        tmp = sequential[3:5]
        self.assertEqual(len(tmp), 2)

        tmp = sequential[-1]
        self.assertEqual(tmp, sequential[9])

        with self.assertRaises(IndexError):
            tmp = sequential[10]

        with self.assertRaises(IndexError):
            tmp = sequential[-11]


if __name__ == '__main__':
    unittest.main()
