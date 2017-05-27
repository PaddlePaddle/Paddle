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

import paddle.v2.dataset.mq2007
import unittest


class TestMQ2007(unittest.TestCase):
    def test_pairwise(self):
        for label, query_left, query_right in paddle.v2.dataset.mq2007.test(
                format="pairwise"):
            self.assertEqual(query_left.shape(), (46, ))
            self.assertEqual(query_right.shape(), (46, ))

    def test_listwise(self):
        for label_array, query_array in paddle.v2.dataset.mq2007.test(
                format="listwise"):
            self.assertEqual(len(label_array), len(query_array))


if __name__ == "__main__":
    unittest.main()
