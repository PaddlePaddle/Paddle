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

# Copyright PaddlePaddle contributors. All Rights Reservedd
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
import os
import unittest
import numpy as np
import paddle.v2.reader.creator


class TestNumpyArray(unittest.TestCase):
    def test_numpy_array(self):
        l = [[1, 2, 3], [4, 5, 6]]
        x = np.array(l, np.int32)
        reader = paddle.v2.reader.creator.np_array(x)
        for idx, e in enumerate(reader()):
            self.assertItemsEqual(e, l[idx])


class TestTextFile(unittest.TestCase):
    def test_text_file(self):
        path = os.path.join(os.path.dirname(__file__), "test_data_creator.txt")
        reader = paddle.v2.reader.creator.text_file(path)
        for idx, e in enumerate(reader()):
            self.assertEqual(e, str(idx * 2) + " " + str(idx * 2 + 1))


class TestRecordIO(unittest.TestCase):
    def do_test(self, path):
        reader = paddle.v2.reader.creator.recordio(path)
        idx = 0
        for e in reader():
            if idx == 0:
                self.assertEqual(e, (1, 2, 3))
            elif idx == 1:
                self.assertEqual(e, (4, 5, 6))
            idx += 1
        self.assertEqual(idx, 2)

    def test_recordIO(self):
        self.do_test(
            os.path.join(
                os.path.dirname(__file__), "test_reader_recordio.dat"))
        self.do_test([
            os.path.join(
                os.path.dirname(__file__), "test_reader_recordio.dat")
        ])


if __name__ == '__main__':
    unittest.main()
