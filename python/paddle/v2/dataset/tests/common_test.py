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

import paddle.v2.dataset.common
import unittest
import tempfile


class TestCommon(unittest.TestCase):
    def test_md5file(self):
        _, temp_path = tempfile.mkstemp()
        with open(temp_path, 'w') as f:
            f.write("Hello\n")
        self.assertEqual('09f7e02f1290be211da707a266f153b3',
                         paddle.v2.dataset.common.md5file(temp_path))

    def test_download(self):
        yi_avatar = 'https://avatars0.githubusercontent.com/u/1548775?v=3&s=460'
        self.assertEqual(
            paddle.v2.dataset.common.DATA_HOME + '/test/1548775?v=3&s=460',
            paddle.v2.dataset.common.download(
                yi_avatar, 'test', 'f75287202d6622414c706c36c16f8e0d'))


if __name__ == '__main__':
    unittest.main()
