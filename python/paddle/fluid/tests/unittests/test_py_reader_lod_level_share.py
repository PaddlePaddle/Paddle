# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import unittest


class TestLoDLevelShare(unittest.TestCase):
    def setUp(self):
        self.use_double_buffer = False

    def test_lod_level_share(self):
        reader = fluid.layers.py_reader(
            capacity=16,
            shapes=([-1, 256], [-1, 512], [-1, 100]),
            dtypes=('float32', 'int64', 'double'),
            lod_levels=(1, 2, 0),
            use_double_buffer=self.use_double_buffer)

        x, y, z = fluid.layers.read_file(reader)
        self.assertEqual(x.lod_level, 1)
        self.assertEqual(y.lod_level, 2)
        self.assertEqual(z.lod_level, 0)


class TestLoDLevelShare2(TestLoDLevelShare):
    def setUp(self):
        self.use_double_buffer = True


if __name__ == '__main__':
    unittest.main()
