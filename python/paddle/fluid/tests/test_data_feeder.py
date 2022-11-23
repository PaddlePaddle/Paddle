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

import paddle.fluid as fluid
import unittest
import paddle

paddle.enable_static()


class TestDataFeeder(unittest.TestCase):

    def test_lod_level_0_converter(self):
        img = fluid.layers.data(name='image', shape=[1, 28, 28])
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        feeder = fluid.DataFeeder([img, label], fluid.CPUPlace())
        result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])

        self.assertEqual(result['image'].shape(), [2, 1, 28, 28])
        self.assertEqual(result['label'].shape(), [2, 1])
        self.assertEqual(result['image'].recursive_sequence_lengths(), [])
        self.assertEqual(result['label'].recursive_sequence_lengths(), [])

        try:
            result = feeder.feed([([0] * 783, [9]), ([1] * 783, [1])])
            self.assertTrue(False)
        except ValueError:
            self.assertTrue(True)

    def test_lod_level_1_converter(self):
        # lod_level = 1
        # each sentence has a different number of words
        sentences = fluid.layers.data(name='sentences',
                                      shape=[1],
                                      dtype='int64',
                                      lod_level=1)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        feeder = fluid.DataFeeder([sentences, label], fluid.CPUPlace())

        # lod = [[0, 3, 5, 9]]
        # data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        # label = [1] * len(data)
        result = feeder.feed([([1, 2, 3], [1]), ([4, 5], [1]),
                              ([6, 7, 8, 9], [1])])

        self.assertEqual(result['sentences'].shape(), [9, 1])
        self.assertEqual(result['label'].shape(), [3, 1])
        self.assertEqual(result['sentences'].recursive_sequence_lengths(),
                         [[3, 2, 4]])
        self.assertEqual(result['label'].recursive_sequence_lengths(), [])

    def test_lod_level_2_converter(self):
        # lod_level = 2
        # paragraphs -> sentences -> words
        paragraphs = fluid.layers.data(name='paragraphs',
                                       shape=[1],
                                       dtype='int64',
                                       lod_level=2)
        label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        feeder = fluid.DataFeeder([paragraphs, label], fluid.CPUPlace())

        # lod = [[0, 2, 3], [0, 3, 5, 9]]
        # data = [[[1, 2, 3], [4, 5]], [[6, 7, 8, 9]]]
        # label = [1] * len(data)
        result = feeder.feed([([[1, 2, 3], [4, 5]], [1]), ([[6, 7, 8,
                                                             9]], [1])])

        self.assertEqual(result['paragraphs'].shape(), [9, 1])
        self.assertEqual(result['label'].shape(), [2, 1])
        self.assertEqual(result['paragraphs'].recursive_sequence_lengths(),
                         [[2, 1], [3, 2, 4]])
        self.assertEqual(result['label'].recursive_sequence_lengths(), [])


if __name__ == '__main__':
    unittest.main()
