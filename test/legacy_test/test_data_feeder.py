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

import unittest

import paddle
from paddle import base

paddle.enable_static()


class TestDataFeeder(unittest.TestCase):
    def test_lod_level_0_converter(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            img = paddle.static.data(name='image', shape=[-1, 1, 28, 28])
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            feeder = base.DataFeeder([img, label], base.CPUPlace())
            result = feeder.feed([([0] * 784, [9]), ([1] * 784, [1])])

            self.assertEqual(result['image'].shape(), [2, 1, 28, 28])
            self.assertEqual(result['label'].shape(), [2, 1])

            try:
                result = feeder.feed([([0] * 783, [9]), ([1] * 783, [1])])
                self.assertTrue(False)
            except ValueError:
                self.assertTrue(True)

    def test_lod_level_1_converter(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # lod_level = 1
            # each sentence has a different number of words
            sentences = paddle.static.data(
                name='sentences', shape=[-1, 1], dtype='int64'
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            feeder = base.DataFeeder([sentences, label], base.CPUPlace())

            # lod = [[0, 3, 5, 9]]
            # data = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
            # label = [1] * len(data)
            result = feeder.feed(
                [([1, 2, 3], [1]), ([4, 5], [1]), ([6, 7, 8, 9], [1])]
            )

            self.assertEqual(result['sentences'].shape(), [9, 1])
            self.assertEqual(result['label'].shape(), [3, 1])

    def test_lod_level_2_converter(self):
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            # lod_level = 2
            # paragraphs -> sentences -> words
            paragraphs = paddle.static.data(
                name='paragraphs', shape=[-1, 1], dtype='int64', lod_level=2
            )
            label = paddle.static.data(
                name='label', shape=[-1, 1], dtype='int64'
            )
            feeder = base.DataFeeder([paragraphs, label], base.CPUPlace())

            # lod = [[0, 2, 3], [0, 3, 5, 9]]
            # data = [[[1, 2, 3], [4, 5]], [[6, 7, 8, 9]]]
            # label = [1] * len(data)
            result = feeder.feed(
                [([[1, 2, 3], [4, 5]], [1]), ([[6, 7, 8, 9]], [1])]
            )

            self.assertEqual(result['paragraphs'].shape(), [9, 1])
            self.assertEqual(result['label'].shape(), [2, 1])

    def test_errors(self):
        def pir_mode_not_supported_str_feed():
            with paddle.pir_utils.IrGuard():
                with paddle.static.program_guard(
                    paddle.static.Program(), paddle.static.Program()
                ):
                    img = paddle.static.data(
                        name='image', shape=[-1, 1, 28, 28]
                    )
                    label = paddle.static.data(
                        name='label', shape=[-1, 1], dtype='int64'
                    )
                    feeder = base.DataFeeder(['image', label], base.CPUPlace())

        self.assertRaises(ValueError, pir_mode_not_supported_str_feed)


if __name__ == '__main__':
    unittest.main()
