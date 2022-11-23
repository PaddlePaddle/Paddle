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

import paddle.dataset.imdb
import unittest
import re

__all__ = []

TRAIN_POS_PATTERN = re.compile(r"aclImdb/train/pos/.*\.txt$")
TRAIN_NEG_PATTERN = re.compile(r"aclImdb/train/neg/.*\.txt$")
TRAIN_PATTERN = re.compile(r"aclImdb/train/.*\.txt$")

TEST_POS_PATTERN = re.compile(r"aclImdb/test/pos/.*\.txt$")
TEST_NEG_PATTERN = re.compile(r"aclImdb/test/neg/.*\.txt$")
TEST_PATTERN = re.compile(r"aclImdb/test/.*\.txt$")


class TestIMDB(unittest.TestCase):
    word_idx = None

    def test_build_dict(self):
        if self.word_idx == None:
            self.word_idx = paddle.dataset.imdb.build_dict(TRAIN_PATTERN, 150)

        self.assertEqual(len(self.word_idx), 7036)

    def check_dataset(self, dataset, expected_size):
        if self.word_idx == None:
            self.word_idx = paddle.dataset.imdb.build_dict(TRAIN_PATTERN, 150)

        sum = 0
        for l in dataset(self.word_idx):
            self.assertEqual(l[1], sum % 2)
            sum += 1
        self.assertEqual(sum, expected_size)

    def test_train(self):
        self.check_dataset(paddle.dataset.imdb.train, 25000)

    def test_test(self):
        self.check_dataset(paddle.dataset.imdb.test, 25000)


if __name__ == '__main__':
    unittest.main()
