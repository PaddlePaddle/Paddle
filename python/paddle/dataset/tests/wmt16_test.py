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

import paddle.dataset.wmt16
import unittest

__all__ = []


class TestWMT16(unittest.TestCase):

    def checkout_one_sample(self, sample):
        # train data has 3 field: source language word indices,
        # target language word indices, and target next word indices.
        self.assertEqual(len(sample), 3)

        # test start mark and end mark in source word indices.
        self.assertEqual(sample[0][0], 0)
        self.assertEqual(sample[0][-1], 1)

        # test start mask in target word indices
        self.assertEqual(sample[1][0], 0)

        # test en mask in target next word indices
        self.assertEqual(sample[2][-1], 1)

    def test_train(self):
        for idx, sample in enumerate(
                paddle.dataset.wmt16.train(src_dict_size=100000,
                                           trg_dict_size=100000)()):
            if idx >= 10: break
            self.checkout_one_sample(sample)

    def test_test(self):
        for idx, sample in enumerate(
                paddle.dataset.wmt16.test(src_dict_size=1000,
                                          trg_dict_size=1000)()):
            if idx >= 10: break
            self.checkout_one_sample(sample)

    def test_val(self):
        for idx, sample in enumerate(
                paddle.dataset.wmt16.validation(src_dict_size=1000,
                                                trg_dict_size=1000)()):
            if idx >= 10: break
            self.checkout_one_sample(sample)

    def test_get_dict(self):
        dict_size = 1000
        word_dict = paddle.dataset.wmt16.get_dict("en", dict_size, True)
        self.assertEqual(len(word_dict), dict_size)
        self.assertEqual(word_dict[0], "<s>")
        self.assertEqual(word_dict[1], "<e>")
        self.assertEqual(word_dict[2], "<unk>")


if __name__ == "__main__":
    unittest.main()
