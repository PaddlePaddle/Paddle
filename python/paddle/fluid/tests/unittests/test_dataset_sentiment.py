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
"""
TestCases for Dataset,
including create, config, run, etc.
"""

from __future__ import print_function
import numpy as np
import unittest
import os
import paddle
import zipfile
import paddle.dataset.common

URL = "https://corpora.bj.bcebos.com/movie_reviews%2Fmovie_reviews.zip"
MD5 = '155de2b77c6834dd8eea7cbe88e93acb'


class TestDatasetSentiment(unittest.TestCase):
    """  TestCases for Sentiment. """

    def setUp(self):
        paddle.dataset.common.download(
            URL, 'corpora', md5sum=MD5, save_name='movie_reviews.zip')
        path = os.path.join(paddle.dataset.common.DATA_HOME, 'corpora')
        filename = os.path.join(path, 'movie_reviews.zip')
        zip_file = zipfile.ZipFile(filename)
        zip_file.extractall(path)
        zip_file.close()

    def test_get_word_dict(self):
        """ Testcase for get_word_dict. """
        words_freq_sorted = paddle.dataset.sentiment.get_word_dict()
        print(words_freq_sorted)
        self.assertTrue(len(words_freq_sorted) == 39768)


if __name__ == '__main__':
    unittest.main()
