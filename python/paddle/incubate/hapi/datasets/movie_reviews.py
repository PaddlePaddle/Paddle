#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os
import six
import numpy as np
import collections
import nltk
from nltk.corpus import movie_reviews
import zipfile
from functools import cmp_to_key
from itertools import chain

import paddle
from paddle.io import Dataset

__all__ = ['MovieReviews']

URL = "https://corpora.bj.bcebos.com/movie_reviews%2Fmovie_reviews.zip"
MD5 = '155de2b77c6834dd8eea7cbe88e93acb'

NUM_TRAINING_INSTANCES = 1600
NUM_TOTAL_INSTANCES = 2000


class MovieReviews(Dataset):
    """
    Implementation of `NLTK movie reviews <http://www.nltk.org/nltk_data/>`_ dataset.

    Args:
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' 'test' mode. Default 'train'.
        download(bool): whether auto download cifar dataset if
            :attr:`data_file` unset. Default True.

    Returns:
        Dataset: instance of movie reviews dataset

    Examples:

        .. code-block:: python

	    import paddle
	    from paddle.incubate.hapi.datasets import MovieReviews

	    class SimpleNet(paddle.nn.Layer):
		def __init__(self):
		    super(SimpleNet, self).__init__()

		def forward(self, word, category):
		    return paddle.sum(word), category

	    paddle.disable_static()

	    movie_reviews = MovieReviews(mode='train')

	    for i in range(10):
		word_list, category = movie_reviews[i]
		word_list = paddle.to_tensor(word_list)
		category = paddle.to_tensor(category)

		model = SimpleNet()
		word_list, category = model(word_list, category)
		print(word_list.numpy().shape, category.numpy())

    """

    def __init__(self, mode='train'):
        assert mode.lower() in ['train', 'test'], \
            "mode should be 'train', 'test', but got {}".format(mode)
        self.mode = mode.lower()

        self._download_data_if_not_yet()

        # read dataset into memory
        self._load_sentiment_data()

    def _get_word_dict(self):
        """
	Sorted the words by the frequency of words which occur in sample
	:return:
	    words_freq_sorted
	"""
        words_freq_sorted = list()
        word_freq_dict = collections.defaultdict(int)

        for category in movie_reviews.categories():
            for field in movie_reviews.fileids(category):
                for words in movie_reviews.words(field):
                    word_freq_dict[words] += 1
        words_sort_list = list(six.iteritems(word_freq_dict))
        words_sort_list.sort(key=cmp_to_key(lambda a, b: b[1] - a[1]))
        for index, word in enumerate(words_sort_list):
            words_freq_sorted.append((word[0], index))
        return words_freq_sorted

    def _sort_files(self):
        """
	Sorted the sample for cross reading the sample
	:return:
	    files_list
	"""
        files_list = list()
        neg_file_list = movie_reviews.fileids('neg')
        pos_file_list = movie_reviews.fileids('pos')
        files_list = list(
            chain.from_iterable(list(zip(neg_file_list, pos_file_list))))
        return files_list

    def _load_sentiment_data(self):
        """
	Load the data set
	:return:
	    data_set
	"""
        self.data = []
        words_ids = dict(self._get_word_dict())
        for sample_file in self._sort_files():
            words_list = list()
            category = 0 if 'neg' in sample_file else 1
            for word in movie_reviews.words(sample_file):
                words_list.append(words_ids[word.lower()])
            self.data.append((words_list, category))

    def _download_data_if_not_yet(self):
        """
	Download the data set, if the data set is not download.
	"""
        try:
            # download and extract movie_reviews.zip
            paddle.dataset.common.download(
                URL, 'corpora', md5sum=MD5, save_name='movie_reviews.zip')
            path = os.path.join(paddle.dataset.common.DATA_HOME, 'corpora')
            filename = os.path.join(path, 'movie_reviews.zip')
            zip_file = zipfile.ZipFile(filename)
            zip_file.extractall(path)
            zip_file.close()
            # make sure that nltk can find the data
            if paddle.dataset.common.DATA_HOME not in nltk.data.path:
                nltk.data.path.append(paddle.dataset.common.DATA_HOME)
            movie_reviews.categories()
        except LookupError:
            print("Downloading movie_reviews data set, please wait.....")
            nltk.download(
                'movie_reviews', download_dir=paddle.dataset.common.DATA_HOME)
            print("Download data set success.....")
            print("Path is " + nltk.data.find('corpora/movie_reviews').path)

    def __getitem__(self, idx):
        if self.mode == 'test':
            idx += NUM_TRAINING_INSTANCES
        data = self.data[idx]
        return np.array(data[0]), np.array(data[1])

    def __len__(self):
        if self.mode == 'train':
            return NUM_TRAINING_INSTANCES
        else:
            return NUM_TOTAL_INSTANCES - NUM_TRAINING_INSTANCES
