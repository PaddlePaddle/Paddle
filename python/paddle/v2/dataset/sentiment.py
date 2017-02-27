# /usr/bin/env python
# -*- coding:utf-8 -*-

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
"""
The script fetch and preprocess movie_reviews data set

that provided by NLTK
"""

import nltk
import numpy as np
from itertools import chain
from nltk.corpus import movie_reviews
from config import DATA_HOME

__all__ = ['train', 'test', 'get_label_dict', 'get_word_dict']
NUM_TRAINING_INSTANCES = 1600
NUM_TOTAL_INSTANCES = 2000


def get_label_dict():
    """
    Define the labels dict for dataset
    """
    label_dict = {'neg': 0, 'pos': 1}
    return label_dict


def download_data_if_not_yet():
    """
    Download the data set, if the data set is not download.
    """
    try:
        # make sure that nltk can find the data
        nltk.data.path.append(DATA_HOME)
        movie_reviews.categories()
    except LookupError:
        print "Downloading movie_reviews data set, please wait....."
        nltk.download('movie_reviews', download_dir=DATA_HOME)
        # make sure that nltk can find the data
        nltk.data.path.append(DATA_HOME)
        print "Download data set success....."
        print "Path is " + nltk.data.find('corpora/movie_reviews').path


def get_word_dict():
    """
    Sorted the words by the frequency of words which occur in sample
    :return:
        words_freq_sorted
    """
    words_freq_sorted = list()
    download_data_if_not_yet()
    words_freq = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    words_sort_list = words_freq.items()
    words_sort_list.sort(cmp=lambda a, b: b[1] - a[1])
    for index, word in enumerate(words_sort_list):
        words_freq_sorted.append((word[0], index + 1))
    return words_freq_sorted


def sort_files():
    """
    Sorted the sample for cross reading the sample
    :return:
        files_list
    """
    files_list = list()
    download_data_if_not_yet()
    neg_file_list = movie_reviews.fileids('neg')
    pos_file_list = movie_reviews.fileids('pos')
    files_list = list(chain.from_iterable(zip(neg_file_list, pos_file_list)))
    return files_list


def load_sentiment_data():
    """
    Load the data set
    :return:
        data_set
    """
    data_set = list()
    download_data_if_not_yet()
    words_ids = dict(get_word_dict())
    for sample_file in sort_files():
        words_list = list()
        category = 0 if 'neg' in sample_file else 1
        for word in movie_reviews.words(sample_file):
            words_list.append(words_ids[word.lower()])
        data_set.append((words_list, category))
    return data_set


data_set = load_sentiment_data()


def reader_creator(data):
    """
    Reader creator, it format data set to numpy
    :param data:
        train data set or test data set
    """
    for each in data:
        list_of_int = np.array(each[0], dtype=np.int32)
        label = each[1]
        yield list_of_int, label


def train():
    """
    Default train set reader creator
    """
    return reader_creator(data_set[0:NUM_TRAINING_INSTANCES])


def test():
    """
    Default test set reader creator
    """
    return reader_creator(data_set[NUM_TRAINING_INSTANCES:])


def unittest():
    assert len(data_set) == NUM_TOTAL_INSTANCES
    assert len(list(train())) == NUM_TRAINING_INSTANCES
    assert len(list(test())) == NUM_TOTAL_INSTANCES - NUM_TRAINING_INSTANCES


if __name__ == '__main__':
    unittest()
