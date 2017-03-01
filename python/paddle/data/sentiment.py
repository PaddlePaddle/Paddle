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
A utility for fetching, reading sentiment data set.

http://ai.stanford.edu/%7Eamaas/data/sentiment
"""

import os
import shutil
import tarfile
from http_download import download
import gzip
import hashlib
import nltk
import collections
import h5py
import numpy
import random

BASE_URL = 'http://ai.stanford.edu/%%7Eamaas/data/sentiment/%s.tar.gz'
RAW_DATA_NAME = 'aclImdb_v1'
RAW_DATA_MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
MOVIE_REVIEW_DATASET = BASE_URL % (RAW_DATA_NAME)

LABEL_DICT = {'neg': 0, 'pos': 1}

__all__ = ['fetch', 'data']


def calculate_md5(fn):
    """
    """
    h = hashlib.md5()
    with open(fn, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def check_path(directory):
    if directory is None:
        directory = os.path.expanduser(
                os.path.join('~', 'paddle_data', 'sentiment_data'))

    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def fetch(download_directory=None):
    """
    According to the source name,set the download path for source,
    download the data from the source url,and return the download path to fetch
    for training api.
    :param download_directory:
    :return:
    """
    download_data_path = os.path.join(
        check_path(download_directory), "%s.tar.gz" % RAW_DATA_NAME)
    if os.path.exists(download_data_path) and \
       calculate_md5(download_data_path) == RAW_DATA_MD5:
        # already download
        g = gzip.open(download_data_path)
        unpack_path = os.path.join(
                download_directory, g.readline().split("/")[0])
        g.close()
        return unpack_path
    download(MOVIE_REVIEW_DATASET, download_data_path)
    g = gzip.open(download_data_path)
    unpack_path = os.path.join(download_directory, g.readline().split("/")[0])
    g.close()
    tar = tarfile.open(download_data_path)
    if not os.path.exists(unpack_path):
        try:
            print "unpack %s, please wait......" % (download_data_path)
            tar.extractall(download_directory)
            print "unpack %s success..........." % (download_data_path)
        except:
            if os.path.exists(unpack_path):
                shutil.rmtree(unpack_path)
            print "unpack %s was fail, please check" % (download_data_path)
        tar.close()
    return unpack_path


def get_file_contents(file_path, word_dict, sentence_list):
    for key, value in LABEL_DICT.iteritems():
        if key in file_path:
            label = LABEL_DICT[key]

    for num, file_name in enumerate(os.listdir(file_path)):
        with open(os.path.join(file_path, file_name)) as file_handler:
            txt = file_handler.readline().decode('utf-8').lower()
            sentence_list.append((label, txt))
            try:
                words = nltk.tokenize.word_tokenize(txt, 'english')
            except LookupError:
                nltk.download('punkt')
                words = nltk.tokenize.word_tokenize(txt, 'english')
            for each_word in words:
                word_dict[each_word] += 1
    num = num + 1
    return num


def get_dataset(
        data_set_path, freq_dict,
        train_sample_list, test_sample_list):
    train_num = 0
    test_num = 0
    data_set_dict = {
            "train_set_neg": "train/neg",
            "train_set_pos": "train/pos",
            "test_set_neg": "test/neg",
            "test_set_pos": "test/pos"}
    for key, value in data_set_dict.iteritems():
        if "train" in value:
            train_path = os.path.join(data_set_path, value)
            train_num = train_num + get_file_contents(
                    train_path, freq_dict, train_sample_list)
        else:
            test_path = os.path.join(data_set_path, value)
            test_num = test_num + get_file_contents(
                    test_path, freq_dict, test_sample_list)
    random.shuffle(train_sample_list)
    random.shuffle(test_sample_list)
    return train_num, test_num


def preprocess(raw_data_directory):
    raw_data_hdf5_name = os.path.join(
            raw_data_directory, "%s.hdf5" % RAW_DATA_NAME)
    sentiment_data_path = fetch(raw_data_directory)
    if not os.path.exists(raw_data_hdf5_name):
        word_freq_dict = collections.defaultdict(int)
        train_sample_list = list()
        test_sample_list = list()
        train_set_num, test_set_num = get_dataset(
                sentiment_data_path, word_freq_dict,
                train_sample_list, test_sample_list)

        word_list_sorted = list()
        for word in word_freq_dict:
            word_list_sorted.append((word, word_freq_dict[word]))

        word_list_sorted.sort(cmp=lambda a, b: a[1] - b[1])
        word_freq_dict = dict(word_list_sorted)

        sentiment_hdf5 = h5py.File(raw_data_hdf5_name, 'w')
        try:
            word_dict_h5 = sentiment_hdf5.create_dataset(
                    'word_dict',
                    shape=(len(word_list_sorted),),
                    dtype=h5py.special_dtype(vlen=str))
            for word_id, word_tuple in enumerate(word_list_sorted):
                word_dict_h5[word_id] = word_tuple[0]

            train_labels = sentiment_hdf5.create_dataset(
                    'train_labels',
                    shape=(train_set_num, 1),
                    dtype=numpy.int8)

            train_sentences = sentiment_hdf5.create_dataset(
                    'train_sentences',
                    shape=(train_set_num,),
                    dtype=h5py.special_dtype(vlen=numpy.int32))
            for idx, each in enumerate(train_sample_list):
                words = nltk.tokenize.word_tokenize(each[1], 'english')
                words = numpy.array(
                        [word_freq_dict[w] for w in words], dtype=numpy.int32)
                train_sentences[idx] = words
                train_labels[idx] = each[0]

            test_labels = sentiment_hdf5.create_dataset(
                    'test_labels',
                    shape=(test_set_num, 1),
                    dtype=numpy.int8)

            test_sentences = sentiment_hdf5.create_dataset(
                    'test_sentences',
                    shape=(test_set_num,),
                    dtype=h5py.special_dtype(vlen=numpy.int32))
            for idx, each in enumerate(test_sample_list):
                words = nltk.tokenize.word_tokenize(each[1])
                words = numpy.array(
                        [word_freq_dict[w] for w in words], dtype=numpy.int32)
                test_sentences[idx] = words
                test_labels[idx] = each[0]
        finally:
            sentiment_hdf5.close()
    return raw_data_hdf5_name


def data(batch_size, sentiment_data_directory=None):
    """
    """
    data_path = check_path(sentiment_data_directory)
    sentiment_hdf5_file = preprocess(data_path)
    sentiment_datasets = h5py.File(sentiment_hdf5_file, 'r')

    train_labels = sentiment_datasets['train_labels']
    train_sentences = sentiment_datasets['train_sentences']
    test_labels = sentiment_datasets['test_labels']
    test_sentences = sentiment_datasets['test_sentences']

    if max(train_labels.shape[0], test_labels.shape[0]) <= batch_size:
        lens = min(train_labels.shape[0], test_labels.shape[0])
    else:
        lens = batch_size
    for index in xrange(lens):
        print(numpy.array(train_sentences[index]), train_labels[index])
        print(numpy.array(test_sentences[index]), test_labels[index])


def train_data(batch_size, sentiment_data_directory=None):
    """
    """
    train_list = list()
    data_path = check_path(sentiment_data_directory)
    sentiment_hdf5_file = preprocess(data_path)
    sentiment_datasets = h5py.File(sentiment_hdf5_file, 'r')
    train_labels = sentiment_datasets['train_labels']
    train_sentences = sentiment_datasets['train_sentences']
    if train_labels.shape[0] <= batch_size:
        lens = train_labels.shape[0]
    else:
        lens = batch_size

    for index in xrange(lens):
        train_list.append((train_sentences[index], train_labels[index]))
    return train_list


def test_data(batch_size, sentiment_data_directory=None):
    """
    """
    test_list = list()
    data_path = check_path(sentiment_data_directory)
    sentiment_hdf5_file = preprocess(data_path)
    sentiment_datasets = h5py.File(sentiment_hdf5_file, 'r')
    test_labels = sentiment_datasets['test_labels']
    test_sentences = sentiment_datasets['test_sentences']
    if test_labels.shape[0] <= batch_size:
        lens = test_labels.shape[0]
    else:
        lens = batch_size

    for index in xrange(lens):
        test_list.append((test_sentences[index], test_labels[index]))
    return test_list


if __name__ == '__main__':
    data(2)
    print train_data(2)
    print test_data(2)
