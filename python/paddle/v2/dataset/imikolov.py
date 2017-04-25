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
imikolov's simple dataset.

This module will download dataset from 
http://www.fit.vutbr.cz/~imikolov/rnnlm/ and parse training set and test set
into paddle reader creators.
"""
import paddle.v2.dataset.common
import collections
import tarfile

__all__ = ['train', 'test', 'build_dict']

URL = 'http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz'
MD5 = '30177ea32e27c525793142b6bf2c8e2d'


def word_count(f, word_freq=None):
    if word_freq is None:
        word_freq = collections.defaultdict(int)

    for l in f:
        for w in l.strip().split():
            word_freq[w] += 1
        word_freq['<s>'] += 1
        word_freq['<e>'] += 1

    return word_freq


def build_dict():
    """
    Build a word dictionary from the corpus,  Keys of the dictionary are words,
    and values are zero-based IDs of these words.
    """
    train_filename = './simple-examples/data/ptb.train.txt'
    test_filename = './simple-examples/data/ptb.valid.txt'
    with tarfile.open(
            paddle.v2.dataset.common.download(
                paddle.v2.dataset.imikolov.URL, 'imikolov',
                paddle.v2.dataset.imikolov.MD5)) as tf:
        trainf = tf.extractfile(train_filename)
        testf = tf.extractfile(test_filename)
        word_freq = word_count(testf, word_count(trainf))
        if '<unk>' in word_freq:
            # remove <unk> for now, since we will set it as last index
            del word_freq['<unk>']

        TYPO_FREQ = 50
        word_freq = filter(lambda x: x[1] > TYPO_FREQ, word_freq.items())

        word_freq_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*word_freq_sorted))
        word_idx = dict(zip(words, xrange(len(words))))
        word_idx['<unk>'] = len(words)

    return word_idx


def reader_creator(filename, word_idx, n):
    def reader():
        with tarfile.open(
                paddle.v2.dataset.common.download(
                    paddle.v2.dataset.imikolov.URL, 'imikolov',
                    paddle.v2.dataset.imikolov.MD5)) as tf:
            f = tf.extractfile(filename)

            UNK = word_idx['<unk>']
            for l in f:
                l = ['<s>'] + l.strip().split() + ['<e>']
                if len(l) >= n:
                    l = [word_idx.get(w, UNK) for w in l]
                    for i in range(n, len(l) + 1):
                        yield tuple(l[i - n:i])

    return reader


def train(word_idx, n):
    """
    imikolov training set creator.

    It returns a reader creator, each sample in the reader is a word ID
    tuple.

    :param word_idx: word dictionary
    :type word_idx: dict
    :param n: sliding window size
    :type n: int
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator('./simple-examples/data/ptb.train.txt', word_idx, n)


def test(word_idx, n):
    """
    imikolov test set creator.

    It returns a reader creator, each sample in the reader is a word ID
    tuple.

    :param word_idx: word dictionary
    :type word_idx: dict
    :param n: sliding window size
    :type n: int
    :return: Test reader creator
    :rtype: callable
    """
    return reader_creator('./simple-examples/data/ptb.valid.txt', word_idx, n)


def fetch():
    paddle.v2.dataset.common.download(URL, "imikolov", MD5)
