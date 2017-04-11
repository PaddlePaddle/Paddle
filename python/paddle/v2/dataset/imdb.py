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
IMDB dataset: http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz

TODO(yuyang18): Complete comments.
"""

import paddle.v2.dataset.common
import collections
import tarfile
import Queue
import re
import string
import threading

__all__ = ['build_dict', 'train', 'test']

URL = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
MD5 = '7c2ac02c03563afcf9b574c7e56c153a'


# Read files that match pattern.  Tokenize and yield each file.
def tokenize(pattern):
    with tarfile.open(paddle.v2.dataset.common.download(URL, 'imdb',
                                                        MD5)) as tarf:
        # Note that we should use tarfile.next(), which does
        # sequential access of member files, other than
        # tarfile.extractfile, which does random access and might
        # destroy hard disks.
        tf = tarf.next()
        while tf != None:
            if bool(pattern.match(tf.name)):
                # newline and punctuations removal and ad-hoc tokenization.
                yield tarf.extractfile(tf).read().rstrip("\n\r").translate(
                    None, string.punctuation).lower().split()
            tf = tarf.next()


def build_dict(pattern, cutoff):
    word_freq = collections.defaultdict(int)
    for doc in tokenize(pattern):
        for word in doc:
            word_freq[word] += 1

    # Not sure if we should prune less-frequent words here.
    word_freq = filter(lambda x: x[1] > cutoff, word_freq.items())

    dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*dictionary))
    word_idx = dict(zip(words, xrange(len(words))))
    word_idx['<unk>'] = len(words)
    return word_idx


def reader_creator(pos_pattern, neg_pattern, word_idx, buffer_size):
    UNK = word_idx['<unk>']

    qs = [Queue.Queue(maxsize=buffer_size), Queue.Queue(maxsize=buffer_size)]

    def load(pattern, queue):
        for doc in tokenize(pattern):
            queue.put(doc)
        queue.put(None)

    def reader():
        # Creates two threads that loads positive and negative samples
        # into qs.
        t0 = threading.Thread(
            target=load, args=(
                pos_pattern,
                qs[0], ))
        t0.daemon = True
        t0.start()

        t1 = threading.Thread(
            target=load, args=(
                neg_pattern,
                qs[1], ))
        t1.daemon = True
        t1.start()

        # Read alternatively from qs[0] and qs[1].
        i = 0
        doc = qs[i].get()
        while doc != None:
            yield [word_idx.get(w, UNK) for w in doc], i % 2
            i += 1
            doc = qs[i % 2].get()

        # If any queue is empty, reads from the other queue.
        i += 1
        doc = qs[i % 2].get()
        while doc != None:
            yield [word_idx.get(w, UNK) for w in doc], i % 2
            doc = qs[i % 2].get()

    return reader()


def train(word_idx):
    return reader_creator(
        re.compile("aclImdb/train/pos/.*\.txt$"),
        re.compile("aclImdb/train/neg/.*\.txt$"), word_idx, 1000)


def test(word_idx):
    return reader_creator(
        re.compile("aclImdb/test/pos/.*\.txt$"),
        re.compile("aclImdb/test/neg/.*\.txt$"), word_idx, 1000)


def word_dict():
    return build_dict(
        re.compile("aclImdb/((train)|(test))/((pos)|(neg))/.*\.txt$"), 150)


def fetch():
    paddle.v2.dataset.common.download(URL, 'imdb', MD5)
