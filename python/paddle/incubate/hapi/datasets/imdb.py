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

import re
import six
import string
import tarfile
import numpy as np
import collections

from paddle.io import Dataset
from .utils import _check_exists_and_download


__all__ = ['Imdb']

URL = 'https://dataset.bj.bcebos.com/imdb%2FaclImdb_v1.tar.gz'
MD5 = '7c2ac02c03563afcf9b574c7e56c153a'


class Imdb(Dataset):
    """
    Implement of IMDB dataset.

    Args:
        data_file(str): path to data tar file, can be set None if
            :attr:`download` is True. Default None
        mode(str): 'train' 'test' mode. Default 'train'.
        cutoff(int): cutoff number for building word dictionary. Default 150.
        download(bool): whether auto download cifar dataset if
            :attr:`data_file` unset. Default
            True

    Examples:

        .. code-block:: python

            from paddle.incubate.hapi.datasets import Imdb

            imdb = Imdb()

            for i in range(len(imdb)):
                sample = imdb[i]
                print(sample)

    """

    def __init__(self,
                 data_file=None,
                 mode='train',
                 cutoff=150,
                 download=True):
        assert mode.lower() in ['train', 'test'], \
            "mode should be 'train', 'test', but got {}".format(mode)
        self.mode = mode.lower()

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file not set and auto download disabled"
            self.data_file = _check_exists_and_download(
                data_file, URL, MD5, 'imdb', download)

        # Build a word dictionary from the corpus
        self.word_idx = self._build_work_dict(cutoff)

        # read dataset into memory
        self._load_anno()

    def _build_work_dict(self, cutoff):
        word_freq = collections.defaultdict(int)
        pattern = re.compile("aclImdb/((train)|(test))/((pos)|(neg))/.*\.txt$")
        for doc in self._tokenize(pattern):
            for word in doc:
                word_freq[word] += 1

        # Not sure if we should prune less-frequent words here.
        word_freq = [x for x in six.iteritems(word_freq) if x[1] > cutoff]

        dictionary = sorted(word_freq, key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*dictionary))
        word_idx = dict(list(zip(words, six.moves.range(len(words)))))
        word_idx['<unk>'] = len(words)
        return word_idx

    def _tokenize(self, pattern):
        data = []
        with tarfile.open(self.data_file) as tarf:
            tf = tarf.next()
            while tf != None:
                if bool(pattern.match(tf.name)):
                    # newline and punctuations removal and ad-hoc tokenization.
                    data.append(tarf.extractfile(tf).read().rstrip(six.b(
                        "\n\r")).translate(
                            None, six.b(string.punctuation)).lower().split())
                tf = tarf.next()
        
        return data

    def _load_anno(self):
        pos_pattern = re.compile("aclImdb/{}/pos/.*\.txt$".format(self.mode))
        neg_pattern = re.compile("aclImdb/{}/neg/.*\.txt$".format(self.mode))

        UNK = self.word_idx['<unk>']

        self.docs = []
        self.labels = []
        for doc in self._tokenize(pos_pattern):
            self.docs.append([self.word_idx.get(w, UNK) for w in doc])
            self.labels.append(0)
        for doc in self._tokenize(neg_pattern):
            self.docs.append([self.word_idx.get(w, UNK) for w in doc])
            self.labels.append(1)

    def __getitem__(self, idx):
        return (np.array(self.docs[idx]), np.array([self.labels[idx]]))
    
    def __len__(self):
        return len(self.docs)

