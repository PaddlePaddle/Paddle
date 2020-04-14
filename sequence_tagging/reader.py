#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
SequenceTagging dataset
"""

from __future__ import division
from __future__ import print_function

import io
import numpy as np

import paddle.fluid as fluid


class LacDataset(object):
    """
    Load lexical analysis dataset
    """

    def __init__(self, args):
        self.word_dict_path = args.word_dict_path
        self.label_dict_path = args.label_dict_path
        self.word_rep_dict_path = args.word_rep_dict_path
        self._load_dict()

    def _load_dict(self):
        self.word2id_dict = self.load_kv_dict(
            self.word_dict_path, reverse=True, value_func=np.int64)
        self.id2word_dict = self.load_kv_dict(self.word_dict_path)
        self.label2id_dict = self.load_kv_dict(
            self.label_dict_path, reverse=True, value_func=np.int64)
        self.id2label_dict = self.load_kv_dict(self.label_dict_path)
        if self.word_rep_dict_path is None:
            self.word_replace_dict = dict()
        else:
            self.word_replace_dict = self.load_kv_dict(self.word_rep_dict_path)

    def load_kv_dict(self,
                     dict_path,
                     reverse=False,
                     delimiter="\t",
                     key_func=None,
                     value_func=None):
        """
        Load key-value dict from file
        """
        result_dict = {}
        for line in io.open(dict_path, "r", encoding='utf8'):
            terms = line.strip("\n").split(delimiter)
            if len(terms) != 2:
                continue
            if reverse:
                value, key = terms
            else:
                key, value = terms
            if key in result_dict:
                raise KeyError("key duplicated with [%s]" % (key))
            if key_func:
                key = key_func(key)
            if value_func:
                value = value_func(value)
            result_dict[key] = value
        return result_dict

    @property
    def vocab_size(self):
        return len(self.word2id_dict.values())

    @property
    def num_labels(self):
        return len(self.label2id_dict.values())

    def get_num_examples(self, filename):
        """num of line of file"""
        return sum(1 for line in io.open(filename, "r", encoding='utf8'))

    def word_to_ids(self, words):
        """convert word to word index"""
        word_ids = []
        for word in words:
            word = self.word_replace_dict.get(word, word)
            if word not in self.word2id_dict:
                word = "OOV"
            word_id = self.word2id_dict[word]
            word_ids.append(word_id)

        return word_ids

    def label_to_ids(self, labels):
        """convert label to label index"""
        label_ids = []
        for label in labels:
            if label not in self.label2id_dict:
                label = "O"
            label_id = self.label2id_dict[label]
            label_ids.append(label_id)
        return label_ids

    def file_reader(self,
                    filename,
                    mode="train",
                    batch_size=32,
                    max_seq_len=126):
        """
        yield (word_idx, target_idx) one by one from file,
            or yield (word_idx, ) in `infer` mode
        """

        def wrapper():
            fread = io.open(filename, "r", encoding="utf-8")
            headline = next(fread)
            headline = headline.strip().split('\t')
            assert len(headline) == 2 and headline[0] == "text_a" and headline[
                1] == "label"
            buf = []
            for line in fread:
                words, labels = line.strip("\n").split("\t")
                if len(words) < 1:
                    continue
                word_ids = self.word_to_ids(words.split("\002"))
                label_ids = self.label_to_ids(labels.split("\002"))
                assert len(word_ids) == len(label_ids)
                word_ids = word_ids[0:max_seq_len]
                words_len = np.int64(len(word_ids))
                word_ids += [0 for _ in range(max_seq_len - words_len)]
                label_ids = label_ids[0:max_seq_len]
                label_ids += [0 for _ in range(max_seq_len - words_len)]
                assert len(word_ids) == len(label_ids)
                yield word_ids, label_ids, words_len
            fread.close()

        return wrapper


def create_lexnet_data_generator(args, reader, file_name, place, mode="train"):
    def wrapper():
        batch_words, batch_labels, seq_lens = [], [], []
        for epoch in xrange(args.epoch):
            for instance in reader.file_reader(
                    file_name, mode, max_seq_len=args.max_seq_len)():
                words, labels, words_len = instance
                if len(seq_lens) < args.batch_size:
                    batch_words.append(words)
                    batch_labels.append(labels)
                    seq_lens.append(words_len)
                if len(seq_lens) == args.batch_size:
                    yield batch_words, batch_labels, seq_lens, batch_labels
                    batch_words, batch_labels, seq_lens = [], [], []

        if len(seq_lens) > 0:
            yield batch_words, batch_labels, seq_lens, batch_labels
            batch_words, batch_labels, seq_lens = [], [], []

    return wrapper


def create_dataloader(generator, place, feed_list=None):
    if not feed_list:
        data_loader = paddle.io.DataLoader.from_generator(
            capacity=50,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
    else:
        data_loader = paddle.io.DataLoader.from_generator(
            feed_list=feed_list,
            capacity=50,
            use_double_buffer=True,
            iterable=True,
            return_list=True)
    data_loader.set_batch_generator(generator, places=place)
    return data_loader


