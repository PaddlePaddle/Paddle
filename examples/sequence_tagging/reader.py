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
import os
import numpy as np
import shutil
from functools import partial

import paddle
from paddle.io import BatchSampler, DataLoader, Dataset
from paddle.fluid.dygraph.parallel import ParallelEnv
from hapi.distributed import DistributedBatchSampler


class LacDataset(Dataset):
    """
    Load lexical analysis dataset
    """

    def __init__(self, args):
        self.word_dict_path = args.word_dict_path
        self.label_dict_path = args.label_dict_path
        self.word_rep_dict_path = args.word_rep_dict_path
        self._load_dict()
        self.examples = []

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
        return max(self.word2id_dict.values()) + 1

    @property
    def num_labels(self):
        return max(self.label2id_dict.values()) + 1

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

    def file_reader(self, filename, phase="train"):
        """
        yield (word_idx, target_idx) one by one from file,
            or yield (word_idx, ) in `infer` mode
        """
        self.phase = phase
        with io.open(filename, "r", encoding="utf8") as fr:
            if phase in ["train", "test"]:
                headline = next(fr)
                headline = headline.strip().split('\t')
                assert len(headline) == 2 and headline[
                    0] == "text_a" and headline[1] == "label"

                for line in fr:
                    line_str = line.strip("\n")
                    if len(line_str) < 1 and len(line_str.split('\t')) < 2:
                        continue

                    self.examples.append(line_str)
            else:
                for idx, line in enumerate(fr):
                    words = line.strip("\n").split("\t")[0]
                    self.examples.append(words)

    def __getitem__(self, idx):
        line_str = self.examples[idx]
        if self.phase in ["train", "test"]:
            words, labels = line_str.split('\t')
            word_ids = self.word_to_ids(words.split("\002"))
            label_ids = self.label_to_ids(labels.split("\002"))
            assert len(word_ids) == len(label_ids)
            return word_ids, label_ids
        else:
            words = [w for w in line_str]
            word_ids = self.word_to_ids(words)
            return word_ids

    def __len__(self):

        return len(self.examples)


def create_lexnet_data_generator(args, insts, phase="train"):
    def padding_data(max_len, batch_data, if_len=False):
        padding_batch_data = []
        padding_lens = []
        for data in batch_data:
            data = data[:max_len]
            if if_len:
                seq_len = np.int64(len(data))
                padding_lens.append(seq_len)
            data += [0 for _ in range(max_len - len(data))]
            padding_batch_data.append(data)
        if if_len:
            return np.array(padding_batch_data), np.array(padding_lens)
        else:
            return np.array(padding_batch_data)

    if phase == "train":
        batch_words = [inst[0] for inst in insts]
        batch_labels = [inst[1] for inst in insts]
        padding_batch_words, padding_lens = padding_data(
            args.max_seq_len, batch_words, if_len=True)
        padding_batch_labels = padding_data(args.max_seq_len, batch_labels)
        return [
            padding_batch_words, padding_lens, padding_batch_labels,
            padding_batch_labels
        ]
    elif phase == "test":
        batch_words = [inst[0] for inst in insts]
        seq_len = [len(inst[0]) for inst in insts]
        max_seq_len = max(seq_len)
        batch_labels = [inst[1] for inst in insts]
        padding_batch_words, padding_lens = padding_data(
            max_seq_len, batch_words, if_len=True)
        padding_batch_labels = padding_data(max_seq_len, batch_labels)
        return [
            padding_batch_words, padding_lens, padding_batch_labels,
            padding_batch_labels
        ]
    else:
        batch_words = insts
        seq_len = [len(inst) for inst in insts]
        max_seq_len = max(seq_len)
        padding_batch_words, padding_lens = padding_data(
            max_seq_len, batch_words, if_len=True)
        return [padding_batch_words, padding_lens]


class LacDataLoader(object):
    def __init__(self,
                 args,
                 place,
                 phase="train",
                 shuffle=False,
                 num_workers=0,
                 drop_last=False):
        assert phase in [
            "train", "test", "predict"
        ], "phase should be in [train, test, predict], but get %s" % phase

        if phase == "train":
            file_name = args.train_file
        elif phase == "test":
            file_name = args.test_file
        elif phase == "predict":
            file_name = args.predict_file

        self.dataset = LacDataset(args)
        self.dataset.file_reader(file_name, phase=phase)

        if phase == "train":
            self.sampler = DistributedBatchSampler(
                dataset=self.dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                drop_last=drop_last)
        else:
            self.sampler = BatchSampler(
                dataset=self.dataset,
                batch_size=args.batch_size,
                shuffle=shuffle,
                drop_last=drop_last)

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            places=place,
            collate_fn=partial(
                create_lexnet_data_generator, args, phase=phase),
            num_workers=num_workers,
            return_list=True)
