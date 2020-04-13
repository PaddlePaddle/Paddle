# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import io
import os
import six
import csv
import glob
import tarfile
import itertools
import leveldb
from functools import partial

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.io import BatchSampler, DataLoader, Dataset
from hapi.distributed import DistributedBatchSampler
from hapi.text.bert.data_processor import DataProcessor, XnliProcessor, ColaProcessor, MrpcProcessor, MnliProcessor
from hapi.text.bert.batching import prepare_batch_data
import hapi.text.tokenizer.tokenization as tokenization

__all__ = [
    'BertInputExample', 'BertInputFeatures', 'SingleSentenceDataset',
    'SentencePairDataset', 'BertDataLoader'
]


class BertInputExample(object):
    def __init__(self, uid, text_a, text_b=None, label=None):
        self.uid = uid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class BertInputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.pos_ids = list(range(len(self.input_ids)))
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example_to_unicode(guid, single_example):
    text_a = tokenization.convert_to_unicode(single_example[0])
    text_b = tokenization.convert_to_unicode(single_example[1])
    label = tokenization.convert_to_unicode(single_example[2])
    return BertInputExample(uid=uid, text_a=text_a, text_b=text_b, label=label)


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `BertInputExample` into a single `BertInputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)
    label_id = label_map[example.label]

    feature = BertInputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)

    return feature


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)

    return features


def _read_tsv(input_file, delimiter="\t", quotechar=None):
    """Reads a tab separated value file."""
    with io.open(input_file, "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


class SingleSentenceDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 label_list,
                 max_seq_length,
                 mode="all_in_memory"):

        assert isinstance(mode,
                          str), "mode of SingleSentenceDataset should be str"
        assert mode in [
            "all_in_memory", "leveldb", "streaming"
        ], "mode of SingleSentenceDataset should be in [all_in_memory, leveldb, streaming], but get" % mode

        self.delimiter = None
        self.mode = mode
        self.examples = []
        self._db = None
        self._line_processor = None

    def load_all_data_in_memory(self,
                                input_file,
                                label_list,
                                max_seq_length,
                                tokenizer,
                                line_processor=None,
                                delimiter="\t",
                                quotechar=None):
        lines = _read_tsv(input_file, delimiter=delimiter, quotechar=quotechar)

        def default_line_processor(line_id, line):
            assert len(line) == 2
            text_a = line[0]
            label = line[1]

            return BertInputExample(
                str(line_id), text_a=text_a, text_b=None, label=label)

        if line_processor is None:
            line_processor = default_line_processor

        for (line_id, line) in enumerate(lines):
            input_example = line_processor(line_id, line)
            if not input_example:
                continue
            input_feature = convert_single_example(
                str(line_id), input_example, label_list, max_seq_length,
                tokenizer)
            self.examples.append(input_feature)

    def prepare_leveldb(self,
                        input_file,
                        leveldb_file,
                        label_list,
                        max_seq_length,
                        tokenizer,
                        line_processor=None,
                        delimiter="\t",
                        quotechar=None):
        def default_line_processor(line_id, line):
            assert len(line) == 2
            text_a = line[0]
            label = line[1]

            return BertInputExample(
                str(line_id), text_a=text_a, text_b=None, label=label)

        if line_processor is None:
            line_processor = default_line_processor

        if not os.path.exists(leveldb_file):
            print("putting data %s into leveldb %s" %
                  (input_file, leveldb_file))
            _example_num = 0
            _db = leveldb.LevelDB(leveldb_file, create_if_missing=True)
            with io.open(input_file, "r", encoding="utf8") as f:
                reader = csv.reader(
                    f, delimiter=delimiter, quotechar=quotechar)
                line_id = 0
                for (_line_id, line) in enumerate(reader):
                    if line_processor(str(_line_id), line) is None:
                        continue

                    line_str = delimiter.join(line)
                    _db.Put(
                        str(line_id).encode("utf8"), line_str.encode("utf8"))
                    line_id += 1
                    _example_num += 1
            _db.Put("_example_num_".encode("utf8"),
                    str(_example_num).encode("utf8"))
        else:
            _db = leveldb.LevelDB(leveldb_file, create_if_missing=False)

        self.label_list = label_list
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.delimiter = delimiter
        self._db = _db
        self._line_processor = line_processor

    def __getitem__(self, idx):

        if self.mode == "all_in_memory":
            return self.examples[idx].input_ids, self.examples[
                idx].pos_ids, self.examples[idx].segment_ids, self.examples[
                    idx].label_id

        if self.mode == "leveldb":
            assert self._db is not None, "you shold call prepare_leveldb before you run dataloader"
            line_str = self._db.Get(str(idx).encode("utf8"))
            line_str = line_str.decode("utf8")

            line = line_str.split(self.delimiter)
            input_example = self._line_processor(str(idx + 1), line)

            input_example = convert_single_example(
                str(idx + 1), input_example, self.label_list,
                self.max_seq_length, self.tokenizer)

            return input_example.input_ids, input_example.pos_ids, input_example.segment_ids, input_example.label_id

    def __len__(self):
        if self.mode == "all_in_memory":
            return len(self.examples)

        if self.mode == "leveldb":
            assert self._db is not None, "you shold call prepare_leveldb before you run dataloader"

            exmaple_num = self._db.Get("_example_num_".encode("utf8"))
            exmaple_num = exmaple_num.decode("utf8")
            return int(exmaple_num)


class SentencePairDataset(Dataset):
    def __init__(self,
                 tokenizer,
                 label_ist,
                 max_seq_length,
                 mode="all_in_memory"):

        assert isinstance(mode,
                          str), "mode of SentencePairDataset should be str"
        assert mode in [
            "all_in_memory", "leveldb"
        ], "mode of SentencePairDataset should be in [all_in_memory, leveldb], but get" % mode

        self.examples = []

    def load_all_data_in_memory(self,
                                input_file,
                                label_list,
                                max_seq_length,
                                tokenizer,
                                line_processor=None,
                                delimiter="\t",
                                quotechar=None):
        lines = _read_tsv(input_file, delimiter=delimiter, quotechar=quotechar)

        def default_line_processor(line_id, line):
            assert len(line) == 3
            text_a = line[0]
            text_b = line[1]
            label = line[2]

            return BertInputExample(
                str(line_id), text_a=text_a, text_b=text_b, label=label)

        if line_processor is None:
            line_processor = default_line_processor

        for (line_id, line) in enumerate(lines):
            input_example = line_processor(line_id, line)
            if not input_example:
                continue
            input_feature = convert_single_example(
                str(line_id), input_example, label_list, max_seq_length,
                tokenizer)
            self.examples.append(input_feature)

    def __getitem__(self, idx):
        return self.examples[idx].input_ids, self.examples[
            idx].pos_ids, self.examples[idx].segment_ids, self.examples[
                idx].label_id

    def __len__(self):
        return len(self.examples)


def _prepare_train_batch(insts,
                         vocab_size=0,
                         pad_id=None,
                         cls_id=None,
                         sep_id=None,
                         mask_id=-1,
                         return_input_mask=True,
                         return_max_len=True,
                         return_num_token=False):

    return prepare_batch_data(
        insts,
        0,
        voc_size=vocab_size,
        pad_id=pad_id,
        cls_id=cls_id,
        sep_id=sep_id,
        mask_id=mask_id,
        return_input_mask=return_input_mask,
        return_max_len=return_max_len,
        return_num_token=return_num_token)


class BertDataLoader(object):
    def __init__(self,
                 input_file,
                 tokenizer,
                 label_list,
                 max_seq_length,
                 batch_size,
                 shuffle=False,
                 drop_last=False,
                 mode="all_in_memory",
                 leveldb_file="./leveldb",
                 line_processor=None,
                 delimiter="\t",
                 quotechar=None,
                 device=fluid.CPUPlace(),
                 num_workers=0,
                 return_list=True):

        self.dataset = SingleSentenceDataset(tokenizer, label_list,
                                             max_seq_length, mode)

        if mode == "all_in_memory":
            self.dataset.load_all_data_in_memory(
                input_file, label_list, max_seq_length, tokenizer,
                line_processor, delimiter, quotechar)
        elif mode == "leveldb":
            #prepare_leveldb(self, input_file, leveldb_file, label_list, max_seq_length, tokenizer, line_processor=None, delimiter="\t", quotechar=None):
            self.dataset.prepare_leveldb(input_file, leveldb_file, label_list,
                                         max_seq_length, tokenizer,
                                         line_processor, delimiter, quotechar)
        else:
            raise ValueError("mode should be in [all_in_memory, leveldb]")

        self.sampler = DistributedBatchSampler(
            self.dataset, batch_size, shuffle=shuffle, drop_last=drop_last)

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            places=device,
            collate_fn=partial(
                _prepare_train_batch,
                vocab_size=-1,
                pad_id=tokenizer.vocab["[PAD]"],
                cls_id=tokenizer.vocab["[CLS]"],
                sep_id=tokenizer.vocab["[SEP]"],
                mask_id=-1,
                return_input_mask=True,
                return_max_len=False,
                return_num_token=False),
            num_workers=num_workers,
            return_list=return_list)


if __name__ == "__main__":
    print("hello world.")
