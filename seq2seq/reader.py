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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import six
import os
import io
import itertools
from functools import partial

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.fluid.io import BatchSampler, DataLoader, Dataset


def create_data_loader(args, device, for_train=True):
    data_loaders = [None, None]
    data_prefixes = [args.train_data_prefix, args.eval_data_prefix
                     ] if args.eval_data_prefix else [args.train_data_prefix]
    for i, data_prefix in enumerate(data_prefixes):
        dataset = Seq2SeqDataset(
            fpattern=data_prefix + "." + args.src_lang,
            trg_fpattern=data_prefix + "." + args.tar_lang,
            src_vocab_fpath=args.vocab_prefix + "." + args.src_lang,
            trg_vocab_fpath=args.vocab_prefix + "." + args.tar_lang,
            token_delimiter=None,
            start_mark="<s>",
            end_mark="</s>",
            unk_mark="<unk>",
            max_length=args.max_len if i == 0 else None,
            truncate=True,
            trg_add_bos_eos=True)
        (args.src_vocab_size, args.tar_vocab_size, bos_id, eos_id,
         unk_id) = dataset.get_vocab_summary()
        batch_sampler = Seq2SeqBatchSampler(
            dataset=dataset,
            use_token_batch=False,
            batch_size=args.batch_size,
            pool_size=args.batch_size * 20,
            sort_type=SortType.POOL,
            shuffle=False if args.enable_ce else True,
            distribute_mode=True if i == 0 else False)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            places=device,
            collate_fn=partial(
                prepare_train_input,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=eos_id),
            num_workers=0,
            return_list=True)
        data_loaders[i] = data_loader
    return data_loaders


def prepare_train_input(insts, bos_id, eos_id, pad_id):
    src, src_length = pad_batch_data(
        [inst[0] for inst in insts], pad_id=pad_id)
    trg, trg_length = pad_batch_data(
        [inst[1] for inst in insts], pad_id=pad_id)
    trg_length = trg_length - 1
    return src, src_length, trg[:, :-1], trg_length, trg[:, 1:, np.newaxis]


def prepare_infer_input(insts, bos_id, eos_id, pad_id):
    src, src_length = pad_batch_data(insts, pad_id=pad_id)
    return src, src_length


def pad_batch_data(insts, pad_id):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    inst_lens = np.array([len(inst) for inst in insts], dtype="int64")
    max_len = np.max(inst_lens)
    inst_data = np.array(
        [inst + [pad_id] * (max_len - len(inst)) for inst in insts],
        dtype="int64")
    return inst_data, inst_lens


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"


class Converter(object):
    def __init__(self, vocab, beg, end, unk, delimiter, add_beg, add_end):
        self._vocab = vocab
        self._beg = beg
        self._end = end
        self._unk = unk
        self._delimiter = delimiter
        self._add_beg = add_beg
        self._add_end = add_end

    def __call__(self, sentence):
        return ([self._beg] if self._add_beg else []) + [
            self._vocab.get(w, self._unk)
            for w in sentence.split(self._delimiter)
        ] + ([self._end] if self._add_end else [])


class ComposedConverter(object):
    def __init__(self, converters):
        self._converters = converters

    def __call__(self, fields):
        return [
            converter(field)
            for field, converter in zip(fields, self._converters)
        ]


class SentenceBatchCreator(object):
    def __init__(self, batch_size):
        self.batch = []
        self._batch_size = batch_size

    def append(self, info):
        self.batch.append(info)
        if len(self.batch) == self._batch_size:
            tmp = self.batch
            self.batch = []
            return tmp


class TokenBatchCreator(object):
    def __init__(self, batch_size):
        self.batch = []
        self.max_len = -1
        self._batch_size = batch_size

    def append(self, info):
        cur_len = info.max_len
        max_len = max(self.max_len, cur_len)
        if max_len * (len(self.batch) + 1) > self._batch_size:
            result = self.batch
            self.batch = [info]
            self.max_len = cur_len
            return result
        else:
            self.max_len = max_len
            self.batch.append(info)


class SampleInfo(object):
    def __init__(self, i, lens):
        self.i = i
        self.lens = lens
        self.max_len = lens[0]  # to be consitent with the original reader

    def get_ranges(self, min_length=None, max_length=None, truncate=False):
        ranges = []
        # source
        if (min_length is None or self.lens[0] >= min_length) and (
                max_length is None or self.lens[0] <= max_length or truncate):
            end = max_length if truncate and max_length else self.lens[0]
            ranges.append([0, end])
        # target
        if len(self.lens) == 2:
            if (min_length is None or self.lens[1] >= min_length) and (
                    max_length is None or self.lens[1] <= max_length + 2 or
                    truncate):
                end = max_length + 2 if truncate and max_length else self.lens[
                    1]
                ranges.append([0, end])
        return ranges if len(ranges) == len(self.lens) else None


class MinMaxFilter(object):
    def __init__(self, max_len, min_len, underlying_creator):
        self._min_len = min_len
        self._max_len = max_len
        self._creator = underlying_creator

    def append(self, info):
        if (self._min_len is None or info.min_len >= self._min_len) and (
                self._max_len is None or info.max_len <= self._max_len):
            return self._creator.append(info)

    @property
    def batch(self):
        return self._creator.batch


class Seq2SeqDataset(Dataset):
    def __init__(self,
                 src_vocab_fpath,
                 trg_vocab_fpath,
                 fpattern,
                 field_delimiter="\t",
                 token_delimiter=" ",
                 start_mark="<s>",
                 end_mark="<e>",
                 unk_mark="<unk>",
                 trg_fpattern=None,
                 trg_add_bos_eos=False,
                 byte_data=False,
                 min_length=None,
                 max_length=None,
                 truncate=False):
        if byte_data:
            # The WMT16 bpe data used here seems including bytes can not be
            # decoded by utf8. Thus convert str to bytes, and use byte data
            field_delimiter = field_delimiter.encode("utf8")
            token_delimiter = token_delimiter.encode("utf8")
            start_mark = start_mark.encode("utf8")
            end_mark = end_mark.encode("utf8")
            unk_mark = unk_mark.encode("utf8")
        self._byte_data = byte_data
        self._src_vocab = self.load_dict(src_vocab_fpath, byte_data=byte_data)
        self._trg_vocab = self.load_dict(trg_vocab_fpath, byte_data=byte_data)
        self._bos_idx = self._src_vocab[start_mark]
        self._eos_idx = self._src_vocab[end_mark]
        self._unk_idx = self._src_vocab[unk_mark]
        self._field_delimiter = field_delimiter
        self._token_delimiter = token_delimiter
        self._min_length = min_length
        self._max_length = max_length
        self._truncate = truncate
        self._trg_add_bos_eos = trg_add_bos_eos
        self.load_src_trg_ids(fpattern, trg_fpattern)

    def load_src_trg_ids(self, fpattern, trg_fpattern=None):
        src_converter = Converter(
            vocab=self._src_vocab,
            beg=self._bos_idx,
            end=self._eos_idx,
            unk=self._unk_idx,
            delimiter=self._token_delimiter,
            add_beg=False,
            add_end=False)

        trg_converter = Converter(
            vocab=self._trg_vocab,
            beg=self._bos_idx,
            end=self._eos_idx,
            unk=self._unk_idx,
            delimiter=self._token_delimiter,
            add_beg=True if self._trg_add_bos_eos else False,
            add_end=True if self._trg_add_bos_eos else False)

        converters = ComposedConverter([src_converter, trg_converter])

        self._src_seq_ids = []
        self._trg_seq_ids = []
        self._sample_infos = []

        slots = [self._src_seq_ids, self._trg_seq_ids]
        for i, line in enumerate(self._load_lines(fpattern, trg_fpattern)):
            fields = converters(line)
            lens = [len(field) for field in fields]
            sample = SampleInfo(i, lens)
            field_ranges = sample.get_ranges(self._min_length,
                                             self._max_length, self._truncate)
            if field_ranges:
                for field, field_range, slot in zip(fields, field_ranges,
                                                    slots):
                    slot.append(field[field_range[0]:field_range[1]])
                self._sample_infos.append(sample)

    def _load_lines(self, fpattern, trg_fpattern=None):
        fpaths = glob.glob(fpattern)
        fpaths = sorted(fpaths)  # TODO: Add custum sort
        assert len(fpaths) > 0, "no matching file to the provided data path"

        (f_mode, f_encoding,
         endl) = ("rb", None, b"\n") if self._byte_data else ("r", "utf8",
                                                              "\n")
        if trg_fpattern is None:
            for fpath in fpaths:
                with io.open(fpath, f_mode, encoding=f_encoding) as f:
                    for line in f:
                        fields = line.strip(endl).split(self._field_delimiter)
                        yield fields
        else:
            # separated source and target language data files
            # assume we can get aligned data by sort the two language files
            # TODO: Need more rigorous check
            trg_fpaths = glob.glob(trg_fpattern)
            trg_fpaths = sorted(trg_fpaths)
            assert len(fpaths) == len(
                trg_fpaths
            ), "the number of source language data files must equal \
                with that of source language"

            for fpath, trg_fpath in zip(fpaths, trg_fpaths):
                with io.open(fpath, f_mode, encoding=f_encoding) as f:
                    with io.open(
                            trg_fpath, f_mode, encoding=f_encoding) as trg_f:
                        for line in zip(f, trg_f):
                            fields = [field.strip(endl) for field in line]
                            yield fields

    @staticmethod
    def load_dict(dict_path, reverse=False, byte_data=False):
        word_dict = {}
        (f_mode, f_encoding,
         endl) = ("rb", None, b"\n") if byte_data else ("r", "utf8", "\n")
        with io.open(dict_path, f_mode, encoding=f_encoding) as fdict:
            for idx, line in enumerate(fdict):
                if reverse:
                    word_dict[idx] = line.strip(endl)
                else:
                    word_dict[line.strip(endl)] = idx
        return word_dict

    def get_vocab_summary(self):
        return len(self._src_vocab), len(
            self._trg_vocab), self._bos_idx, self._eos_idx, self._unk_idx

    def __getitem__(self, idx):
        return (self._src_seq_ids[idx], self._trg_seq_ids[idx]
                ) if self._trg_seq_ids else self._src_seq_ids[idx]

    def __len__(self):
        return len(self._sample_infos)


class Seq2SeqBatchSampler(BatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 pool_size=10000,
                 sort_type=SortType.NONE,
                 min_length=None,
                 max_length=None,
                 shuffle=False,
                 shuffle_batch=False,
                 use_token_batch=False,
                 clip_last_batch=False,
                 distribute_mode=True,
                 seed=0):
        for arg, value in locals().items():
            if arg != "self":
                setattr(self, "_" + arg, value)
        self._random = np.random
        self._random.seed(seed)
        # for multi-devices
        self._distribute_mode = distribute_mode
        self._nranks = ParallelEnv().nranks
        self._local_rank = ParallelEnv().local_rank
        self._device_id = ParallelEnv().dev_id

    def __iter__(self):
        # global sort or global shuffle
        if self._sort_type == SortType.GLOBAL:
            infos = sorted(
                self._dataset._sample_infos, key=lambda x: x.max_len)
        else:
            if self._shuffle:
                infos = self._dataset._sample_infos
                self._random.shuffle(infos)
            else:
                infos = self._dataset._sample_infos

            if self._sort_type == SortType.POOL:
                reverse = True
                for i in range(0, len(infos), self._pool_size):
                    # to avoid placing short next to long sentences
                    reverse = False  # not reverse
                    infos[i:i + self._pool_size] = sorted(
                        infos[i:i + self._pool_size],
                        key=lambda x: x.max_len,
                        reverse=reverse)

        batches = []
        batch_creator = TokenBatchCreator(
            self.
            _batch_size) if self._use_token_batch else SentenceBatchCreator(
                self._batch_size * self._nranks)
        batch_creator = MinMaxFilter(self._max_length, self._min_length,
                                     batch_creator)

        for info in infos:
            batch = batch_creator.append(info)
            if batch is not None:
                batches.append(batch)

        if not self._clip_last_batch and len(batch_creator.batch) != 0:
            batches.append(batch_creator.batch)

        if self._shuffle_batch:
            self._random.shuffle(batches)

        if not self._use_token_batch:
            # when producing batches according to sequence number, to confirm
            # neighbor batches which would be feed and run parallel have similar
            # length (thus similar computational cost) after shuffle, we as take
            # them as a whole when shuffling and split here
            batches = [[
                batch[self._batch_size * i:self._batch_size * (i + 1)]
                for i in range(self._nranks)
            ] for batch in batches]
            batches = list(itertools.chain.from_iterable(batches))

        # for multi-device
        for batch_id, batch in enumerate(batches):
            if not self._distribute_mode or (
                    batch_id % self._nranks == self._local_rank):
                batch_indices = [info.i for info in batch]
                yield batch_indices
        if self._distribute_mode and len(batches) % self._nranks != 0:
            if self._local_rank >= len(batches) % self._nranks:
                # use previous data to pad
                yield batch_indices

    def __len__(self):
        if not self._use_token_batch:
            batch_number = (
                len(self._dataset) + self._batch_size * self._nranks - 1) // (
                    self._batch_size * self._nranks)
        else:
            # TODO(guosheng): fix the uncertain length
            batch_number = 1
        return batch_number
