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

import glob
import sys
import os
import io
import itertools
from functools import partial

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.parallel import ParallelEnv
from paddle.io import BatchSampler, DataLoader, Dataset


def create_data_loader(args, device):
    data_loaders = [(None, None)] * 2
    data_files = [args.training_file, args.validation_file
                  ] if args.validation_file else [args.training_file]
    for i, data_file in enumerate(data_files):
        dataset = Seq2SeqDataset(
            fpattern=data_file,
            src_vocab_fpath=args.src_vocab_fpath,
            trg_vocab_fpath=args.trg_vocab_fpath,
            token_delimiter=args.token_delimiter,
            start_mark=args.special_token[0],
            end_mark=args.special_token[1],
            unk_mark=args.special_token[2],
            byte_data=True)
        args.src_vocab_size, args.trg_vocab_size, args.bos_idx, args.eos_idx, \
            args.unk_idx = dataset.get_vocab_summary()
        batch_sampler = Seq2SeqBatchSampler(
            dataset=dataset,
            use_token_batch=args.use_token_batch,
            batch_size=args.batch_size,
            pool_size=args.pool_size,
            sort_type=args.sort_type,
            shuffle=args.shuffle,
            shuffle_batch=args.shuffle_batch,
            max_length=args.max_length,
            distribute_mode=True
            if i == 0 else False)  # every device eval all data
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            places=device,
            collate_fn=partial(
                prepare_train_input,
                bos_idx=args.bos_idx,
                eos_idx=args.eos_idx,
                src_pad_idx=args.eos_idx,
                trg_pad_idx=args.eos_idx,
                n_head=args.n_head),
            num_workers=0,  # TODO: use multi-process
            return_list=True)
        data_loaders[i] = (data_loader, batch_sampler.__len__)
    return data_loaders


def prepare_train_input(insts, bos_idx, eos_idx, src_pad_idx, trg_pad_idx,
                        n_head):
    """
    Put all padded data needed by training into a list.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] + [eos_idx] for inst in insts],
        src_pad_idx,
        n_head,
        is_target=False)
    src_word = src_word.reshape(-1, src_max_len)
    src_pos = src_pos.reshape(-1, src_max_len)
    trg_word, trg_pos, trg_slf_attn_bias, trg_max_len = pad_batch_data(
        [[bos_idx] + inst[1] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=True)
    trg_word = trg_word.reshape(-1, trg_max_len)
    trg_pos = trg_pos.reshape(-1, trg_max_len)

    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, trg_max_len, 1]).astype("float32")

    lbl_word, lbl_weight, num_token = pad_batch_data(
        [inst[1] + [eos_idx] for inst in insts],
        trg_pad_idx,
        n_head,
        is_target=False,
        is_label=True,
        return_attn_bias=False,
        return_max_len=False,
        return_num_token=True)
    lbl_word = lbl_word.reshape(-1, 1)
    lbl_weight = lbl_weight.reshape(-1, 1)

    data_inputs = [
        src_word, src_pos, src_slf_attn_bias, trg_word, trg_pos,
        trg_slf_attn_bias, trg_src_attn_bias, lbl_word, lbl_weight
    ]

    return data_inputs


def prepare_infer_input(insts, bos_idx, eos_idx, src_pad_idx, n_head):
    """
    Put all padded data needed by beam search decoder into a list.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] + [eos_idx] for inst in insts],
        src_pad_idx,
        n_head,
        is_target=False)
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, 1, 1]).astype("float32")
    src_word = src_word.reshape(-1, src_max_len)
    src_pos = src_pos.reshape(-1, src_max_len)

    data_inputs = [src_word, src_pos, src_slf_attn_bias, trg_src_attn_bias]
    return data_inputs


def pad_batch_data(insts,
                   pad_idx,
                   n_head,
                   is_target=False,
                   is_label=False,
                   return_attn_bias=True,
                   return_max_len=True,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and attention bias.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.
    inst_data = np.array(
        [inst + [pad_idx] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_data.astype("int64").reshape([-1, 1])]
    if is_label:  # label weight
        inst_weight = np.array([[1.] * len(inst) + [0.] * (max_len - len(inst))
                                for inst in insts])
        return_list += [inst_weight.astype("float32").reshape([-1, 1])]
    else:  # position data
        inst_pos = np.array([
            list(range(0, len(inst))) + [0] * (max_len - len(inst))
            for inst in insts
        ])
        return_list += [inst_pos.astype("int64").reshape([-1, 1])]
    if return_attn_bias:
        if is_target:
            # This is used to avoid attention on paddings and subsequent
            # words.
            slf_attn_bias_data = np.ones(
                (inst_data.shape[0], max_len, max_len))
            slf_attn_bias_data = np.triu(slf_attn_bias_data,
                                         1).reshape([-1, 1, max_len, max_len])
            slf_attn_bias_data = np.tile(slf_attn_bias_data,
                                         [1, n_head, 1, 1]) * [-1e9]
        else:
            # This is used to avoid attention on paddings.
            slf_attn_bias_data = np.array([[0] * len(inst) + [-1e9] *
                                           (max_len - len(inst))
                                           for inst in insts])
            slf_attn_bias_data = np.tile(
                slf_attn_bias_data.reshape([-1, 1, 1, max_len]),
                [1, n_head, max_len, 1])
        return_list += [slf_attn_bias_data.astype("float32")]
    if return_max_len:
        return_list += [max_len]
    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]
    return return_list if len(return_list) > 1 else return_list[0]


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
        # take bos and eos into account
        self.min_len = min(lens[0] + 1, lens[1] + 2)
        self.max_len = max(lens[0] + 1, lens[1] + 2)


class MinMaxFilter(object):
    def __init__(self, max_len, min_len, underlying_creator):
        self._min_len = min_len
        self._max_len = max_len
        self._creator = underlying_creator

    def append(self, info):
        if info.max_len > self._max_len or info.min_len < self._min_len:
            return
        else:
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
                 byte_data=False):
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
            add_beg=False,
            add_end=False)

        converters = ComposedConverter([src_converter, trg_converter])

        self._src_seq_ids = []
        self._trg_seq_ids = []
        self._sample_infos = []

        slots = [self._src_seq_ids, self._trg_seq_ids]
        for i, line in enumerate(self._load_lines(fpattern, trg_fpattern)):
            lens = []
            for field, slot in zip(converters(line), slots):
                slot.append(field)
                lens.append(len(field))
            self._sample_infos.append(SampleInfo(i, lens))

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
                 min_length=0,
                 max_length=100,
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
                    reverse = not reverse
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
        self.batch_number = (len(batches) + self._nranks - 1) // self._nranks

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
        if hasattr(self, "batch_number"):  #
            return self.batch_number
        if not self._use_token_batch:
            batch_number = (
                len(self._dataset) + self._batch_size * self._nranks - 1) // (
                    self._batch_size * self._nranks)
        else:
            # for uncertain batch number, the actual value is self.batch_number
            batch_number = sys.maxsize
        return batch_number
