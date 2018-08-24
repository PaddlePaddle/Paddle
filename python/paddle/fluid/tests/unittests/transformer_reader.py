# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os
import random
import tarfile
import cPickle


class SortType(object):
    GLOBAL = 'global'
    POOL = 'pool'
    NONE = "none"


class Converter(object):
    def __init__(self, vocab, beg, end, unk, delimiter):
        self._vocab = vocab
        self._beg = beg
        self._end = end
        self._unk = unk
        self._delimiter = delimiter

    def __call__(self, sentence):
        return [self._beg] + [
            self._vocab.get(w, self._unk)
            for w in sentence.split(self._delimiter)
        ] + [self._end]


class ComposedConverter(object):
    def __init__(self, converters):
        self._converters = converters

    def __call__(self, parallel_sentence):
        return [
            self._converters[i](parallel_sentence[i])
            for i in range(len(self._converters))
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
    def __init__(self, i, max_len, min_len):
        self.i = i
        self.min_len = min_len
        self.max_len = max_len


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


class DataReader(object):
    """
    The data reader loads all data from files and produces batches of data
    in the way corresponding to settings.

    An example of returning a generator producing data batches whose data
    is shuffled in each pass and sorted in each pool:

    ```
    train_data = DataReader(
        src_vocab_fpath='data/src_vocab_file',
        trg_vocab_fpath='data/trg_vocab_file',
        fpattern='data/part-*',
        use_token_batch=True,
        batch_size=2000,
        pool_size=10000,
        sort_type=SortType.POOL,
        shuffle=True,
        shuffle_batch=True,
        start_mark='<s>',
        end_mark='<e>',
        unk_mark='<unk>',
        clip_last_batch=False).batch_generator
    ```

    :param src_vocab_fpath: The path of vocabulary file of source language.
    :type src_vocab_fpath: basestring
    :param trg_vocab_fpath: The path of vocabulary file of target language.
    :type trg_vocab_fpath: basestring
    :param fpattern: The pattern to match data files.
    :type fpattern: basestring
    :param batch_size: The number of sequences contained in a mini-batch.
        or the maximum number of tokens (include paddings) contained in a
        mini-batch.
    :type batch_size: int
    :param pool_size: The size of pool buffer.
    :type pool_size: int
    :param sort_type: The grain to sort by length: 'global' for all
        instances; 'pool' for instances in pool; 'none' for no sort.
    :type sort_type: basestring
    :param clip_last_batch: Whether to clip the last uncompleted batch.
    :type clip_last_batch: bool
    :param tar_fname: The data file in tar if fpattern matches a tar file.
    :type tar_fname: basestring
    :param min_length: The minimum length used to filt sequences.
    :type min_length: int
    :param max_length: The maximum length used to filt sequences.
    :type max_length: int
    :param shuffle: Whether to shuffle all instances.
    :type shuffle: bool
    :param shuffle_batch: Whether to shuffle the generated batches.
    :type shuffle_batch: bool
    :param use_token_batch: Whether to produce batch data according to
        token number.
    :type use_token_batch: bool
    :param field_delimiter: The delimiter used to split source and target in
        each line of data file.
    :type field_delimiter: basestring
    :param token_delimiter: The delimiter used to split tokens in source or
        target sentences.
    :type token_delimiter: basestring
    :param start_mark: The token representing for the beginning of
        sentences in dictionary.
    :type start_mark: basestring
    :param end_mark: The token representing for the end of sentences
        in dictionary.
    :type end_mark: basestring
    :param unk_mark: The token representing for unknown word in dictionary.
    :type unk_mark: basestring
    :param seed: The seed for random.
    :type seed: int
    """

    def __init__(self,
                 src_vocab_fpath,
                 trg_vocab_fpath,
                 fpattern,
                 batch_size,
                 pool_size,
                 sort_type=SortType.GLOBAL,
                 clip_last_batch=True,
                 tar_fname=None,
                 min_length=0,
                 max_length=100,
                 shuffle=True,
                 shuffle_batch=False,
                 use_token_batch=False,
                 field_delimiter="\t",
                 token_delimiter=" ",
                 start_mark="<s>",
                 end_mark="<e>",
                 unk_mark="<unk>",
                 seed=0):
        self._src_vocab = self.load_dict(src_vocab_fpath)
        self._only_src = True
        if trg_vocab_fpath is not None:
            self._trg_vocab = self.load_dict(trg_vocab_fpath)
            self._only_src = False
        self._pool_size = pool_size
        self._batch_size = batch_size
        self._use_token_batch = use_token_batch
        self._sort_type = sort_type
        self._clip_last_batch = clip_last_batch
        self._shuffle = shuffle
        self._shuffle_batch = shuffle_batch
        self._min_length = min_length
        self._max_length = max_length
        self._field_delimiter = field_delimiter
        self._token_delimiter = token_delimiter
        self.load_src_trg_ids(end_mark, fpattern, start_mark, tar_fname,
                              unk_mark)
        self._random = random.Random(x=seed)

    def load_src_trg_ids(self, end_mark, fpattern, start_mark, tar_fname,
                         unk_mark):
        converters = [
            Converter(
                vocab=self._src_vocab,
                beg=self._src_vocab[start_mark],
                end=self._src_vocab[end_mark],
                unk=self._src_vocab[unk_mark],
                delimiter=self._token_delimiter)
        ]
        if not self._only_src:
            converters.append(
                Converter(
                    vocab=self._trg_vocab,
                    beg=self._trg_vocab[start_mark],
                    end=self._trg_vocab[end_mark],
                    unk=self._trg_vocab[unk_mark],
                    delimiter=self._token_delimiter))

        converters = ComposedConverter(converters)

        self._src_seq_ids = []
        self._trg_seq_ids = None if self._only_src else []
        self._sample_infos = []

        for i, line in enumerate(self._load_lines(fpattern, tar_fname)):
            src_trg_ids = converters(line)
            self._src_seq_ids.append(src_trg_ids[0])
            lens = [len(src_trg_ids[0])]
            if not self._only_src:
                self._trg_seq_ids.append(src_trg_ids[1])
                lens.append(len(src_trg_ids[1]))
            self._sample_infos.append(SampleInfo(i, max(lens), min(lens)))

    def _load_lines(self, fpattern, tar_fname):
        fpaths = glob.glob(fpattern)

        if len(fpaths) == 1 and tarfile.is_tarfile(fpaths[0]):
            if tar_fname is None:
                raise Exception("If tar file provided, please set tar_fname.")

            f = tarfile.open(fpaths[0], "r")
            for line in f.extractfile(tar_fname):
                fields = line.strip("\n").split(self._field_delimiter)
                if (not self._only_src and len(fields) == 2) or (
                        self._only_src and len(fields) == 1):
                    yield fields
        else:
            for fpath in fpaths:
                if not os.path.isfile(fpath):
                    raise IOError("Invalid file: %s" % fpath)

                with open(fpath, "r") as f:
                    for line in f:
                        fields = line.strip("\n").split(self._field_delimiter)
                        if (not self._only_src and len(fields) == 2) or (
                                self._only_src and len(fields) == 1):
                            yield fields

    @staticmethod
    def load_dict(dict_path, reverse=False):
        word_dict = {}
        with open(dict_path, "r") as fdict:
            for idx, line in enumerate(fdict):
                if reverse:
                    word_dict[idx] = line.strip("\n")
                else:
                    word_dict[line.strip("\n")] = idx
        return word_dict

    def batch_generator(self):
        # global sort or global shuffle
        if self._sort_type == SortType.GLOBAL:
            infos = sorted(
                self._sample_infos, key=lambda x: x.max_len, reverse=True)
        else:
            if self._shuffle:
                infos = self._sample_infos
                self._random.shuffle(infos)
            else:
                infos = self._sample_infos

            if self._sort_type == SortType.POOL:
                for i in range(0, len(infos), self._pool_size):
                    infos[i:i + self._pool_size] = sorted(
                        infos[i:i + self._pool_size], key=lambda x: x.max_len)

        # concat batch
        batches = []
        batch_creator = TokenBatchCreator(
            self._batch_size
        ) if self._use_token_batch else SentenceBatchCreator(self._batch_size)
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

        for batch in batches:
            batch_ids = [info.i for info in batch]

            if self._only_src:
                yield [[self._src_seq_ids[idx]] for idx in batch_ids]
            else:
                yield [(self._src_seq_ids[idx], self._trg_seq_ids[idx][:-1],
                        self._trg_seq_ids[idx][1:]) for idx in batch_ids]
