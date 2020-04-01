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

import collections
import os
import io
import sys
import numpy as np

Py3 = sys.version_info[0] == 3

UNK_ID = 0


def _read_words(filename):
    data = []
    with io.open(filename, "r", encoding='utf-8') as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace(u"\n", u"<eos>").split()


def read_all_line(filenam):
    data = []
    with io.open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            data.append(line.strip())


def _build_vocab(filename):

    vocab_dict = {}
    ids = 0
    with io.open(filename, "r", encoding='utf-8') as f:
        for line in f.readlines():
            vocab_dict[line.strip()] = ids
            ids += 1

    print("vocab word num", ids)

    return vocab_dict


def _para_file_to_ids(src_file, tar_file, src_vocab, tar_vocab):

    src_data = []
    with io.open(src_file, "r", encoding='utf-8') as f_src:
        for line in f_src.readlines():
            arra = line.strip().split()
            ids = [src_vocab[w] if w in src_vocab else UNK_ID for w in arra]
            ids = ids

            src_data.append(ids)

    tar_data = []
    with io.open(tar_file, "r", encoding='utf-8') as f_tar:
        for line in f_tar.readlines():
            arra = line.strip().split()
            ids = [tar_vocab[w] if w in tar_vocab else UNK_ID for w in arra]

            ids = [1] + ids + [2]

            tar_data.append(ids)

    return src_data, tar_data


def filter_len(src, tar, max_sequence_len=50):
    new_src = []
    new_tar = []

    for id1, id2 in zip(src, tar):
        if len(id1) > max_sequence_len:
            id1 = id1[:max_sequence_len]
        if len(id2) > max_sequence_len + 2:
            id2 = id2[:max_sequence_len + 2]

        new_src.append(id1)
        new_tar.append(id2)

    return new_src, new_tar


def raw_data(src_lang,
             tar_lang,
             vocab_prefix,
             train_prefix,
             eval_prefix,
             test_prefix,
             max_sequence_len=50):

    src_vocab_file = vocab_prefix + "." + src_lang
    tar_vocab_file = vocab_prefix + "." + tar_lang

    src_train_file = train_prefix + "." + src_lang
    tar_train_file = train_prefix + "." + tar_lang

    src_eval_file = eval_prefix + "." + src_lang
    tar_eval_file = eval_prefix + "." + tar_lang

    src_test_file = test_prefix + "." + src_lang
    tar_test_file = test_prefix + "." + tar_lang

    src_vocab = _build_vocab(src_vocab_file)
    tar_vocab = _build_vocab(tar_vocab_file)

    train_src, train_tar = _para_file_to_ids( src_train_file, tar_train_file, \
                                              src_vocab, tar_vocab )
    train_src, train_tar = filter_len(
        train_src, train_tar, max_sequence_len=max_sequence_len)
    eval_src, eval_tar = _para_file_to_ids( src_eval_file, tar_eval_file, \
                                              src_vocab, tar_vocab )

    test_src, test_tar = _para_file_to_ids( src_test_file, tar_test_file, \
                                              src_vocab, tar_vocab )

    return ( train_src, train_tar), (eval_src, eval_tar), (test_src, test_tar),\
            (src_vocab, tar_vocab)


def raw_mono_data(vocab_file, file_path):

    src_vocab = _build_vocab(vocab_file)

    test_src, test_tar = _para_file_to_ids( file_path, file_path, \
                                              src_vocab, src_vocab )

    return (test_src, test_tar)


def get_data_iter(raw_data,
                  batch_size,
                  mode='train',
                  enable_ce=False,
                  cache_num=20):

    src_data, tar_data = raw_data

    data_len = len(src_data)

    index = np.arange(data_len)
    if mode == "train" and not enable_ce:
        np.random.shuffle(index)

    def to_pad_np(data, source=False):
        max_len = 0
        bs = min(batch_size, len(data))
        for ele in data:
            if len(ele) > max_len:
                max_len = len(ele)

        ids = np.ones((bs, max_len), dtype='int64') * 2
        mask = np.zeros((bs), dtype='int32')

        for i, ele in enumerate(data):
            ids[i, :len(ele)] = ele
            if not source:
                mask[i] = len(ele) - 1
            else:
                mask[i] = len(ele)

        return ids, mask

    b_src = []

    if mode != "train":
        cache_num = 1
    for j in range(data_len):
        if len(b_src) == batch_size * cache_num:
            # build batch size

            # sort
            if mode == 'infer':
                new_cache = b_src
            else:
                new_cache = sorted(b_src, key=lambda k: len(k[0]))

            for i in range(cache_num):
                batch_data = new_cache[i * batch_size:(i + 1) * batch_size]
                src_cache = [w[0] for w in batch_data]
                tar_cache = [w[1] for w in batch_data]
                src_ids, src_mask = to_pad_np(src_cache, source=True)
                tar_ids, tar_mask = to_pad_np(tar_cache)
                yield (src_ids, src_mask, tar_ids, tar_mask)

            b_src = []

        b_src.append((src_data[index[j]], tar_data[index[j]]))
    if len(b_src) == batch_size * cache_num or mode == 'infer':
        if mode == 'infer':
            new_cache = b_src
        else:
            new_cache = sorted(b_src, key=lambda k: len(k[0]))

        for i in range(cache_num):
            batch_end = min(len(new_cache), (i + 1) * batch_size)
            batch_data = new_cache[i * batch_size:batch_end]
            src_cache = [w[0] for w in batch_data]
            tar_cache = [w[1] for w in batch_data]
            src_ids, src_mask = to_pad_np(src_cache, source=True)
            tar_ids, tar_mask = to_pad_np(tar_cache)
            yield (src_ids, src_mask, tar_ids, tar_mask)
