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

import io
import sys
import random


def str2bool(v):
    return v.lower() in ("true", "t", "1")


def data_reader(file_path, word_dict, num_examples, phrase, epoch, padding_size, shuffle=False):
    unk_id = len(word_dict)
    all_data = []
    with io.open(file_path, "r", encoding='utf8') as fin:
        for line in fin:
            if line.startswith('text_a'):
                continue
            cols = line.strip().split("\t")
            if len(cols) != 2:
                sys.stderr.write("[NOTICE] Error Format Line!")
                continue
            label = [int(cols[1])]
            wids = [
                word_dict[x] if x in word_dict else unk_id
                for x in cols[0].split(" ")
            ]
            wids = wids[:padding_size]
            while len(wids) < padding_size:
                wids.append(unk_id)
            all_data.append((wids, label))

    if shuffle:
        if phrase == "train":
            random.shuffle(all_data)

    num_examples[phrase] = len(all_data)

    def reader():
        for epoch_index in range(epoch):
            for doc, label in all_data:
                yield doc, label

    return reader


def load_vocab(file_path):
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for line in f:
            if line.strip() not in vocab:
                vocab[line.strip()] = wid
                wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab
