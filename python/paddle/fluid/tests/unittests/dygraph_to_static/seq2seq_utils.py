# -*- coding: utf-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

import numpy as np

SEED = 2020


def build_fake_sentence(seed):
    random = np.random.RandomState(seed)
    sentence_len = random.randint(5, 15)
    token_ids = [random.randint(0, 1000) for _ in range(sentence_len - 1)]
    return token_ids


def get_data_iter(batch_size, mode='train', cache_num=20):

    self_random = np.random.RandomState(SEED)

    def to_pad_np(data, source=False):
        max_len = 0
        bs = min(batch_size, len(data))
        for ele in data:
            if len(ele) > max_len:
                max_len = len(ele)

        ids = np.ones((bs, max_len), dtype='int64') * 2
        mask = np.zeros((bs), dtype='int32')

        for i, ele in enumerate(data):
            ids[i, : len(ele)] = ele
            if not source:
                mask[i] = len(ele) - 1
            else:
                mask[i] = len(ele)

        return ids, mask

    b_src = []

    if mode != "train":
        cache_num = 1
    data_len = 1000
    for j in range(data_len):
        if len(b_src) == batch_size * cache_num:
            if mode == 'infer':
                new_cache = b_src
            else:
                new_cache = sorted(b_src, key=lambda k: len(k[0]))

            for i in range(cache_num):
                batch_data = new_cache[i * batch_size : (i + 1) * batch_size]
                src_cache = [w[0] for w in batch_data]
                tar_cache = [w[1] for w in batch_data]
                src_ids, src_mask = to_pad_np(src_cache, source=True)
                tar_ids, tar_mask = to_pad_np(tar_cache)
                yield (src_ids, src_mask, tar_ids, tar_mask)

            b_src = []
        src_seed = self_random.randint(0, data_len)
        tar_seed = self_random.randint(0, data_len)
        src_data = build_fake_sentence(src_seed)
        tar_data = build_fake_sentence(tar_seed)
        b_src.append((src_data, tar_data))

    if len(b_src) == batch_size * cache_num or mode == 'infer':
        if mode == 'infer':
            new_cache = b_src
        else:
            new_cache = sorted(b_src, key=lambda k: len(k[0]))

        for i in range(cache_num):
            batch_end = min(len(new_cache), (i + 1) * batch_size)
            batch_data = new_cache[i * batch_size : batch_end]
            src_cache = [w[0] for w in batch_data]
            tar_cache = [w[1] for w in batch_data]
            src_ids, src_mask = to_pad_np(src_cache, source=True)
            tar_ids, tar_mask = to_pad_np(tar_cache)
            yield (src_ids, src_mask, tar_ids, tar_mask)


class Seq2SeqModelHyperParams:
    # Whether use attention model
    attention = False

    # learning rate for optimizer
    learning_rate = 0.01

    # layers number of encoder and decoder
    num_layers = 2

    # hidden size of encoder and decoder
    hidden_size = 8

    src_vocab_size = 1000
    tar_vocab_size = 1000
    batch_size = 8
    max_epoch = 12

    # max length for source and target sentence
    max_len = 30

    # drop probability
    dropout = 0.0

    # init scale for parameter
    init_scale = 0.1

    # max grad norm for global norm clip
    max_grad_norm = 5.0

    # model path for model to save

    base_model_path = "dy2stat/model/base_seq2seq"
    attn_model_path = "dy2stat/model/attn_seq2seq"

    # reload model to inference
    reload_model = "model/epoch_0.pdparams"

    beam_size = 4

    max_seq_len = 3
