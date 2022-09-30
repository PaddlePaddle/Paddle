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

import numpy as np
import random

SEED = 2020


def get_bert_config():
    bert_config = {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 2,
        "initializer_range": 0.02,
        "intermediate_size": 72,
        "max_position_embeddings": 512,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pooler_fc_size": 2,
        "pooler_num_attention_heads": 2,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 8,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 21128
    }
    return bert_config


def mask(batch_tokens, total_token_num, vocab_size, CLS=1, SEP=2, MASK=3):
    """
    Add mask for batch_tokens, return out, mask_label, mask_pos;
    Note: mask_pos responding the batch_tokens after padded;
    """
    max_len = max([len(sent) for sent in batch_tokens])
    mask_label = []
    mask_pos = []
    # NOTE: numpy random is not thread-safe, for async DataLoader,
    # using np.random.seed() directly is risky, using RandomState
    # class is a better way
    self_random = np.random.RandomState(SEED)
    prob_mask = self_random.rand(total_token_num)
    # Note: the first token is [CLS], so [low=1]
    replace_ids = self_random.randint(1, high=vocab_size, size=total_token_num)
    pre_sent_len = 0
    prob_index = 0
    for sent_index, sent in enumerate(batch_tokens):
        mask_flag = False
        prob_index += pre_sent_len
        for token_index, token in enumerate(sent):
            prob = prob_mask[prob_index + token_index]
            if prob > 0.15:
                continue
            elif 0.03 < prob <= 0.15:
                # mask
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    sent[token_index] = MASK
                    mask_flag = True
                    mask_pos.append(sent_index * max_len + token_index)
            elif 0.015 < prob <= 0.03:
                # random replace
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    sent[token_index] = replace_ids[prob_index + token_index]
                    mask_flag = True
                    mask_pos.append(sent_index * max_len + token_index)
            else:
                # keep the original token
                if token != SEP and token != CLS:
                    mask_label.append(sent[token_index])
                    mask_pos.append(sent_index * max_len + token_index)
        pre_sent_len = len(sent)

        # ensure at least mask one word in a sentence
        while not mask_flag:
            token_index = int(self_random.randint(1, high=len(sent) - 1,
                                                  size=1))
            if sent[token_index] != SEP and sent[token_index] != CLS:
                mask_label.append(sent[token_index])
                sent[token_index] = MASK
                mask_flag = True
                mask_pos.append(sent_index * max_len + token_index)
    mask_label = np.array(mask_label).astype("int64").reshape([-1, 1])
    mask_pos = np.array(mask_pos).astype("int64").reshape([-1, 1])
    return batch_tokens, mask_label, mask_pos


def pad_batch_data(insts,
                   pad_idx=0,
                   return_pos=False,
                   return_input_mask=False,
                   return_max_len=False,
                   return_num_token=False):
    """
    Pad the instances to the max sequence length in batch, and generate the
    corresponding position data and input mask.
    """
    return_list = []
    max_len = max(len(inst) for inst in insts)
    # Any token included in dict can be used to pad, since the paddings' loss
    # will be masked out by weights and make no effect on parameter gradients.

    inst_data = np.array([
        list(inst) + list([pad_idx] * (max_len - len(inst))) for inst in insts
    ])
    return_list += [inst_data.astype("int64").reshape([-1, max_len])]

    # position data
    if return_pos:
        inst_pos = np.array([
            list(range(0, len(inst))) + [pad_idx] * (max_len - len(inst))
            for inst in insts
        ])

        return_list += [inst_pos.astype("int64").reshape([-1, max_len])]

    if return_input_mask:
        # This is used to avoid attention on paddings.
        input_mask_data = np.array(
            [[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
        input_mask_data = np.expand_dims(input_mask_data, axis=-1)
        return_list += [input_mask_data.astype("float32")]

    if return_max_len:
        return_list += [max_len]

    if return_num_token:
        num_token = 0
        for inst in insts:
            num_token += len(inst)
        return_list += [num_token]

    return return_list if len(return_list) > 1 else return_list[0]


def prepare_batch_data(insts,
                       total_token_num,
                       voc_size=0,
                       pad_id=None,
                       cls_id=None,
                       sep_id=None,
                       mask_id=None,
                       return_input_mask=True,
                       return_max_len=True,
                       return_num_token=False):
    """
    1. generate Tensor of data
    2. generate Tensor of position
    3. generate self attention mask, [shape: batch_size *  max_len * max_len]
    """

    batch_src_ids = [inst[0] for inst in insts]
    batch_sent_ids = [inst[1] for inst in insts]
    batch_pos_ids = [inst[2] for inst in insts]
    labels_list = []

    for i in range(3, len(insts[0]), 1):
        labels = [inst[i] for inst in insts]
        labels = np.array(labels).astype("int64").reshape([-1, 1])
        labels_list.append(labels)

    # First step: do mask without padding
    if mask_id >= 0:
        out, mask_label, mask_pos = mask(batch_src_ids,
                                         total_token_num,
                                         vocab_size=voc_size,
                                         CLS=cls_id,
                                         SEP=sep_id,
                                         MASK=mask_id)
    else:
        out = batch_src_ids
    # Second step: padding
    src_id, self_input_mask = pad_batch_data(out,
                                             pad_idx=pad_id,
                                             return_input_mask=True)
    pos_id = pad_batch_data(batch_pos_ids,
                            pad_idx=pad_id,
                            return_pos=False,
                            return_input_mask=False)
    sent_id = pad_batch_data(batch_sent_ids,
                             pad_idx=pad_id,
                             return_pos=False,
                             return_input_mask=False)

    if mask_id >= 0:
        return_list = [
            src_id, pos_id, sent_id, self_input_mask, mask_label, mask_pos
        ] + labels_list
    else:
        return_list = [src_id, pos_id, sent_id, self_input_mask] + labels_list

    res = return_list if len(return_list) > 1 else return_list[0]
    return res


class DataReader(object):

    def __init__(self,
                 batch_size=4096,
                 in_tokens=True,
                 max_seq_len=512,
                 shuffle_files=False,
                 epoch=100,
                 voc_size=0,
                 is_test=False,
                 generate_neg_sample=False):

        self.batch_size = batch_size
        self.in_tokens = in_tokens
        self.shuffle_files = shuffle_files
        self.epoch = epoch
        self.current_epoch = 0
        self.current_file_index = 0
        self.total_file = 0
        self.current_file = None
        self.voc_size = voc_size
        self.max_seq_len = max_seq_len

        self.pad_id = 0
        self.cls_id = 101
        self.sep_id = 102
        self.mask_id = 103
        self.is_test = is_test
        self.generate_neg_sample = generate_neg_sample
        if self.in_tokens:
            assert self.batch_size >= self.max_seq_len, "The number of " \
                                                        "tokens in batch should not be smaller than max seq length."

        if self.is_test:
            self.epoch = 1
            self.shuffle_files = False

    def build_fake_data(self):
        for _ in range(1000000):
            # NOTE: python random has bug in python2,
            # we should avoid using random module,
            # please using numpy.random
            self_random = np.random.RandomState(SEED)
            sent0_len = self_random.randint(50, 100)
            sent1_len = self_random.randint(50, 100)

            token_ids = [1] \
                        + [self_random.randint(0, 10000) for i in range(sent0_len-1)] \
                        + [self_random.randint(0, 10000) for i in range(sent1_len-1)] \
                        + [2]

            sent_ids = [0 for i in range(sent0_len)
                        ] + [1 for i in range(sent1_len)]
            pos_ids = [i for i in range(sent0_len + sent1_len)]
            label = 1
            yield token_ids, sent_ids, pos_ids, label

    def data_generator(self):

        def wrapper():

            def reader():
                for epoch in range(self.epoch):
                    self.current_epoch = epoch + 1
                    sample_generator = self.build_fake_data()
                    for sample in sample_generator:
                        if sample is None:
                            continue
                        yield sample

            def batch_reader(reader, batch_size, in_tokens):
                batch, total_token_num, max_len = [], 0, 0
                for parsed_line in reader():
                    token_ids, sent_ids, pos_ids, label = parsed_line
                    max_len = max(max_len, len(token_ids))
                    if in_tokens:
                        to_append = (len(batch) + 1) * max_len <= batch_size
                    else:
                        to_append = len(batch) < batch_size
                    if to_append:
                        batch.append(parsed_line)
                        total_token_num += len(token_ids)
                    else:
                        yield batch, total_token_num
                        batch, total_token_num, max_len = [
                            parsed_line
                        ], len(token_ids), len(token_ids)

                if len(batch) > 0:
                    yield batch, total_token_num

            for batch_data, total_token_num in batch_reader(
                    reader, self.batch_size, self.in_tokens):
                yield prepare_batch_data(batch_data,
                                         total_token_num,
                                         voc_size=self.voc_size,
                                         pad_id=self.pad_id,
                                         cls_id=self.cls_id,
                                         sep_id=self.sep_id,
                                         mask_id=self.mask_id,
                                         return_input_mask=True,
                                         return_max_len=False,
                                         return_num_token=False)

        return wrapper


class ModelHyperParams(object):
    generate_neg_sample = False
    epoch = 100
    max_seq_len = 512
    batch_size = 8192
    in_tokens = True


def get_feed_data_reader(bert_config):
    args = ModelHyperParams()
    data_reader = DataReader(batch_size=args.batch_size,
                             in_tokens=args.in_tokens,
                             voc_size=bert_config['vocab_size'],
                             epoch=args.epoch,
                             max_seq_len=args.max_seq_len,
                             generate_neg_sample=args.generate_neg_sample)

    return data_reader
