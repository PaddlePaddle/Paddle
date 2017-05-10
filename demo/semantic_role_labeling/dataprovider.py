# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

from paddle.trainer.PyDataProvider2 import *

UNK_IDX = 0


def hook(settings, word_dict, label_dict, predicate_dict, **kwargs):
    settings.word_dict = word_dict
    settings.label_dict = label_dict
    settings.predicate_dict = predicate_dict

    #all inputs are integral and sequential type
    settings.slots = [
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(predicate_dict)), integer_value_sequence(2),
        integer_value_sequence(len(label_dict))
    ]


def get_batch_size(yeild_data):
    return len(yeild_data[0])


@provider(
    init_hook=hook,
    should_shuffle=True,
    calc_batch_size=get_batch_size,
    can_over_batch_size=True,
    cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_name):
    with open(file_name, 'r') as fdata:
        for line in fdata:
            sentence, predicate, ctx_n2, ctx_n1, ctx_0, ctx_p1, ctx_p2,  mark, label = \
                line.strip().split('\t')

            words = sentence.split()
            sen_len = len(words)
            word_slot = [settings.word_dict.get(w, UNK_IDX) for w in words]

            predicate_slot = [settings.predicate_dict.get(predicate)] * sen_len
            ctx_n2_slot = [settings.word_dict.get(ctx_n2, UNK_IDX)] * sen_len
            ctx_n1_slot = [settings.word_dict.get(ctx_n1, UNK_IDX)] * sen_len
            ctx_0_slot = [settings.word_dict.get(ctx_0, UNK_IDX)] * sen_len
            ctx_p1_slot = [settings.word_dict.get(ctx_p1, UNK_IDX)] * sen_len
            ctx_p2_slot = [settings.word_dict.get(ctx_p2, UNK_IDX)] * sen_len

            marks = mark.split()
            mark_slot = [int(w) for w in marks]

            label_list = label.split()
            label_slot = [settings.label_dict.get(w) for w in label_list]
            yield word_slot, ctx_n2_slot, ctx_n1_slot, \
                  ctx_0_slot, ctx_p1_slot, ctx_p2_slot, predicate_slot, mark_slot, label_slot
