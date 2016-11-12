# Copyright (c) 2016 Baidu, Inc. All Rights Reserved
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


def hook(settings, word_dict, label_dict, **kwargs):
    settings.word_dict = word_dict
    settings.label_dict = label_dict
    #all inputs are integral and sequential type
    settings.slots = [
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)),
        integer_value_sequence(len(word_dict)), integer_value_sequence(2),
        integer_value_sequence(len(label_dict))
    ]


@provider(init_hook=hook)
def process(obj, file_name):
    with open(file_name, 'r') as fdata:
        for line in fdata:
            sentence, predicate, ctx_n1, ctx_0, ctx_p1, mark, label = \
                line.strip().split('\t')

            words = sentence.split()
            sen_len = len(words)
            word_slot = [obj.word_dict.get(w, UNK_IDX) for w in words]

            predicate_slot = [obj.word_dict.get(predicate, UNK_IDX)] * sen_len
            ctx_n1_slot = [obj.word_dict.get(ctx_n1, UNK_IDX)] * sen_len
            ctx_0_slot = [obj.word_dict.get(ctx_0, UNK_IDX)] * sen_len
            ctx_p1_slot = [obj.word_dict.get(ctx_p1, UNK_IDX)] * sen_len

            marks = mark.split()
            mark_slot = [int(w) for w in marks]

            label_list = label.split()
            label_slot = [obj.label_dict.get(w) for w in label_list]

            yield word_slot, predicate_slot, ctx_n1_slot, \
                  ctx_0_slot, ctx_p1_slot, mark_slot, label_slot
