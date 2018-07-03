#  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from paddle.trainer.PyDataProvider2 import *

# Note that each config should has an independent provider
# in current design of PyDataProvider2.
#######################################################
data = [
    [[[1, 3, 2], [4, 5, 2]], 0],
    [[[0, 2], [2, 5], [0, 1, 2]], 1],
]


# Used for sequence_nest_rnn.conf
@provider(
    input_types=[integer_value_sub_sequence(10), integer_value(3)],
    should_shuffle=False)
def process_subseq(settings, file_name):
    for d in data:
        yield d


# Used for sequence_rnn.conf
@provider(
    input_types=[integer_value_sequence(10), integer_value(3)],
    should_shuffle=False)
def process_seq(settings, file_name):
    for d in data:
        seq = []
        for subseq in d[0]:
            seq += subseq
        yield seq, d[1]


# Used for sequence_nest_rnn_multi_input.conf
@provider(
    input_types=[integer_value_sub_sequence(10), integer_value(3)],
    should_shuffle=False)
def process_subseq2(settings, file_name):
    for d in data:
        yield d


# Used for sequence_rnn_multi_input.conf
@provider(
    input_types=[integer_value_sequence(10), integer_value(3)],
    should_shuffle=False)
def process_seq2(settings, file_name):
    for d in data:
        seq = []
        for subseq in d[0]:
            seq += subseq
        yield seq, d[1]


###########################################################
data2 = [
    [[[1, 2], [4, 5, 2]], [[5, 4, 1], [3, 1]], 0],
    [[[0, 2], [2, 5], [0, 1, 2]], [[1, 5], [4], [2, 3, 6, 1]], 1],
]


# Used for sequence_nest_rnn_multi_unequalength_inputs.conf
@provider(
    input_types=[
        integer_value_sub_sequence(10), integer_value_sub_sequence(10),
        integer_value(2)
    ],
    should_shuffle=False)
def process_unequalength_subseq(settings, file_name):
    for d in data2:
        yield d


# Used for sequence_rnn_multi_unequalength_inputs.conf
@provider(
    input_types=[
        integer_value_sequence(10), integer_value_sequence(10), integer_value(2)
    ],
    should_shuffle=False)
def process_unequalength_seq(settings, file_name):
    for d in data2:
        words1 = reduce(lambda x, y: x + y, d[0])
        words2 = reduce(lambda x, y: x + y, d[1])
        yield words1, words2, d[2]


###########################################################
data3 = [
    [[[1, 2], [4, 5, 2]], [1, 2], 0],
    [[[0, 2], [2, 5], [0, 1, 2]], [2, 3, 0], 1],
]


# Used for sequence_nest_mixed_inputs.conf
@provider(
    input_types=[
        integer_value_sub_sequence(10), integer_value_sequence(10),
        integer_value(2)
    ],
    should_shuffle=False)
def process_mixed(settings, file_name):
    for d in data3:
        yield d
