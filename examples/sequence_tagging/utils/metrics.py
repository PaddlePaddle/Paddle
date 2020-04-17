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

import sys

import paddle.fluid as fluid

__all__ = ['chunk_count', "build_chunk"]


def build_chunk(data_list, id2label_dict): 
    """
    Assembly entity
    """
    tag_list = [id2label_dict.get(str(id)) for id in data_list]
    ner_dict = {}
    ner_str = ""
    ner_start = 0
    for i in range(len(tag_list)): 
        tag = tag_list[i]
        if tag == u"O": 
            if i != 0: 
                key = "%d_%d" % (ner_start, i - 1)
                ner_dict[key] = ner_str
            ner_start = i
            ner_str = tag 
        elif tag.endswith(u"B"): 
            if i != 0: 
                key = "%d_%d" % (ner_start, i - 1)
                ner_dict[key] = ner_str
            ner_start = i
            ner_str = tag.split('-')[0]
        elif tag.endswith(u"I"): 
            if tag.split('-')[0] != ner_str: 
                if i != 0: 
                    key = "%d_%d" % (ner_start, i - 1)
                    ner_dict[key] = ner_str
                ner_start = i
                ner_str = tag.split('-')[0]
    return ner_dict
                    

def chunk_count(infer_numpy, label_numpy, seq_len, id2label_dict):
    """
    calculate num_correct_chunks num_error_chunks total_num for metrics
    """
    num_infer_chunks, num_label_chunks, num_correct_chunks = 0, 0, 0
    assert infer_numpy.shape[0] == label_numpy.shape[0]

    for i in range(infer_numpy.shape[0]): 
        infer_list = infer_numpy[i][: seq_len[i]]
        label_list = label_numpy[i][: seq_len[i]]
        infer_dict = build_chunk(infer_list, id2label_dict)
        num_infer_chunks += len(infer_dict)
        label_dict = build_chunk(label_list, id2label_dict)
        num_label_chunks += len(label_dict)
        for key in infer_dict: 
            if key in label_dict and label_dict[key] == infer_dict[key]: 
                num_correct_chunks += 1
    return num_infer_chunks, num_label_chunks, num_correct_chunks

