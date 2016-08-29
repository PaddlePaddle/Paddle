#!/usr/bin/env python
#coding=utf-8

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

import os
import sys

from paddle.trainer.PyDataProviderWrapper import *

@init_hook_wrapper
def hook(obj, dict_file, **kwargs):
    obj.word_dict = dict_file
    obj.slots = [IndexSlot(len(obj.word_dict)), IndexSlot(3)]
    obj.logger.info('dict len : %d' % (len(obj.word_dict)))

@provider(use_seq=True, init_hook=hook)
def process(obj, file_name):
    with open(file_name, 'r') as fdata:
        for line in fdata:
            label, comment = line.strip().split('\t')
            label = int(''.join(label.split()))
            words = comment.split()
            word_slot = [obj.word_dict[w] for w in words if w in obj.word_dict]
            yield word_slot, [label]

## for hierarchical sequence network
@provider(use_seq=True, init_hook=hook)
def process2(obj, file_name):
    with open(file_name) as fdata:
        label_list = []
        word_slot_list = []
        for line in fdata:
            if (len(line)) > 1:
                label,comment = line.strip().split('\t')
                label = int(''.join(label.split()))
                words = comment.split()
                word_slot = [obj.word_dict[w] for w in words if w in obj.word_dict]
                label_list.append([label])
                word_slot_list.append(word_slot)
            else:
                yield word_slot_list, label_list
                label_list = []
                word_slot_list = []
