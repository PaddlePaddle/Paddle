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
import os
import sys

from paddle.trainer.PyDataProvider2 import *


def hook(settings, dict_file, **kwargs):
    settings.word_dict = dict_file
    settings.input_types = [
        integer_value_sequence(len(settings.word_dict)), integer_value(3)
    ]
    settings.logger.info('dict len : %d' % (len(settings.word_dict)))


@provider(init_hook=hook, should_shuffle=False)
def process(settings, file_name):
    with open(file_name, 'r') as fdata:
        for line in fdata:
            label, comment = line.strip().split('\t')
            label = int(''.join(label.split()))
            words = comment.split()
            words = [
                settings.word_dict[w] for w in words if w in settings.word_dict
            ]
            yield words, label


## for hierarchical sequence network
def hook2(settings, dict_file, **kwargs):
    settings.word_dict = dict_file
    settings.input_types = [
        integer_value_sub_sequence(len(settings.word_dict)),
        integer_value_sequence(3)
    ]
    settings.logger.info('dict len : %d' % (len(settings.word_dict)))


@provider(init_hook=hook2, should_shuffle=False)
def process2(settings, file_name):
    with open(file_name) as fdata:
        labels = []
        sentences = []
        for line in fdata:
            if (len(line)) > 1:
                label, comment = line.strip().split('\t')
                label = int(''.join(label.split()))
                words = comment.split()
                words = [
                    settings.word_dict[w] for w in words
                    if w in settings.word_dict
                ]
                labels.append(label)
                sentences.append(words)
            else:
                yield sentences, labels
                labels = []
                sentences = []
