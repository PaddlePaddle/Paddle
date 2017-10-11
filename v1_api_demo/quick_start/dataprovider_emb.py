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


def initializer(settings, dictionary, **kwargs):
    settings.word_dict = dictionary
    settings.input_types = {
        # Define the type of the first input as sequence of integer.
        # The value of the integers range from 0 to len(dictrionary)-1
        'word': integer_value_sequence(len(dictionary)),
        # Define the second input for label id
        'label': integer_value(2)
    }


@provider(init_hook=initializer, cache=CacheType.CACHE_PASS_IN_MEM)
def process(settings, file_name):
    with open(file_name, 'r') as f:
        for line in f:
            label, comment = line.strip().split('\t')
            words = comment.split()
            word_slot = [settings.word_dict.get(w, UNK_IDX) for w in words]
            yield {'word': word_slot, 'label': int(label)}


def predict_initializer(settings, dictionary, **kwargs):
    settings.word_dict = dictionary
    settings.input_types = {'word': integer_value_sequence(len(dictionary))}


@provider(init_hook=predict_initializer, should_shuffle=False)
def process_predict(settings, file_name):
    with open(file_name, 'r') as f:
        for line in f:
            comment = line.strip().split()
            word_slot = [settings.word_dict.get(w, UNK_IDX) for w in comment]
            yield {'word': word_slot}
