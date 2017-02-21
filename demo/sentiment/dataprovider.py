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


def hook(settings, dictionary, **kwargs):
    settings.word_dict = dictionary
    settings.input_types = [
        integer_value_sequence(len(settings.word_dict)), integer_value(2)
    ]
    settings.logger.info('dict len : %d' % (len(settings.word_dict)))


@provider(init_hook=hook)
def process(settings, file_name):
    with open(file_name, 'r') as fdata:
        for line_count, line in enumerate(fdata):
            label, comment = line.strip().split('\t\t')
            label = int(label)
            words = comment.split()
            word_slot = [
                settings.word_dict[w] for w in words if w in settings.word_dict
            ]
            if not word_slot:
                continue
            yield word_slot, label
