#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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


def on_init(settings, dictionary, **kwargs):
    # on_init will invoke when data provider is initialized. The dictionary
    # is passed from trainer_config, and is a dict object with type
    # (word string => word id).

    # set input types in runtime. It will do the same thing as
    # @provider(input_types) will do, but it is set dynamically during runtime.
    settings.input_types = {
        # The text is a sequence of integer values, and each value is a word id.
        # The whole sequence is the sentences that we want to predict its
        # sentimental.
        'data': integer_value_sequence(len(dictionary)),  # text input
        'label': integer_value(2)  # label positive/negative
    }

    # save dictionary as settings.dictionary. 
    # It will be used in process method.
    settings.dictionary = dictionary


@provider(init_hook=on_init)
def process(settings, filename):
    f = open(filename, 'r')

    for line in f:  # read each line of file
        label, sentence = line.split('\t')  # get label and sentence
        words = sentence.split(' ')  # get words

        # convert word string to word id
        # the word not in dictionary will be ignored.
        word_ids = []

        for each_word in words:
            if each_word in settings.dictionary:
                word_ids.append(settings.dictionary[each_word])

        # give data to paddle.
        yield word_ids, int(label)

    f.close()
