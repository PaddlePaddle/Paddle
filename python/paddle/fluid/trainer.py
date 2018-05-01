#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = [
    'Event',
    'Trainer',
]


class Event(object):
    BEGIN_EPOCH = 0
    END_EPOCH = 1
    BEGIN_STEP = 2
    END_STEP = 3

    def __init__(self):
        self.step = 0
        self.epoch = 0
        self.type = Event.BEGIN_EPOCH


class Trainer(object):
    def __init__(self, network_func, optimizer, params=None, place=None):
        # we need to generate a framework.Program by calling
        # network_func reference: fluid.program_guard in test_word2vec.py
        # move the default_main_program to self.program
        # and run the default_startup program on an empty
        self.network_func = network_func
        self.optimizer = optimizer
        self.params = params
        self.place = place
        # TODO(helin): support distributed training

    def train(self, reader, num_epochs, event_handler):
        pass

    def test(self, reader):
        pass
