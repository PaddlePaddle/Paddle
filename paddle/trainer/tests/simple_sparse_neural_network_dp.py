#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from paddle.trainer.PyDataProvider2 import provider, integer_sequence, integer_value
import random


def init_hook(settings, is_train, **kwargs):
    settings.is_train = is_train


@provider(
    input_types={'word_ids': integer_value(8191),
                 'label': integer_value(10)},
    min_pool_size=0,
    init_hook=init_hook)
def process(settings, filename):
    if settings.is_train:
        data_size = 2**10
    else:
        data_size = 2**5

    for _ in xrange(data_size):
        yield random.randint(0, 8190), random.randint(0, 9)
