# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import six


def fake_imdb_reader(word_dict_size,
                     sample_num,
                     lower_seq_len=100,
                     upper_seq_len=200,
                     class_dim=2):
    def __reader__():
        for _ in six.moves.range(sample_num):
            length = np.random.random_integers(
                low=lower_seq_len, high=upper_seq_len, size=[1])[0]
            ids = np.random.random_integers(
                low=0, high=word_dict_size - 1, size=[length]).astype('int64')
            label = np.random.random_integers(
                low=0, high=class_dim - 1, size=[1]).astype('int64')[0]
            yield ids, label

    return __reader__
