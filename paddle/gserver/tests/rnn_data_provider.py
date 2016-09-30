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

from paddle.trainer.PyDataProvider2 import *

data = [
    [[[1, 3, 2], [4, 5, 2]], 0],
    [[[0, 2], [2, 5], [0, 1, 2]], 1],
]


@provider(input_types=[integer_value_sub_sequence(10),
                       integer_value(2)],
          should_shuffle=False)
def process_subseq(settings, file_name):
    for d in data:
        yield d


@provider(input_types=[integer_value_sequence(10),
                       integer_value(2)],
          should_shuffle=False)
def process_seq(settings, file_name):
    for d in data:
        seq = []
        for subseq in d[0]:
            seq += subseq
        yield seq, d[1]
