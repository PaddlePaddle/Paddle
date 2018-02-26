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

from paddle.trainer_config_helpers import *

settings(batch_size=1000, learning_rate=1e-5)

din = data_layer(name='data', size=30)

seq_op = [first_seq, last_seq]

agg_level = [AggregateLevel.TO_SEQUENCE, AggregateLevel.TO_NO_SEQUENCE]

opts = []

for op in seq_op:
    for al in agg_level:
        opts.append(op(input=din, agg_level=al))

for op in seq_op:
    opts.append(
        op(input=din, agg_level=AggregateLevel.TO_NO_SEQUENCE, stride=5))

outputs(opts)
