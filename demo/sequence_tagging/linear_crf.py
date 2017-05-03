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

from paddle.trainer_config_helpers import *

import math

define_py_data_sources2(
    train_list="data/train.list",
    test_list="data/test.list",
    module="dataprovider",
    obj="process")

batch_size = 1
settings(
    learning_method=MomentumOptimizer(),
    batch_size=batch_size,
    regularization=L2Regularization(batch_size * 1e-4),
    model_average=ModelAverage(0.5),
    learning_rate=1e-1,
    learning_rate_decay_a=1e-5,
    learning_rate_decay_b=0.25, )

num_label_types = 23


def get_simd_size(size):
    return int(math.ceil(float(size) / 8)) * 8


# Currently, in order to use sparse_update=True,
# the size has to be aligned.
num_label_types = get_simd_size(num_label_types)

features = data_layer(name="features", size=76328)
word = data_layer(name="word", size=6778)
pos = data_layer(name="pos", size=44)
chunk = data_layer(name="chunk", size=num_label_types)

crf_input = fc_layer(
    input=features,
    size=num_label_types,
    act=LinearActivation(),
    bias_attr=False,
    param_attr=ParamAttr(
        initial_std=0, sparse_update=True))

crf = crf_layer(
    input=crf_input,
    label=chunk,
    param_attr=ParamAttr(
        name="crfw", initial_std=0), )

crf_decoding = crf_decoding_layer(
    size=num_label_types,
    input=crf_input,
    label=chunk,
    param_attr=ParamAttr(name="crfw"), )

sum_evaluator(
    name="error",
    input=crf_decoding, )

chunk_evaluator(
    name="chunk_f1",
    input=crf_decoding,
    label=chunk,
    chunk_scheme="IOB",
    num_chunk_types=11, )

inputs(word, pos, chunk, features)
outputs(crf)
