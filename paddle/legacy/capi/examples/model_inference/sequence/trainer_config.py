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

WORD_DIM = 3000

sentence = data_layer(name='sentence', size=WORD_DIM)
sentence_embedding = embedding_layer(
    input=sentence,
    size=64,
    param_attr=ParameterAttribute(
        initial_max=1.0, initial_min=0.5))
lstm = simple_lstm(input=sentence_embedding, size=64)
lstm_last = last_seq(input=lstm)
outputs(fc_layer(input=lstm_last, size=2, act=SoftmaxActivation()))
