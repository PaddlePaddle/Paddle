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

...  # the settings and define data provider is omitted.
DICT_DIM = 3000  # dictionary dimension.
word_ids = data_layer('word_ids', size=DICT_DIM)

emb = embedding_layer(
    input=word_ids, size=256, param_attr=ParamAttr(sparse_update=True))
emb_sum = pooling_layer(input=emb, pooling_type=SumPooling())
predict = fc_layer(input=emb_sum, size=DICT_DIM, act=Softmax())
outputs(
    classification_cost(
        input=predict, label=data_layer(
            'label', size=DICT_DIM)))
