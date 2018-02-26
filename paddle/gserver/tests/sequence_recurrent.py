#  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.trainer_config_helpers import *

######################## data source ################################
dict_path = 'gserver/tests/Sequence/tour_dict_phrase.dict'
dict_file = dict()
for line_count, line in enumerate(open(dict_path, "r")):
    dict_file[line.strip()] = line_count

define_py_data_sources2(
    train_list='gserver/tests/Sequence/train.list',
    test_list=None,
    module='sequenceGen',
    obj='process',
    args={"dict_file": dict_file})

settings(batch_size=5)
######################## network configure ################################
dict_dim = len(open(dict_path, 'r').readlines())
word_dim = 128
hidden_dim = 128
label_dim = 3

# This config is designed to be equivalent with sequence_recurrent_group.py

data = data_layer(name="word", size=dict_dim)

emb = embedding_layer(
    input=data, size=word_dim, param_attr=ParamAttr(name="emb"))

recurrent = recurrent_layer(input=emb, bias_attr=False, act=SoftmaxActivation())

recurrent_last = last_seq(input=recurrent)

with mixed_layer(
        size=label_dim, act=SoftmaxActivation(), bias_attr=True) as output:
    output += full_matrix_projection(input=recurrent_last)

outputs(
    classification_cost(
        input=output, label=data_layer(
            name="label", size=1)))
