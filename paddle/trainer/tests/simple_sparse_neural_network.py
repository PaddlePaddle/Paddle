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

from paddle.trainer_config_helpers import *

settings(batch_size=17, learning_method=AdaGradOptimizer(), learning_rate=1e-4)

file_list = 'trainer/tests/fake_file_list.list'

define_py_data_sources2(
    train_list=file_list,
    test_list=file_list,
    module="simple_sparse_neural_network_dp",
    obj="process")

embedding = embedding_layer(
    input=data_layer(
        name="word_ids", size=8191),
    size=128,
    param_attr=ParamAttr(sparse_update=True))
prediction = fc_layer(input=embedding, size=10, act=SoftmaxActivation())

outputs(
    classification_cost(
        input=prediction, label=data_layer(
            name='label', size=10)))
