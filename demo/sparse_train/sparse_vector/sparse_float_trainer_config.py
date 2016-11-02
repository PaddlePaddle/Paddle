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

from paddle.trainer_config_helpers import *

label_size = 1
data_size = 10000000


""" Algorithm Configuration """
settings(learning_rate=1e-3,
         learning_method=MomentumOptimizer(momentum=0.9, sparse=True),
         batch_size=200)

""" Data Configuration """
define_py_data_sources2(train_list='train.list',
                        test_list=None,
                        module='sparse_float_data_provider',
                        obj='process')

""" Model Configuration """
data = data_layer(name='data',
                       size=data_size)
label = data_layer(name='label',
                   size=label_size)

hidden1 = fc_layer(input=data,
                   size=512,
                   param_attr=ParameterAttribute(sparse_update=True))
hidden2 = fc_layer(input=hidden1,
                   size=1)
outputs(regression_cost(input=hidden2, label=label))
