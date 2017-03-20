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

# 1. read data. Suppose you saved above python code as dataprovider.py
define_py_data_sources2(
    train_list=['no_matter.txt'],
    test_list=None,
    module='dataprovider',
    obj='process',
    args={})

# 2. learning algorithm
settings(batch_size=12, learning_rate=1e-3, learning_method=MomentumOptimizer())

# 3. Network configuration
x = data_layer(name='x', size=1)
y = data_layer(name='y', size=1)
y_predict = fc_layer(
    input=x,
    param_attr=ParamAttr(name='w'),
    size=1,
    act=LinearActivation(),
    bias_attr=ParamAttr(name='b'))
cost = mse_cost(input=y_predict, label=y)
outputs(cost)
