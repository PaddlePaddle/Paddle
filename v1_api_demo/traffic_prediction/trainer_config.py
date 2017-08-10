# Copyright (c) 2016 PaddlePaddle Authors, Inc. All Rights Reserved
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

################################### DATA Configuration #############################################
is_predict = get_config_arg('is_predict', bool, False)
trn = './data/train.list' if not is_predict else None
tst = './data/test.list' if not is_predict else './data/pred.list'
process = 'process' if not is_predict else 'process_predict'
define_py_data_sources2(
    train_list=trn, test_list=tst, module="dataprovider", obj=process)
################################### Parameter Configuaration #######################################
TERM_NUM = 24
FORECASTING_NUM = 24
emb_size = 16
batch_size = 128 if not is_predict else 1
settings(
    batch_size=batch_size,
    learning_rate=1e-3,
    learning_method=RMSPropOptimizer())
################################### Algorithm Configuration ########################################

output_label = []

link_encode = data_layer(name='link_encode', size=TERM_NUM)
for i in xrange(FORECASTING_NUM):
    # Each task share same weight.
    link_param = ParamAttr(
        name='_link_vec.w', initial_max=1.0, initial_min=-1.0)
    link_vec = fc_layer(input=link_encode, size=emb_size, param_attr=link_param)
    score = fc_layer(input=link_vec, size=4, act=SoftmaxActivation())
    if is_predict:
        maxid = maxid_layer(score)
        output_label.append(maxid)
    else:
        # Multi-task training.
        label = data_layer(name='label_%dmin' % ((i + 1) * 5), size=4)
        cls = classification_cost(
            input=score, name="cost_%dmin" % ((i + 1) * 5), label=label)
        output_label.append(cls)
outputs(output_label)
