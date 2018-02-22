# edit-mode: -*- python -*-

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
"""
This configuration is a demonstration of how to implement the stacked LSTM
with residual connections, i.e. an LSTM layer takes the sum of the hidden states
and inputs of the previous LSTM layer instead of only the hidden states.
This architecture is from:
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi,
Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey,
Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Lukasz Kaiser,
Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens,
George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa,
Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean. 2016.
Google's Neural Machine Translation System: Bridging the Gap between Human and
Machine Translation. In arXiv https://arxiv.org/pdf/1609.08144v2.pdf
Different from the architecture described in the paper, we use a stack single
direction LSTM layers as the first layer instead of bi-directional LSTM. Also,
since this is a demo code, to reduce computation time, we stacked 4 layers
instead of 8 layers.
"""

from paddle.trainer_config_helpers import *

dict_file = "./data/dict.txt"
word_dict = dict()
with open(dict_file, 'r') as f:
    for i, line in enumerate(f):
        w = line.strip().split()[0]
        word_dict[w] = i

is_predict = get_config_arg('is_predict', bool, False)
trn = 'data/train.list' if not is_predict else None
tst = 'data/test.list' if not is_predict else 'data/pred.list'
process = 'process' if not is_predict else 'process_predict'
define_py_data_sources2(
    train_list=trn,
    test_list=tst,
    module="dataprovider_emb",
    obj=process,
    args={"dictionary": word_dict})

batch_size = 128 if not is_predict else 1
settings(
    batch_size=batch_size,
    learning_rate=2e-3,
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25)

bias_attr = ParamAttr(initial_std=0., l2_rate=0.)

data = data_layer(name="word", size=len(word_dict))
emb = embedding_layer(input=data, size=128)
lstm = simple_lstm(input=emb, size=128, lstm_cell_attr=ExtraAttr(drop_rate=0.1))

previous_input, previous_hidden_state = emb, lstm

for i in range(3):
    # The input to the current layer is the sum of the hidden state
    # and input of the previous layer.
    current_input = addto_layer(input=[previous_input, previous_hidden_state])
    hidden_state = simple_lstm(
        input=current_input, size=128, lstm_cell_attr=ExtraAttr(drop_rate=0.1))
    previous_input, previous_hidden_state = current_input, hidden_state

lstm = previous_hidden_state

lstm_last = pooling_layer(input=lstm, pooling_type=MaxPooling())
output = fc_layer(
    input=lstm_last, size=2, bias_attr=bias_attr, act=SoftmaxActivation())

if is_predict:
    maxid = maxid_layer(output)
    outputs([maxid, output])
else:
    label = data_layer(name="label", size=2)
    cls = classification_cost(input=output, label=label)
    outputs(cls)
