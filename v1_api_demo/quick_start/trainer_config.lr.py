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

from paddle.trainer_config_helpers import *

dict_file = get_config_arg('dict_file', str, "./data/dict.txt")
word_dict = dict()
with open(dict_file, 'r') as f:
    for i, line in enumerate(f):
        w = line.strip().split()[0]
        word_dict[w] = i

is_predict = get_config_arg('is_predict', bool, False)
trn = 'data/train.list' if not is_predict else None
tst = 'data/test.list' if not is_predict else 'data/pred.list'
process = 'process' if not is_predict else 'process_predict'

# define the data sources for the model.
# We need to use different process for training and prediction.
# For training, the input data includes both word IDs and labels.
# For prediction, the input data only includs word Ids.
define_py_data_sources2(
    train_list=trn,
    test_list=tst,
    module="dataprovider_bow",
    obj=process,
    args={"dictionary": word_dict})

batch_size = 128 if not is_predict else 1
settings(
    batch_size=batch_size,
    learning_rate=2e-3,
    learning_method=AdamOptimizer(),
    regularization=L2Regularization(8e-4),
    gradient_clipping_threshold=25)

# Define the data for text features. The size of the data layer is the number
# of words in the dictionary.
data = data_layer(name="word", size=len(word_dict))

# Define a fully connected layer with logistic activation.
# (also called softmax activation).
output = fc_layer(input=data, size=2, act=SoftmaxActivation())

if not is_predict:
    # For training, we need label and cost

    # define the category id for each example.
    # The size of the data layer is the number of labels.
    label = data_layer(name="label", size=2)

    # Define cross-entropy classification loss and error.
    cls = classification_cost(input=output, label=label)
    outputs(cls)
else:
    # For prediction, no label is needed. We need to output
    # We need to output classification result, and class probabilities.
    maxid = maxid_layer(output)
    outputs([maxid, output])
