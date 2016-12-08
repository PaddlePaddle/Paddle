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

is_predict = get_config_arg("is_predict", bool, False)

####################Data Configuration ##################
if not is_predict:
    data_dir = 'data/cifar-out/batches/'
    meta_path = data_dir + 'batches.meta'

    args = {
        'meta': meta_path,
        'mean_img_size': 32,
        'img_size': 32,
        'num_classes': 10,
        'use_jpeg': 1,
        'color': "color"
    }

    define_py_data_sources2(
        train_list="train.list",
        test_list="train.list",
        module='image_provider',
        obj='processData',
        args=args)

######################Algorithm Configuration #############
settings(
    batch_size=128,
    learning_rate=0.1 / 128.0,
    learning_method=MomentumOptimizer(0.9),
    regularization=L2Regularization(0.0005 * 128))

#######################Network Configuration #############
data_size = 3 * 32 * 32
label_size = 10
img = data_layer(name='image', size=data_size)
# small_vgg is predefined in trainer_config_helpers.networks
predict = small_vgg(input_image=img, num_channels=3, num_classes=label_size)

if not is_predict:
    lbl = data_layer(name="label", size=label_size)
    outputs(classification_cost(input=predict, label=lbl))
else:
    outputs(predict)
