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
    data_dir = './data/'
    define_py_data_sources2(
        train_list=data_dir + 'train.list',
        test_list=data_dir + 'test.list',
        module='mnist_provider',
        obj='process')

######################Algorithm Configuration #############
settings(batch_size=50, learning_rate=0.001, learning_method=AdamOptimizer())

#######################Network Configuration #############

data_size = 1 * 28 * 28
label_size = 10
img = data_layer(name='pixel', size=data_size)


# light cnn
# A shallower cnn model: [CNN, BN, ReLU, Max-Pooling] x4 + FC x1
# Easier to train for mnist dataset and quite efficient
# Final performance is close to deeper ones on tasks such as digital and character classification 
def light_cnn(input_image, num_channels, num_classes):
    def __light__(ipt,
                  num_filter=128,
                  times=1,
                  conv_filter_size=3,
                  dropouts=0,
                  num_channels_=None):
        return img_conv_group(
            input=ipt,
            num_channels=num_channels_,
            pool_size=2,
            pool_stride=2,
            conv_padding=0,
            conv_num_filter=[num_filter] * times,
            conv_filter_size=conv_filter_size,
            conv_act=ReluActivation(),
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type=MaxPooling())

    tmp = __light__(input_image, num_filter=128, num_channels_=num_channels)
    tmp = __light__(tmp, num_filter=128)
    tmp = __light__(tmp, num_filter=128)
    tmp = __light__(tmp, num_filter=128, conv_filter_size=1)

    tmp = fc_layer(input=tmp, size=num_classes, act=SoftmaxActivation())
    return tmp


predict = light_cnn(input_image=img, num_channels=1, num_classes=label_size)

if not is_predict:
    lbl = data_layer(name="label", size=label_size)
    inputs(img, lbl)
    outputs(classification_cost(input=predict, label=lbl))
else:
    outputs(predict)
