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

import numpy as np
import os
from paddle.trainer.config_parser import *
from paddle.utils.preprocess_img import \
    ImageClassificationDatasetCreater
from paddle.trainer_config_helpers import *


def image_data(data_dir,
               processed_image_size,
               overwrite=False,
               color=True,
               train_list="batches/train.list",
               test_list="batches/test.list",
               meta_file="batches/batches.meta",
               use_jpeg=1):
    """
    Predefined image data provider for image classification.
    train_list: a text file containing a list of training batches.
    test_list: a text file containing a list of test batches.
    processed_image_size: all the input images will be resized into this size.
       If the image is not square. Then the shorter edge will be resized into
       this size, and the aspect ratio is kept the same.
    color: whether the images are color or gray.
    meta_path: the path of the meta file that stores the mean image file and
               other dataset information, such as the size of images,
               the size of the mean image, the number of classes.
    async_load_data: whether to load image data asynchronuously.
    """
    data_creator = ImageClassificationDatasetCreater(
        data_dir, processed_image_size, color)
    batch_data_dir = data_dir
    train_list = os.path.join(batch_data_dir, train_list)
    test_list = os.path.join(batch_data_dir, test_list)
    meta_path = os.path.join(batch_data_dir, meta_file)
    image_size = processed_image_size
    conf = np.load(meta_path)
    mean_image_size = conf["mean_image_size"]
    is_color = conf["color"]
    num_classes = conf["num_classes"]
    color_string = "color" if is_color else "gray"

    args = {
        'meta': meta_path,
        'mean_img_size': mean_image_size,
        'img_size': image_size,
        'num_classes': num_classes,
        'use_jpeg': use_jpeg != 0,
        'color': color_string
    }

    define_py_data_sources2(
        train_list,
        test_list,
        module='image_provider',
        obj='processData',
        args=args)
    return {
        "image_size": image_size,
        "num_classes": num_classes,
        "is_color": is_color
    }


def get_extra_layer_attr(drop_rate):
    if drop_rate == 0:
        return None
    else:
        return ExtraLayerAttribute(drop_rate=drop_rate)


def image_data_layers(image_size, num_classes, is_color=False,
                      is_predict=False):
    """
    Data layers for image classification.
    image_size: image size.
    num_classes: num of classes.
    is_color: whether the input images are color.
    is_predict: whether the network is used for prediction.
    """
    num_image_channels = 3 if is_color else 1
    data_input = data_layer("input",
                            image_size * image_size * num_image_channels)
    if is_predict:
        return data_input, None, num_image_channels
    else:
        label_input = data_layer("label", 1)
        return data_input, label_input, num_image_channels


def simple_conv_net(data_conf, is_color=False):
    """
    A Wrapper for a simple network for MNIST digit recognition.
    It contains two convolutional layers, one fully conencted layer, and
    one softmax layer.
    data_conf is a dictionary with the following keys:
        image_size: image size.
        num_classes: num of classes.
        is_color: whether the input images are color.
    """
    for k, v in data_conf.iteritems():
        globals()[k] = v
    data_input, label_input, num_image_channels = \
        image_data_layers(image_size, num_classes, is_color, is_predict)
    filter_sizes = [5, 5]
    num_channels = [32, 64]
    strides = [1, 1]
    fc_dims = [500]
    conv_bn_pool1 = img_conv_bn_pool(
        name="g1",
        input=data_input,
        filter_size=filter_sizes[0],
        num_channel=num_image_channels,
        num_filters=num_channels[0],
        conv_stride=1,
        conv_padding=0,
        pool_size=3,
        pool_stride=2,
        act=ReluActivation())
    conv_bn_pool2 = img_conv_bn_pool(
        name="g2",
        input=conv_bn_pool1,
        filter_size=filter_sizes[1],
        num_channel=num_channels[0],
        num_filters=num_channels[1],
        conv_stride=1,
        conv_padding=0,
        pool_size=3,
        pool_stride=2,
        act=ReluActivation())
    fc3 = fc_layer(
        name="fc3", input=conv_bn_pool2, dim=fc_dims[0], act=ReluActivation())
    fc3_dropped = dropout_layer(name="fc3_dropped", input=fc3, dropout_rate=0.5)
    output = fc_layer(
        name="output",
        input=fc3_dropped,
        dim=fc_dims[0],
        act=SoftmaxActivation())
    if is_predict:
        end_of_network(output)
    else:
        cost = classify(name="cost", input=output, label=label_input)
        end_of_network(cost)


def conv_layer_group(prefix_num,
                     num_layers,
                     input,
                     input_channels,
                     output_channels,
                     drop_rates=[],
                     strides=[],
                     with_bn=[]):
    """
    A set of convolution layers, and batch normalization layers,
    followed by one pooling layer.
    It is utilized in VGG network for image classifcation.
    prefix_num: the prefix number of the layer names.
                For example, if prefix_num = 1, the first convolutioal layer's
                name will be conv_1_1.
    num_layers: number of the convolutional layers.
    input: the name of the input layer.
    input_channels: the number of channels of the input feature map.
    output_channels: the number of channels of the output feature map.
    drop_rates: the drop rates of the BN layers. It will be all zero by default.
    strides: the stride of the convolution for the layers.
             It will be all 1 by  default.
    with_bn: whether to use Batch Normalization for Conv layers.
             By default,  it is all false.
    """
    if len(drop_rates) == 0: drop_rates = [0] * num_layers
    if len(strides) == 0: strides = [1] * num_layers
    if len(with_bn) == 0: with_bn = [False] * num_layers
    assert (len(drop_rates) == num_layers)
    assert (len(strides) == num_layers)

    for i in range(1, num_layers + 1):
        if i == 1:
            i_conv_in = input
        else:
            i_conv_in = group_output
        i_channels_conv = input_channels if i == 1 else output_channels
        conv_act = LinearActivation() if with_bn[i - 1] else ReluActivation()
        conv_output = img_conv_layer(
            name="conv%d_%d" % (prefix_num, i),
            input=i_conv_in,
            filter_size=3,
            num_channels=i_channels_conv,
            num_filters=output_channels,
            stride=strides[i - 1],
            padding=1,
            act=conv_act)
        if with_bn[i - 1]:
            bn = batch_norm_layer(
                name="conv%d_%d_bn" % (prefix_num, i),
                input=conv_output,
                num_channels=output_channels,
                act=ReluActivation(),
                layer_attr=get_extra_layer_attr(drop_rate=drop_rates[i - 1]))
            group_output = bn
        else:
            group_output = conv_output
    pool = img_pool_layer(
        name="pool%d" % prefix_num,
        input=group_output,
        pool_size=2,
        num_channels=output_channels,
        stride=2)
    return pool


def vgg_conv_net(image_size,
                 num_classes,
                 num_layers,
                 channels,
                 strides,
                 with_bn,
                 fc_dims,
                 drop_rates,
                 drop_rates_fc=[],
                 is_color=True,
                 is_predict=False):
    """
    A Wrapper for a VGG network for image classification.
    It is a set of convolutional groups followed by several fully
    connected layers, and a cross-entropy classifiation loss.
    The detailed architecture of the paper can be found here:
      Very Deep Convolutional Networks for Large-Scale Visual Recognition
      http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    image_size: image size.
    num_classes: num of classes.
    num_layers: the number of layers for all the convolution groups.
    channels: the number of output filters for all the convolution groups.
    with_bn: whether each layer of a convolution group is followed by a
    batch normalization.
    drop_rates: the dropout rates for all the convolutional layers.
    fc_dims: the dimension for all the fully connected layers.
    is_color: whether the input images are color.
    """
    data_input, label_input, num_image_channels = \
        image_data_layers(image_size, num_classes, is_color, is_predict)
    assert (len(num_layers) == len(channels))
    assert (len(num_layers) == len(strides))
    assert (len(num_layers) == len(with_bn))
    num_fc_layers = len(fc_dims)
    assert (num_fc_layers + 1 == len(drop_rates_fc))

    for i in range(len(num_layers)):
        input_layer = data_input if i == 0 else group_output
        input_channels = 3 if i == 0 else channels[i - 1]
        group_output = conv_layer_group(
            prefix_num=i + 1,
            num_layers=num_layers[i],
            input=input_layer,
            input_channels=input_channels,
            output_channels=channels[i],
            drop_rates=drop_rates[i],
            strides=strides[i],
            with_bn=with_bn[i])
    conv_output_name = group_output
    if drop_rates_fc[0] != 0.0:
        dropped_pool_name = "pool_dropped"
        conv_output_name = dropout_layer(
            name=dropped_pool_name,
            input=conv_output_name,
            dropout_rate=drop_rates_fc[0])
    for i in range(len(fc_dims)):
        input_layer_name = conv_output_name if i == 0 else fc_output
        active_type = LinearActivation() if i == len(
            fc_dims) - 1 else ReluActivation()
        drop_rate = 0.0 if i == len(fc_dims) - 1 else drop_rates_fc[i + 1]
        fc_output = fc_layer(
            name="fc%d" % (i + 1),
            input=input_layer_name,
            size=fc_dims[i],
            act=active_type,
            layer_attr=get_extra_layer_attr(drop_rate))
    bn = batch_norm_layer(
        name="fc_bn",
        input=fc_output,
        num_channels=fc_dims[len(fc_dims) - 1],
        act=ReluActivation(),
        layer_attr=get_extra_layer_attr(drop_rate=drop_rates_fc[-1]))
    output = fc_layer(
        name="output", input=bn, size=num_classes, act=SoftmaxActivation())
    if is_predict:
        outputs(output)
    else:
        cost = classification_cost(name="cost", input=output, label=label_input)
        outputs(cost)


def vgg16_conv_net(image_size, num_classes, is_color=True, is_predict=False):
    """
    A Wrapper for a 16 layers VGG network for image classification.
    The detailed architecture of the paper can be found here:
      Very Deep Convolutional Networks for Large-Scale Visual Recognition
      http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    image_size: image size.
    num_classes: num of classes.
    is_color: whether the input images are color.
    """
    vgg_conv_net(image_size, num_classes,
                 num_layers=[2, 2, 3, 3, 3],
                 channels=[64, 128, 256, 512, 512],
                 strides=[[], [], [], [], []],
                 with_bn=[[False, True], [False, True], [False, False, True], \
                          [False, False, True], [False, False, True]],
                 drop_rates=[[]] * 5,
                 drop_rates_fc=[0.0, 0.5, 0.5],
                 fc_dims=[4096, 4096],
                 is_predict=is_predict)


def small_vgg(data_conf, is_predict=False):
    """
    A Wrapper for a small VGG network for CIFAR-10 image classification.
    The detailed architecture of the paper can be found here:
      92.45% on CIFAR-10 in Torch
      http://torch.ch/blog/2015/07/30/cifar.html
    Due to the constraints of CuDNN, it only has four convolutional groups
    rather than five.
    Thus, it only achieves 91.2% test accuracy and 98.1% training accuracy.
    data_conf is a dictionary with the following keys:
        image_size: image size.
        num_classes: num of classes.
        is_color: whether the input images are color.
    """
    for k, v in data_conf.iteritems():
        globals()[k] = v
    vgg_conv_net(image_size, num_classes,
                 num_layers=[2, 2, 3, 3],
                 channels=[64, 128, 256, 512],
                 strides=[[], [], [], []],
                 with_bn=[[True, True], [True, True], [True, True, True], \
                          [True, True, True]],
                 drop_rates=[[0.3, 0.0], [0.4, 0.0],
                             [0.4, 0.4, 0.0], [0.4, 0.4, 0.0]],
                 drop_rates_fc=[0.5, 0.5],
                 fc_dims=[512],
                 is_predict=is_predict)


def training_settings(learning_rate=0.1,
                      batch_size=128,
                      algorithm="sgd",
                      momentum=0.9,
                      decay_rate=0.001):
    """
    Training settings.
    learning_rate: learning rate of the training.
    batch_size: the size of each training batch.
    algorithm: training algorithm, can be
       - sgd
       - adagrad
       - adadelta
       - rmsprop
    momentum: momentum of the training algorithm.
    decay_rate: weight decay rate.
    """
    Settings(
        algorithm=algorithm,
        batch_size=batch_size,
        learning_rate=learning_rate / float(batch_size))
    default_momentum(momentum)
    default_decay_rate(decay_rate * batch_size)
