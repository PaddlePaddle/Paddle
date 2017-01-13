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

mode = get_config_arg("mode", str, "generator")
dataSource = get_config_arg("data", str, "mnist")
assert mode in set([
    "generator", "discriminator", "generator_training", "discriminator_training"
])

is_generator_training = mode == "generator_training"
is_discriminator_training = mode == "discriminator_training"
is_generator = mode == "generator"
is_discriminator = mode == "discriminator"

# The network structure below follows the dcgan paper 
# (https://arxiv.org/abs/1511.06434)

print('mode=%s' % mode)
# the dim of the noise (z) as the input of the generator network
noise_dim = 100
# the number of filters in the layer in generator/discriminator that is 
# closet to the image
gf_dim = 64
df_dim = 64
if dataSource == "mnist":
    sample_dim = 28  # image dim
    c_dim = 1  # image color
else:
    sample_dim = 32
    c_dim = 3
s2, s4 = int(sample_dim / 2), int(sample_dim / 4),
s8, s16 = int(sample_dim / 8), int(sample_dim / 16)

settings(
    batch_size=128,
    learning_rate=2e-4,
    learning_method=AdamOptimizer(beta1=0.5))


def conv_bn(input,
            channels,
            imgSize,
            num_filters,
            output_x,
            stride,
            name,
            param_attr,
            bias_attr,
            param_attr_bn,
            bn,
            trans=False,
            act=ReluActivation()):
    """
    conv_bn is a utility function that constructs a convolution/deconv layer 
    with an optional batch_norm layer

    :param bn: whether to use batch_norm_layer
    :type bn: bool
    :param trans: whether to use conv (False) or deconv (True)
    :type trans: bool
    """

    # calculate the filter_size and padding size based on the given
    # imgSize and ouput size
    tmp = imgSize - (output_x - 1) * stride
    if tmp <= 1 or tmp > 5:
        raise ValueError("conv input-output dimension does not fit")
    elif tmp <= 3:
        filter_size = tmp + 2
        padding = 1
    else:
        filter_size = tmp
        padding = 0

    print(imgSize, output_x, stride, filter_size, padding)

    if trans:
        nameApx = "_convt"
    else:
        nameApx = "_conv"

    if bn:
        conv = img_conv_layer(
            input,
            filter_size=filter_size,
            num_filters=num_filters,
            name=name + nameApx,
            num_channels=channels,
            act=LinearActivation(),
            groups=1,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr,
            param_attr=param_attr,
            shared_biases=True,
            layer_attr=None,
            filter_size_y=None,
            stride_y=None,
            padding_y=None,
            trans=trans)

        conv_bn = batch_norm_layer(
            conv,
            act=act,
            name=name + nameApx + "_bn",
            bias_attr=bias_attr,
            param_attr=param_attr_bn,
            use_global_stats=False)

        return conv_bn
    else:
        conv = img_conv_layer(
            input,
            filter_size=filter_size,
            num_filters=num_filters,
            name=name + nameApx,
            num_channels=channels,
            act=act,
            groups=1,
            stride=stride,
            padding=padding,
            bias_attr=bias_attr,
            param_attr=param_attr,
            shared_biases=True,
            layer_attr=None,
            filter_size_y=None,
            stride_y=None,
            padding_y=None,
            trans=trans)
        return conv


def generator(noise):
    """
    generator generates a sample given noise
    """
    param_attr = ParamAttr(
        is_static=is_discriminator_training, initial_mean=0.0, initial_std=0.02)
    bias_attr = ParamAttr(
        is_static=is_discriminator_training, initial_mean=0.0, initial_std=0.0)

    param_attr_bn = ParamAttr(
        is_static=is_discriminator_training, initial_mean=1.0, initial_std=0.02)

    h1 = fc_layer(
        input=noise,
        name="gen_layer_h1",
        size=s8 * s8 * gf_dim * 4,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=LinearActivation())

    h1_bn = batch_norm_layer(
        h1,
        act=ReluActivation(),
        name="gen_layer_h1_bn",
        bias_attr=bias_attr,
        param_attr=param_attr_bn,
        use_global_stats=False)

    h2_bn = conv_bn(
        h1_bn,
        channels=gf_dim * 4,
        output_x=s8,
        num_filters=gf_dim * 2,
        imgSize=s4,
        stride=2,
        name="gen_layer_h2",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=True,
        trans=True)

    h3_bn = conv_bn(
        h2_bn,
        channels=gf_dim * 2,
        output_x=s4,
        num_filters=gf_dim,
        imgSize=s2,
        stride=2,
        name="gen_layer_h3",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=True,
        trans=True)

    return conv_bn(
        h3_bn,
        channels=gf_dim,
        output_x=s2,
        num_filters=c_dim,
        imgSize=sample_dim,
        stride=2,
        name="gen_layer_h4",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False,
        trans=True,
        act=TanhActivation())


def discriminator(sample):
    """
    discriminator ouputs the probablity of a sample is from generator
    or real data.
    The output has two dimenstional: dimension 0 is the probablity
    of the sample is from generator and dimension 1 is the probabblity
    of the sample is from real data.
    """
    param_attr = ParamAttr(
        is_static=is_generator_training, initial_mean=0.0, initial_std=0.02)
    bias_attr = ParamAttr(
        is_static=is_generator_training, initial_mean=0.0, initial_std=0.0)

    param_attr_bn = ParamAttr(
        is_static=is_generator_training, initial_mean=1.0, initial_std=0.02)

    h0 = conv_bn(
        sample,
        channels=c_dim,
        imgSize=sample_dim,
        num_filters=df_dim,
        output_x=s2,
        stride=2,
        name="dis_h0",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=False)

    h1_bn = conv_bn(
        h0,
        channels=df_dim,
        imgSize=s2,
        num_filters=df_dim * 2,
        output_x=s4,
        stride=2,
        name="dis_h1",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=True)

    h2_bn = conv_bn(
        h1_bn,
        channels=df_dim * 2,
        imgSize=s4,
        num_filters=df_dim * 4,
        output_x=s8,
        stride=2,
        name="dis_h2",
        param_attr=param_attr,
        bias_attr=bias_attr,
        param_attr_bn=param_attr_bn,
        bn=True)

    return fc_layer(
        input=h2_bn,
        name="dis_prob",
        size=2,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=SoftmaxActivation())


if is_generator_training:
    noise = data_layer(name="noise", size=noise_dim)
    sample = generator(noise)

if is_discriminator_training:
    sample = data_layer(name="sample", size=sample_dim * sample_dim * c_dim)

if is_generator_training or is_discriminator_training:
    label = data_layer(name="label", size=1)
    prob = discriminator(sample)
    cost = cross_entropy(input=prob, label=label)
    classification_error_evaluator(
        input=prob, label=label, name=mode + '_error')
    outputs(cost)

if is_generator:
    noise = data_layer(name="noise", size=noise_dim)
    outputs(generator(noise))
