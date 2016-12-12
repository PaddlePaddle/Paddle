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
assert mode in set([
    "generator", "discriminator", "generator_training", "discriminator_training"
])

is_generator_training = mode == "generator_training"
is_discriminator_training = mode == "discriminator_training"
is_generator = mode == "generator"
is_discriminator = mode == "discriminator"

# The network structure below follows the ref https://arxiv.org/abs/1406.2661
# Here we used two hidden layers and batch_norm

print('mode=%s' % mode)
# the dim of the noise (z) as the input of the generator network
noise_dim = 10
# the dim of the hidden layer
hidden_dim = 10
# the dim of the generated sample
sample_dim = 2

settings(
    batch_size=128,
    learning_rate=1e-4,
    learning_method=AdamOptimizer(beta1=0.5))


def discriminator(sample):
    """
    discriminator ouputs the probablity of a sample is from generator
    or real data.
    The output has two dimenstional: dimension 0 is the probablity
    of the sample is from generator and dimension 1 is the probabblity
    of the sample is from real data.
    """
    param_attr = ParamAttr(is_static=is_generator_training)
    bias_attr = ParamAttr(
        is_static=is_generator_training, initial_mean=1.0, initial_std=0)

    hidden = fc_layer(
        input=sample,
        name="dis_hidden",
        size=hidden_dim,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    hidden2 = fc_layer(
        input=hidden,
        name="dis_hidden2",
        size=hidden_dim,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=LinearActivation())

    hidden_bn = batch_norm_layer(
        hidden2,
        act=ReluActivation(),
        name="dis_hidden_bn",
        bias_attr=bias_attr,
        param_attr=ParamAttr(
            is_static=is_generator_training, initial_mean=1.0,
            initial_std=0.02),
        use_global_stats=False)

    return fc_layer(
        input=hidden_bn,
        name="dis_prob",
        size=2,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=SoftmaxActivation())


def generator(noise):
    """
    generator generates a sample given noise
    """
    param_attr = ParamAttr(is_static=is_discriminator_training)
    bias_attr = ParamAttr(
        is_static=is_discriminator_training, initial_mean=1.0, initial_std=0)

    hidden = fc_layer(
        input=noise,
        name="gen_layer_hidden",
        size=hidden_dim,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=ReluActivation())

    hidden2 = fc_layer(
        input=hidden,
        name="gen_hidden2",
        size=hidden_dim,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=LinearActivation())

    hidden_bn = batch_norm_layer(
        hidden2,
        act=ReluActivation(),
        name="gen_layer_hidden_bn",
        bias_attr=bias_attr,
        param_attr=ParamAttr(
            is_static=is_discriminator_training,
            initial_mean=1.0,
            initial_std=0.02),
        use_global_stats=False)

    return fc_layer(
        input=hidden_bn,
        name="gen_layer1",
        size=sample_dim,
        bias_attr=bias_attr,
        param_attr=param_attr,
        act=LinearActivation())


if is_generator_training:
    noise = data_layer(name="noise", size=noise_dim)
    sample = generator(noise)

if is_discriminator_training:
    sample = data_layer(name="sample", size=sample_dim)

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
