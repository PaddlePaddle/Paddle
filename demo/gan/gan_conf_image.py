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
from paddle.trainer_config_helpers.layers import img_convTrans_layer
from paddle.trainer_config_helpers.activations import LinearActivation
from numpy.distutils.system_info import tmp

mode = get_config_arg("mode", str, "generator")
assert mode in set(["generator",
                    "discriminator",
                    "generator_training",
                    "discriminator_training"])

is_generator_training = mode == "generator_training"
is_discriminator_training = mode == "discriminator_training"
is_generator = mode == "generator"
is_discriminator = mode == "discriminator"

print('mode=%s' % mode)
noise_dim = 100
gf_dim = 64
df_dim = 64
sample_dim = 28 # image dim
c_dim = 1 # image color
s2, s4 = int(sample_dim/2), int(sample_dim/4), 
s8, s16 = int(sample_dim/8), int(sample_dim/16)

settings(
    batch_size=100,
    learning_rate=1e-4,
    learning_method=AdamOptimizer()
)

def convTrans_bn(input, channels, output_x, num_filters, imgSize, stride, name, 
                 param_attr, bias_attr, param_attr_bn):
    tmp =  imgSize - (output_x - 1) * stride
    if tmp <= 1 or tmp > 5:
        raise ValueError("convTrans input-output dimension does not fit")
    elif tmp <= 3:
        filter_size = tmp + 2
        padding = 1
    else:
        filter_size = tmp
        padding = 0
        
        
    convTrans = img_convTrans_layer(input, filter_size=filter_size, 
                   num_filters=num_filters,
                   name=name + "_convt", num_channels=channels,
                   act=LinearActivation(), groups=1, stride=stride, 
                   padding=padding, bias_attr=bias_attr,
                   param_attr=param_attr, shared_biases=True, layer_attr=None,
                   filter_size_y=None, stride_y=None, padding_y=None)
    
    convTrans_bn = batch_norm_layer(convTrans, 
                     act=ReluActivation(), 
                     name=name + "_convt_bn", 
                     bias_attr=bias_attr, 
                     param_attr=param_attr_bn,
                     use_global_stats=False)
    
    return convTrans_bn

def conv_bn(input, channels, imgSize, num_filters, output_x, stride, name, 
                 param_attr, bias_attr, param_attr_bn, bn):
    tmp =  imgSize - (output_x - 1) * stride
    if tmp <= 1 or tmp > 5:
        raise ValueError("conv input-output dimension does not fit")
    elif tmp <= 3:
        filter_size = tmp + 2
        padding = 1
    else:
        filter_size = tmp
        padding = 0

    print (imgSize, output_x, stride, filter_size, padding)
        
    if bn:
        conv = img_conv_layer(input, filter_size=filter_size, 
                   num_filters=num_filters,
                   name=name + "_conv", num_channels=channels,
                   act=LinearActivation(), groups=1, stride=stride, 
                   padding=padding, bias_attr=bias_attr,
                   param_attr=param_attr, shared_biases=True, layer_attr=None,
                   filter_size_y=None, stride_y=None, padding_y=None)
        
        conv_bn = batch_norm_layer(conv, 
                         act=ReluActivation(), 
                         name=name + "_conv_bn", 
                         bias_attr=bias_attr, 
                         param_attr=param_attr_bn,
                         use_global_stats=False)
        
        return conv_bn
    else:
        conv = img_conv_layer(input, filter_size=filter_size, 
                   num_filters=num_filters,
                   name=name + "_conv", num_channels=channels,
                   act=ReluActivation(), groups=1, stride=stride, 
                   padding=padding, bias_attr=bias_attr,
                   param_attr=param_attr, shared_biases=True, layer_attr=None,
                   filter_size_y=None, stride_y=None, padding_y=None)
        return conv
    
def generator(noise):
    """
    generator generates a sample given noise
    """
    param_attr = ParamAttr(is_static=is_discriminator_training)
    bias_attr = ParamAttr(is_static=is_discriminator_training,
                           initial_mean=1.0,
                           initial_std=0)
    
    param_attr_bn=ParamAttr(is_static=is_discriminator_training,
                           initial_mean=1.0,
                           initial_std=0.02)
    
    h1 = fc_layer(input=noise,
                    name="gen_layer_h1",
                    size=s8 * s8 * gf_dim * 4,
                    bias_attr=bias_attr,
                    param_attr=param_attr,
                    #act=ReluActivation())
                    act=LinearActivation())
    
    h1_bn = batch_norm_layer(h1, 
                     act=ReluActivation(), 
                     name="gen_layer_h1_bn", 
                     bias_attr=bias_attr, 
                     param_attr=param_attr_bn,
                     use_global_stats=False)
    
    h2_bn = convTrans_bn(h1_bn, 
                        channels=gf_dim*4, 
                        output_x=s8,
                        num_filters=gf_dim*2, 
                        imgSize=s4,
                        stride=2,
                        name="gen_layer_h2", 
                        param_attr=param_attr, 
                        bias_attr=bias_attr, 
                        param_attr_bn=param_attr_bn)
    
    h3_bn = convTrans_bn(h2_bn, 
                        channels=gf_dim*2, 
                        output_x=s4,
                        num_filters=gf_dim, 
                        imgSize=s2,
                        stride=2,
                        name="gen_layer_h3", 
                        param_attr=param_attr, 
                        bias_attr=bias_attr, 
                        param_attr_bn=param_attr_bn)
     
    
    return convTrans_bn(h3_bn,
                        channels=gf_dim, 
                        output_x=s2,
                        num_filters=c_dim, 
                        imgSize=sample_dim,
                        stride=2,
                        name="gen_layer_h4", 
                        param_attr=param_attr, 
                        bias_attr=bias_attr, 
                        param_attr_bn=param_attr_bn)


def discriminator(sample):
    """
    discriminator ouputs the probablity of a sample is from generator
    or real data.
    The output has two dimenstional: dimension 0 is the probablity
    of the sample is from generator and dimension 1 is the probabblity
    of the sample is from real data.
    """
    param_attr = ParamAttr(is_static=is_generator_training)
    bias_attr = ParamAttr(is_static=is_generator_training,
                          initial_mean=1.0,
                          initial_std=0)
    
    param_attr_bn=ParamAttr(is_static=is_generator_training,
                           initial_mean=1.0,
                           initial_std=0.02)
    
    h0 = conv_bn(sample, 
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
    
    h1_bn = conv_bn(h0, 
                 channels=df_dim,
                 imgSize=s2,
                 num_filters=df_dim*2, 
                 output_x=s4, 
                 stride=2, 
                 name="dis_h1", 
                 param_attr=param_attr, 
                 bias_attr=bias_attr, 
                 param_attr_bn=param_attr_bn, 
                 bn=True)

    h2_bn = conv_bn(h1_bn, 
                 channels=df_dim*2,
                 imgSize=s4,
                 num_filters=df_dim*4, 
                 output_x=s8, 
                 stride=2, 
                 name="dis_h2", 
                 param_attr=param_attr, 
                 bias_attr=bias_attr, 
                 param_attr_bn=param_attr_bn, 
                 bn=True)
        
    return fc_layer(input=h2_bn, name="dis_prob", size=2,
                    bias_attr=bias_attr,
                    param_attr=param_attr,
                    act=SoftmaxActivation())



if is_generator_training:
    noise = data_layer(name="noise", size=noise_dim)
    sample = generator(noise)

if is_discriminator_training:
    sample = data_layer(name="sample", size=sample_dim * sample_dim*c_dim)

if is_generator_training or is_discriminator_training:
    label = data_layer(name="label", size=1)
    prob = discriminator(sample)
    cost = cross_entropy(input=prob, label=label)
    classification_error_evaluator(input=prob, label=label, name=mode+'_error')
    outputs(cost)

    
if is_generator:
    noise = data_layer(name="noise", size=noise_dim)
    outputs(generator(noise))
