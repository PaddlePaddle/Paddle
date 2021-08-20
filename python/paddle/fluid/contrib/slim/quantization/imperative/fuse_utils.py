#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import paddle
import paddle.nn as nn
from . import utils


class Identity(nn.Layer):
    '''a layer to replace bn or relu layers'''

    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def fuse_layers(model, layers_to_fuse, inplace=False):
    '''fuse layers in layers_to_fuse'''
    if inplace == False:
        model = copy.deepcopy(model)
    for layers in layers_to_fuse:
        _fuse_layers(model, layers)
    return model


def _fuse_layers(model, layers_list):
    '''fuse all the layers in layers_list'''
    lay_list = []
    for layer_name in layers_list:
        parent_layer, sub_name = utils.find_parent_layer_and_sub_name(
            model, layer_name)
        lay_list.append(getattr(parent_layer, sub_name))
    new_layers = fuse_func(lay_list)
    for i, item in enumerate(layers_list):
        parent_layer, sub_name = utils.find_parent_layer_and_sub_name(model,
                                                                      item)
        setattr(parent_layer, sub_name, new_layers[i])


def fuse_func(lay_list):
    '''choose the fuser method and fuse layers'''
    types = tuple(type(m) for m in lay_list)
    fuser_method = layer_list_to_fuse_method.get(types, None)
    new_layers = [None] * len(lay_list)
    fused = fuser_method(*lay_list)
    for handle_id, pre_hook_fn in lay_list[0]._forward_pre_hooks.items():
        fused.register_forward_pre_hook(pre_hook_fn)
        del lay_list[0]._forward_pre_hooks[handle_id]
    for handle_id, hook_fn in lay_list[-1]._forward_post_hooks.items():
        fused.register_forward_post_hook(hook_fn)
        del lay_list[-1]._forward_post_hooks[handle_id]
    new_layers[0] = fused
    for i in range(1, len(lay_list)):
        identity = Identity()
        identity.training = lay_list[0].training
        new_layers[i] = identity
    return new_layers


def fuse_conv_bn(conv, bn):
    '''fuse conv and bn for train or eval'''
    assert(conv.training == bn.training),\
        "Conv and BN both must be in the same mode (train or eval)."
    if conv.training:
        assert bn._num_features == conv._out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        raise NotImplementedError
    else:
        return fuse_conv_bn_eval(conv, bn)


def fuse_conv_bn_eval(conv, bn):
    '''fuse conv and bn for eval'''
    assert (not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_weight, fused_bias = fuse_conv_bn_weights(
        fused_conv.weight, fused_conv.bias, bn._mean, bn._variance, bn._epsilon,
        bn.weight, bn.bias)
    fused_conv.weight.set_value(fused_weight)
    if fused_conv.bias is None:
        fused_conv.bias = paddle.create_parameter(
            shape=[fused_conv._out_channels], is_bias=True, dtype='float32')
    fused_conv.bias.set_value(fused_bias)
    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    '''fuse weights and bias of conv and bn'''
    if conv_b is None:
        conv_b = paddle.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = paddle.ones_like(bn_rm)
    if bn_b is None:
        bn_b = paddle.zeros_like(bn_rm)
    bn_var_rsqrt = paddle.rsqrt(bn_rv + bn_eps)
    conv_w = conv_w * \
        (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    return conv_w, conv_b


def fuse_linear_bn(linear, bn):
    '''fuse linear and bn'''
    assert (linear.training == bn.training),\
        "Linear and BN both must be in the same mode (train or eval)."
    if linear.training:
        assert bn._num_features == linear.weight.shape[
            1], 'Output channel of Linear must match num_features of BatchNorm'
        raise NotImplementedError
    else:
        return fuse_linear_bn_eval(linear, bn)


def fuse_linear_bn_eval(linear, bn):
    '''fuse linear and bn for eval'''
    assert (not (linear.training or bn.training)), "Fusion only for eval!"
    fused_linear = copy.deepcopy(linear)

    fused_weight, fused_bias = fuse_linear_bn_weights(
        fused_linear.weight, fused_linear.bias, bn._mean, bn._variance,
        bn._epsilon, bn.weight, bn.bias)
    fused_linear.weight.set_value(fused_weight)
    if fused_linear.bias is None:
        fused_linear.bias = paddle.create_parameter(
            shape=[fused_linear.weight.shape[1]], is_bias=True, dtype='float32')
    fused_linear.bias.set_value(fused_bias)
    return fused_linear


def fuse_linear_bn_weights(linear_w, linear_b, bn_rm, bn_rv, bn_eps, bn_w,
                           bn_b):
    '''fuse weights and bias of linear and bn'''
    if linear_b is None:
        linear_b = paddle.zeros_like(bn_rm)
    bn_scale = bn_w * paddle.rsqrt(bn_rv + bn_eps)
    fused_w = linear_w * bn_scale.unsqueeze(-1)
    fused_b = (linear_b - bn_rm) * bn_scale + bn_b
    return fused_w, fused_b


layer_list_to_fuse_method = {
    (nn.Conv2D, nn.BatchNorm2D): fuse_conv_bn,
    (nn.Linear, nn.BatchNorm1D): fuse_linear_bn,
}
