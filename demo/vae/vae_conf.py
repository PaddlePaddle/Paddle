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
import numpy as np

is_generating = get_config_arg("is_generating", bool, False)

settings(batch_size=32, learning_rate=1e-3, learning_method=AdamOptimizer())

X_dim = 28 * 28
h_dim = 128
z_dim = 100


def reparameterization(mu, logvar):
    eps = ParamAttr(initial_mean=0., initial_std=1)
    with mixed_layer() as sigma:
        sigma += dotmul_projection(layer_math.exp(logvar) * 0.5, param_attr=eps)
    return mu + sigma


def q_func(X):
    """
    xavier initialization
    """
    param_attr = ParamAttr(
        name='share.w', initial_mean=0., initial_std=1. / np.sqrt(X_dim / 2.))
    mu_param = ParamAttr(
        name='mu.w', initial_mean=0., initial_std=1. / np.sqrt(h_dim / 2.))
    logvar_param = ParamAttr(
        name='logvar.w', initial_mean=0., initial_std=1. / np.sqrt(h_dim / 2.))

    bias_attr = ParamAttr(name='share.bias', initial_mean=0., initial_std=0.)
    mu_bias = ParamAttr(name='mu.bias', initial_mean=0., initial_std=0.)
    logvar_bias = ParamAttr(name='logvar.bias', initial_mean=0., initial_std=0.)

    share_layer = fc_layer(
        X,
        size=h_dim,
        param_attr=param_attr,
        bias_attr=bias_attr,
        act=ReluActivation())

    return (fc_layer(
        share_layer,
        size=z_dim,
        param_attr=mu_param,
        bias_attr=mu_bias,
        act=LinearActivation()), fc_layer(
            share_layer,
            size=z_dim,
            param_attr=logvar_param,
            bias_attr=logvar_bias,
            act=LinearActivation()))


def generator(z):

    hidden_param = ParamAttr(
        name='hidden.w', initial_mean=0., initial_std=1. / np.sqrt(z_dim / 2.))
    hidden_bias = ParamAttr(name='hidden.bias', initial_mean=0., initial_std=0.)
    prob_param = ParamAttr(
        name='prob.w', initial_mean=0., initial_std=1. / np.sqrt(h_dim / 2.))
    prob_bias = ParamAttr(name='prob.bias', initial_mean=0., initial_std=0.)

    hidden_layer = fc_layer(
        z,
        size=h_dim,
        act=ReluActivation(),
        param_attr=hidden_param,
        bias_attr=hidden_bias)
    prob = fc_layer(
        hidden_layer,
        size=X_dim,
        act=SigmoidActivation(),
        param_attr=prob_param,
        bias_attr=prob_bias)

    return prob


def reconstruct_error(prob, X):
    cost = multi_binary_label_cross_entropy(input=prob, label=X)
    return cost


def KL_loss(mu, logvar):
    with mixed_layer() as mu_square:
        mu_square += dotmul_operator(mu, mu, scale=1.)

    cost = 0.5 * sum_cost(layer_math.exp(logvar) + mu_square - 1. - logvar)

    return cost


if not is_generating:
    x_batch = data_layer(name='x_batch', size=X_dim)
    mu, logvar = q_func(x_batch)
    z_samples = reparameterization(mu, logvar)
    prob = generator(z_samples)
    outputs(reconstruct_error(prob, x_batch) + KL_loss(mu, logvar))
else:
    z_samples = data_layer(name='noise', size=z_dim)
    outputs(generator(z_samples))
