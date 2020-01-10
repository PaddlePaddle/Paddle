# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
from paddle import fluid
from .layers import Layer
from .. import layers as F
from ..framework import dygraph_only

__all__ = ['WeightNormWrapper']


def norm(param, dim, power=2):
    """Compute norm of a tensor for dim.
    
    Args:
        param (Variable): The input tensor.
        dim (int | NoneType): The dimension along which the norm is computed. If dim is an integer, it should be within [-R, R) where R in the rank of **param**.
        power (float): order of norm, default is 2.
    
    Returns:
        Variable: If dim is an interger, the norm along dim. If dim is None, the norm of the tensor.
    """

    powered = F.pow(param, power)
    powered_norm = F.reduce_sum(powered, dim=dim, keep_dim=False)
    norm_ = F.pow(powered_norm, 1. / power)
    return norm_


def norm_except(param, dim, power=2):
    """Compute norm of a tensor for all dimensions except **dim**.

    Args:
        param (Variable): The input tensor.
        dim (int): The dimension to exclude when computing norm. It should be within [-R, R) where R in the rank of **param**.
        power (float): order of norm, default is 2.
    
    Returns:
        Variable: shape(param.shape[dim], ), an 1D tensor, the computed norm.
    """

    shape = param.shape
    ndim = len(shape)

    if dim is None:
        return norm(param, dim, power)
    elif dim == 0:
        param_matrix = F.reshape(param, (shape[0], np.prod(shape[1:])))
        return norm(param_matrix, dim=1, power=power)
    elif dim == -1 or dim == ndim - 1:
        param_matrix = F.reshape(param, (np.prod(shape[:-1]), shape[-1]))
        return norm(param_matrix, dim=0, power=power)
    else:
        perm = list(range(ndim))
        perm[0] = dim
        perm[dim] = 0
        transposed_param = F.transpose(param, perm)
        return norm_except(transposed_param, dim=0, power=power)


def compute_weight(v, g, dim, power, epsilon=1e-12):
    assert len(g.shape) == 1, "magnitude should be a vector"
    v_normalized = F.elementwise_div(
        v, (norm_except(v, dim, power) + epsilon), axis=dim)
    weight = F.elementwise_mul(v_normalized, g, axis=dim)
    return weight


class WeightNormWrapper(Layer):
    """A wrapper to apply weight norm to a layer's **weight**.
    Weight Norm is a reparameterization of the weight vectors
    in a neural network that decouples the magnitude of those weight vectors from
    their direction. Weight Norm has been implemented as discussed in this
    paper: `Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks <https://arxiv.org/pdf/1602.07868.pdf>`_.
    
    Parameters:
        layer (Layer): A `dygraph.Layer` object. We assumes it is a simple layer 
        that does not contain other sublayers, and it has a weight parameter with 
        name `param_name`.
        param_name (str, optional): Defaults to "weight". The name of the parameter
        in *layer* to which weight normalization is applied.
        dim (int, optional): Defaults to -1. The dimension to exclude when computing
        norm. Typically, the dim to except is the dim representing the output features,
        for Linear, the output dim is 1, for Convolutions the output dim is 0, and 
        for ConvTransposes, the output dim is 1.
        power (int, optional): Defaults to 2. The order of the norm.

    Attributes:
        **layer** (Layer): The wrapped layer.
        **weight_v** (Parameter): The Parameter v, where `v / ||v||` represents the
        directions of the weight vectors.
        **weight_g** (Parameter): The Parameter g, which represents the magnitudes
        of the weight vectors.
        **weigth_norm_applied** (bool): A flag indicating whether to compute parameter
        from **weight_v** and **weight_g**. If True, parameter is computed from 
        **weight_v** and **weight_g** before layer's forward. If False, it indicates 
        that weight normalization is removed, the parameter is computed and set back 
        to layer, the forward of this Wrapper is just calling the layer's forward method.
    
    Returns:
        None
    
    Raises:
        ValueError: if dim is out of range [-R, R), where R is the rank of the 
        parameter to which weight normalization is applied.
    """

    @dygraph_only
    def __init__(self, layer, param_name="weight", dim=-1, power=2):
        super(WeightNormWrapper, self).__init__()

        self.param_name = param_name
        rank = len(getattr(layer, param_name).shape)
        if dim < -rank or dim >= rank:
            raise ValueError("The dim to expect should be in "
                             "range [-{}, {})".format(rank, rank))
        self.dim = dim
        self.power = power
        self.layer = layer

        w_v = param_name + "_v"
        w_g = param_name + "_g"

        original_weight = getattr(layer, param_name)
        self.add_parameter(
            w_v,
            self.create_parameter(
                original_weight.shape, dtype=original_weight.dtype))
        F.assign(original_weight, getattr(self, w_v))
        delattr(layer, param_name)
        magnitude = norm_except(getattr(self, w_v), self.dim, self.power)
        self.add_parameter(
            w_g, self.create_parameter(
                magnitude.shape, dtype=magnitude.dtype))
        F.assign(magnitude, getattr(self, w_g))

        self.weigth_norm_applied = True

    def hook(self):
        # hook to compute weight with weight_v & weight_g
        w_v = self.param_name + "_v"
        w_g = self.param_name + "_g"
        weight = compute_weight(
            getattr(self, w_v), getattr(self, w_g), self.dim, self.power)
        setattr(self.layer, self.param_name, weight)

    def remove_weight_norm(self):
        self.hook()
        self.weigth_norm_applied = False

    def forward(self, *args, **kwargs):
        if self.weigth_norm_applied:
            self.hook()
        return self.layer(*args, **kwargs)

    def __getattr__(self, key):
        """
        This is used for attr forwarding. A WeightNormWrapper object wraps a layer.
        With this, the wrapper can access the original layer's attributes through 
        a proxy.
        """
        if key in self._parameters:
            return self._parameters[key]
        elif key in self._sub_layers:
            return self._sub_layers[key]
        elif key is "layer":
            return self._sub_layers["layer"]
        else:
            return getattr(
                object.__getattribute__(self, "_sub_layers")["layer"], key)
