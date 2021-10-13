# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from paddle.fluid.contrib import sparsity

__all__ = ['add_supported_layer']


def _default_pruning(weight_nparray, m, n, func_name, param_name):

    checked_func_name = sparsity.CheckMethod.get_checking_method(func_name)

    # The double transpose ops here make sure pruning direction consistent with cuSparseLt.
    # SPMMA in cuSparseLt: D = (AxB) + C, where matrix A (mxk) is sparse matrix.
    # cuSparseLt would prune matrix A along k dimension.
    # In sparse training, layer weight matriices is viewed sparse matrix A, so
    # the math fomula should be 'Act(WX + b)'. However, default fomula in PaddlePaddle
    #  is 'Act(XW + b)'. For enabling SPMMA, weights and inputs should be transposed 
    # for computing, Act( (W^T X^T)^T + b). Therefore, we have to prune alog k dimension 
    # of W^T, which is m dimension of W. Moreove, all mask generating functions in 
    # sparsity/utils is row-major pruning. That is the reason we have to transpose weight 
    # matrices beforce invoking create_mask. Then we transpose the result maks to make 
    # sure its shape to be the same as the input weight.
    weight_sparse_mask = sparsity.create_mask(
        weight_nparray.T, func_name=func_name, n=n, m=m).T
    weight_pruned_nparray = np.multiply(weight_nparray, weight_sparse_mask)
    assert sparsity.check_sparsity(weight_pruned_nparray.T,  n=n, m=m, func_name=checked_func_name), \
                    'Pruning {} weight matrix failure!!!'.format(param_name)
    return weight_pruned_nparray, weight_sparse_mask


# When value of given key in this DICT is None, 
# ASP will call default pruning function in pruning stage.
# SUPPORTED_LAYERS = set(['fc', 'linear', 'conv'])
SUPPORTED_LAYERS_AND_PRUNE_FUNC_MAP = {
    'fc': _default_pruning,
    'linear': _default_pruning,
    'conv2d': _default_pruning
}


def add_supported_layer(layer, pruning_func=None):
    r"""
    Add supported layers and its corresponding pruning functino.

    Args:
        name (string|Layer): The name or type of layer, needed to support. If layer is `Layer` then 
        it would be turn to string internally. ASP would use this name to match parameter's name and call 
        its the corresponding pruning function.
        pruning_func (function, optional): a function type which receives five argument (weight_nparray,
        m, n, func_name, param_name), weight_nparray is a nparray of weight, param_name is the name of weight,
        m, n, and func_name, please see `prune_model` for details.
    """
    name = None
    if isinstance(layer, str):
        name = layer
    elif issubclass(layer, paddle.fluid.dygraph.layers.Layer):
        name = paddle.fluid.dygraph.layers._convert_camel_to_snake(
            layer.__name__)
    else:
        assert "The type of layer should be string of Layer, but got {}!".format(
            type(layer))
    if pruning_func is None:
        pruning_func = _default_pruning

    SUPPORTED_LAYERS_AND_PRUNE_FUNC_MAP.update({name: pruning_func})
