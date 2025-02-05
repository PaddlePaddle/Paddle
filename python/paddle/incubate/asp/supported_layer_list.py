# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation.  All rights reserved.
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

from __future__ import annotations

import copy
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

import paddle
from paddle.base.log_helper import get_logger
from paddle.incubate import asp

if TYPE_CHECKING:
    import numpy.typing as npt

    from paddle.nn import Layer

    from .utils import MaskAlgo

__all__ = []

_logger = get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s'
)


def _default_pruning(
    weight_nparray: npt.NDArray[Any],
    m: int,
    n: int,
    func_name: MaskAlgo,
    param_name: str,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    # if the to-be-pruned dimension's size is smaller than m, we don't prune it. This strong assertion is required by the inference from cuSparseLT.
    shape = weight_nparray.shape
    weight_pruned_nparray = copy.deepcopy(weight_nparray)
    weight_sparse_mask = np.ones_like(weight_pruned_nparray)
    exclude_cond_shape2 = len(shape) == 2 and shape[0] < m
    exclude_cond_shape4 = len(shape) == 4 and shape[1] < m
    if exclude_cond_shape2:
        _logger.warning(
            f'{param_name} is not pruned because the first dimension of {shape} is smaller than {m}'
        )
        return weight_pruned_nparray, weight_sparse_mask
    if exclude_cond_shape4:
        _logger.warning(
            f'{param_name} is not pruned because the second dimension of {shape} is smaller than {m}'
        )
        return weight_pruned_nparray, weight_sparse_mask

    checked_func_name = asp.CheckMethod.get_checking_method(func_name)

    # The double transpose ops here make sure pruning direction consistent with cuSparseLt.
    # SPMMA in cuSparseLt: D = (AxB) + C, where matrix A (mxk) is sparse matrix.
    # cuSparseLt would prune matrix A along k dimension.
    # In sparse training, layer weight matrices is viewed sparse matrix A, so
    # the math formula should be 'Act(WX + b)'. However, default formula in PaddlePaddle
    #  is 'Act(XW + b)'. For enabling SPMMA, weights and inputs should be transposed
    # for computing, Act( (W^T X^T)^T + b). Therefore, we have to prune along k dimension
    # of W^T, which is m dimension of W. Moreover, all mask generating functions in
    # asp/utils is row-major pruning. That is the reason we have to transpose weight
    # matrices before invoking create_mask. Then we transpose the result mask to make
    # sure its shape to be the same as the input weight.
    weight_sparse_mask = asp.create_mask(
        weight_nparray.T, func_name=func_name, n=n, m=m
    ).T
    weight_pruned_nparray = np.multiply(weight_nparray, weight_sparse_mask)
    assert asp.check_sparsity(
        weight_pruned_nparray.T, n=n, m=m, func_name=checked_func_name
    ), f'Pruning {param_name} weight matrix failure!!!'
    return weight_pruned_nparray, weight_sparse_mask


# When value of given key in this DICT is None,
# ASP will call default pruning function in pruning stage.
_supported_layers_and_prune_func_map_lock = threading.Lock()
supported_layers_and_prune_func_map = {}


def add_supported_layer(
    layer: Layer | type[Layer] | str,
    pruning_func: (
        Callable[
            [npt.NDArray[Any], int, int, MaskAlgo, str],
            tuple[npt.NDArray[Any], npt.NDArray[Any]],
        ]
        | None
    ) = None,
) -> None:
    r"""

    Add supported layers and its corresponding pruning function.

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
    elif isinstance(layer, paddle.nn.Layer):
        name = paddle.nn.layer.layers._convert_camel_to_snake(
            type(layer).__name__
        )
    elif issubclass(layer, paddle.nn.Layer):
        name = paddle.nn.layer.layers._convert_camel_to_snake(layer.__name__)
    else:
        assert f"The type of layer should be string of Layer, but got {type(layer)}!"
    if pruning_func is None:
        pruning_func = _default_pruning
    _supported_layers_and_prune_func_map_lock.acquire()
    supported_layers_and_prune_func_map.update({name: pruning_func})
    _supported_layers_and_prune_func_map_lock.release()


add_supported_layer('fc')
add_supported_layer('linear')
add_supported_layer('conv2d')
