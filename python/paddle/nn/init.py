# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import paddle

from .initializer.initializer import calculate_gain, compute_fans


def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(
            paddle.uniform(shape=tensor.shape, dtype=tensor.dtype, min=a, max=b)
        )
        return tensor


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError(
            f"Mode {mode} not supported, please use one of {valid_modes}"
        )

    fan_in, fan_out = compute_fans(tensor)

    return fan_in if mode == "fan_in" else fan_out


def kaiming_uniform_(
    tensor: paddle.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> paddle.Tensor:
    """Modify tensor inplace using Kaiming uniform method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        a (float, optional): The negative slope of the rectifier used after this layer.
            Defaults to 0.
        mode (str, optional): Mode to compute the fan. Choose from ["fan_in", "fan_out"].
            When set to 'fan_in', the fan_in parameter is used for initialization.
            When set to 'fan_out', the out_features of trainable Tensor will be used.
            Default is 'fan_in'.
        nonlinearity (str, optional): Nonlinearity method name. Defaults to "leaky_relu".

    Returns:
        Tensor: Initialized tensor.
    """
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    k = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -k, k)
