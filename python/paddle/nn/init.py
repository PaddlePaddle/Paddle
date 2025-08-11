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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import paddle

from .initializer.initializer import calculate_gain as calculate_gain_
from .initializer.kaiming import KaimingNormal, KaimingUniform
from .initializer.uniform import Uniform
from .initializer.xavier import XavierNormal, XavierUniform

calculate_gain = calculate_gain_


def kaiming_uniform_(
    tensor: paddle.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    block: paddle.pir.Block | None = None,
) -> paddle.Tensor | None:
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
        block (Block|None, optional): The block in which initialization ops
                should be added. Used in static graph only, default None.

    Returns:
        Tensor: Initialized tensor.
    """
    init = KaimingUniform(
        negative_slope=a, nonlinearity=nonlinearity, mode=mode
    )

    return init(tensor, block=block)


def kaiming_normal_(
    tensor: paddle.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    block: paddle.pir.Block | None = None,
) -> paddle.Tensor | None:
    """Modify tensor inplace using Kaiming normal method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        a (float, optional): The negative slope of the rectifier used after this layer.
            Defaults to 0.
        mode (str, optional): Mode to compute the fan. Choose from ["fan_in", "fan_out"].
            When set to 'fan_in', the fan_in parameter is used for initialization.
            When set to 'fan_out', the out_features of trainable Tensor will be used.
            Default is 'fan_in'.
        nonlinearity (str, optional): Nonlinearity method name. Defaults to "leaky_relu".
        block (Block|None, optional): The block in which initialization ops
                should be added. Used in static graph only, default None.

    Returns:
        Tensor: Initialized tensor.
    """
    init = KaimingNormal(negative_slope=a, nonlinearity=nonlinearity, mode=mode)

    return init(tensor, block=block)


def xavier_uniform_(
    tensor: paddle.Tensor,
    gain: float = 1.0,
    fan_in: float | None = None,
    fan_out: float | None = None,
    block: paddle.pir.Block | None = None,
) -> paddle.Tensor | None:
    """Modify tensor inplace using Xavier uniform method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        gain (float, optional): Scaling Tensor. Default is 1.0.
        fan_in (float|None, optional): fan_in for Xavier initialization, which is
                inferred from the Tensor. Default is None.
        fan_out (float|None, optional): fan_out for Xavier initialization, which is
                 inferred from the Tensor. Default is None.
        block (Block|None, optional): The block in which initialization ops
                should be added. Used in static graph only, default None.

    Returns:
        Tensor: Initialized tensor.
    """
    init = XavierUniform(
        gain=gain,
        fan_in=fan_in,
        fan_out=fan_out,
    )

    return init(tensor, block=block)


def xavier_normal_(
    tensor: paddle.Tensor,
    gain: float = 1.0,
    fan_in: float | None = None,
    fan_out: float | None = None,
    block: paddle.pir.Block | None = None,
) -> paddle.Tensor | None:
    """Modify tensor inplace using Xavier normal method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        gain (float, optional): Scaling Tensor. Default is 1.0.
        fan_in (float|None, optional): fan_in for Xavier initialization, which is
                inferred from the Tensor. Default is None.
        fan_out (float|None, optional): fan_out for Xavier initialization, which is
                 inferred from the Tensor. Default is None.
        block (Block|None, optional): The block in which initialization ops
                should be added. Used in static graph only, default None.

    Returns:
        Tensor: Initialized tensor.
    """
    init = XavierNormal(
        gain=gain,
        fan_in=fan_in,
        fan_out=fan_out,
    )

    return init(tensor, block=block)


def uniform_(
    tensor: paddle.Tensor,
    a: float = 0.0,
    b: float = 1.0,
    block: paddle.pir.Block | None = None,
) -> paddle.Tensor | None:
    """Modify tensor inplace using uniform method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        low (float, optional): Lower boundary of the uniform distribution. Default is :math:`-1.0`.
        high (float, optional): Upper boundary of the uniform distribution. Default is :math:`1.0`.
        block (Block|None, optional): The block in which initialization ops
                should be added. Used in static graph only, default None.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Uniform(low=a, high=b)

    return init(tensor, block=block)
