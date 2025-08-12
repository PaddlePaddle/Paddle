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

import paddle

from ..base.framework import in_dygraph_mode, in_pir_mode
from .initializer.constant import Constant
from .initializer.dirac import Dirac
from .initializer.initializer import calculate_gain  # noqa: F401
from .initializer.kaiming import KaimingNormal, KaimingUniform
from .initializer.normal import Normal, TruncatedNormal
from .initializer.orthogonal import Orthogonal
from .initializer.uniform import Uniform
from .initializer.xavier import XavierNormal, XavierUniform


def kaiming_uniform_(
    tensor: paddle.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
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

    Returns:
        Tensor: Initialized tensor.
    """
    init = KaimingUniform(
        negative_slope=a, nonlinearity=nonlinearity, mode=mode
    )

    return init(tensor)


def kaiming_normal_(
    tensor: paddle.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
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

    Returns:
        Tensor: Initialized tensor.
    """
    init = KaimingNormal(negative_slope=a, nonlinearity=nonlinearity, mode=mode)

    return init(tensor)


def xavier_uniform_(
    tensor: paddle.Tensor,
    gain: float = 1.0,
    fan_in: float | None = None,
    fan_out: float | None = None,
) -> paddle.Tensor | None:
    """Modify tensor inplace using Xavier uniform method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        gain (float, optional): Scaling Tensor. Default is 1.0.
        fan_in (float|None, optional): fan_in for Xavier initialization, which is
                inferred from the Tensor. Default is None.
        fan_out (float|None, optional): fan_out for Xavier initialization, which is
                 inferred from the Tensor. Default is None.

    Returns:
        Tensor: Initialized tensor.
    """
    init = XavierUniform(
        gain=gain,
        fan_in=fan_in,
        fan_out=fan_out,
    )

    return init(tensor)


def xavier_normal_(
    tensor: paddle.Tensor,
    gain: float = 1.0,
    fan_in: float | None = None,
    fan_out: float | None = None,
) -> paddle.Tensor | None:
    """Modify tensor inplace using Xavier normal method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        gain (float, optional): Scaling Tensor. Default is 1.0.
        fan_in (float|None, optional): fan_in for Xavier initialization, which is
                inferred from the Tensor. Default is None.
        fan_out (float|None, optional): fan_out for Xavier initialization, which is
                 inferred from the Tensor. Default is None.

    Returns:
        Tensor: Initialized tensor.
    """
    init = XavierNormal(
        gain=gain,
        fan_in=fan_in,
        fan_out=fan_out,
    )

    return init(tensor)


def uniform_(
    tensor: paddle.Tensor,
    a: float = 0.0,
    b: float = 1.0,
) -> paddle.Tensor | None:
    """Modify tensor inplace using uniform method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        low (float, optional): Lower boundary of the uniform distribution. Default is :math:`-1.0`.
        high (float, optional): Upper boundary of the uniform distribution. Default is :math:`1.0`.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Uniform(low=a, high=b)

    return init(tensor)


def normal_(
    tensor: paddle.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
) -> paddle.Tensor | None:
    """Modify tensor inplace using normal method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        mean (float|complex, optional): mean of the normal distribution. Default is 0.0.
        std (float, optional): standard deviation of the normal distribution. Default is 1.0.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Normal(mean=mean, std=std)

    return init(tensor)


def trunc_normal_(
    tensor: paddle.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> paddle.Tensor | None:
    """Modify tensor inplace using truncated normal method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        mean (float|complex, optional): mean of the normal distribution. Default is 0.0.
        std (float, optional): standard deviation of the normal distribution. Default is 1.0.
        a (float, optional): The minimum cutoff value. Default is -2.0.
        b (float, optional): The maximum cutoff value. Default is 2.0.

    Returns:
        Tensor: Initialized tensor.
    """
    init = TruncatedNormal(mean=mean, std=std, a=a, b=b)

    return init(tensor)


def constant_(
    tensor: paddle.Tensor,
    val: float,
) -> paddle.Tensor | None:
    """Modify tensor inplace using constant method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        value (float32|float64, optional): constant value to initialize the parameter.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Constant(value=val)

    return init(tensor)


def ones_(
    tensor: paddle.Tensor,
) -> paddle.Tensor | None:
    """Fill the input Tensor with the scalar value 1.

    Args:
        tensor (Tensor):  Paddle Tensor.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Constant(value=1.0)

    return init(tensor)


def zeros_(
    tensor: paddle.Tensor,
) -> paddle.Tensor | None:
    """Fill the input Tensor with the scalar value 0.

    Args:
        tensor (Tensor):  Paddle Tensor.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Constant(value=0.0)

    return init(tensor)


def dirac_(
    tensor: paddle.Tensor,
    groups: int = 1,
) -> paddle.Tensor | None:
    """Initialize the 3D/4D/5D Tensor with Dirac delta function.

    Args:
        tensor (Tensor):  Paddle Tensor.
        groups (int|None, optional): 0-dimension of the Tensor will be divided by groups,
            each group has the same value. Default: 1.
    Returns:
        Tensor: Initialized tensor.
    """
    init = Dirac(groups=groups)

    return init(tensor)


def eye_(
    tensor: paddle.Tensor,
) -> paddle.Tensor | None:
    """Fill the 2-dimensional input Tensor with the identity matrix.

    Args:
        tensor (Tensor):  Paddle Tensor.
    Returns:
        Tensor: Initialized tensor.
    """

    if len(tensor.shape) != 2:
        raise AssertionError(
            f"Only support 2 dimensional tensor, but got {len(tensor.shape)}."
        )

    if in_dygraph_mode():
        new_tensor = paddle.eye(
            tensor.shape[0], tensor.shape[1], dtype=tensor.dtype
        )
        new_tensor._share_underline_tensor_to(tensor)
        return None
    elif in_pir_mode():
        new_tensor = paddle.eye(
            tensor.shape[0], tensor.shape[1], dtype=tensor.dtype
        )
        return new_tensor
    else:
        raise NotImplementedError(
            'Only support run in dygraph mode or PIR mode.'
        )


def orthogonal_(
    tensor: paddle.Tensor,
    gain: float = 1,
) -> paddle.Tensor | None:
    """Fill the input Tensor with a (semi) orthogonal matrix.

    Args:
        tensor (Tensor):  Paddle Tensor.
        gain(float, optional): The multiplication coefficient for initialized tensor. Default: 1.0.
    Returns:
        Tensor: Initialized tensor.
    """
    init = Orthogonal(gain=gain)
    return init(tensor)
