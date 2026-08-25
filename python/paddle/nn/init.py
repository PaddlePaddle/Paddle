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
    from collections.abc import Callable
    from typing import TypeVar

    from typing_extensions import ParamSpec

    _InputT = ParamSpec("_InputT")
    _RetT = TypeVar("_RetT")

import math
import warnings

import numpy as np

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


def _calculate_fan_in_and_fan_out(var: paddle.Tensor) -> tuple[int, int]:
    """Compute the fan_in and the fan_out for layers

    This method computes the fan_in and the fan_out
    for neural network layers, if not specified. It is
    not possible to perfectly estimate fan_in and fan_out.
    This method will estimate it correctly for matrix multiply and
    convolutions.

    Args:
        var: variable for which fan_in and fan_out have to be computed.

    Returns:
        tuple of two integers (fan_in, fan_out).
    """
    shape = var.shape
    if not shape or len(shape) == 0:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        # This is the case for simple matrix multiply
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assume this to be a convolutional kernel
        # In PaddlePaddle, the shape of the kernel is like:
        # [num_filters, num_filter_channels, ...] where the remaining
        # dimensions are the filter_size
        receptive_field_size = np.prod(shape[2:])
        fan_in = int(shape[1] * receptive_field_size)
        fan_out = int(shape[0] * receptive_field_size)
    return (fan_in, fan_out)


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
    init = KaimingUniform(
        negative_slope=a, nonlinearity=nonlinearity, mode=mode
    )

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def kaiming_normal_(
    tensor: paddle.Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
) -> paddle.Tensor:
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

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def xavier_uniform_(
    tensor: paddle.Tensor,
    gain: float = 1.0,
    fan_in: float | None = None,
    fan_out: float | None = None,
) -> paddle.Tensor:
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

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def xavier_normal_(
    tensor: paddle.Tensor,
    gain: float = 1.0,
    fan_in: float | None = None,
    fan_out: float | None = None,
) -> paddle.Tensor:
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

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def uniform_(
    tensor: paddle.Tensor,
    a: float = 0.0,
    b: float = 1.0,
) -> paddle.Tensor:
    """Modify tensor inplace using uniform method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        low (float, optional): Lower boundary of the uniform distribution. Default is :math:`-1.0`.
        high (float, optional): Upper boundary of the uniform distribution. Default is :math:`1.0`.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Uniform(low=a, high=b)

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def normal_(
    tensor: paddle.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
) -> paddle.Tensor:
    """Modify tensor inplace using normal method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        mean (float|complex, optional): mean of the normal distribution. Default is 0.0.
        std (float, optional): standard deviation of the normal distribution. Default is 1.0.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Normal(mean=mean, std=std)

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def _no_grad_trunc_normal_(
    tensor: paddle.Tensor,
    mean: float,
    std: float,
    a: float,
    b: float,
) -> paddle.Tensor:
    """Fill ``tensor`` in place with samples from a truncated normal distribution.

    The truncation is done by rejection sampling, so the returned values follow
    the exact truncated normal distribution instead of being clipped to
    ``[a, b]``. Two sampling schemes are used depending on how much probability
    mass the normal distribution puts inside ``[a, b]``:

    - the interval covers enough mass (``p > 0.3``): draw from the normal
      distribution and resample the out-of-range elements until none is left;
    - otherwise: draw uniformly from ``[a, b]`` and accept each sample with
      probability proportional to the normal density, which stays efficient for
      narrow or far-out intervals.

    Args:
        tensor (Tensor): Paddle Tensor to be filled in place.
        mean (float): mean of the normal distribution.
        std (float): standard deviation of the normal distribution.
        a (float): The minimum cutoff value.
        b (float): The maximum cutoff value.

    Returns:
        Tensor: ``tensor`` itself.
    """

    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with paddle.no_grad():
        p = norm_cdf((b - mean) / std) - norm_cdf((a - mean) / std)

        if p > 0.3:
            # Cast bounds to tensor dtype so the rejection mask is consistent
            # with the tensor's representable values.
            lo = paddle.to_tensor(
                a, dtype=tensor.dtype, place=paddle.CPUPlace()
            ).item()
            hi = paddle.to_tensor(
                b, dtype=tensor.dtype, place=paddle.CPUPlace()
            ).item()
            tensor.normal_(mean, std)
            result = tensor
            while True:
                mask = (result < lo) | (result > hi)
                if not bool(mask.any()):
                    break
                result = paddle.where(
                    mask,
                    paddle.empty_like(result).normal_(mean, std),
                    result,
                )
            if tensor is not result:
                paddle.assign(result, tensor)
        else:
            mode = max(a, min(mean, b))
            log_peak = -0.5 * ((mode - mean) / std) ** 2
            # Dividing by a scalar is a multiplication by its reciprocal, and
            # the reciprocal is taken before narrowing to the tensor dtype.
            inv_std = 1.0 / std

            def log_accept_ratio(x: paddle.Tensor) -> paddle.Tensor:
                # log_pdf - log_peak, with log_pdf = -0.5 * ((x - mean) / std) ** 2
                return ((x - mean) * inv_std) ** 2 * (-0.5) - log_peak

            def uniform_ab_(x: paddle.Tensor) -> paddle.Tensor:
                if a == b:
                    # `uniform_` rejects an empty range, but the degenerate
                    # interval still has to draw (and therefore consume) one
                    # sample per element.
                    x.uniform_(0.0, 1.0)
                    return x.fill_(a)
                return x.uniform_(a, b)

            accept_buf = paddle.empty_like(tensor)

            # First iteration: sample directly into tensor to avoid a where()
            # plus an assign() if all samples are accepted.
            uniform_ab_(tensor)
            threshold = log_accept_ratio(tensor)
            pending = accept_buf.uniform_(0.0, 1.0).log_() > threshold
            if bool(pending.any()):
                result = tensor
                candidates = paddle.empty_like(tensor)
                while True:
                    uniform_ab_(candidates)
                    result = paddle.where(pending, candidates, result)
                    threshold = log_accept_ratio(candidates)
                    pending = paddle.where(
                        pending,
                        accept_buf.uniform_(0.0, 1.0).log_() > threshold,
                        pending,
                    )
                    if not bool(pending.any()):
                        break
                paddle.assign(result, tensor)

        return tensor


def trunc_normal_(
    tensor: paddle.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> paddle.Tensor:
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
    if in_dygraph_mode():
        return _no_grad_trunc_normal_(tensor, mean, std, a, b)

    init = TruncatedNormal(mean=mean, std=std, a=a, b=b)
    return init(tensor)


def constant_(
    tensor: paddle.Tensor,
    val: float,
) -> paddle.Tensor:
    """Modify tensor inplace using constant method.

    Args:
        tensor (Tensor):  Paddle Tensor.
        value (float32|float64, optional): constant value to initialize the parameter.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Constant(value=val)

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def ones_(
    tensor: paddle.Tensor,
) -> paddle.Tensor:
    """Fill the input Tensor with the scalar value 1.

    Args:
        tensor (Tensor):  Paddle Tensor.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Constant(value=1.0)

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def zeros_(
    tensor: paddle.Tensor,
) -> paddle.Tensor:
    """Fill the input Tensor with the scalar value 0.

    Args:
        tensor (Tensor):  Paddle Tensor.

    Returns:
        Tensor: Initialized tensor.
    """
    init = Constant(value=0.0)

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def dirac_(
    tensor: paddle.Tensor,
    groups: int = 1,
) -> paddle.Tensor:
    """Initialize the 3D/4D/5D Tensor with Dirac delta function.

    Args:
        tensor (Tensor):  Paddle Tensor.
        groups (int|None, optional): 0-dimension of the Tensor will be divided by groups,
            each group has the same value. Default: 1.
    Returns:
        Tensor: Initialized tensor.
    """
    init = Dirac(groups=groups)

    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def eye_(
    tensor: paddle.Tensor,
) -> paddle.Tensor:
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
        return tensor
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
) -> paddle.Tensor:
    """Fill the input Tensor with a (semi) orthogonal matrix.

    Args:
        tensor (Tensor):  Paddle Tensor.
        gain(float, optional): The multiplication coefficient for initialized tensor. Default: 1.0.
    Returns:
        Tensor: Initialized tensor.
    """
    init = Orthogonal(gain=gain)
    if in_dygraph_mode():
        init(tensor)
        return tensor
    return init(tensor)


def sparse_(
    tensor: paddle.Tensor, sparsity: float, std: float = 0.01
) -> paddle.Tensor:
    """Fill the 2D input Tensor as a sparse matrix.

    The non-zero elements will be drawn from the normal distribution.

    Args:
        tensor (Tensor): Paddle Tensor with 2 dimensions.
        sparsity (float): The fraction of elements in each column to be set to zero.
        std (float): the standard deviation of the normal distribution used to generate
            the non-zero values. Default is 0.01.

    Examples:
        >>> tensor = paddle.empty(3, 5)
        >>> result = paddle.nn.init.sparse_(tensor, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    rows, cols = tensor.shape
    num_zeros = math.ceil(sparsity * rows)

    with paddle.no_grad():
        tensor = normal_(tensor, mean=0, std=std)
        for col_idx in range(cols):
            row_indices = paddle.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor


def _make_deprecate(func: Callable[_InputT, _RetT]) -> Callable[_InputT, _RetT]:
    new_name = func.__name__
    old_name = new_name[:-1]

    def deprecated_init(*args: _InputT.args, **kwargs: _InputT.kwargs) -> _RetT:
        warnings.warn(
            f"`nn.init.{old_name}` is now deprecated in favor of `nn.init.{new_name}`.",
            FutureWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    deprecated_init.__doc__ = rf"""
    {old_name}(...)

    .. warning::
        This method is now deprecated in favor of :func:`paddle.nn.init.{new_name}`.

    See :func:`~paddle.nn.init.{new_name}` for details."""
    deprecated_init.__name__ = old_name
    return deprecated_init


uniform = _make_deprecate(uniform_)
normal = _make_deprecate(normal_)
constant = _make_deprecate(constant_)
eye = _make_deprecate(eye_)
dirac = _make_deprecate(dirac_)
xavier_uniform = _make_deprecate(xavier_uniform_)
xavier_normal = _make_deprecate(xavier_normal_)
kaiming_uniform = _make_deprecate(kaiming_uniform_)
kaiming_normal = _make_deprecate(kaiming_normal_)
orthogonal = _make_deprecate(orthogonal_)
sparse = _make_deprecate(sparse_)
