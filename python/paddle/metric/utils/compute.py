# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Compute utilities for paddle.metric."""


from typing_extensions import Literal

import paddle
from paddle.metric.utils.prints import rank_zero_warn


def _safe_matmul(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """Safe matrix multiplication. Casts float16 to float32 for computation."""
    if x.dtype == paddle.float16 or y.dtype == paddle.float16:
        return (x.cast("float32") @ y.T.cast("float32")).cast("float16")
    return x @ y.T


def _safe_xlogy(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """Compute x * log(y). Returns 0 if x=0."""
    res = x * paddle.log(y)
    res = paddle.where(x == 0, paddle.zeros_like(res), res)
    return res


def _safe_divide(
    num: paddle.Tensor,
    denom: paddle.Tensor,
    zero_division: float | Literal["warn", "nan"] = 0.0,
) -> paddle.Tensor:
    """Safe division, by preventing division by zero.

    Args:
        num: numerator tensor
        denom: denominator tensor, which may contain zeros
        zero_division: value to replace elements divided by zero
    """
    num = num if num.is_floating_point() else num.cast("float32")
    denom = denom if denom.is_floating_point() else denom.cast("float32")
    if isinstance(zero_division, (float, int)) or zero_division == "warn":
        if zero_division == "warn" and paddle.any(denom == 0):
            rank_zero_warn(
                "Detected zero division in _safe_divide. Setting 0/0 to 0.0"
            )
        zero_division = 0.0 if zero_division == "warn" else zero_division
        zero_division_tensor = paddle.to_tensor(zero_division, dtype=num.dtype)
        return paddle.where(denom != 0, num / denom, zero_division_tensor)
    return paddle.divide(num, denom)


def _adjust_weights_safe_divide(
    score: paddle.Tensor,
    average: str | None,
    multilabel: bool,
    tp: paddle.Tensor,
    fp: paddle.Tensor,
    fn: paddle.Tensor,
    top_k: int = 1,
) -> paddle.Tensor:
    if average is None or average == "none":
        return score
    if average == "weighted":
        weights = tp + fn
    else:
        weights = paddle.ones_like(score)
        if not multilabel:
            weights = paddle.where(
                (tp + fp + fn == 0) if top_k == 1 else (tp + fn == 0),
                paddle.zeros_like(weights),
                weights,
            )
    return _safe_divide(
        weights * score, weights.sum(-1, keepdim=True).expand_as(score)
    ).sum(-1)


def _auc_format_inputs(
    x: paddle.Tensor, y: paddle.Tensor
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Check that auc input is correct."""
    x = x.squeeze() if x.ndim > 1 else x
    y = y.squeeze() if y.ndim > 1 else y
    if x.ndim > 1 or y.ndim > 1:
        raise ValueError(
            f"Expected both `x` and `y` tensor to be 1d, but got tensors with dimension {x.ndim} and {y.ndim}"
        )
    if x.numel() != y.numel():
        raise ValueError(
            f"Expected the same number of elements in `x` and `y` tensor but received {x.numel()} and {y.numel()}"
        )
    return x, y


def _auc_compute_without_check(
    x: paddle.Tensor, y: paddle.Tensor, direction: float, axis: int = -1
) -> paddle.Tensor:
    """Compute area under the curve using the trapezoidal rule."""
    with paddle.no_grad():
        auc_score = paddle.trapezoid(y=y, x=x, axis=axis) * direction
    return auc_score


def _auc_compute(
    x: paddle.Tensor, y: paddle.Tensor, reorder: bool = False
) -> paddle.Tensor:
    """Compute area under the curve using the trapezoidal rule."""
    with paddle.no_grad():
        if reorder:
            x_idx = paddle.argsort(x, stable=True)
            x = x[x_idx]
            y = y[x_idx]
        dx = x[1:] - x[:-1]
        if (dx < 0).any():
            if (dx <= 0).all():
                direction = -1.0
            else:
                raise ValueError(
                    "The `x` tensor is neither increasing or decreasing. Try setting reorder=True."
                )
        else:
            direction = 1.0
        return _auc_compute_without_check(x, y, direction)


def auc(
    x: paddle.Tensor, y: paddle.Tensor, reorder: bool = False
) -> paddle.Tensor:
    """Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x: x-coordinates, must be either increasing or decreasing
        y: y-coordinates
        reorder: if True, will reorder the arrays to make it either increasing or decreasing
    """
    x, y = _auc_format_inputs(x, y)
    return _auc_compute(x, y, reorder=reorder)


def interp(
    x: paddle.Tensor, xp: paddle.Tensor, fp: paddle.Tensor
) -> paddle.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample points.

    Args:
        x: x-coordinates at which to evaluate the interpolated values
        xp: x-coordinates of the data points, must be increasing
        fp: y-coordinates of the data points, same length as xp
    """
    m = _safe_divide(fp[1:] - fp[:-1], xp[1:] - xp[:-1])
    b = fp[:-1] - m * xp[:-1]
    indices = (
        paddle.sum(
            paddle.greater_equal(x.unsqueeze(1), xp.unsqueeze(0)), axis=1
        )
        - 1
    )
    indices = paddle.clip(indices, 0, len(m) - 1)
    return m[indices] * x + b[indices]


def normalize_logits_if_needed(
    tensor: paddle.Tensor,
    normalization: Literal["sigmoid", "softmax"] | None,
) -> paddle.Tensor:
    """Normalize logits if needed.

    If input tensor is outside the [0,1] we assume that logits are provided and apply normalization.
    """
    if not normalization:
        return tensor
    condition = ((tensor < 0) | (tensor > 1)).any()
    if normalization == "sigmoid":
        normalized = paddle.nn.functional.sigmoid(tensor)
    else:
        normalized = paddle.nn.functional.softmax(tensor, axis=1)
    return paddle.where(condition, normalized, tensor)
