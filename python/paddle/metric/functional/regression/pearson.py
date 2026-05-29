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

import math

import paddle
from paddle.metric.functional.regression.utils import (
    _check_data_shape_to_num_outputs,
)
from paddle.metric.utils import rank_zero_warn
from paddle.metric.utils.checks import _check_same_shape


def _pearson_corrcoef_update(
    preds: paddle.Tensor,
    target: paddle.Tensor,
    mean_x: paddle.Tensor,
    mean_y: paddle.Tensor,
    max_abs_dev_x: paddle.Tensor,
    max_abs_dev_y: paddle.Tensor,
    var_x: paddle.Tensor,
    var_y: paddle.Tensor,
    corr_xy: paddle.Tensor,
    num_prior: paddle.Tensor,
    num_outputs: int,
) -> tuple[
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
    paddle.Tensor,
]:
    """Update and returns variables required to compute Pearson Correlation Coefficient.

    Check for same shape of input tensors.

    Args:
        preds: estimated scores
        target: ground truth scores
        mean_x: current mean estimate of x tensor
        mean_y: current mean estimate of y tensor
        max_abs_dev_x: current maximum absolute value of x tensor
        max_abs_dev_y: current maximum absolute value of y tensor
        var_x: current variance estimate of x tensor
        var_y: current variance estimate of y tensor
        corr_xy: current covariance estimate between x and y tensor
        num_prior: current number of observed observations
        num_outputs: Number of outputs in multioutput setting

    """
    _check_same_shape(preds, target)
    _check_data_shape_to_num_outputs(preds, target, num_outputs)
    num_obs = preds.shape[0]
    batch_mean_x = preds.mean(0)
    batch_mean_y = target.mean(0)
    delta_x = batch_mean_x - mean_x
    delta_y = batch_mean_y - mean_y
    n_total = num_prior + num_obs
    mx_new = mean_x + delta_x * num_obs / n_total
    my_new = mean_y + delta_y * num_obs / n_total
    if num_obs == 1:
        delta2_x = batch_mean_x - mx_new
        delta2_y = batch_mean_y - my_new
        var_x = var_x + delta2_x * delta_x
        var_y = var_y + delta2_y * delta_y
        corr_xy = corr_xy + delta_x * delta2_y
    else:
        preds_centered = preds - batch_mean_x
        target_centered = target - batch_mean_y
        batch_var_x = (preds_centered**2).sum(0)
        batch_var_y = (target_centered**2).sum(0)
        batch_cov_xy = (preds_centered * target_centered).sum(0)
        correction = num_prior * num_obs / n_total
        var_x = var_x + batch_var_x + delta_x**2 * correction
        var_y = var_y + batch_var_y + delta_y**2 * correction
        corr_xy = corr_xy + batch_cov_xy + delta_x * delta_y * correction
    max_abs_dev_x = paddle.maximum(
        max_abs_dev_x, paddle.max((preds - mx_new).abs(), axis=0)
    )
    max_abs_dev_y = paddle.maximum(
        max_abs_dev_y, paddle.max((target - my_new).abs(), axis=0)
    )
    return (
        mx_new,
        my_new,
        max_abs_dev_x,
        max_abs_dev_y,
        var_x,
        var_y,
        corr_xy,
        n_total,
    )


def _pearson_corrcoef_compute(
    max_abs_dev_x: paddle.Tensor,
    max_abs_dev_y: paddle.Tensor,
    var_x: paddle.Tensor,
    var_y: paddle.Tensor,
    corr_xy: paddle.Tensor,
    nb: paddle.Tensor,
) -> paddle.Tensor:
    """Compute the final pearson correlation based on accumulated statistics.

    Args:
        max_abs_dev_x: maximum absolute value of x tensor
        max_abs_dev_y: maximum absolute value of y tensor
        var_x: variance estimate of x tensor
        var_y: variance estimate of y tensor
        corr_xy: covariance estimate between x and y tensor
        nb: number of observations

    """
    var_x = var_x / (nb - 1)
    var_y = var_y / (nb - 1)
    corr_xy = corr_xy / (nb - 1)
    if var_x.dtype == paddle.float16 and var_x.device == paddle.device("cpu"):
        var_x = var_x.bfloat16()
        var_y = var_y.bfloat16()
    var_x = var_x * paddle.pow(max_abs_dev_x, -2)
    var_y = var_y * paddle.pow(max_abs_dev_y, -2)
    corr_xy = corr_xy / (max_abs_dev_x * max_abs_dev_y)
    bound = math.sqrt(paddle.finfo(var_x.dtype).eps)
    if (
        (var_x < bound).any()
        or (var_y < bound).any()
        or ~paddle.isfinite(var_x).any()
        or ~paddle.isfinite(var_y).any()
        or ~paddle.isfinite(corr_xy).any()
    ):
        rank_zero_warn(
            f"The variance of predictions or target is close to zero. This can cause instability in Pearson correlationcoefficient, leading to wrong results. Consider re-scaling the input if possible or computing using alarger dtype (currently using {var_x.dtype}). Setting the correlation coefficient to nan.",
            UserWarning,
        )
    zero_var_mask = (
        (var_x < bound)
        | (var_y < bound)
        | ~paddle.isfinite(var_x)
        | ~paddle.isfinite(var_y)
        | ~paddle.isfinite(corr_xy)
    )
    corrcoef = paddle.full_like(
        corr_xy, float("nan"), device=corr_xy.device, dtype=corr_xy.dtype
    )
    valid_mask = ~zero_var_mask
    if valid_mask.any():
        corrcoef[valid_mask] = (
            (
                corr_xy[valid_mask]
                / (var_x[valid_mask] * var_y[valid_mask]).sqrt()
            )
            .squeeze()
            .to(corrcoef.dtype)
        )
        corrcoef = paddle.clamp(corrcoef, -1.0, 1.0)
    return corrcoef.squeeze()


def pearson_corrcoef(
    preds: paddle.Tensor, target: paddle.Tensor
) -> paddle.Tensor:
    """Compute pearson correlation coefficient.

    Args:
        preds: estimated scores
        target: ground truth scores

    Example (single output regression):
        >>> from paddle.metric.functional.regression import pearson_corrcoef
        >>> target = paddle.to_tensor([3, -0.5, 2, 7])
        >>> preds = paddle.to_tensor([2.5, 0.0, 2, 8])
        >>> pearson_corrcoef(preds, target)
        tensor(0.9849)

    Example (multi output regression):
        >>> from paddle.metric.functional.regression import pearson_corrcoef
        >>> target = paddle.to_tensor([[3, -0.5], [2, 7]])
        >>> preds = paddle.to_tensor([[2.5, 0.0], [2, 8]])
        >>> pearson_corrcoef(preds, target)
        tensor([1., 1.])

    """
    d = preds.shape[1] if preds.ndim == 2 else 1
    _temp = paddle.zeros(d, dtype=preds.dtype, device=preds.place)
    mean_x, mean_y, var_x = _temp.clone(), _temp.clone(), _temp.clone()
    var_y, corr_xy, nb = _temp.clone(), _temp.clone(), _temp.clone()
    max_abs_dev_x, max_abs_dev_y = _temp.clone(), _temp.clone()
    (
        _,
        _,
        max_abs_dev_x,
        max_abs_dev_y,
        var_x,
        var_y,
        corr_xy,
        nb,
    ) = _pearson_corrcoef_update(
        preds=preds,
        target=target,
        mean_x=mean_x,
        mean_y=mean_y,
        max_abs_dev_x=max_abs_dev_x,
        max_abs_dev_y=max_abs_dev_y,
        var_x=var_x,
        var_y=var_y,
        corr_xy=corr_xy,
        num_prior=nb,
        num_outputs=1 if preds.ndim == 1 else preds.shape[-1],
    )
    return _pearson_corrcoef_compute(
        max_abs_dev_x, max_abs_dev_y, var_x, var_y, corr_xy, nb
    )
