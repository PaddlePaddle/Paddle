# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

# This file contains composite rules of nonbasic operations. There are some notes:
# 1. When define composite rule of some op, you can only use primitive ops defined in primitives.py.
# 2. The name and args of target op must be corresponding with standard description of op in
#    ops.yaml or legacy_ops.yaml.


from .primitives import *  # noqa: F403
from .primreg import REGISTER_COMPOSITE, lookup_composite


def _composite(op, *args):
    _lowerrule = lookup_composite(op.type)
    return _lowerrule(op, *args)


@REGISTER_COMPOSITE('softmax')
def softmax_composite(x, axis):
    """define composite rule of op softmax"""
    if not x.shape:
        # do not return 1, to ensure gradients
        res = divide(x + 1e-5, x + 1e-5)
        return res
    max_temp = max(x, axis, keepdim=True)
    max_temp.stop_gradient = True
    molecular = exp(x - max_temp)
    denominator = sum(molecular, axis=axis, keepdim=True)
    res = divide(molecular, denominator)
    return res


@REGISTER_COMPOSITE('batch_norm')
def composite_batchnorm(
    x,
    run_mean,
    run_var,
    scale,
    bias,
    is_test,
    momentum,
    epsilon,
    data_layout,
    use_global_stats,
    trainable_statistics,
):
    """define composite rule of op batch_norm"""

    feature_axis = (
        1 if data_layout in ('NC', 'NCL', 'NCHW', 'NCHWD') else len(x.shape) - 1
    )
    if use_global_stats is None:
        use_global_stats = is_test
        trainable_statistics = False
    else:
        trainable_statistics = not use_global_stats

    use_run_stat = (is_test and (not trainable_statistics)) or use_global_stats
    reduce_axes = tuple(i for i in range(len(x.shape)) if i != feature_axis)
    stats_shape = tuple(
        1 if i in reduce_axes else s for i, s in enumerate(x.shape)
    )

    batch_mean = zeros(run_mean.shape, run_mean.dtype)
    batch_var = zeros(run_var.shape, run_var.dtype)
    if not use_run_stat:
        batch_mean = mean(x, reduce_axes, keepdim=True)
        temp = mean(x * x, reduce_axes, keepdim=True)
        batch_var = temp - batch_mean * batch_mean

        x_hat = (x - reshape(batch_mean, stats_shape)) / sqrt(
            reshape(batch_var, stats_shape) + epsilon
        )

        run_mean = momentum * run_mean + (1 - momentum) * reshape(
            batch_mean, run_mean.shape
        )
        run_var = momentum * run_var + (1 - momentum) * reshape(
            batch_var, run_var.shape
        )
    else:
        x_hat = (x - reshape(run_mean, stats_shape)) / sqrt(
            reshape(run_var, stats_shape) + epsilon
        )
    y = reshape(scale, stats_shape) * x_hat + reshape(bias, stats_shape)

    # add op assign to detach tensor in void unsafe change outside the rule.
    batch_mean_ = assign(reshape(batch_mean, run_mean.shape))
    batch_var_ = assign(reshape(batch_var, run_var.shape))
    run_mean_ = assign(run_mean)
    run_var_ = assign(run_var)

    # reserve_space is not needed in composite rule, but still ruturn None to keep same as phi op defination.
    reserve_space = None

    return y, run_mean_, run_var_, batch_mean_, batch_var_, reserve_space


@REGISTER_COMPOSITE('gelu')
def gelu_composite(x, approximate):
    """define composite rule of op gelu"""
    M_SQRT1_2 = (
        0.70710678118654752440  # /* 1/sqrt(2) */ copy from gelu-kernel.cc
    )
    M_2_SQRTPI = 1.12837916709551257390  # /* 2/sqrt(pi) */
    one = ones(x.shape, x.dtype)
    half = full(x.shape, 0.5, x.dtype)
    if approximate:
        # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / \pi) * (x + 0.044715 * x^{3})))
        kAlpha = full(x.shape, M_2_SQRTPI * M_SQRT1_2, x.dtype)
        GELU_CONSTANT = full(x.shape, 0.044715, x.dtype)
        tanh_out = tanh(kAlpha * (x + GELU_CONSTANT * x * x * x))
        out = x * half * (one + tanh_out)
        return out

    else:
        # gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
        cdf = half * (one + erf(x * full(x.shape, M_SQRT1_2, x.dtype)))
        out = x * cdf
        return out
