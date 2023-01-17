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
from paddle.fluid.layers import fill_constant  # noqa: F401
from paddle.tensor import mean, ones, reshape, sqrt, square, zeros

from .primitives import *  # noqa: F403
from .primreg import REGISTER_COMPOSITE, lookup_composite


def _composite(op, *args):
    _lowerrule = lookup_composite(op.type)
    return _lowerrule(op, *args)


@REGISTER_COMPOSITE('softmax')
def softmax_composite(x, axis):
    """define composite rule of op softmax"""
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
    variance,
    scale,
    bias,
    is_test,
    momentum,
    epsilon,
    data_layout,
    use_global_stats,
    trainable_statistics,
):
    """define composite rule of op softmax"""
    # reserve_space = (
    #     None if len(op.output_names) == 5 else get_output_var_list(op)[1]
    # )
    print(
        "x",
        x,
        '\n',
        "run_mean",
        run_mean,
        '\n',
        "variance",
        variance,
        '\n',
        "scale",
        scale,
        '\n',
        "bias",
        bias,
        '\n',
        "is_test",
        is_test,
        '\n',
        "momentum",
        momentum,
        '\n',
        "epsilon",
        epsilon,
        '\n',
        "data_layout",
        data_layout,
        '\n',
        "use_global_stats",
        use_global_stats,
        '\n',
        "trainable_statistics",
        trainable_statistics,
    )

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
    batch_var = zeros(variance.shape, variance.dtype)
    # breakpoint()
    if not use_run_stat:
        print("in========================")
        batch_mean = mean(x, reduce_axes, keepdim=True)
        batch_var = mean(
            square(subtract(x, broadcast_to(batch_mean, x.shape))),
            reduce_axes,
            keepdim=True,
        )
        x_hat = divide(
            subtract(x, broadcast_to(batch_mean, x.shape)),
            sqrt(
                add(
                    broadcast_to(batch_var, x.shape),
                    fill_constant(x.shape, batch_var.dtype, epsilon),
                )
            ),
        )

        momentum = fill_constant(run_mean.shape, run_mean.dtype, momentum)
        run_mean = add(
            multiply(momentum, run_mean),
            multiply(
                subtract(ones(run_mean.shape, run_mean.dtype), momentum),
                reshape(batch_mean, run_mean.shape),
            ),
        )
        variance = add(
            multiply(momentum, variance),
            multiply(
                subtract(ones(variance.shape, variance.dtype), momentum),
                reshape(batch_var, variance.shape),
            ),
        )
    else:
        x_hat = divide(
            subtract(x, broadcast_to(reshape(run_mean, stats_shape), x.shape)),
            sqrt(
                add(
                    broadcast_to(reshape(variance, stats_shape), x.shape),
                    fill_constant(x.shape, x.dtype, epsilon),
                )
            ),
        )
    y = add(
        multiply(broadcast_to(reshape(scale, stats_shape), x_hat.shape), x_hat),
        broadcast_to(reshape(bias, stats_shape), x_hat.shape),
    )

    # if reserve_space:
    if trainable_statistics or not is_test:
        return run_mean, None, batch_mean, batch_var, variance, y
    else:
        return run_mean, batch_mean, batch_var, variance, y
