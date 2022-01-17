#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import contextlib
from collections import abc
from enum import Enum
from math import inf
import numpy as np
from types import MethodType

import paddle
import paddle.distributed as dist
from paddle import _C_ops
from paddle.fluid import core
from paddle.fluid import layers
from paddle.fluid.dygraph import to_variable
from paddle.fluid.framework import dygraph_only
from paddle.fluid.dygraph import base as imperative_base
from paddle.distributed.collective import _get_global_group


class Taskflow:
    """
    Task flows, one way linked list for task acquisition.
    """

    def __init__(self, task, callback):
        self.task = task
        self.callback = callback


class Type(Enum):
    """
    Type of trainable parameters
    """
    fp16 = paddle.float16
    fp32 = paddle.float32


class ShardingClipGrad:
    def __init__(self, clip, device, group):
        self._clip = clip
        self._device = device
        self._group = group

    @imperative_base.no_grad
    def _dygraph_clip(self, params_grads):
        params_and_grads = []

        sum_square_fp16 = []
        sum_square_fp32 = []

        for p, g in params_grads:
            if g is None or getattr(p, 'need_clip', True) is False:
                continue

            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = layers.get_tensor_from_selected_rows(
                    layers.merge_selected_rows(g))
            square = layers.square(merge_grad)
            sum_square = layers.reduce_sum(square)

            if p.dtype == paddle.float16:
                sum_square_fp16.append(sum_square)
            elif p.dtype == paddle.float32:
                sum_square_fp32.append(sum_square)

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_fp16) == 0:
            global_norm_fp16 = paddle.to_tensor([0.], dtype=paddle.float32)
        else:
            global_norm_fp16 = layers.concat(sum_square_fp16)
            global_norm_fp16 = layers.reduce_sum(global_norm_fp16)
            global_norm_fp16 = paddle.cast(
                global_norm_fp16, dtype=paddle.float32)

        # global norm of non-distributed FP32 params_and_grads
        global_norm_fp32 = layers.concat(sum_square_fp32) if len(
            sum_square_fp32) != 0 else paddle.to_tensor(
                [0.], dtype=paddle.float32)
        global_norm_fp32 = layers.reduce_sum(global_norm_fp32)

        global_norm_var = global_norm_fp16 + global_norm_fp32

        # add all reduce to get global norm of distributed params_and_grads
        dev_id = int(self._device.split(":")[1])
        with device_guard(dev_id, "gpu"):
            paddle.distributed.all_reduce(global_norm_var, group=self._group)

        global_norm_var = layers.sqrt(global_norm_var)
        max_global_norm = layers.fill_constant(
            shape=[1], dtype=global_norm_var.dtype, value=self.clip_norm)

        clip_var = layers.elementwise_div(
            x=max_global_norm,
            y=layers.elementwise_max(
                x=global_norm_var, y=max_global_norm))
        clip_var_fp16 = paddle.cast(clip_var, paddle.float16)

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                params_and_grads.append((p, g))
                continue
            if p.dtype == paddle.float16:
                new_grad = layers.elementwise_mul(x=g, y=clip_var_fp16)
            else:
                new_grad = layers.elementwise_mul(x=g, y=clip_var)
            params_and_grads.append((p, new_grad))

        return params_and_grads

    def __getattr__(self, item):
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        return self._dygraph_clip(params_grads)


@contextlib.contextmanager
def device_guard(dev_id=0, device="cpu"):
    origin_device = paddle.device.get_device()
    if device == "cpu":
        paddle.set_device(device)
    elif device == "gpu":
        paddle.set_device("gpu:{}".format(dev_id))
    try:
        yield
    finally:
        paddle.set_device(origin_device)


@dygraph_only
def ShardingScaler(scaler):
    def unscale_method(self, optimizer):
        if not self._enable:
            return
        param_grads = []
        param_grads_fp16 = []
        param_grads_fp32 = []
        if hasattr(optimizer, "update_slice"):
            optimizer.update_slice()
            optimizer.update_scaler = True

        if getattr(optimizer._optim, '_param_groups', None) and isinstance(
                optimizer._optim._param_groups[0], dict):

            for group in optimizer._optim._param_groups:
                for param in group['params']:
                    if param._grad_ivar() is not None:
                        param_grads.append(param._grad_ivar())
                        if param._grad_ivar(
                        ).dtype in [core.VarDesc.VarType.FP16, paddle.float16]:
                            param_grads_fp16.append(param._grad_ivar())
                        else:
                            param_grads_fp32.append(param._grad_ivar())
        else:
            for param in optimizer._optim._parameter_list:
                if param.grad is not None:
                    param_grads.append(param.grad)
                    if param.grad.dtype in [
                            core.VarDesc.VarType.FP16, paddle.float16
                    ]:
                        param_grads_fp16.append(param.grad)
                    else:
                        param_grads_fp32.append(param.grad)

        temp_found_inf_fp16 = to_variable(np.array([0]).astype(np.bool))
        temp_found_inf_fp32 = to_variable(np.array([0]).astype(np.bool))

        device = "cpu" if optimizer.offload else "gpu"
        dev_id = 0 if device == "cpu" else int(paddle.get_device().split(":")[
            1])

        with device_guard(dev_id, device):
            if len(param_grads_fp16):
                _C_ops.check_finite_and_unscale(param_grads_fp16, self._scale,
                                                param_grads_fp16,
                                                temp_found_inf_fp16)
            if len(param_grads_fp32):
                _C_ops.check_finite_and_unscale(param_grads_fp32, self._scale,
                                                param_grads_fp32,
                                                temp_found_inf_fp32)

        self._found_inf = 1 if temp_found_inf_fp16 or temp_found_inf_fp32 else 0
        is_found_inf = paddle.to_tensor([self._found_inf], dtype="int32")

        paddle.distributed.all_reduce(
            is_found_inf,
            op=paddle.distributed.ReduceOp.MAX,
            group=optimizer.group)
        self._found_inf = is_found_inf.numpy()[0]

    scaler._unscale = MethodType(unscale_method, scaler)
    return scaler
