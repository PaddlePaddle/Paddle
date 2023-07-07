# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import framework
from paddle.autograd import no_grad
from paddle.distributed import fleet
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm, clip

from ...base.topology import ParallelMode
from ...utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
    sharding_reduce_gradients,
)
from ...utils.log_util import logger
from ...utils.mix_precision_utils import MixPrecisionOptimizer

__all__ = []


def _obtain_optimizer_parameters_list(optimizer):
    if getattr(optimizer, '_param_groups', None) and isinstance(
        optimizer._param_groups[0], dict
    ):
        parameters_list = []
        for group in optimizer._param_groups:
            for param in group['params']:
                parameters_list.append(param)
    else:
        parameters_list = list(optimizer._parameter_list)

    return parameters_list


class HybridParallelClipGrad:
    def __init__(self, clip, hcg):
        self._clip = clip
        self._hcg = hcg

    @no_grad()
    def _dygraph_clip(self, params_grads):
        sum_square_dist_fp16 = []
        sum_square_dist_bf16 = []
        sum_square_dist_fp32 = []
        sum_square_not_dist_fp16 = []
        sum_square_not_dist_bf16 = []
        sum_square_not_dist_fp32 = []

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = clip.merge_selected_rows(g)
                merge_grad = clip.get_tensor_from_selected_rows(merge_grad)
            square = paddle.square(merge_grad)
            sum_square = paddle.sum(square)

            not_shared_enable = (not hasattr(p, 'is_firstly_shared')) or (
                hasattr(p, 'is_firstly_shared')
                and getattr(p, 'is_firstly_shared', True)
            )

            if not_shared_enable:
                if p.is_distributed:
                    if g.dtype == paddle.float16:
                        sum_square_dist_fp16.append(sum_square)
                    elif g.dtype == paddle.bfloat16:
                        sum_square_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_dist_fp32.append(sum_square)
                else:
                    if g.dtype == paddle.float16:
                        sum_square_not_dist_fp16.append(sum_square)
                    if g.dtype == paddle.bfloat16:
                        sum_square_not_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_not_dist_fp32.append(sum_square)

        # global norm of distributed FP16 params_and_grads
        if len(sum_square_dist_fp16) == 0:
            global_norm_dist_fp16 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_dist_fp16 = paddle.add_n(sum_square_dist_fp16)
            global_norm_dist_fp16 = paddle.cast(
                global_norm_dist_fp16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_not_dist_fp16) == 0:
            global_norm_not_dist_fp16 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_not_dist_fp16 = paddle.add_n(sum_square_not_dist_fp16)
            global_norm_not_dist_fp16 = paddle.cast(
                global_norm_not_dist_fp16, dtype=paddle.float32
            )

        # global norm of distributed BF16 params_and_grads
        if len(sum_square_dist_bf16) == 0:
            global_norm_dist_bf16 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_dist_bf16 = paddle.add_n(sum_square_dist_bf16)
            global_norm_dist_bf16 = paddle.cast(
                global_norm_dist_bf16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_not_dist_bf16) == 0:
            global_norm_not_dist_bf16 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_not_dist_bf16 = paddle.add_n(sum_square_not_dist_bf16)
            global_norm_not_dist_bf16 = paddle.cast(
                global_norm_not_dist_bf16, dtype=paddle.float32
            )

        # global norm of distributed FP32 params_and_grads
        if len(sum_square_dist_fp32) == 0:
            global_norm_dist_fp32 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_dist_fp32 = paddle.add_n(sum_square_dist_fp32)

        # global norm of non-distributed FP32 params_and_grads
        if len(sum_square_not_dist_fp32) == 0:
            global_norm_not_dist_fp32 = paddle.to_tensor(
                np.array(0.0), dtype=paddle.float32
            )
        else:
            global_norm_not_dist_fp32 = paddle.add_n(sum_square_not_dist_fp32)

        global_norm_var_dist = (
            global_norm_dist_fp16
            + global_norm_dist_bf16
            + global_norm_dist_fp32
        )
        global_norm_var_not_dist = (
            global_norm_not_dist_fp16
            + global_norm_not_dist_bf16
            + global_norm_not_dist_fp32
        )

        # add all reduce to get global norm of distributed params_and_grads
        if self._hcg.get_model_parallel_world_size() > 1:
            paddle.distributed.all_reduce(
                global_norm_var_dist, group=self._hcg.get_check_parallel_group()
            )

        # add all reduce to get global norm of non-distributed params_and_grads in groups of pp
        if self._hcg.get_pipe_parallel_world_size() > 1:
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_pipe_parallel_group(),
            )

        # In Sharding mode, param and grad is mapping different rank in optimizer.
        # ClipGradByGlobalNorm need allreduce to get globol norm
        if self._hcg.get_sharding_parallel_world_size() > 1:
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_sharding_parallel_group(),
            )

        global_norm_var_fp32 = paddle.sqrt(
            global_norm_var_dist + global_norm_var_not_dist
        )

        max_global_norm = paddle.full(
            shape=[],
            dtype=global_norm_var_fp32.dtype,
            fill_value=self.clip_norm,
        )
        clip_var = paddle.divide(
            x=max_global_norm,
            y=paddle.maximum(x=global_norm_var_fp32, y=max_global_norm)
            + paddle.to_tensor(np.array(1.0e-6), dtype=paddle.float32),
        )
        clip_var_fp16 = paddle.cast(clip_var, paddle.float16)

        # bf16 is not supported on XPU now
        if not (
            paddle.is_compiled_with_xpu()
            or isinstance(
                paddle.framework._current_expected_place(), paddle.CustomPlace
            )
        ):
            clip_var_bf16 = paddle.cast(clip_var, paddle.bfloat16)
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            if g.dtype == paddle.float16:
                g.scale_(clip_var_fp16)
            elif g.dtype == paddle.bfloat16:
                if paddle.is_compiled_with_xpu():
                    raise NotImplementedError(
                        "BF16 is not supported on XPU now"
                    )
                g.scale_(clip_var_bf16)
            else:
                g.scale_(clip_var)
            p._reset_grad_inplace_version(True)

        return params_grads

    def __getattr__(self, item):
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        return self._dygraph_clip(params_grads)


class HybridParallelOptimizer:
    # adapter wrapper for optimizer
    def __init__(self, optimizer, hcg, strategy):
        self._inner_opt = optimizer
        self._strategy = strategy
        self._hcg = hcg

        self._use_dp_mode = (
            self._hcg.get_parallel_mode() == ParallelMode.DATA_PARALLEL
        )

        self._need_dp = self._hcg.get_data_parallel_world_size() > 1

        # NOTE(shenliang03): Because of the pure DataParallel mode, the gradient synchronization
        # is achieved through reducer, so there is no need to call fuse_allreduce in optimizer.
        self._dp_enable = not self._use_dp_mode and self._need_dp

        self._sharding_enable = self._hcg.get_sharding_parallel_world_size() > 1

        if (
            isinstance(self._inner_opt._grad_clip, ClipGradByGlobalNorm)
            and not self._use_dp_mode
        ):
            logger.warning(
                "While using ClipGradByGlobalNorm in TensorParallel, PipelineParallel "
                "or Sharding, the grad clip of original optimizer will be changed."
            )

            inner_opt = (
                self._inner_opt._inner_optimizer
                if self._sharding_enable
                else self._inner_opt
            )

            if isinstance(inner_opt, MixPrecisionOptimizer):
                inner_opt = inner_opt._inner_opt

            if (
                inner_opt._parameter_list
                and not isinstance(inner_opt._parameter_list[0], dict)
                and len(
                    [
                        p
                        for p in inner_opt._parameter_list
                        if hasattr(p, "main_grad")
                    ]
                )
                > 0
            ):
                inner_opt._grad_clip = HybridParallelClipGrad(
                    inner_opt._grad_clip, hcg
                )
            else:
                inner_opt._grad_clip = HybridParallelClipGrad(
                    inner_opt._grad_clip, hcg
                )
                if inner_opt._parameter_list and isinstance(
                    inner_opt._parameter_list[0], dict
                ):
                    for item in inner_opt._param_groups:
                        if "grad_clip" in item.keys():
                            item["grad_clip"] = HybridParallelClipGrad(
                                inner_opt._grad_clip, hcg
                            )

    def _insert_sync(self, sync_var, src, mp_group, sync_mode):
        if sync_mode == "broadcast":
            paddle.distributed.broadcast(
                sync_var, src=src, group=mp_group, sync_op=True
            )
        else:
            paddle.distributed.all_reduce(
                sync_var, group=mp_group, sync_op=True
            )
            sync_var.scale_(1.0 / mp_group.nranks)

    def _filter_fn(self, param, strategy):
        p_name = param.name
        tar_param = strategy.sync_param_name
        if param.is_distributed is False:
            for tar in tar_param:
                if tar in p_name:
                    return True
        return False

    def _step(self, parameters_list):
        mp_group = self._hcg.get_model_parallel_group()
        src_rank = self._hcg.get_model_parallel_group_src_rank()
        params = None
        mp_configs = None

        if mp_group.nranks > 1:
            mp_configs = fleet.fleet._user_defined_strategy.hybrid_configs[
                "mp_configs"
            ]

        if mp_configs and (
            mp_configs.sync_param
            or mp_configs.sync_grad
            or mp_configs.sync_moment
        ):
            params = sorted(
                [
                    p
                    for p in parameters_list
                    if self._filter_fn(p, fleet.fleet._user_defined_strategy)
                ],
                key=lambda p: p.name,
            )

        # Grad sync before opt
        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_grad:
            for p in params:
                if hasattr(p, "main_grad") and p.main_grad is not None:
                    assert p.grad is None
                    self._insert_sync(
                        p.main_grad, src_rank, mp_group, mp_configs.sync_mode
                    )
                elif p.grad is not None:
                    self._insert_sync(
                        p.grad, src_rank, mp_group, mp_configs.sync_mode
                    )

        self._inner_opt.step()

        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_param:
            for p in params:
                # Param sync after opt
                self._insert_sync(p, src_rank, mp_group, mp_configs.sync_mode)

                # Master param sync after opt
                if (
                    hasattr(self._inner_opt, "_multi_precision")
                    and self._inner_opt._multi_precision
                    and p.name in self._inner_opt._master_weights
                ):
                    self._insert_sync(
                        self._inner_opt._master_weights[p.name],
                        src_rank,
                        mp_group,
                        mp_configs.sync_mode,
                    )

        # Moment sync after opt
        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_moment:
            for p in params:
                # support opt state of adam and adamw to broadcast now.
                if isinstance(
                    self._inner_opt,
                    (paddle.optimizer.Adam, paddle.optimizer.AdamW),
                ):
                    if (
                        p.name
                        in self._inner_opt._accumulators[
                            self._inner_opt._moment1_acc_str
                        ]
                    ):
                        moment1 = self._inner_opt._get_accumulator(
                            self._inner_opt._moment1_acc_str, p
                        )
                        self._insert_sync(
                            moment1, src_rank, mp_group, mp_configs.sync_mode
                        )

                    if (
                        p.name
                        in self._inner_opt._accumulators[
                            self._inner_opt._moment2_acc_str
                        ]
                    ):
                        moment2 = self._inner_opt._get_accumulator(
                            self._inner_opt._moment2_acc_str, p
                        )
                        self._insert_sync(
                            moment2, src_rank, mp_group, mp_configs.sync_mode
                        )

    @no_grad()
    @framework.dygraph_only
    def step(self):
        parameters_list = _obtain_optimizer_parameters_list(self._inner_opt)
        if self._sharding_enable:
            sharding_reduce_gradients(list(parameters_list), self._hcg)

        if self._dp_enable:
            fused_allreduce_gradients(list(parameters_list), self._hcg)

        self._step(parameters_list)

    @no_grad()
    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):

        # minimize does not support parameters in the form of param_group,
        # so no need use _obtain_optimizer_parameters_list
        parameter_list = (
            parameters
            if parameters
            else _obtain_optimizer_parameters_list(self._inner_opt)
        )

        # Here sharding should use global parameter list
        if self._sharding_enable:
            sharding_reduce_gradients(list(parameter_list), self._hcg)

        if self._dp_enable:
            fused_allreduce_gradients(list(parameter_list), self._hcg)

        return self._inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set
        )

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
