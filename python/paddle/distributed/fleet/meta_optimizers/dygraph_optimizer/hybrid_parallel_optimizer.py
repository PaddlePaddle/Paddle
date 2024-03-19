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


import distutils.util
import os

import numpy as np

import paddle
from paddle import framework
from paddle.autograd import no_grad
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    obtain_optimizer_parameters_list,
)
from paddle.framework import core
from paddle.nn import ClipGradByGlobalNorm, clip

from ...base.topology import ParallelMode
from ...utils import timer_helper as timer
from ...utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
    unwrap_optimizer,
)
from ...utils.log_util import logger
from ...utils.mix_precision_utils import MixPrecisionOptimizer

__all__ = []

g_shard_norm_align_dp = int(os.environ.get("FLAGS_shard_norm_align_dp", 0))


class HybridParallelClipGrad:
    def __init__(self, clip, hcg, timers=None):
        self._clip = clip
        self._hcg = hcg
        self._vpp_chunk_num = None
        self._force_align_vpp_grad_sum_order = distutils.util.strtobool(
            os.getenv('FLAGS_force_align_vpp_grad_sum_order', '0')
        )
        self._timers = timers

        if hasattr(self._clip, "clip_norm") and self._clip.clip_norm <= 0:
            raise ValueError(
                "It's not supported when clip_norm of grad clip is set to 0 or negative number."
            )

    def _get_vpp_chunk_num(self, params_grads):
        chunk_num = -1
        for p, g in params_grads:
            if g is None:
                continue
            chunk_info = getattr(p, '_chunk_info', {})
            cur_chunk_num = chunk_info.get('chunk_num', -1)
            if chunk_num < 0:
                chunk_num = cur_chunk_num
            else:
                assert chunk_num == cur_chunk_num
        return chunk_num

    @no_grad()
    def _vpp_dygraph_clip(self, params_grads, chunk_num):
        pp_group = self._hcg.get_pipe_parallel_group()
        pp_rank = self._hcg.get_stage_id()
        pp_size = self._hcg.get_pipe_parallel_world_size()

        if self._vpp_chunk_num is None:
            all_chunk_nums = []
            paddle.distributed.all_gather_object(
                all_chunk_nums, chunk_num, group=pp_group
            )
            assert all([chunk_num == n for n in all_chunk_nums])
            self._vpp_chunk_num = chunk_num
        else:
            assert self._vpp_chunk_num == chunk_num

        sum_square_metas = []
        for p, g in params_grads:
            if g is None:
                continue

            not_shared_enable = (not hasattr(p, 'is_firstly_shared')) or (
                hasattr(p, 'is_firstly_shared')
                and getattr(p, 'is_firstly_shared', True)
            )

            chunk_id = p._chunk_info['chunk_id']
            if not_shared_enable:
                if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                    merge_grad = clip.merge_selected_rows(g)
                    g = clip.get_tensor_from_selected_rows(merge_grad)
                square = paddle.square(g)
                sum_square = paddle.sum(square)
                layer_id = chunk_id * pp_size + pp_rank
                sum_square_metas.append(
                    [layer_id, p.is_distributed, sum_square.numpy()]
                )

        all_sum_square_metas = []
        paddle.distributed.all_gather_object(
            all_sum_square_metas,
            sum_square_metas,
            group=pp_group,
        )

        # order: FP16, BF16, FP32
        sum_square_dist = [[], [], []]
        sum_square_not_dist = [[], [], []]

        pp_stage = self._hcg.get_stage_id()
        for i, metas in enumerate(all_sum_square_metas):
            for layer_id, is_distributed, sum_square in metas:
                rank = layer_id // chunk_num
                assert rank < pp_size
                if rank != pp_rank:
                    continue
                if sum_square.dtype == np.float32:
                    idx = 2
                elif sum_square.dtype == np.float16:
                    idx = 0
                else:
                    assert (
                        sum_square.dtype == np.uint16
                    ), "The data type of grad must be FP32, FP16 or BF16, but got {}".format(
                        sum_square.dtype
                    )
                    idx = 1

                if is_distributed:
                    sum_square_dist[idx].append(sum_square)
                else:
                    sum_square_not_dist[idx].append(sum_square)

        global_norm_var_dist = self._add_sum_squares(sum_square_dist)
        global_norm_var_not_dist = self._add_sum_squares(sum_square_not_dist)

        return self._comm_and_clip(
            params_grads, global_norm_var_dist, global_norm_var_not_dist
        )

    def _add_sum_squares(self, sum_squares):
        norm_sum = None
        for sq in sum_squares:
            if len(sq) == 0:
                continue

            sq = np.concatenate(sq, axis=0).flatten()
            sq = paddle.to_tensor(sq)
            sq = paddle.sum(sq)
            if sq.dtype != paddle.float32:
                sq = sq.astype(paddle.float32)

            if norm_sum is None:
                norm_sum = sq
            else:
                norm_sum = norm_sum + sq

        if norm_sum is None:
            norm_sum = paddle.to_tensor([0.0], dtype=paddle.float32)

        return norm_sum

    @no_grad()
    def _dygraph_clip(self, params_grads):

        if self._timers:
            self._timers("dygraph-clip").start()

        if self._force_align_vpp_grad_sum_order:
            chunk_num = self._get_vpp_chunk_num(params_grads)
            if chunk_num > 0:
                return self._vpp_dygraph_clip(params_grads, chunk_num)

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
                [0.0], dtype=paddle.float32
            )
        else:
            global_norm_dist_fp16 = paddle.concat(sum_square_dist_fp16)
            global_norm_dist_fp16 = paddle.sum(global_norm_dist_fp16)
            global_norm_dist_fp16 = paddle.cast(
                global_norm_dist_fp16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_not_dist_fp16) == 0:
            global_norm_not_dist_fp16 = paddle.to_tensor(
                [0.0], dtype=paddle.float32
            )
        else:
            global_norm_not_dist_fp16 = paddle.concat(sum_square_not_dist_fp16)
            global_norm_not_dist_fp16 = paddle.sum(global_norm_not_dist_fp16)
            global_norm_not_dist_fp16 = paddle.cast(
                global_norm_not_dist_fp16, dtype=paddle.float32
            )

        # global norm of distributed BF16 params_and_grads
        if len(sum_square_dist_bf16) == 0:
            global_norm_dist_bf16 = paddle.to_tensor(
                [0.0], dtype=paddle.float32
            )
        else:
            global_norm_dist_bf16 = paddle.concat(sum_square_dist_bf16)
            global_norm_dist_bf16 = paddle.sum(global_norm_dist_bf16)
            global_norm_dist_bf16 = paddle.cast(
                global_norm_dist_bf16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_not_dist_bf16) == 0:
            global_norm_not_dist_bf16 = paddle.to_tensor(
                [0.0], dtype=paddle.float32
            )
        else:
            global_norm_not_dist_bf16 = paddle.concat(sum_square_not_dist_bf16)
            global_norm_not_dist_bf16 = paddle.sum(global_norm_not_dist_bf16)
            global_norm_not_dist_bf16 = paddle.cast(
                global_norm_not_dist_bf16, dtype=paddle.float32
            )

        # global norm of distributed FP32 params_and_grads
        global_norm_dist_fp32 = (
            paddle.concat(sum_square_dist_fp32)
            if len(sum_square_dist_fp32) != 0
            else paddle.to_tensor([0.0], dtype=paddle.float32)
        )
        global_norm_dist_fp32 = paddle.sum(global_norm_dist_fp32)

        # global norm of non-distributed FP32 params_and_grads
        global_norm_not_dist_fp32 = (
            paddle.concat(sum_square_not_dist_fp32)
            if len(sum_square_not_dist_fp32) != 0
            else paddle.to_tensor([0.0], dtype=paddle.float32)
        )
        global_norm_not_dist_fp32 = paddle.sum(global_norm_not_dist_fp32)

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

        result = self._comm_and_clip(
            params_grads, global_norm_var_dist, global_norm_var_not_dist
        )
        if self._timers:
            self._timers("dygraph-clip").stop()

        return result

    def _global_norm(self, global_norm_var_dist, global_norm_var_not_dist):
        sharding_flag = self._hcg.get_sharding_parallel_world_size() > 1
        dp_flag = self._hcg.get_data_parallel_world_size() > 1
        mp_flag = self._hcg.get_model_parallel_world_size() > 1
        pp_flag = self._hcg.get_pipe_parallel_world_size() > 1

        # not g_shard_norm_align_dp, grads are sharded among sharding group
        if sharding_flag and not g_shard_norm_align_dp:
            # norm of mp distributed variable
            if mp_flag:
                # dist should reduce among sharding group and mp group、pp group latter
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_sharding_parallel_group(),
                )
            # not dist only reduce among sharding group and pp group later
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_sharding_parallel_group(),
            )

        # norm of mp distributed variable
        if mp_flag:
            # the else branch would suffice, but this branch remains here for number precision backward compatibility
            if not (dp_flag and sharding_flag):
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_check_parallel_group(sharding_flag),
                )
            else:
                # global_norm_var_dist should all reduce among model parallel group and pp group
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_model_parallel_group(),
                )
                if pp_flag:
                    paddle.distributed.all_reduce(
                        global_norm_var_dist,
                        group=self._hcg.get_pipe_parallel_group(),
                    )

        # add all reduce to get global norm of non-distributed params_and_grads in groups of pp
        if pp_flag:
            paddle.distributed.all_reduce(
                global_norm_var_not_dist,
                group=self._hcg.get_pipe_parallel_group(),
            )

    def _comm_and_clip(
        self, params_grads, global_norm_var_dist, global_norm_var_not_dist
    ):

        self._global_norm(global_norm_var_dist, global_norm_var_not_dist)

        global_norm_var_fp32 = paddle.sqrt(
            global_norm_var_dist + global_norm_var_not_dist
        )

        max_global_norm = paddle.full(
            shape=[1],
            dtype=global_norm_var_fp32.dtype,
            fill_value=self.clip_norm,
        )
        clip_var = paddle.divide(
            x=max_global_norm,
            y=paddle.maximum(x=global_norm_var_fp32, y=max_global_norm)
            + paddle.to_tensor([1.0e-6], dtype=paddle.float32),
        )
        clip_var_fp16 = paddle.cast(clip_var, paddle.float16)

        # bf16 is not supported on XPU now
        if not paddle.is_compiled_with_xpu():
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
        # Note: Only sharding stage 1 is considered in HybridParallelOptimizer.
        # The sharding stage2 and stage3 optimizers are invoked in other api.
        if hcg.get_sharding_parallel_world_size() > 1:
            split_param = strategy.hybrid_configs[
                'sharding_configs'
            ].split_param
            ShardingOptimizer = (
                DygraphShardingOptimizerV2
                if split_param
                else DygraphShardingOptimizer
            )
            optimizer = ShardingOptimizer(optimizer, hcg)

        self._enable_timer = strategy.hybrid_configs["enable_optimizer_timer"]

        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self._timers = timer.get_timers()
        else:
            self._timers = None

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
            inner_opt = unwrap_optimizer(
                self._inner_opt,
                (
                    MixPrecisionOptimizer,
                    DygraphShardingOptimizer,
                    DygraphShardingOptimizerV2,
                ),
            )

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
                    inner_opt._grad_clip, hcg, self._timers
                )
            else:
                inner_opt._grad_clip = HybridParallelClipGrad(
                    inner_opt._grad_clip, hcg, self._timers
                )
                if inner_opt._parameter_list and isinstance(
                    inner_opt._parameter_list[0], dict
                ):
                    for item in inner_opt._param_groups:
                        if "grad_clip" in item.keys():
                            item["grad_clip"] = HybridParallelClipGrad(
                                inner_opt._grad_clip, hcg, self._timers
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
        parameter_list = list(obtain_optimizer_parameters_list(self._inner_opt))
        dp_parameter_list = parameter_list
        if self._sharding_enable:
            assert isinstance(
                self._inner_opt,
                (DygraphShardingOptimizer, DygraphShardingOptimizerV2),
            )
            self._inner_opt.reduce_gradients(parameter_list, self._hcg)
            # dp sync later do not need to use global parameter list
            if not g_shard_norm_align_dp:
                dp_parameter_list = self._inner_opt.filter_parameters(
                    parameter_list, self._hcg
                )

        if self._dp_enable:
            fused_allreduce_gradients(dp_parameter_list, self._hcg)

        self._step(parameter_list)

    @no_grad()
    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):

        # minimize does not support parameters in the form of param_group,
        # so no need use _obtain_optimizer_parameters_list
        parameter_list = (
            parameters if parameters else self._inner_opt._parameter_list
        )
        parameter_list = list(parameter_list)
        dp_parameter_list = parameter_list
        # Here sharding should use global parameter list
        if self._sharding_enable:
            assert isinstance(
                self._inner_opt,
                (DygraphShardingOptimizer, DygraphShardingOptimizerV2),
            )
            self._inner_opt.reduce_gradients(parameter_list, self._hcg)
            # dp sync later do not need to use global parameter list
            if not g_shard_norm_align_dp:
                dp_parameter_list = self._inner_opt.filter_parameters(
                    parameter_list, self._hcg
                )

        if self._dp_enable:
            fused_allreduce_gradients(dp_parameter_list, self._hcg)

        return self._inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set
        )

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
