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


import copy
import os
from collections import defaultdict

import paddle
import paddle.distributed as dist
from paddle import framework
from paddle.autograd import no_grad
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.dygraph_sharding_optimizer import (
    DygraphShardingOptimizer,
    DygraphShardingOptimizerV2,
)
from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    MuonShardingOptimizer,
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
from ...utils.log_util import get_sync_logger, logger
from ...utils.mix_precision_utils import MixPrecisionOptimizer

g_profile_optimizer_details_steps = int(
    os.getenv("FLAGS_profile_optimizer_details_steps", "0")
)

__all__ = []

SHARED_WEIGHT_SYNC_PREFIX = "@SHARED_WEIGHT"


class HybridParallelClipGrad:
    def __init__(self, clip, hcg, split_norm_comm=False, timers=None):
        self._clip = clip
        self._hcg = hcg
        self.not_sharding_stage1 = True
        self._timers = timers
        self.processed_steps = 0
        self.split_norm_comm = split_norm_comm

    def _global_norm(self, global_norm_var_dist, global_norm_var_not_dist):
        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting to calculate global norm.")
        # sharding first
        sharding_flag = self._hcg.get_sharding_parallel_world_size() > 1
        dp_flag = self._hcg.get_data_parallel_world_size() > 1
        mp_flag = self._hcg.get_model_parallel_world_size() > 1
        pp_flag = self._hcg.get_pipe_parallel_world_size() > 1

        # add all reduce to get global norm of distributed params_and_grads
        if sharding_flag:
            # norm of mp distributed variable
            # dist should reduce among sharding group, mp group, pp group
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
            # dist should reduce among sharding group, mp group, pp group

            # the else branch would suffice, but this branch remains here for number precision backward compatibility
            if not (dp_flag and sharding_flag) and not self.split_norm_comm:
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_check_parallel_group(sharding_flag),
                )
            else:
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

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finished calculating global norm.")
        self.processed_steps += 1

    @no_grad()
    def _dygraph_clip(self, params_grads):
        if self._timers:
            self._timers("dygraph-clip").start()
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
            sum_square = clip._squared_l2_norm(merge_grad)

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

        def async_add_n(var_list):
            return paddle.stack(var_list).sum()

        # global norm of distributed FP16 params_and_grads
        if len(sum_square_dist_fp16) == 0:
            global_norm_dist_fp16 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_dist_fp16 = async_add_n(sum_square_dist_fp16)
            global_norm_dist_fp16 = paddle.cast(
                global_norm_dist_fp16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_not_dist_fp16) == 0:
            global_norm_not_dist_fp16 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_not_dist_fp16 = async_add_n(sum_square_not_dist_fp16)
            global_norm_not_dist_fp16 = paddle.cast(
                global_norm_not_dist_fp16, dtype=paddle.float32
            )

        # global norm of distributed BF16 params_and_grads
        if len(sum_square_dist_bf16) == 0:
            global_norm_dist_bf16 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_dist_bf16 = async_add_n(sum_square_dist_bf16)
            global_norm_dist_bf16 = paddle.cast(
                global_norm_dist_bf16, dtype=paddle.float32
            )

        # global norm of non-distributed FP16 params_and_grads
        if len(sum_square_not_dist_bf16) == 0:
            global_norm_not_dist_bf16 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_not_dist_bf16 = async_add_n(sum_square_not_dist_bf16)
            global_norm_not_dist_bf16 = paddle.cast(
                global_norm_not_dist_bf16, dtype=paddle.float32
            )

        # global norm of distributed FP32 params_and_grads
        if len(sum_square_dist_fp32) == 0:
            global_norm_dist_fp32 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_dist_fp32 = async_add_n(sum_square_dist_fp32)

        # global norm of non-distributed FP32 params_and_grads
        if len(sum_square_not_dist_fp32) == 0:
            global_norm_not_dist_fp32 = paddle.zeros((1,), dtype=paddle.float32)
        else:
            global_norm_not_dist_fp32 = async_add_n(sum_square_not_dist_fp32)

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

    def _comm_and_clip(
        self, params_grads, global_norm_var_dist, global_norm_var_not_dist
    ):
        self._global_norm(global_norm_var_dist, global_norm_var_not_dist)

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
            + paddle.full(shape=[], dtype=paddle.float32, fill_value=1.0e-6),
        )
        clip_var_fp16 = paddle.cast(clip_var, paddle.float16)

        if not isinstance(
            paddle.framework._current_expected_place(), paddle.CustomPlace
        ) or paddle.framework._current_expected_place().get_device_type() in [
            'npu',
            'iluvatar_gpu',
            'metax_gpu',
        ]:
            clip_var_bf16 = paddle.cast(clip_var, paddle.bfloat16)
        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, 'need_clip', True) is False:
                continue
            if g.dtype == paddle.float16:
                g.multiply_(clip_var_fp16)
            elif g.dtype == paddle.bfloat16:
                g.multiply_(clip_var_bf16)
            else:
                g.multiply_(clip_var)
            p._reset_grad_inplace_version(True)

        return params_grads

    def __getattr__(self, item):
        return getattr(self._clip, item)

    def __call__(self, params_grads):
        return self._dygraph_clip(params_grads)


class ClipGradByAdaptiveNorm(clip.ClipGradBase):
    """AdaGC: https://arxiv.org/abs/2502.11034"""

    def __init__(
        self,
        clip_ratio=1.04,
        start_clip_steps=100,
        beta=0.99,
        epsilon=1e-8,
        shard_clip=False,
        enable_record=False,
        enable_record_clip_history=False,
        verbose=False,
    ):
        super().__init__()
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.epsilon = epsilon
        self.state = defaultdict(dict)
        self.start_clip_steps = start_clip_steps
        self.shard_clip = shard_clip
        self.enable_record = enable_record
        self.steps = 0
        self.enable_record_clip_history = enable_record_clip_history
        self.verbose = verbose
        self.keys = [
            "clip_ratio",
            "beta",
            "epsilon",
            "start_clip_steps",
            "shard_clip",
            "enable_record",
            "steps",
            "enable_record_clip_history",
        ]

        self._hcg = fleet.get_hybrid_communicate_group()
        # When moe-group is dp-group, extra grad-norm communication is needed; otherwise `HybridParallelClipGrad` already handles grad-norm communication
        if (
            hasattr(self._hcg, "get_moe_sharding_parallel_world_size")
            and self._hcg.get_moe_sharding_parallel_world_size() > 0
        ):
            # hybrid expert parallel
            self.moe_group = self._hcg.get_expert_parallel_group()
            self.moe_sharding_group = (
                self._hcg.get_moe_sharding_parallel_group()
            )
        else:
            self.moe_group = (
                self._hcg.get_data_parallel_group()
                if self._hcg.get_data_parallel_world_size()
                else None
            )
            self.moe_sharding_group = None
        self.stat = {}  # for logging

        self.need_sync = False
        if not self.shard_clip:
            self.mp_size = self._hcg.get_model_parallel_world_size()
            self.mp_group = self._hcg.get_model_parallel_group()
            self.sharding_size = self._hcg.get_sharding_parallel_world_size()
            self.sharding_group = self._hcg.get_sharding_parallel_group()
            if self.mp_size > 1 or self.sharding_size > 1:
                self.need_sync = True

        if start_clip_steps < 0:
            raise ValueError(
                f"start_clip_steps {start_clip_steps}, please start_clip_steps >= 0."
            )

    def __str__(self):
        return f"ClipGradByAdaptiveNorm, clip_ratio={self.clip_ratio}, beta={self.beta}, start_clip_steps={self.start_clip_steps}, \
            shard_clip={self.shard_clip}, enable_record={self.enable_record}"

    def _clip_by_norm(self, param, grad, norm_value, global_norm):
        """
        Core clipping function
        """
        state = self.state[param.name]

        if "norm_value" not in state:
            state["norm_value"] = norm_value

        if "clip_times" not in state:
            state["clip_times"] = 0

        if self.enable_record_clip_history:
            if "clip_history" not in state:
                state["clip_history"] = {}

        avg_norm_value = state["norm_value"]

        if self.enable_record:
            if "norm_history" not in state:
                state["norm_history"] = {}
            state["norm_history"][self.steps] = [
                float(norm_value),
                float(avg_norm_value),
            ]

        if self.steps <= self.start_clip_steps:
            clip_coeff = 1.0 / (global_norm + self.epsilon)
            if clip_coeff < 1.0:
                grad.multiply_(clip_coeff)
                param._reset_grad_inplace_version(True)

            if norm_value < state["norm_value"]:
                state["norm_value"] = norm_value
        else:
            if norm_value > self.clip_ratio * avg_norm_value:
                # clip grad
                coef = (self.clip_ratio * avg_norm_value) / (
                    norm_value + self.epsilon
                )
                grad.multiply_(coef)
                param._reset_grad_inplace_version(True)
                norm_value_old = norm_value
                norm_value = self.clip_ratio * avg_norm_value
                state["clip_times"] = state["clip_times"] + 1
                if self.enable_record_clip_history:
                    state["clip_history"][self.steps] = [
                        float(norm_value_old),
                        float(norm_value),
                    ]
                if self.verbose:
                    logger.info(
                        "{} gradclip {} times, clip from {} to {}".format(
                            param.name,
                            state["clip_times"],
                            float(norm_value_old),
                            float(norm_value),
                        )
                    )

                    logger.info(
                        "{} steps {}, gradclip {} times, clip_ratio {}, clip from {} to {}".format(
                            param.name,
                            self.steps,
                            state["clip_times"],
                            self.clip_ratio,
                            float(norm_value_old),
                            float(norm_value),
                        )
                    )
            state["norm_value"] = avg_norm_value * self.beta + norm_value * (
                1.0 - self.beta
            )

        return grad

    def _global_norm(
        self,
        global_norm_var_dist,
        global_norm_var_not_dist,
        global_norm_var_dist_moe,
        global_norm_var_not_dist_moe,
    ):

        # sharding first
        sharding_flag = self._hcg.get_sharding_parallel_world_size() > 1
        dp_flag = self._hcg.get_data_parallel_world_size() > 1
        mp_flag = self._hcg.get_model_parallel_world_size() > 1
        pp_flag = self._hcg.get_pipe_parallel_world_size() > 1

        """do comm"""
        logger.info(
            f"before reduce: dist-moe-grad-norm={global_norm_var_dist_moe.item()} "
            f"before reduce: non-dist-moe-grad-norm={global_norm_var_not_dist_moe.item()}"
        )

        if self.moe_sharding_group:
            dist.all_reduce(
                global_norm_var_dist_moe,
                op=dist.ReduceOp.SUM,
                group=self.moe_sharding_group,
            )
            dist.all_reduce(
                global_norm_var_not_dist_moe,
                op=dist.ReduceOp.SUM,
                group=self.moe_sharding_group,
            )

        if self.moe_group:
            dist.all_reduce(
                global_norm_var_dist_moe,
                op=dist.ReduceOp.SUM,
                group=self.moe_group,
            )
            dist.all_reduce(
                global_norm_var_not_dist_moe,
                op=dist.ReduceOp.SUM,
                group=self.moe_group,
            )

        if pp_flag:
            paddle.distributed.all_reduce(
                global_norm_var_dist_moe,
                group=self._hcg.get_pipe_parallel_group(),
            )
            paddle.distributed.all_reduce(
                global_norm_var_not_dist_moe,
                group=self._hcg.get_pipe_parallel_group(),
            )

        logger.info(
            f"after reduce: dist-moe-grad-norm={global_norm_var_dist_moe.item()} "
            f"after reduce: non-dist-moe-grad-norm={global_norm_var_not_dist_moe.item()}"
        )

        logger.info(
            f"before reduce: dist-grad-norm={global_norm_var_dist.item()} "
            f"before reduce: non-dist-grad-norm={global_norm_var_not_dist.item()}"
        )
        # add all reduce to get global norm of distributed params_and_grads
        if sharding_flag:
            # norm of mp distributed variable
            if mp_flag:
                # dist should reduce among sharding group, mp group, pp group
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
            # dist should reduce among sharding group, mp group, pp group

            # the else branch would suffice, but this branch remains here for number precision backward compatibility
            # Note (GuoxiaWang): don't use this branch
            if False:  # not (dp_flag and sharding_flag):
                paddle.distributed.all_reduce(
                    global_norm_var_dist,
                    group=self._hcg.get_check_parallel_group(sharding_flag),
                )
            else:
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

        logger.info(
            f"after reduce: dist-grad-norm={global_norm_var_dist.item()} "
            f"after reduce: non-dist-grad-norm={global_norm_var_not_dist.item()}"
        )

    def _comm_and_clip(
        self,
        params_grads,
        global_norm_var_dist,
        global_norm_var_not_dist,
        global_norm_var_dist_moe,
        global_norm_var_not_dist_moe,
    ):
        self._global_norm(
            global_norm_var_dist,
            global_norm_var_not_dist,
            global_norm_var_dist_moe,
            global_norm_var_not_dist_moe,
        )

        global_norm_var_fp32 = paddle.sqrt(
            global_norm_var_dist
            + global_norm_var_not_dist
            + global_norm_var_dist_moe
            + global_norm_var_not_dist_moe
        )

        global_norm_var_fp32 = global_norm_var_fp32.astype("float32")
        return global_norm_var_fp32

    @no_grad()
    def _collect_norm(self, params_grads):
        sum_square_dist_fp16 = []
        sum_square_dist_bf16 = []
        sum_square_dist_fp32 = []

        sum_square_dist_moe_fp16 = []
        sum_square_dist_moe_bf16 = []
        sum_square_dist_moe_fp32 = []

        sum_square_not_dist_fp16 = []
        sum_square_not_dist_bf16 = []
        sum_square_not_dist_fp32 = []

        sum_square_not_dist_moe_fp16 = []
        sum_square_not_dist_moe_bf16 = []
        sum_square_not_dist_moe_fp32 = []

        for p, g in params_grads:
            if g is None:
                continue
            if getattr(p, "need_clip", True) is False:
                continue
            merge_grad = g
            if g.type == core.VarDesc.VarType.SELECTED_ROWS:
                merge_grad = clip.merge_selected_rows(g)
                merge_grad = clip.get_tensor_from_selected_rows(merge_grad)
            sum_square = clip._squared_l2_norm(merge_grad)

            not_shared_enable = (not hasattr(p, "is_firstly_shared")) or (
                hasattr(p, "is_firstly_shared")
                and getattr(p, "is_firstly_shared", True)
            )

            if not_shared_enable:
                if getattr(p, "no_sync", False):
                    if p.is_distributed:
                        if g.dtype == paddle.float16:
                            sum_square_dist_moe_fp16.append(sum_square)
                        elif g.dtype == paddle.bfloat16:
                            sum_square_dist_moe_bf16.append(sum_square)
                        elif g.dtype == paddle.float32:
                            sum_square_dist_moe_fp32.append(sum_square)
                    else:
                        if g.dtype == paddle.float16:
                            sum_square_not_dist_moe_fp16.append(sum_square)
                        elif g.dtype == paddle.bfloat16:
                            sum_square_not_dist_moe_bf16.append(sum_square)
                        elif g.dtype == paddle.float32:
                            sum_square_not_dist_moe_fp32.append(sum_square)

                elif p.is_distributed:
                    if g.dtype == paddle.float16:
                        sum_square_dist_fp16.append(sum_square)
                    elif g.dtype == paddle.bfloat16:
                        sum_square_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_dist_fp32.append(sum_square)
                else:
                    assert not getattr(p, "no_sync", False), (
                        f"moe param should be distributed, got: {p.name}, shape={p.shape}"
                    )
                    if g.dtype == paddle.float16:
                        sum_square_not_dist_fp16.append(sum_square)
                    if g.dtype == paddle.bfloat16:
                        sum_square_not_dist_bf16.append(sum_square)
                    elif g.dtype == paddle.float32:
                        sum_square_not_dist_fp32.append(sum_square)
            else:
                assert not getattr(p, "no_sync", False), (
                    "moe don't know share param"
                )

        def add_n_list(tensor_list):
            if not tensor_list:
                return paddle.zeros((1,), dtype=paddle.float32)
            return paddle.add_n(tensor_list).cast(paddle.float32)

        # moe global norm of distributed FP16 params_and_grads
        global_norm_dist_moe_fp16 = add_n_list(
            sum_square_dist_moe_fp16,
        )
        global_norm_not_dist_moe_fp16 = add_n_list(
            sum_square_not_dist_moe_fp16,
        )
        global_norm_dist_fp16 = add_n_list(
            sum_square_dist_fp16,
        )
        global_norm_not_dist_fp16 = add_n_list(
            sum_square_not_dist_fp16,
        )

        global_norm_dist_moe_bf16 = add_n_list(
            sum_square_dist_moe_bf16,
        )
        global_norm_not_dist_moe_bf16 = add_n_list(
            sum_square_not_dist_moe_bf16,
        )
        global_norm_dist_bf16 = add_n_list(
            sum_square_dist_bf16,
        )
        global_norm_not_dist_bf16 = add_n_list(
            sum_square_not_dist_bf16,
        )

        global_norm_dist_moe_fp32 = add_n_list(
            sum_square_dist_moe_fp32,
        )
        global_norm_not_dist_moe_fp32 = add_n_list(
            sum_square_not_dist_moe_fp32,
        )
        global_norm_dist_fp32 = add_n_list(
            sum_square_dist_fp32,
        )
        global_norm_not_dist_fp32 = add_n_list(
            sum_square_not_dist_fp32,
        )

        global_norm_var_dist_moe = (
            global_norm_dist_moe_fp16
            + global_norm_dist_moe_bf16
            + global_norm_dist_moe_fp32
        )

        global_norm_var_not_dist_moe = (
            global_norm_not_dist_moe_fp16
            + global_norm_not_dist_moe_bf16
            + global_norm_not_dist_moe_fp32
        )

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
        global_norm_var_fp32 = self._comm_and_clip(
            params_grads,
            global_norm_var_dist,
            global_norm_var_not_dist,
            global_norm_var_dist_moe,
            global_norm_var_not_dist_moe,
        )

        return global_norm_var_fp32

    @no_grad()
    def _dygraph_clip(self, params_grads):
        global_norm_tensor = self._collect_norm(params_grads)
        self.stat["global_grad_norm"] = global_norm_tensor.item()

        if self.verbose and global_norm_tensor is not None:
            logger.info(
                f"step: {self.steps}, global norm: {float(global_norm_tensor)}"
            )

        assert self.sharding_stage1_v2, "AdaGC only supports ShardingV2 now!"

        # Initialize per-group norm_tensor containers
        group_idx_to_norm_tensor = {}
        for group_idx, num_params in self.group_idx_to_param_num.items():
            # Create num_params zero FloatTensors
            group_idx_to_norm_tensor[group_idx] = [
                paddle.zeros([1], dtype=paddle.float32)
                for _ in range(num_params)
            ]

        # Step 2&3: Iterate over param-grad list, compute each grad's norm, and store into the corresponding norm_tensor
        for param, grad in params_grads:
            if grad is None:
                continue
            if getattr(param, "need_clip", True) is False:
                continue

            group_idx = self.param_name_to_group_idx[param.name]
            local_idx = self.group_index_to_name_to_local_idx[group_idx][
                param.name
            ]

            assert grad.dtype == paddle.float32, (
                "AdaGC not support main_grad now"
            )
            norm_squared_value = clip._squared_l2_norm(grad)

            comm_group = self.group_idx_to_comm_group[group_idx]
            if (
                self.need_sync
                and param.is_distributed
                and comm_group is not self.moe_group
            ):
                # Dense params: can be TP-sharded or non-sharded
                # For non-sharded params, divide by TP size first for unified TP communication later
                group_idx_to_norm_tensor[group_idx][local_idx] = (
                    norm_squared_value / self.mp_size
                )
            else:
                # For expert params, can directly do all-reduce
                # For TP params, can also directly do all-reduce
                group_idx_to_norm_tensor[group_idx][local_idx] = (
                    norm_squared_value
                )

        # Step 4: Reduce norm_tensor for each group
        for group_idx, norm_tensor in group_idx_to_norm_tensor.items():
            num = len(norm_tensor)
            norm_tensor = paddle.concat(norm_tensor, axis=0)
            comm_group = self.group_idx_to_comm_group[group_idx]
            if (
                comm_group is self.moe_group
                and self._hcg.get_moe_sharding_parallel_world_size() > 1
            ):
                dist.all_reduce(norm_tensor, group=comm_group)
            else:
                if self.need_sync and self.mp_size > 1:
                    dist.all_reduce(norm_tensor, group=self.mp_group)
                if self.sharding_size > 1:
                    dist.all_reduce(norm_tensor, group=comm_group)

            group_idx_to_norm_tensor[group_idx] = paddle.split(
                paddle.sqrt(norm_tensor), num, axis=0
            )

        params_and_grads = []
        for param, grad in params_grads:
            if grad is None:
                continue
            if getattr(param, "need_clip", True) is False:
                continue
            group_idx = self.param_name_to_group_idx[param.name]
            local_idx = self.group_index_to_name_to_local_idx[group_idx][
                param.name
            ]
            norm_value = group_idx_to_norm_tensor[group_idx][local_idx]

            new_grad = self._clip_by_norm(
                param, grad, norm_value, global_norm_tensor
            )
            params_and_grads.append((param, new_grad))

        self.steps += 1
        return params_and_grads

    @framework.dygraph_only
    def state_dict(self):
        """
        Get state dict
        """
        state_dict = dict(self.state.items())
        for key in self.keys:
            state_dict[key] = self.__dict__[key]
        return state_dict

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        """
        Set state dict
        """
        if len(state_dict) == 0 or state_dict is None:
            logger.info("state_dict is empty, please check if it is right.")

        for key in self.keys:
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
            else:
                logger.info(f"Can't find [ {key} ] in state_dict")

        for k in state_dict:
            if k in self.keys:
                continue
            self.state[k] = copy.deepcopy(state_dict[k])


class HybridParallelOptimizer:
    # adapter wrapper for optimizer
    def __init__(self, optimizer, hcg, strategy):
        # Note: Only sharding stage 1 is considered in HybridParallelOptimizer.
        # The sharding stage2 and stage3 optimizers are invoked in other api.
        if hcg.get_sharding_parallel_world_size() > 1:
            split_param = strategy.hybrid_configs[
                'sharding_configs'
            ].split_param
            use_muon_sharding = getattr(strategy, "use_muon_sharding", False)
            if use_muon_sharding:
                ShardingOptimizer = MuonShardingOptimizer
            elif split_param:
                ShardingOptimizer = DygraphShardingOptimizerV2
            else:
                ShardingOptimizer = DygraphShardingOptimizer
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

        self._sep_enable = self._hcg.get_sep_parallel_world_size() > 1

        split_norm_comm = strategy.hybrid_configs["split_norm_comm"]

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
                    MuonShardingOptimizer,
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
                    inner_opt._grad_clip, hcg, split_norm_comm, self._timers
                )
            else:
                inner_opt._grad_clip = HybridParallelClipGrad(
                    inner_opt._grad_clip, hcg, split_norm_comm, self._timers
                )
                if inner_opt._parameter_list and isinstance(
                    inner_opt._parameter_list[0], dict
                ):
                    for item in inner_opt._param_groups:
                        if "grad_clip" in item.keys():
                            item["grad_clip"] = HybridParallelClipGrad(
                                inner_opt._grad_clip,
                                hcg,
                                split_norm_comm,
                                self._timers,
                            )
        self.processed_steps = 0

    def _set_all_gather_overlap_forward(
        self, all_gather_overlap_forward, layers=None
    ):
        self._all_gather_overlap_forward = all_gather_overlap_forward
        if self._all_gather_overlap_forward:
            self._layers = layers
            self._inner_opt._set_all_gather_overlap_forward(
                self._all_gather_overlap_forward, self._layers
            )

    def _set_broadcast_overlap(self, broadcast_overlap, layers=None):
        self._broadcast_overlap = broadcast_overlap
        if self._broadcast_overlap:
            self._layers = layers
            self._inner_opt._set_broadcast_overlap(
                self._broadcast_overlap, self._layers
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
            sync_var.multiply_(
                paddle.full(
                    shape=[],
                    dtype=sync_var.dtype,
                    fill_value=(1.0 / mp_group.nranks),
                )
            )

    def _pp_filter_fn(self, param):
        color = getattr(param, 'color', -1)
        if isinstance(color, dict):
            color_color = color.get('color', -1)
            if SHARED_WEIGHT_SYNC_PREFIX in str(color_color):
                return True
        return False

    def _mp_filter_fn(self, param, strategy):
        p_name = param.name
        tar_param = strategy.sync_param_name
        if param.is_distributed is False:
            for tar in tar_param:
                if tar in p_name:
                    return True
        return False

    def syc_grad(self, param, src_rank, group, sync_mode):
        if hasattr(param, "main_grad") and param.main_grad is not None:
            assert param.grad is None
            self._insert_sync(param.main_grad, src_rank, group, sync_mode)
        elif param.grad is not None:
            self._insert_sync(param.grad, src_rank, group, sync_mode)

    def syc_param(self, param, src_rank, group, sync_mode):
        # Param sync after opt
        self._insert_sync(param, src_rank, group, sync_mode)

    def syc_master_weight(self, param, src_rank, group, sync_mode):
        # Master param sync after opt
        if (
            hasattr(self._inner_opt, "_multi_precision")
            and self._inner_opt._multi_precision
            and param.name in self._inner_opt._master_weights
        ):
            self._insert_sync(
                self._inner_opt._master_weights[param.name],
                src_rank,
                group,
                sync_mode,
            )

    def syc_moment(self, param, src_rank, group, sync_mode):
        _OPTIMIZER_TYPES = (paddle.optimizer.Adam, paddle.optimizer.AdamW)

        def recursive_isinstance(opt):
            return isinstance(opt, _OPTIMIZER_TYPES) or (
                hasattr(opt, "_inner_opt")
                and recursive_isinstance(opt._inner_opt)
            )

        if recursive_isinstance(self._inner_opt):
            if (
                param.name
                in self._inner_opt._accumulators[
                    self._inner_opt._moment1_acc_str
                ]
            ):
                moment1 = self._inner_opt._get_accumulator(
                    self._inner_opt._moment1_acc_str, param
                )
                self._insert_sync(moment1, src_rank, group, sync_mode)

            if (
                param.name
                in self._inner_opt._accumulators[
                    self._inner_opt._moment2_acc_str
                ]
            ):
                moment2 = self._inner_opt._get_accumulator(
                    self._inner_opt._moment2_acc_str, param
                )
                self._insert_sync(moment2, src_rank, group, sync_mode)

    def _sync_mp_grads(self, params, mp_configs):
        mp_group = self._hcg.get_model_parallel_group()
        src_rank = self._hcg.get_model_parallel_group_src_rank()

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting mp grad sync")

        # Grad sync before opt
        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_grad:
            for p in params:
                self.syc_grad(p, src_rank, mp_group, mp_configs.sync_mode)

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finished mp grad sync")

    def _sync_mp_params_and_moments(self, params, mp_configs):
        mp_group = self._hcg.get_model_parallel_group()
        src_rank = self._hcg.get_model_parallel_group_src_rank()

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting mp params sync")

        # syc param and master weight after opt
        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_param:
            for p in params:
                self.syc_param(p, src_rank, mp_group, mp_configs.sync_mode)
                self.syc_master_weight(
                    p, src_rank, mp_group, mp_configs.sync_mode
                )

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finished mp params sync")
            get_sync_logger().info("Starting mp moments sync")

        # Moment sync after opt
        if mp_group.nranks > 1 and mp_configs and mp_configs.sync_moment:
            for p in params:
                self.syc_moment(p, src_rank, mp_group, mp_configs.sync_mode)
        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finished mp moments sync")

    def _get_pp_sync_params(self, parameters_list):
        pp_group = self._hcg.get_pipe_parallel_group()
        params = None
        pp_configs = None

        if pp_group.nranks > 1:
            pp_configs = fleet.fleet._user_defined_strategy.hybrid_configs[
                "pp_configs"
            ]

        if pp_configs and (pp_configs.sync_param or pp_configs.sync_moment):
            params = sorted(
                [p for p in parameters_list if self._pp_filter_fn(p)],
                key=lambda p: p.color["shared_weight_name"],
            )
        return params, pp_configs

    def _sync_pp_params_and_moments(self, params, pp_configs):
        pp_group = self._hcg.get_pipe_parallel_group()

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting pp param and master weight sync")

        # syc param and master weight after opt
        if pp_group.nranks > 1 and pp_configs and pp_configs.sync_param:
            for p in params:
                assert hasattr(p, 'color') and 'broadcast_group' in p.color, (
                    f"{p.name} has no color"
                )
                broadcast_group = p.color["broadcast_group"]
                src_rank = min(broadcast_group.ranks)
                self.syc_param(
                    p, src_rank, broadcast_group, pp_configs.sync_mode
                )
                self.syc_master_weight(
                    p, src_rank, broadcast_group, pp_configs.sync_mode
                )

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting pp param and master weight sync")
            get_sync_logger().info("Finished pp moments sync")

        # Moment sync after opt
        if pp_group.nranks > 1 and pp_configs and pp_configs.sync_moment:
            for p in params:
                assert hasattr(p, 'color') and 'broadcast_group' in p.color, (
                    f"{p.name} has no color"
                )
                broadcast_group = p.color["broadcast_group"]
                src_rank = min(broadcast_group.ranks)
                self.syc_moment(
                    p, src_rank, broadcast_group, pp_configs.sync_mode
                )

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finished pp moments sync")

    def _get_mp_sync_params(self, parameters_list):
        mp_group = self._hcg.get_model_parallel_group()
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
                    if self._mp_filter_fn(p, fleet.fleet._user_defined_strategy)
                ],
                key=lambda p: p.name,
            )
        return params, mp_configs

    def _step(self, parameters_list):
        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Starting hybridoptimizer step")

        # Sync non-model-parallel parameters' grads/weights/moments for MP group consistency.
        mp_params, mp_configs = self._get_mp_sync_params(parameters_list)
        # Sync PP shared params' weights and moments to ensure consistency within the PP group.
        # Note: Grads are synced in the pipeline parallel for compatibility.
        pp_params, pp_configs = self._get_pp_sync_params(parameters_list)

        self._sync_mp_grads(mp_params, mp_configs)

        self._inner_opt.step()

        self._sync_mp_params_and_moments(mp_params, mp_configs)
        self._sync_pp_params_and_moments(pp_params, pp_configs)

        if self.processed_steps < g_profile_optimizer_details_steps:
            get_sync_logger().info("Finishing hybridoptimizer step")
        self.processed_steps += 1

    def _hybrid_sync_grad(self, parameter_list):
        dp_parameter_list = parameter_list
        if self._sharding_enable:
            assert isinstance(
                self._inner_opt,
                (
                    DygraphShardingOptimizer,
                    DygraphShardingOptimizerV2,
                    MuonShardingOptimizer,
                ),
            )
            self._inner_opt.reduce_gradients(parameter_list, self._hcg)
            dp_parameter_list = self._inner_opt.filter_parameters(
                parameter_list, self._hcg
            )
        if self._dp_enable or self._sep_enable:
            fused_allreduce_gradients(dp_parameter_list, self._hcg)

    @no_grad()
    @framework.dygraph_only
    def step(self):
        parameter_list = list(obtain_optimizer_parameters_list(self._inner_opt))
        self._hybrid_sync_grad(parameter_list)
        self._step(parameter_list)

    @no_grad()
    def minimize(
        self, loss, startup_program=None, parameters=None, no_grad_set=None
    ):
        # minimize does not support parameters in the form of param_group,
        # so no need use _obtain_optimizer_parameters_list
        parameter_list = (
            parameters
            if parameters
            else obtain_optimizer_parameters_list(self._inner_opt)
        )
        parameter_list = list(parameter_list)
        self._hybrid_sync_grad(parameter_list)
        return self._inner_opt.minimize(
            loss, startup_program, parameter_list, no_grad_set
        )

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
