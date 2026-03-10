# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
DygraphShardingOptimizerV3: Hybrid Tensor-wise + Element-wise Sharding
=======================================================================

Designed for Muon optimizer compatibility:
  - 2D (Muon) parameters: assigned as *whole tensors* to ranks (like V1).
    This avoids the expensive sharding gather in Muon's _muon_update.
  - Non-2D (AdamW) parameters: split element-wise via reduce-scatter (like V2).
    This provides memory balancing across ranks.
  - MoE expert parameters: also 2D (Muon), assigned as whole tensors but
    using moe_sharding_group for communication instead of regular sharding_group.

The key insight is that Muon requires the full 2D matrix for Newton-Schulz
orthogonalisation, so keeping 2D params whole on each rank eliminates the
need for gather_varlen communication during the optimizer step.
"""

import os
import warnings
from collections import defaultdict
from functools import reduce as functools_reduce

import paddle
from paddle import framework
from paddle.base.framework import EagerParamBase
from paddle.distributed import fleet
from paddle.distributed.communication.reduce import (
    ReduceOp,
    is_avg_reduce_op_supported,
)
from paddle.distributed.fleet.utils.muon_comm_utils import should_use_muon

from ...utils import timer_helper as timer
from ...utils.log_util import logger
from ...utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
)


def _is_trainable(param):
    return not param.stop_gradient


class DygraphShardingOptimizerV3:
    """
    Hybrid sharding optimizer for Muon:
    - 2D (Muon) parameters: tensor-wise assignment to ranks (no cross-rank split).
      Gradient communication uses reduce; parameter sync uses broadcast.
    - Non-2D (AdamW) parameters: element-wise split across ranks (like V2).
      Gradient communication uses reduce-scatter; parameter sync uses all-gather.
    - MoE expert 2D parameters: tensor-wise assignment within moe_sharding_group.
      Uses separate communication group (moe_sharding_group) for reduce/broadcast.

    This avoids the expensive gather_varlen in Muon's _muon_update while
    maintaining memory balance across ranks.
    """

    def __init__(self, optimizer, hcg):
        logger.info("init DygraphShardingOptimizerV3")

        if isinstance(optimizer._parameter_list[0], dict):
            raise TypeError(
                "Do not support param_groups now, please set optimizer._parameter_list as a list of Parameter"
            )
        if not hasattr(optimizer, '_apply_optimize') or not callable(
            optimizer._apply_optimize
        ):
            raise ValueError(
                "the optimizer object should have _apply_optimize function"
            )

        self._inner_opt = optimizer
        self._hcg = hcg
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()

        strategy = fleet.fleet._user_defined_strategy
        sharding_configs = strategy.hybrid_configs['sharding_configs']

        self.tensor_fusion = sharding_configs.tensor_fusion
        self.accumulate_steps = sharding_configs.accumulate_steps
        self.comm_overlap = sharding_configs.comm_overlap
        self.comm_buffer_size_MB = sharding_configs.comm_buffer_size_MB
        self.use_reduce_avg = sharding_configs.use_reduce_avg

        if self.use_reduce_avg and (not is_avg_reduce_op_supported()):
            self.use_reduce_avg = False
            warnings.warn(
                "nccl reduce_avg requires paddle compiled with cuda and nccl>=2.10.0, "
                "please check compilation setups."
            )

        pp_overlap = strategy.hybrid_configs['pp_configs'].sharding_comm_overlap
        self.pp_overlap = pp_overlap

        self._use_main_grad = hasattr(optimizer._parameter_list[0], "main_grad")

        # The full original parameter list
        self._parameter_list = list(optimizer._parameter_list)
        self._origin_parameter_list = list(optimizer._parameter_list)

        # MoE sharding group info
        self._moe_sharding_group = None
        self._moe_sharding_world_size = 0
        self._moe_sharding_rank = 0
        if hasattr(hcg, "get_moe_sharding_parallel_world_size"):
            self._moe_sharding_world_size = (
                hcg.get_moe_sharding_parallel_world_size()
            )
            if self._moe_sharding_world_size > 0:
                self._moe_sharding_group = hcg.get_moe_sharding_parallel_group()
                self._moe_sharding_rank = self._moe_sharding_group.rank

        # ---- Step 1: Separate params into categories ----
        # - _params_2d:      Non-MoE 2D (Muon) params → whole tensor, sharding_group
        # - _params_2d_moe:  MoE expert params (2D or 3D fused) → whole tensor, moe_sharding_group
        #                    3D fused experts [n_experts, H, I] are split into per-expert
        #                    2D slices at optimizer step time for individual Muon update.
        # - _params_1d:      Non-MoE non-2D (AdamW) params → element-wise split via FusedCommBuffer
        self._params_2d = []
        self._params_2d_moe = []
        self._params_1d = []

        for p in self._parameter_list:
            if not _is_trainable(p):
                continue
            color = getattr(p, 'color', -1)
            if isinstance(color, dict):
                color_val = color.get('color', -1)
            else:
                color_val = color

            if color_val == 'moe_expert':
                # All MoE expert params go to _params_2d_moe (whole tensor path).
                # - 2D experts: Muon updates directly.
                # - 3D fused experts [n_experts, H, I]: each expert's 2D slice
                #   is updated individually by Muon in step().
                self._params_2d_moe.append(p)
            elif should_use_muon(p.name, p.shape):
                self._params_2d.append(p)
            else:
                self._params_1d.append(p)

        # ---- Step 2a: Greedy assign non-MoE 2D params to sharding ranks ----
        self._rank2params_2d = self._partition_2d_parameters(
            self._params_2d, self._sharding_world_size, label="non-MoE"
        )
        self._param2rank_2d = {}
        for rank, params in self._rank2params_2d.items():
            for p in params:
                self._param2rank_2d[p.name] = rank

        # ---- Step 2b: Partition MoE expert 2D params within moe_sharding_group ----
        self._rank2params_2d_moe = {}
        self._param2rank_2d_moe = {}
        if self._params_2d_moe:
            if self._moe_sharding_world_size > 1:
                # Need to partition MoE expert params across moe_sharding ranks
                self._rank2params_2d_moe = self._partition_2d_parameters(
                    self._params_2d_moe,
                    self._moe_sharding_world_size,
                    label="MoE",
                )
            else:
                # moe_sharding_degree=1: each rank owns all its local experts, no partition needed
                # All MoE expert params stay on rank 0 (the only rank in moe_sharding_group)
                self._rank2params_2d_moe = {0: list(self._params_2d_moe)}
            for rank, params in self._rank2params_2d_moe.items():
                for p in params:
                    self._param2rank_2d_moe[p.name] = rank

            # Assert: when moe_sharding_degree > 1, expert count must be divisible
            if self._moe_sharding_world_size > 1:
                n_experts_local = len(self._params_2d_moe)
                assert n_experts_local % self._moe_sharding_world_size == 0, (
                    f"Number of local MoE expert params ({n_experts_local}) must be "
                    f"divisible by moe_sharding_degree ({self._moe_sharding_world_size}). "
                    f"Please adjust expert_model_parallel_size or moe_sharding_degree."
                )

        # ---- Step 3: Build comm buffers for 1D params (V2-style) ----
        self._slice_params = {}
        self._comm_buffer_list = []
        self._local_parameter_list_1d = [
            self._create_slice_param(p) for p in self._params_1d
        ]

        self.param2bucket = {}
        self.sd_release_grads = (
            strategy.hybrid_configs['pp_configs'].release_gradients
            or sharding_configs.release_gradients
        )
        self._build_1d_comm_buffers()

        # ---- Step 4: Build the optimizer's parameter list ----
        # The optimizer should see:
        #   - Non-MoE 2D params assigned to this rank (as whole tensors)
        #   - MoE expert 2D params assigned to this rank in moe_sharding_group
        #   - 1D slice_params for all non-2D params (element-wise shards)
        local_2d_params = list(
            self._rank2params_2d.get(self._sharding_rank, [])
        )

        if self._moe_sharding_world_size > 1:
            local_2d_moe_params = list(
                self._rank2params_2d_moe.get(self._moe_sharding_rank, [])
            )
        else:
            # moe_sharding_degree=1: this rank owns all its MoE expert params
            local_2d_moe_params = list(self._rank2params_2d_moe.get(0, []))

        local_opt_params = (
            local_2d_params
            + local_2d_moe_params
            + list(self._local_parameter_list_1d)
        )

        self._set_inner_opt_attr('_parameter_list', local_opt_params)
        self._set_inner_opt_attr('_param_groups', local_opt_params)

        # For external iteration (clear_grad, etc.), expose all params
        self._local_parameter_list = local_opt_params

        self._enable_timer = strategy.hybrid_configs.get(
            "enable_optimizer_timer", False
        )
        if self._enable_timer:
            if not timer.is_timer_initialized():
                timer.set_timers()
            self.timers = timer.get_timers()

        logger.info(
            f"ShardingV3: rank={self._sharding_rank}, "
            f"non-MoE 2D on this rank={len(local_2d_params)}, "
            f"MoE 2D on this rank={len(local_2d_moe_params)}, "
            f"1D slice params={len(self._local_parameter_list_1d)}, "
            f"total non-MoE 2D={len(self._params_2d)}, "
            f"total MoE 2D={len(self._params_2d_moe)}, "
            f"total 1D={len(self._params_1d)}, "
            f"moe_sharding_degree={self._moe_sharding_world_size}"
        )

        # --- [VERIFY] Print group info for each category ---
        _sg = hcg.get_sharding_parallel_group()
        logger.info(
            f"[V3-init rank={self._sharding_rank}] "
            f"sharding_group: ranks={_sg.ranks} nranks={_sg.nranks}"
        )
        if self._moe_sharding_group is not None:
            logger.info(
                f"[V3-init rank={self._sharding_rank}] "
                f"moe_sharding_group: ranks={self._moe_sharding_group.ranks} "
                f"nranks={self._moe_sharding_group.nranks}"
            )

        # --- [VERIFY] Print param categorization: name, shape, category, group ---
        # Non-MoE 2D params (Muon, sharding_group)
        logger.info(
            f"[V3-init rank={self._sharding_rank}] "
            f"CATEGORY non_moe_2d (Muon, sharding_group nranks={_sg.nranks}): "
            f"total={len(self._params_2d)}"
        )
        for _p in self._params_2d[:5]:
            logger.info(
                f"[V3-init rank={self._sharding_rank}]   non_moe_2d: "
                f"name={_p.name!r} shape={list(_p.shape)} "
                f"should_muon={should_use_muon(_p.name, _p.shape)} "
                f"owner_rank={self._param2rank_2d.get(_p.name, -1)}"
            )

        # MoE 2D params (Muon, moe_sharding_group)
        _moe_sg_nranks = (
            self._moe_sharding_group.nranks if self._moe_sharding_group else 0
        )
        logger.info(
            f"[V3-init rank={self._sharding_rank}] "
            f"CATEGORY moe_2d (Muon, moe_sharding_group nranks={_moe_sg_nranks}): "
            f"total={len(self._params_2d_moe)}"
        )
        for _p in self._params_2d_moe[:3]:
            logger.info(
                f"[V3-init rank={self._sharding_rank}]   moe_2d: "
                f"name={_p.name!r} shape={list(_p.shape)} "
                f"should_muon={should_use_muon(_p.name, _p.shape)} "
                f"no_sync={getattr(_p, 'no_sync', None)} "
                f"owner_rank={self._param2rank_2d_moe.get(_p.name, -1)}"
            )

        # 1D params (AdamW, sharding_group via FusedCommBuffer)
        logger.info(
            f"[V3-init rank={self._sharding_rank}] "
            f"CATEGORY 1d_adamw (AdamW, sharding_group reduce-scatter): "
            f"total={len(self._params_1d)}"
        )
        for _p in self._params_1d[:5]:
            _color = getattr(_p, 'color', None)
            _color_group = (
                _color.get('group', None) if isinstance(_color, dict) else None
            )
            _cg_nranks = (
                _color_group.nranks if _color_group is not None else _sg.nranks
            )
            logger.info(
                f"[V3-init rank={self._sharding_rank}]   1d_adamw: "
                f"name={_p.name!r} shape={list(_p.shape)} "
                f"should_muon={should_use_muon(_p.name, _p.shape)} "
                f"color={getattr(_p, 'color', None)} "
                f"comm_group_nranks={_cg_nranks}"
            )

        # --- [SLICE SIZE SUMMARY] Per-rank slice param sizes within this PP stage ---
        import math as _math

        _sg_group = hcg.get_sharding_parallel_group()
        _N = self._sharding_world_size

        # 2D (non-MoE) params owned by this rank
        _local_2d_numel = sum(
            int(functools_reduce(lambda x, y: x * y, p.shape, 1))
            for p in self._rank2params_2d.get(self._sharding_rank, [])
        )
        # 2D (MoE) params owned by this rank
        _moe_rank_key = (
            self._moe_sharding_rank if self._moe_sharding_world_size > 1 else 0
        )
        _local_2d_moe_numel = sum(
            int(functools_reduce(lambda x, y: x * y, p.shape, 1))
            for p in self._rank2params_2d_moe.get(_moe_rank_key, [])
        )
        # 1D (AdamW) slice: each rank owns ceil(param.numel / world_size) elements per param.
        # Sum over all 1D params in this sharding group (same color).
        _local_1d_numel = sum(
            _math.ceil(
                int(functools_reduce(lambda x, y: x * y, p.shape, 1)) / _N
            )
            for p in self._params_1d
        )

        _local_total_numel = (
            _local_2d_numel + _local_2d_moe_numel + _local_1d_numel
        )
        _local_total_MB = (
            _local_total_numel * 2 / (1024 * 1024)
        )  # bf16/fp16 = 2 bytes

        # All-gather total numel from all sharding ranks in this PP stage
        _local_numel_tensor = paddle.to_tensor(
            [_local_total_numel], dtype='int64'
        )
        _all_numel_list = []
        paddle.distributed.all_gather(
            _all_numel_list, _local_numel_tensor, group=_sg_group
        )
        _all_numel = [int(t.item()) for t in _all_numel_list]
        _all_MB = [n * 2 / (1024 * 1024) for n in _all_numel]

        _max_MB = max(_all_MB)
        _min_MB = min(_all_MB)
        _imbalance = (_max_MB - _min_MB) / _max_MB if _max_MB > 0 else 0.0

        if self._sharding_rank == 0:
            logger.info(
                f"[ShardingV3 SliceSize] PP-stage sharding_group ranks={_sg_group.ranks} | "
                f"per-rank MB: {[f'{mb:.1f}' for mb in _all_MB]} | "
                f"max memory diff={_imbalance * 100:.2f}%"
            )

    # ------------------------------------------------------------------
    # 2D partition (V1-style greedy)
    # ------------------------------------------------------------------

    def _partition_2d_parameters(self, params, world_size, label=""):
        """Partition 2D parameters among ranks using greedy bin-packing."""
        mapping = {}
        for rank in range(world_size):
            mapping[rank] = []
        sizes = [0] * world_size

        parameters = list(params)
        parameters.sort(
            key=lambda p: functools_reduce(lambda x, y: x * y, p.shape),
            reverse=True,
        )

        for param in parameters:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = functools_reduce(lambda x, y: x * y, param.shape, 1)
            sizes[rank] += numel

        for r in range(world_size):
            logger.info(
                f"ShardingV3 {label} 2D partition: rank {r} -> {len(mapping[r])} params, "
                f"{sizes[r]:,} elements"
            )

        return mapping

    # ------------------------------------------------------------------
    # 1D slice creation (V2-style)
    # ------------------------------------------------------------------

    def _create_slice_param(self, param):
        """Create a placeholder slice parameter for 1D (element-wise) sharding."""
        slice_param = EagerParamBase(shape=[1], dtype=param.dtype)
        slice_param.name = param.name

        def copy_attr(attr_name):
            if hasattr(param, attr_name):
                setattr(slice_param, attr_name, getattr(param, attr_name))

        copy_attr("is_distributed")
        copy_attr("optimize_attr")
        copy_attr("do_model_average")
        copy_attr("need_clip")
        copy_attr("no_sync")

        self._slice_params[param.name] = slice_param
        return slice_param

    def _build_1d_comm_buffers(self):
        """Build communication buffers for 1D (AdamW) parameters using reduce-scatter."""
        if self.pp_overlap:
            return

        comm_group = self._hcg.get_sharding_parallel_group()
        group_size = (
            self.comm_buffer_size_MB * 1024 * 1024
            if self.comm_buffer_size_MB > 0
            else 256 * 1024 * 1024
        )

        # Group 1D params by color (for MoE compatibility)
        color_dict = defaultdict(list)
        for param in self._params_1d:
            color = getattr(param, 'color', -1)
            color_group = comm_group
            if isinstance(color, dict):
                color_color = color.get('color', -1)
                color_group = color.get('group', comm_group)
            else:
                color_color = color
            color_dict[(color_color, color_group)].append(param)

        if not self.comm_overlap:
            for color, params in color_dict.items():
                params.sort(key=lambda x: str(x.dtype))

        group_idx = 0
        for color, params in color_dict.items():
            g_color = color[0]
            g_group = color[1]
            logger.info(
                f"ShardingV3 1D Buffer: Color {g_color}, Group {g_group}"
            )
            # --- [VERIFY] Log which group each 1D buffer uses ---
            logger.info(
                f"[V3-1d-buffer rank={self._sharding_rank}] "
                f"color={g_color!r} group_ranks={g_group.ranks} "
                f"group_nranks={g_group.nranks} param_count={len(params)} "
                f"sample_names={[p.name for p in params[:3]]}"
            )
            var_groups = assign_group_by_size(params, group_size)
            for _, parameters in var_groups.items():
                buffer = FusedCommBuffer(
                    group_idx,
                    parameters,
                    g_group,
                    self.accumulate_steps,
                    act=HOOK_ACTION.REDUCE_SCATTER,
                    release_grads=self.sd_release_grads,
                    use_reduce_avg=self.use_reduce_avg,
                    free_grads_in_comm=False,
                    init_slice_param=False,
                    slice_params=self._slice_params,
                )
                group_idx += 1
                self._comm_buffer_list.append(buffer)

                for p in parameters:
                    if p.name in self.param2bucket:
                        self.param2bucket[p.name].append(buffer)
                    else:
                        self.param2bucket[p.name] = [buffer]

        self._comm_buffer_list.sort(key=lambda x: x._dst)

    # ------------------------------------------------------------------
    # Gradient communication
    # ------------------------------------------------------------------

    def _get_param_grad(self, param):
        if not param.trainable:
            return None
        if hasattr(param, "main_grad"):
            assert param._grad_ivar() is None, (
                "param.grad should be None when using main_grad"
            )
            return param.main_grad
        return param._grad_ivar()

    def _reduce_2d_grads(self, params, param2rank, comm_group):
        """Reduce gradients for 2D params to their owner rank within comm_group."""
        for param in params:
            g_var = self._get_param_grad(param)
            if g_var is None:
                if hasattr(param, "main_grad"):
                    g_var = paddle.zeros_like(param, dtype=paddle.float32)
                    param.main_grad = g_var
                else:
                    g_var = paddle.zeros_like(param, dtype=param.dtype)
                    param.grad = g_var

            reduce_op = ReduceOp.AVG
            if not self.use_reduce_avg:
                nranks = comm_group.nranks
                g_var.scale_(1.0 / nranks)
                reduce_op = ReduceOp.SUM

            if paddle.distributed.in_auto_parallel_align_mode():
                reduce_op = ReduceOp.SUM

            param_rank = param2rank[param.name]
            paddle.distributed.reduce(
                g_var,
                dst=comm_group.ranks[param_rank],
                op=reduce_op,
                group=comm_group,
                sync_op=True,
            )

    def reduce_gradients(self, parameter_list, hcg):
        """Reduce gradients: reduce for 2D params, reduce-scatter for 1D params."""
        logger.debug("ShardingV3: start gradient sync")

        if (
            paddle.is_compiled_with_xpu()
            and os.getenv("XPU_CDNN_CLUSTER_PARALLEL") is not None
        ):
            paddle.device.synchronize()

        with framework.no_grad():
            # --- Non-MoE 2D params: reduce to owner rank via sharding_group ---
            sharding_group = hcg.get_sharding_parallel_group()
            self._reduce_2d_grads(
                self._params_2d, self._param2rank_2d, sharding_group
            )

            # --- MoE expert 2D params: reduce to owner rank via moe_sharding_group ---
            if self._params_2d_moe and self._moe_sharding_group is not None:
                if self._moe_sharding_world_size > 1:
                    self._reduce_2d_grads(
                        self._params_2d_moe,
                        self._param2rank_2d_moe,
                        self._moe_sharding_group,
                    )
                # When moe_sharding_degree=1, no reduce needed (single rank group)

            # --- 1D params: reduce-scatter via comm buffers ---
            for comm_buffer in self._comm_buffer_list:
                if self.sd_release_grads and comm_buffer.grad_storage is None:
                    if comm_buffer.need_reduce_scale_sync():
                        for param in comm_buffer.params:
                            comm_buffer._copy_grad_to_buffer(param)

                if not self.comm_overlap:
                    comm_buffer._comm_grads()
                comm_buffer.scale_grads()

    def filter_parameters(self, parameter_list, hcg):
        """Filter parameters: return local 2D params + initialized 1D slices."""
        sharding_rank = hcg.get_sharding_parallel_rank()
        local_2d = [
            p
            for p in parameter_list
            if p.name in self._param2rank_2d
            and self._param2rank_2d[p.name] == sharding_rank
        ]
        # Also include MoE 2D params owned by this rank
        if self._moe_sharding_world_size > 1:
            moe_rank = self._moe_sharding_rank
        else:
            moe_rank = 0
        local_2d_moe = [
            p
            for p in parameter_list
            if p.name in self._param2rank_2d_moe
            and self._param2rank_2d_moe[p.name] == moe_rank
        ]
        local_1d = [
            self._slice_params[p.name]
            for p in parameter_list
            if p.name in self._slice_params
        ]
        local_1d = [p for p in local_1d if p._is_initialized()]
        return local_2d + local_2d_moe + local_1d

    # ------------------------------------------------------------------
    # Parameter sync after optimizer step
    # ------------------------------------------------------------------

    def _broadcast_2d_params(self, rank2params, comm_group):
        """Broadcast 2D params from owner ranks within comm_group."""
        broadcast_tasks = []
        for rank, params in rank2params.items():
            src_rank = comm_group.ranks[rank]
            for param in params:
                if param.stop_gradient:
                    continue
                task = paddle.distributed.broadcast(
                    param,
                    src=src_rank,
                    group=comm_group,
                    sync_op=False,
                )
                broadcast_tasks.append(task)
        return broadcast_tasks

    def _sharding_sync_parameters(self):
        """Sync parameters: broadcast 2D, all-gather 1D."""
        logger.debug("ShardingV3: start parameter sync")
        comm_group = self._hcg.get_sharding_parallel_group()

        with framework.no_grad():
            all_tasks = []

            # --- Non-MoE 2D params: broadcast from owner via sharding_group ---
            all_tasks.extend(
                self._broadcast_2d_params(self._rank2params_2d, comm_group)
            )

            # --- MoE expert 2D params: broadcast from owner via moe_sharding_group ---
            if self._params_2d_moe and self._moe_sharding_group is not None:
                if self._moe_sharding_world_size > 1:
                    all_tasks.extend(
                        self._broadcast_2d_params(
                            self._rank2params_2d_moe, self._moe_sharding_group
                        )
                    )
                # When moe_sharding_degree=1, no broadcast needed (single rank group)

            for task in all_tasks:
                task.wait()

            # --- 1D params: all-gather via comm buffers ---
            for comm_buffer in self._comm_buffer_list:
                comm_buffer.sync_params()

    # ------------------------------------------------------------------
    # Clear gradients
    # ------------------------------------------------------------------

    def clear_grad(self, set_to_zero=True):
        """Clear gradients for all parameters."""
        # Clear 2D param grads (non-MoE + MoE)
        all_2d_params = list(self._params_2d) + list(self._params_2d_moe)
        for p in all_2d_params:
            if hasattr(p, "main_grad") and p.main_grad is not None:
                assert p._grad_ivar() is None
                if set_to_zero:
                    p.main_grad.zero_()
                else:
                    p.main_grad._clear()
                    p.main_grad = None
            elif not hasattr(p, "main_grad"):
                p.clear_gradient(set_to_zero)

        # 1D params are managed by comm buffers
        if self.sd_release_grads and not self.pp_overlap:
            for comm_buffer in self._comm_buffer_list:
                if comm_buffer.need_reduce_scale_sync():
                    comm_buffer._clear_grad_storage()

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def _collect_comm_buffers(self):
        """Collect communication buffers (for PP overlap compatibility)."""
        if self._comm_buffer_list:
            return
        for param in self._params_1d:
            if not hasattr(param, "comm_buffer_ref"):
                continue
            comm_buffer_ref = param.comm_buffer_ref
            del param.comm_buffer_ref
            comm_buffer = comm_buffer_ref()
            self._comm_buffer_list.append(comm_buffer)

        for bucket in self._comm_buffer_list:
            for p in bucket._params:
                if p.name in self.param2bucket:
                    self.param2bucket[p.name].append(bucket)
                else:
                    self.param2bucket[p.name] = [bucket]

    def _assign_slice_grad(self):
        """Assign gradients from comm buffers to slice params for 1D params."""
        for comm_buffer in self._comm_buffer_list:
            for param in comm_buffer.params:
                if param.name in self._slice_params:
                    slice_param = self._slice_params[param.name]
                    if self.sd_release_grads and hasattr(
                        slice_param, "main_grad"
                    ):
                        if not slice_param.main_grad._is_initialized():
                            del slice_param.main_grad
                    comm_buffer.assign_slice_grad(param, slice_param)

    def step(self):
        """Optimizer step: update local 2D params and 1D slices, then sync."""
        self._collect_comm_buffers()
        self._assign_slice_grad()

        if not isinstance(self._origin_parameter_list[0], dict):
            params_grads = []

            # --- Non-MoE 2D params on this rank: full tensors ---
            local_2d = self._rank2params_2d.get(self._sharding_rank, [])
            for param in local_2d:
                if param.stop_gradient:
                    continue
                grad_var = param._grad_ivar()
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad_var = param.main_grad
                if grad_var is not None:
                    params_grads.append((param, grad_var))

            # --- MoE expert params on this rank ---
            # Pass the original param (2D or 3D) directly to the optimizer.
            # _muon_update already handles both shapes:
            #   - 2D [H, I]: standard Newton-Schulz
            #   - 3D [n_experts, H, I]: per-expert Newton-Schulz loop (Step 4)
            # Keeping the original name avoids registering _expert_N accumulator
            # keys that are absent from model_sharded_state_dict, which would
            # break sharded_state_dict (checkpoint save).
            #
            # NOTE: MoEHybridParallelClipGrad._dygraph_clip has `assert 0` for
            # params with is_moe_param=True.  Temporarily clear is_moe_param so
            # clip skips the assert and routes via the no_sync/is_distributed
            # branch instead.  We restore the attribute after _apply_optimize.
            if self._moe_sharding_world_size > 1:
                local_2d_moe = self._rank2params_2d_moe.get(
                    self._moe_sharding_rank, []
                )
            else:
                local_2d_moe = self._rank2params_2d_moe.get(0, [])
            moe_params_to_restore = []  # params that had is_moe_param=True
            for param in local_2d_moe:
                if param.stop_gradient:
                    continue
                grad_var = param._grad_ivar()
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad_var = param.main_grad
                if grad_var is None:
                    continue
                if getattr(param, "is_moe_param", False):
                    param.is_moe_param = False
                    moe_params_to_restore.append(param)
                params_grads.append((param, grad_var))

            # --- 1D params: slice params (element-wise shards) ---
            for param in self._params_1d:
                if param.stop_gradient:
                    continue
                if param.name not in self._slice_params:
                    continue
                slice_p = self._slice_params[param.name]
                grad_var = slice_p._grad_ivar()
                if (
                    hasattr(slice_p, "main_grad")
                    and slice_p.main_grad is not None
                ):
                    grad_var = slice_p.main_grad
                if grad_var is not None:
                    params_grads.append((slice_p, grad_var))

            self._apply_optimize(
                loss=None,
                startup_program=None,
                params_grads=params_grads,
            )

            # Restore is_moe_param on MoE params after optimizer step.
            for _p in moe_params_to_restore:
                _p.is_moe_param = True

        # Sync parameters across sharding ranks
        self._sharding_sync_parameters()

    # ------------------------------------------------------------------
    # State dict (checkpoint save/load)
    # ------------------------------------------------------------------

    @framework.dygraph_only
    def set_state_dict(self, state_dict):
        inner_state = {}
        # Local parameters = local 2D + local MoE 2D + 1D slice params
        local_2d = list(self._rank2params_2d.get(self._sharding_rank, []))
        if self._moe_sharding_world_size > 1:
            local_2d_moe = list(
                self._rank2params_2d_moe.get(self._moe_sharding_rank, [])
            )
        else:
            local_2d_moe = list(self._rank2params_2d_moe.get(0, []))
        parameters = local_2d + local_2d_moe
        # Add 1D params (use original param names for matching)
        for p in self._params_1d:
            parameters.append(p)

        if "LR_Scheduler" in state_dict:
            inner_state["LR_Scheduler"] = state_dict.pop("LR_Scheduler")

        if "master_weights" in state_dict:
            master = state_dict.pop("master_weights")
            inner_state["master_weights"] = {}
            for p in parameters:
                for k, v in master.items():
                    if p.name == k:
                        v.name = self._inner_opt._gen_master_weight_var_name(p)
                        inner_state["master_weights"][k] = v

        for p in parameters:
            for k, v in state_dict.items():
                if p.name in k:
                    inner_state[k] = v

        self._inner_opt.set_state_dict(inner_state)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _set_inner_opt_attr(self, attr_name, value):
        inner_opt = self._inner_opt
        inner_opt_name = '_inner_opt'
        if not isinstance(attr_name, str):
            raise TypeError(
                f"attr_name should be str type, but is {type(attr_name)}"
            )
        while hasattr(inner_opt, attr_name):
            setattr(inner_opt, attr_name, value)
            inner_opt = getattr(inner_opt, inner_opt_name, None)
            if inner_opt is None:
                break

    def sharded_state_dict(self, model_sharded_state_dict):
        """Build a sharded optimizer state dict for flex checkpoint save/load.

        Overrides the inner Muon optimizer's sharded_state_dict to handle V3's
        hybrid sharding scheme:
          - 2D Muon params (non-MoE and MoE): whole tensor, shape matches
            model's local_shape. Handled by delegating to the inner Muon's
            sharded_state_dict after filtering out 1D param states.
          - 1D AdamW params: accumulators are 1D shards (from reduce-scatter);
            wrapped with is_flattened=True + flattened_range, like V2.
        """
        import paddle as _paddle
        from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
            ShardedWeight,
            create_sharded_weight_with_new_local,
        )

        # ---- Step 1: Collect flattened_range for each 1D (AdamW) param ----
        # Identical logic to DygraphShardingOptimizerV2.sharded_state_dict.
        param_slice_info = {}  # param_name -> slice(begin, end)
        padded_param = set()
        for buffer in self._comm_buffer_list:
            for (
                param_name,
                grad_view,
            ) in buffer._sharding_param_grad_view.items():
                numel = grad_view._param.numel().item()
                param_begin = grad_view._param_begin
                param_end = grad_view._param_end
                index = grad_view._index
                padding_begin = index + numel
                flattened_range = slice(
                    param_begin - index,
                    max(
                        min(padding_begin - index, param_end - index),
                        param_begin - index,
                    ),
                )
                if param_end > padding_begin:
                    padded_param.add(param_name)
                param_slice_info[param_name] = flattened_range

        # ---- Step 2: Build static_name → struct_name mapping ----
        model_sharded_sorted = dict(sorted(model_sharded_state_dict.items()))
        static_to_struct = {}
        for struct_name, sw in model_sharded_sorted.items():
            if sw.local_tensor.name not in static_to_struct:
                static_to_struct[sw.local_tensor.name] = struct_name

        # ---- Step 3: Process all optimizer states ----
        _FP32_MASTER = "fp32_master_0"
        _optimizer_scalar_names = ["beta1_pow_acc_0", "beta2_pow_acc_0"]
        _optimizer_vector_names = ["moment1_0", "moment2_0"]

        def _split_state_name(vname):
            if _FP32_MASTER in vname:
                return tuple(vname.split("_" + _FP32_MASTER + "_", 1))
            for suffix in _optimizer_scalar_names + _optimizer_vector_names:
                if vname.endswith(suffix):
                    return vname[: -(len(suffix) + 1)], suffix
            raise ValueError(
                f"Cannot parse optimizer state variable name: {vname!r}"
            )

        optimizer_state_dict = self._inner_opt.state_dict()
        master_weights = optimizer_state_dict.pop("master_weights", None)
        optimizer_state_dict.pop("LR_Scheduler", None)

        sharded_state = {}

        for key, tensor in optimizer_state_dict.items():
            static_name, state_type = _split_state_name(key)
            if static_name not in static_to_struct:
                logger.error(
                    f"[V3-sharded_state_dict] KeyError: static_name={static_name!r} "
                    f"not found in model_sharded_state_dict."
                )
                continue

            struct_name = static_to_struct[static_name]
            sharded_param = model_sharded_sorted[struct_name]
            unified_name = f"{struct_name}.{state_type}"

            is_1d_param = static_name in param_slice_info

            if state_type in _optimizer_vector_names:
                if is_1d_param:
                    # 1D AdamW shard: wrap with is_flattened=True (like V2)
                    flattened_range = param_slice_info[static_name]
                    if flattened_range.stop - flattened_range.start == 0:
                        continue
                    is_padded = static_name in padded_param
                    if is_padded:
                        local_tensor = _paddle.slice(
                            tensor,
                            axes=[0],
                            starts=[0],
                            ends=[flattened_range.stop - flattened_range.start],
                        )
                    else:
                        local_tensor = tensor
                    sharded_state[unified_name] = ShardedWeight(
                        key=unified_name,
                        local_tensor=local_tensor,
                        local_shape=sharded_param.local_shape,
                        global_shape=sharded_param.global_shape,
                        global_offset=sharded_param.global_offset,
                        is_flattened=True,
                        flattened_range=flattened_range,
                    )
                elif tensor.is_dist():
                    sharded_state[unified_name] = ShardedWeight(
                        key=unified_name,
                        local_tensor=tensor,
                        local_shape=tensor.shape,
                        global_shape=tensor.shape,
                        global_offset=sharded_param.global_offset,
                    )
                else:
                    # 2D Muon param (non-MoE or MoE): shape may differ between
                    # Python param.shape (3D view) and model storage (2D).
                    # Reshape if numel matches but shape doesn't.
                    target_shape = sharded_param.local_shape
                    if (
                        tuple(tensor.shape) != tuple(target_shape)
                        and tensor.numel()
                        == _paddle.to_tensor(list(target_shape)).prod().item()
                    ):
                        tensor = tensor.reshape(target_shape)
                    sharded_state[unified_name] = (
                        create_sharded_weight_with_new_local(
                            unified_name, tensor, sharded_param
                        )
                    )
            else:
                # Scalar states (beta_pow): replicated
                sharded_state[unified_name] = ShardedWeight(
                    key=unified_name,
                    local_tensor=tensor,
                    local_shape=(1,),
                    global_shape=(1,),
                    global_offset=(0,),
                )

        # FP32 master weights
        if master_weights:
            for weight_key, tensor in master_weights.items():
                if weight_key not in static_to_struct:
                    continue
                struct_name = static_to_struct[weight_key]
                sharded_param = model_sharded_sorted[struct_name]
                unified_name = f"{struct_name}.w_0"
                is_1d_param = weight_key in param_slice_info

                if is_1d_param:
                    flattened_range = param_slice_info[weight_key]
                    if flattened_range.stop - flattened_range.start == 0:
                        continue
                    is_padded = weight_key in padded_param
                    if is_padded:
                        local_tensor = _paddle.slice(
                            tensor,
                            axes=[0],
                            starts=[0],
                            ends=[flattened_range.stop - flattened_range.start],
                        )
                    else:
                        local_tensor = tensor
                    sharded_state[unified_name] = ShardedWeight(
                        key=unified_name,
                        local_tensor=local_tensor,
                        local_shape=sharded_param.local_shape,
                        global_shape=sharded_param.global_shape,
                        global_offset=sharded_param.global_offset,
                        is_flattened=True,
                        flattened_range=flattened_range,
                    )
                elif tensor.is_dist():
                    sharded_state[unified_name] = ShardedWeight(
                        key=unified_name,
                        local_tensor=tensor,
                        local_shape=tensor.shape,
                        global_shape=tensor.shape,
                        global_offset=sharded_param.global_offset,
                    )
                else:
                    # Same reshape logic as for optimizer vector states:
                    # FP32 master weight may be 3D (e.g. grouped_gemm_experts
                    # [n_experts, H, I]) while model storage is 2D [n_experts*H, I].
                    target_shape = sharded_param.local_shape
                    if (
                        tuple(tensor.shape) != tuple(target_shape)
                        and tensor.numel()
                        == _paddle.to_tensor(list(target_shape)).prod().item()
                    ):
                        tensor = tensor.reshape(target_shape)
                    sharded_state[unified_name] = (
                        create_sharded_weight_with_new_local(
                            unified_name, tensor, sharded_param
                        )
                    )

        return sharded_state

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
