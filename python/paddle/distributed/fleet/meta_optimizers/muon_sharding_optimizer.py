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
MuonShardingOptimizer (Sharding Stage1 V3): Hybrid Tensor-wise + Element-wise Sharding
==================================================================

Designed for Muon optimizer compatibility:
  - 2D (Muon) parameters: assigned as *whole tensors* to ranks (like V1).
    This avoids the expensive sharding gather in Muon's _muon_update.
  - Non-2D (AdamW) parameters: split element-wise via reduce-scatter (like V2).
    This provides memory balancing across ranks.

The key insight is that Muon requires the full 2D matrix for Newton-Schulz
orthogonalisation, so keeping 2D params whole on each rank eliminates the
need for gather_varlen communication during the optimizer step.

Parameters are grouped by their `color` attribute, which specifies the
communication group to use:
  - color=None or -1: default sharding_group
  - color='moe_expert': moe_sharding_group
  - color=<custom>: hcg.get_<custom>_parallel_group() (extensible design)
"""

import math
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
from paddle.distributed.fleet.utils import timer_helper as timer
from paddle.distributed.fleet.utils.log_util import logger
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
    assign_group_by_size,
)

g_shard_bypass_dygraph_optimizer = int(
    os.environ.get("FLAGS_shard_bypass_dygraph_optimizer", 0)
)
g_shard_fused_gradient = int(os.environ.get("FLAGS_shard_fused_gradient", 0))


def _is_trainable(param):
    return not param.stop_gradient


class MuonShardingOptimizer:
    """
    Hybrid sharding optimizer for Muon:
    - 2D (Muon) parameters: tensor-wise assignment to ranks (no cross-rank split).
      Gradient communication uses reduce; parameter sync uses broadcast.
    - Non-2D (AdamW) parameters: element-wise split across ranks (like V2).
      Gradient communication uses reduce-scatter; parameter sync uses all-gather.

    Parameters are grouped by `color` attribute to determine the communication
    group. Each color group has its own 2D parameter partition and communication.

    This avoids the expensive gather_varlen in Muon's _muon_update while
    maintaining memory balance across ranks.
    """

    def __init__(self, optimizer, hcg=None):
        logger.info("init MuonShardingOptimizer")

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
        # Get hcg from fleet if not provided
        if hcg is None:
            hcg = fleet.fleet._hcg
        self._hcg = hcg
        self._sharding_world_size = self._hcg.get_sharding_parallel_world_size()
        self._sharding_rank = self._hcg.get_sharding_parallel_rank()
        self._global_rank = paddle.distributed.get_rank()

        # Temporarily: TP is not supported in MuonShardingOptimizer
        _tp_world_size = self._hcg.get_model_parallel_world_size()
        assert _tp_world_size == 1, (
            f"MuonShardingOptimizer does not support tensor parallelism yet. "
            f"Got tp_world_size={_tp_world_size}. Please set tensor_parallel_degree=1."
        )

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

        # Build color -> group_info mapping
        self._color_to_group_info = self._build_color_to_group_info(hcg)

        # Extract MoE group info from color_to_group_info for backward compatibility
        moe_info = self._color_to_group_info.get('moe_expert', {})
        self._moe_sharding_world_size = moe_info.get('world_size', 1)
        self._moe_sharding_rank = moe_info.get('rank', 0)
        self._moe_sharding_group = moe_info.get('group', None)

        # Get muon_param_info_map from Muon optimizer
        # This map contains use_muon field for each parameter, determined by Trainer
        self._muon_param_info_map = getattr(
            optimizer, '_muon_param_info_map', {}
        )

        # ---- Step 1: Separate params into categories by color ----
        # Parameters are grouped by their `color` attribute:
        # - color=None or -1: default sharding_group (key: None)
        # - color='moe_expert': moe_sharding_group (key: 'moe_expert')
        # - color=<custom>: corresponding parallel group (key: <custom>)
        #
        # For each color group:
        # - 2D (Muon) params: whole tensor, assigned to ranks via tensor-wise partition
        # - non-2D (AdamW) params: element-wise split via FusedCommBuffer
        #
        # This design is extensible: adding a new communication group only requires
        # setting the `color` attribute on parameters, no code changes needed here.
        self._params_2d_by_color = defaultdict(
            list
        )  # color -> list of 2D params
        self._params_1d = []  # All non-2D params (single list, sharding_group only)
        self.clear_color = set()
        self._color_to_comm_buffer_list = {}
        for p in self._parameter_list:
            if not _is_trainable(p):
                continue

            # Extract color value
            color = getattr(p, 'color', -1)
            if isinstance(color, dict):
                color_val = color.get('color', -1)
            else:
                color_val = color

            # Normalize color: treat None/-1 as default (None key)
            if color_val == -1 or color_val is None:
                color_key = None
            else:
                color_key = color_val

            # Check if this color group supports 2D tensor-wise partition
            group_info = self._color_to_group_info.get(color_key)

            param_info = self._muon_param_info_map.get(p.name)
            assert param_info is not None, (
                f"Parameter {p.name!r} (shape={list(p.shape)}) has no muon_param_info. "
                f"Trainer._build_muon_param_info_map must set muon_param_info on all "
                f"trainable parameters before MuonShardingOptimizer is constructed."
            )
            use_muon = param_info.use_muon

            if use_muon:
                self._params_2d_by_color[color_key].append(p)
            else:
                # Non-2D params always go to 1D element-wise split (sharding_group only)
                self._params_1d.append(p)

        # ---- Step 2: Partition 2D params for each color group ----
        # For each color, compute rank-to-params and param-to-rank mappings
        self._rank2params_2d_by_color = {}  # color -> {rank -> [params]}
        self._param2rank_2d_by_color = {}  # color -> {param_name -> rank}

        for color_key, params_2d in self._params_2d_by_color.items():
            group_info = self._color_to_group_info.get(color_key, {})
            world_size = group_info.get('world_size', 1)

            if world_size <= 1:
                # No partition needed, all params stay on rank 0
                self._rank2params_2d_by_color[color_key] = {0: list(params_2d)}
                self._param2rank_2d_by_color[color_key] = {
                    p.name: 0 for p in params_2d
                }
            else:
                # Greedy partition across ranks
                label = color_key if color_key else "default"
                self._rank2params_2d_by_color[color_key] = (
                    self._partition_2d_parameters(
                        list(params_2d), world_size, label=label
                    )
                )
                self._param2rank_2d_by_color[color_key] = {}
                for rank, params in self._rank2params_2d_by_color[
                    color_key
                ].items():
                    for p in params:
                        self._param2rank_2d_by_color[color_key][p.name] = rank

        # add sort 2d params
        for color_key, params_2d in self._params_2d_by_color.items():
            params_2d.sort(
                key=lambda p: self._param2rank_2d_by_color[color_key][p.name]
            )

        # ---- Backward compatibility: expose legacy attributes ----
        # These are kept for any external code that might reference them
        self._params_2d = self._params_2d_by_color.get(None, [])
        self._params_2d_moe = self._params_2d_by_color.get('moe_expert', [])
        self._rank2params_2d = self._rank2params_2d_by_color.get(None, {0: []})
        self._param2rank_2d = self._param2rank_2d_by_color.get(None, {})
        self._rank2params_2d_moe = self._rank2params_2d_by_color.get(
            'moe_expert', {0: []}
        )
        self._param2rank_2d_moe = self._param2rank_2d_by_color.get(
            'moe_expert', {}
        )

        self._use_fuse_gradients = g_shard_fused_gradient
        # ---- Build comm buffers for 2D params (V1-style) ----
        if self._use_fuse_gradients:
            if not hasattr(self, 'comm_buffer_2d'):
                self.comm_buffer_2d = self._build_2d_comm_buffers()
                self.comm_buffer_2d.sort(key=lambda x: x._dst)

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

        # --- [SLICE SIZE SUMMARY] Per-rank slice param sizes within this PP stage ---
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
            math.ceil(
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
                f"[MuonSharding global_rank={self._global_rank} sharding_rank={self._sharding_rank}] "
                f"SliceSize sharding_group ranks={_sg_group.ranks} | "
                f"per-rank MB: {[f'{mb:.1f}' for mb in _all_MB]} | "
                f"max memory diff={_imbalance * 100:.2f}%"
            )

    # ------------------------------------------------------------------
    # 2D partition (V1-style greedy)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_color_to_group_info(hcg):
        """Build a mapping from color to communication group info.

        Returns:
            dict: {
                None: {'group': sharding_group, 'world_size': N, 'rank': r},
                'moe_expert': {'group': moe_sharding_group, 'world_size': M, 'rank': s},
                # Future colors can be added here
            }
        """
        color_to_info = {}

        # Default sharding group
        sharding_world_size = hcg.get_sharding_parallel_world_size()
        sharding_group = hcg.get_sharding_parallel_group()
        color_to_info[None] = {
            'group': sharding_group,
            'world_size': sharding_world_size,
            'rank': sharding_group.rank if sharding_group else 0,
        }

        # MoE sharding group (if available)
        if hasattr(hcg, "get_moe_sharding_parallel_world_size"):
            moe_world_size = hcg.get_moe_sharding_parallel_world_size()
            if moe_world_size > 0:
                moe_group = hcg.get_moe_sharding_parallel_group()
                color_to_info['moe_expert'] = {
                    'group': moe_group,
                    'world_size': moe_world_size,
                    'rank': moe_group.rank if moe_group else 0,
                }

        # Future: Add more color -> group mappings here as needed
        # Example:
        # if hasattr(hcg, "get_custom_parallel_world_size"):
        #     custom_world_size = hcg.get_custom_parallel_world_size()
        #     if custom_world_size > 0:
        #         custom_group = hcg.get_custom_parallel_group()
        #         color_to_info['custom'] = {
        #             'group': custom_group,
        #             'world_size': custom_world_size,
        #             'rank': custom_group.rank if custom_group else 0,
        #         }

        return color_to_info

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

        return mapping

    def _build_2d_comm_buffers(self):
        """Build communication buffers for 2D (Tensor-wise) parameters using all-reduce."""
        group_size = (
            self.comm_buffer_size_MB * 1024 * 1024
            if self.comm_buffer_size_MB > 0
            else 256 * 1024 * 1024
        )
        comm_buffers = []

        for color_key, params_2d in self._params_2d_by_color.items():
            group_info = self._color_to_group_info.get(color_key, {})
            comm_group = group_info.get('group', None)

            fused_parameter_group = defaultdict(list)

            for p in params_2d:
                dst_rank = self._param2rank_2d_by_color[color_key][p.name]
                fused_parameter_group[dst_rank].append(p)

            absolute_dst_ranks = {
                rank: comm_group.ranks[rank] for rank in fused_parameter_group
            }

            for dst, params in fused_parameter_group.items():
                var_groups = assign_group_by_size(params, group_size)
                abs_dst = absolute_dst_ranks[dst]

                buffer = [
                    FusedCommBuffer(
                        group_idx,
                        parameters,
                        comm_group,
                        self.accumulate_steps,
                        act=HOOK_ACTION.REDUCE,
                        dst=abs_dst,
                        release_grads=False,
                        use_reduce_avg=True,
                    )
                    for group_idx, parameters in var_groups.items()
                ]
                comm_buffers.extend(buffer)

        return comm_buffers

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
                if g_color not in self._color_to_comm_buffer_list.keys():
                    self._color_to_comm_buffer_list[g_color] = []
                self._color_to_comm_buffer_list[g_color].append(buffer)
                for p in parameters:
                    if p.name in self.param2bucket:
                        self.param2bucket[p.name].append(buffer)
                    else:
                        self.param2bucket[p.name] = [buffer]

        self._comm_buffer_list.sort(key=lambda x: x._dst)

    def clear_param_storage(self, color):
        self.clear_color.add(color)
        if color in self._color_to_comm_buffer_list.keys():
            for comm_buffer in self._color_to_comm_buffer_list[color]:
                for param in comm_buffer.params:
                    grad_view = comm_buffer._sharding_param_grad_view[
                        param.name
                    ]
                    slice_param = self._slice_params[param.name]
                    if (
                        not g_shard_bypass_dygraph_optimizer
                        and grad_view._param_begin < grad_view._param_end
                    ):
                        grad_view.fill_slice_param(slice_param)
                        self._create_master_weight(slice_param)
                    slice_param._clear_dataptr()
                comm_buffer._clear_param_storage()

    def reset_param_storage(self):
        for color in self.clear_color:
            if color is None:
                continue
            if color in self._color_to_comm_buffer_list.keys():
                for comm_buffer in self._color_to_comm_buffer_list[color]:
                    comm_buffer._reset_param_storage()

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
        if (
            paddle.is_compiled_with_xpu()
            and os.getenv("XPU_CDNN_CLUSTER_PARALLEL") is not None
        ):
            paddle.device.synchronize()

        with framework.no_grad():
            # --- 2D params: reduce via comm buffers | per tensors ---
            if self._use_fuse_gradients:
                for comm_buffer in self.comm_buffer_2d:
                    comm_buffer._comm_grads()
            else:
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

            # wait for all comm_buffer tasks to finish
            if self._use_fuse_gradients:
                for comm_buffer in self.comm_buffer_2d:
                    comm_buffer.scale_grads()
            for comm_buffer in self._comm_buffer_list:
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

        def clear_grad_func(p):
            if hasattr(p, "main_grad") and p.main_grad is not None:
                assert p._grad_ivar() is None
                if set_to_zero:
                    p.main_grad.zero_()
                else:
                    p.main_grad._clear()
                    p.main_grad = None
            elif not hasattr(p, "main_grad"):
                if self.tensor_fusion:
                    if set_to_zero:
                        p.grad.zero_()
                    else:
                        p.grad._clear()
                        p.grad = None
                else:
                    p.clear_gradient(set_to_zero)

        for p in self._parameter_list:
            clear_grad_func(p)

        # 1D params are managed by comm buffers
        if self.sd_release_grads and not self.pp_overlap:
            for comm_buffer in self._comm_buffer_list:
                if comm_buffer.need_reduce_scale_sync():
                    comm_buffer._clear_grad_storage()

            if self._use_fuse_gradients:
                for comm_buffer in self.comm_buffer_2d:
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
            if self._moe_sharding_world_size > 1:
                local_2d_moe = self._rank2params_2d_moe.get(
                    self._moe_sharding_rank, []
                )
            else:
                local_2d_moe = self._rank2params_2d_moe.get(0, [])

            for param in local_2d_moe:
                if param.stop_gradient:
                    continue
                grad_var = param._grad_ivar()
                if hasattr(param, "main_grad") and param.main_grad is not None:
                    grad_var = param.main_grad
                if grad_var is None:
                    continue
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

        def _make_2d_entry(uname, t, sp):
            """Reshape tensor if numel matches but shape differs, then wrap as ShardedWeight."""
            target = sp.local_shape
            if (
                tuple(t.shape) != tuple(target)
                and t.numel() == _paddle.to_tensor(list(target)).prod().item()
            ):
                t = t.reshape(target)
            return create_sharded_weight_with_new_local(uname, t, sp)

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
                    sharded_state[unified_name] = _make_2d_entry(
                        unified_name, tensor, sharded_param
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
                    sharded_state[unified_name] = _make_2d_entry(
                        unified_name, tensor, sharded_param
                    )

        return sharded_state

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)
