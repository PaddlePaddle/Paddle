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

import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

import paddle
from paddle.autograd import PyLayer
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    align,
    alignment,
    get_current_device_type,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
from paddle.distributed.fsdp._fsdp_context import (
    register_fsdp_context,
)


class BufferState(Enum):
    # Buffer status for lazy double buffer mechanism
    #
    # State transitions:
    #     FREED ──all_gather──> USING ──computation done──> READY ──release──> FREED
    #                             ^                            │
    #                             │         (reuse)            │
    #                             └────────────────────────────┘

    FREED = 1  # Released, buffer data is sharded, tmp_buffer not allocated
    USING = 2  # Unsharded and actively in use
    READY = 3  # Unsharded, marked for lazy release, can be reused
    SYNCING = 4  # Communication in progress


@dataclass
class BufferGroup:
    params: list = field(default_factory=list)
    dtype: object = None
    trainable: bool = None
    fsdp_unit_id: int = None
    is_tie: bool = False
    is_expert_param: bool = False
    fsdp_group: object = None
    params_buffer: 'TensorFusionBuffer' = None
    grads_buffer: 'TensorFusionBuffer' = None
    grads_use_sum: int = 0
    grads_use_cnt: int = 0


class TensorFusionBuffer:
    def __init__(
        self,
        unique_key,
        params,
        fsdp_group,
        dtype,
        is_params=False,
        main_grad_dtype=None,
    ):
        # Calculate total buffer size needed (with padding)
        self.unique_key = unique_key
        self.fsdp_group = fsdp_group
        self.fsdp_degree = fsdp_group.nranks
        self.is_sharded = fsdp_group.nranks > 1
        self.is_params = is_params
        self.dtype = dtype
        self.main_grad_dtype = (
            main_grad_dtype if main_grad_dtype is not None else dtype
        )
        self.total_buffer_size = 0
        self.param_offsets = {}
        self.tmp_data_buffer = None
        self.comm_task = None
        self.reduce_pending = False
        self.trainable = params[0].trainable

        for param in params:
            self.param_offsets[param.name] = self.total_buffer_size
            self.total_buffer_size += self.get_padded_size(param)

        self.shard_start = 0
        self.shard_end = self.total_buffer_size

        if is_params:
            # Create fused params_buffer
            # TODO(lizhenxing): Build full params_buffer on CPU and only move shards to GPU to minimize mem peaks
            self.data_buffer = paddle.zeros(
                shape=[self.total_buffer_size],
                dtype=dtype,
            )
            # Use BufferState enum instead of is_shard boolean, initial state is FREED (sharded)
            self.status = BufferState.FREED

            for param in params:
                offset = self.param_offsets[param.name]
                stop_gradient = param.stop_gradient
                _shape = param.shape
                param.stop_gradient = True
                param.flatten_()
                paddle.assign(
                    param,
                    self.data_buffer._slice(
                        offset,
                        offset + param._numel(),
                    ),
                )
                param._clear_data()
                param.stop_gradient = stop_gradient
                param.get_tensor()._set_dims(_shape)
            paddle.device.cuda.empty_cache()

            if self.fsdp_degree > 1:
                curr_rank = paddle.distributed.get_rank(fsdp_group)
                total_nums = self.data_buffer.shape[0]
                piece_len = (
                    total_nums + self.fsdp_degree - 1
                ) // self.fsdp_degree
                start = curr_rank * piece_len
                end = min(start + piece_len, total_nums)
                self.shard_start = start
                self.shard_end = end
                self.data_buffer = paddle.slice(
                    self.data_buffer, [0], [start], [end]
                ).clone()
                paddle.device.cuda.empty_cache()

            # Init params_buffer attr
            self.data_buffer.name = "fuse_params_" + str(unique_key)
            self.data_buffer.stop_gradient = params[0].stop_gradient
            self.data_buffer.optimize_attr = params[0].optimize_attr
        else:
            # Create fused grads_buffer with shard
            self.data_buffer = paddle.zeros(
                shape=[self.total_buffer_size // self.fsdp_degree],
                dtype=self.main_grad_dtype,
            )

            if self.is_sharded:
                shard_size = self.total_buffer_size // self.fsdp_degree
                self.shard_start = (
                    paddle.distributed.get_rank(fsdp_group) * shard_size
                )
                self.shard_end = self.shard_start + shard_size

            # Register get_main_grad method for each param, returns view_slice of grad_buffer
            for param in params:
                if param.trainable:
                    param._fusion_buffer = self
                    param._param_offsets = self.param_offsets

                    def get_grad_from_tmp_buf(param):
                        tmp_buffer = param._fusion_buffer.get_tmp_buffer()
                        offset = param._param_offsets[param.name]
                        main_grad = paddle._C_ops.view_slice(
                            tmp_buffer,
                            offset,
                            offset + param._numel(),
                        )
                        return main_grad

                    param.get_main_grad = get_grad_from_tmp_buf.__get__(param)

    def get_padded_size(self, param):
        size = np.prod(param.shape)
        align_size = (
            alignment[get_current_device_type()]
            // align[param.dtype]
            * self.fsdp_degree
        )
        return ((size + align_size - 1) // align_size) * align_size

    def param_shard(self, param):
        """Intersect a param's slot in the fused buffer with this rank's shard, or None."""
        offset = self.param_offsets[param.name]
        numel = int(np.prod(param.shape))
        begin = max(offset, self.shard_start)
        end = min(offset + numel, self.shard_end)
        if end <= begin:
            return None
        return (
            begin - self.shard_start,
            end - self.shard_start,
            slice(begin - offset, end - offset),
        )

    def get_tmp_buffer(self):
        if not self.is_sharded:
            return self.data_buffer
        # Reuse tmp_buffer if exists, else create
        if self.tmp_data_buffer is None:
            self.tmp_data_buffer = paddle.zeros(
                shape=[self.total_buffer_size], dtype=self.dtype
            )
        return self.tmp_data_buffer

    def clear_tmp_buffer(self):
        if self.tmp_data_buffer is not None:
            self.tmp_data_buffer._clear_data()
            self.tmp_data_buffer = None
            # paddle.device.cuda.empty_cache()


class FSDPBufferManager:
    def __init__(self, model, fsdp_unit_layers=None, moe_layers_name=None):
        self.model = model
        self.hcg = fleet.get_hybrid_communicate_group()
        self.dp_group = self.hcg.get_data_parallel_group()
        self._fsdp_group = self.hcg.get_sharding_parallel_group()
        use_ep = (
            hasattr(self.hcg, "get_expert_parallel_world_size")
            and self.hcg.get_expert_parallel_world_size() > 1
        )
        self._expert_fsdp_group = (
            self.hcg.get_moe_sharding_parallel_group() if use_ep else None
        )
        self.main_grad_dtype = paddle.float32

        paddle.device.cuda.empty_cache()

        # Layer types to wrap as FSDP sharding layers
        # Note: 'Qwen3VLTextDecoderLayer' is temporary; fleet models all use 'TransformerLayer'
        self.fsdp_unit_layers = fsdp_unit_layers or [
            'TransformerLayer',
            'Qwen3VLTextDecoderLayer',
            'Qwen3MoeDecoderLayer',
        ]
        self.moe_layers_name = moe_layers_name or [
            'Qwen3MoeMLP',
            'StandardMLPExpert',
        ]

        # Get tie_param_name if using tie_weights
        self.tie_param_name = None
        if hasattr(self.model, "get_input_embeddings"):
            input_embeddings = self.model.get_input_embeddings()
            if input_embeddings is not None and hasattr(
                input_embeddings, "weight"
            ):
                self.tie_param_name = input_embeddings.weight.name

        # Create buffer_groups
        for m in self.model.modules():
            if type(m).__name__ in self.moe_layers_name:
                for p in m.parameters():
                    p.is_moe_param = True
        self.buffer_groups = self._build_groups()
        self.param_to_buffer_id = {}

        # Create params_buffer, grads_buffer with groups
        for gid, group in enumerate(self.buffer_groups):
            params = group.params
            fsdp_group = group.fsdp_group

            group.params_buffer = TensorFusionBuffer(
                gid,
                params,
                fsdp_group,
                params[0].dtype,
                is_params=True,
            )

            if not params[0].stop_gradient:
                group.grads_buffer = TensorFusionBuffer(
                    gid,
                    params,
                    fsdp_group,
                    params[0].dtype,
                    main_grad_dtype=self.main_grad_dtype,
                )
            else:
                group.grads_buffer = None

            group.grads_use_sum = len(params)

            for param in params:
                self.param_to_buffer_id[param.name] = gid

    def _build_groups(self):
        param_to_unit_id = {}
        for unit_id, m in enumerate(self.model.modules()):
            if type(m).__name__ in self.fsdp_unit_layers:
                for p in m.parameters():
                    param_to_unit_id[p.name] = unit_id

        param_groups = []
        for param in self.model.parameters():
            name = param.name
            is_tie = (
                self.tie_param_name is not None and name == self.tie_param_name
            )
            is_expert = getattr(param, "is_moe_param", False) or getattr(
                param, "expert", False
            )
            color = getattr(param, "color", None)
            if isinstance(color, dict) and color.get("group") is not None:
                fsdp_group = color["group"]
            elif is_expert and self._expert_fsdp_group is not None:
                fsdp_group = self._expert_fsdp_group
            else:
                fsdp_group = self._fsdp_group

            param_attrs = {
                "dtype": param.dtype,
                "trainable": param.trainable,
                "fsdp_unit_id": param_to_unit_id.get(name),
                "is_tie": is_tie,
                "is_expert_param": is_expert,
                "fsdp_group": fsdp_group,
            }

            found_group = False
            for param_group in param_groups:
                if (
                    param_group.dtype == param_attrs["dtype"]
                    and param_group.trainable == param_attrs["trainable"]
                    and param_group.fsdp_unit_id == param_attrs["fsdp_unit_id"]
                    and param_group.is_tie == param_attrs["is_tie"]
                    and param_group.is_expert_param
                    == param_attrs["is_expert_param"]
                    and param_group.fsdp_group is param_attrs["fsdp_group"]
                ):
                    param_group.params.append(param)
                    found_group = True
                    break

            # Create new group if no matching
            if not found_group:
                param_groups.append(BufferGroup(params=[param], **param_attrs))

        def group_sort_key(group):
            priority = 0 if group.is_tie else (1 if not group.trainable else 2)
            return (
                priority,
                group.fsdp_unit_id
                if group.fsdp_unit_id is not None
                else float('inf'),
                group.is_expert_param,
            )

        # Buffer them by execution order
        return [g for g in sorted(param_groups, key=group_sort_key) if g.params]


class FSDPCommManager:
    def __init__(
        self,
        buffer_manager,
        enable_overlap=True,
        double_buffer_limit=2,
    ):
        self.buffer_manager = buffer_manager
        self.enable_overlap = enable_overlap
        self.grad_reduce_queue = []

        # for double buffer mechanism config
        self.double_buffer_limit = double_buffer_limit
        self.buffer_cnt_in_using = 0
        self.need_zero_grads = True
        self._last_backward_unit_id = None

    def _flush_expert_grads_after_unit(self, unit_id):
        # Backward walks the FSDP units from last to first, so every unit after
        # `unit_id` is done and its expert grads are final. Reducing here gives
        # expert params their overlap back on a structural trigger.
        if unit_id is None:
            return
        groups = self.buffer_manager.buffer_groups
        for group in reversed(groups):
            grads_buffer = group.grads_buffer
            if (
                not group.is_expert_param
                or group.fsdp_unit_id is None
                or grads_buffer is None
                or grads_buffer.tmp_data_buffer is None
            ):
                continue
            if group.fsdp_unit_id > unit_id:
                self._reduce_group_grads(group)

    def _release_one_buffer_if_needed(self):
        # Release a buffer with the READY status if needed
        while self.buffer_cnt_in_using >= self.double_buffer_limit:
            found = False
            for gid_idx, group in enumerate(self.buffer_manager.buffer_groups):
                if not group.params_buffer.is_sharded:
                    continue
                if group.params_buffer.status == BufferState.READY:
                    group.params_buffer.clear_tmp_buffer()
                    group.params_buffer.status = BufferState.FREED
                    self.buffer_cnt_in_using -= 1
                    found = True
                    break
            if not found:
                break

    def _next_buffer_id(self, gid, is_backward):
        # Get next buffer id for prefetch
        if is_backward:
            next_gid = gid - 1
            # Search backward for trainable buffer_groups
            while (
                next_gid >= 0
                and not self.buffer_manager.buffer_groups[
                    next_gid
                ].params_buffer.trainable
            ):
                next_gid -= 1
            return max(next_gid, 0)
        else:
            return min(gid + 1, len(self.buffer_manager.buffer_groups) - 1)

    def all_gather_params(self, params, is_backward=False):
        if len(params) == 0:
            return
        for param in params:
            gid = self.buffer_manager.param_to_buffer_id[param.name]
            group = self.buffer_manager.buffer_groups[gid]
            params_buffer = group.params_buffer
            # Use group-specific fsdp_group
            fsdp_group = group.fsdp_group

            if (
                is_backward
                and group.fsdp_unit_id != self._last_backward_unit_id
            ):
                self._last_backward_unit_id = group.fsdp_unit_id
                self._flush_expert_grads_after_unit(group.fsdp_unit_id)

            # Double buffer: reuse buffer if status is READY
            if params_buffer.status == BufferState.READY:
                # Reuse: READY -> USING, no need to all_gather again
                params_buffer.status = BufferState.USING

            # Overlap prefetch comm
            if self.enable_overlap:
                prefetch_count = 2
                curr_next_gid = gid
                for _ in range(prefetch_count):
                    next_gid = self._next_buffer_id(curr_next_gid, is_backward)
                    next_group = self.buffer_manager.buffer_groups[next_gid]
                    next_params_buffer = next_group.params_buffer
                    next_fsdp_group = next_group.fsdp_group
                    if (
                        next_params_buffer.status == BufferState.FREED
                        and next_fsdp_group.nranks > 1
                    ):
                        # Check double_buffer_limit before prefetch
                        self._release_one_buffer_if_needed()
                        next_params_buffer.status = BufferState.SYNCING
                        tmp_buffer_prefetch = (
                            next_params_buffer.get_tmp_buffer()
                        )
                        next_params_buffer.comm_task = (
                            paddle.distributed.all_gather(
                                tmp_buffer_prefetch,
                                next_params_buffer.data_buffer,
                                group=next_fsdp_group,
                                sync_op=False,
                            )
                        )
                        self.buffer_cnt_in_using += 1

            # Wait for async comm to complete: SYNCING -> USING
            if params_buffer.status == BufferState.SYNCING:
                params_buffer.status = BufferState.USING
                params_buffer.comm_task.wait()
                params_buffer.comm_task = None

            tmp_buffer = params_buffer.get_tmp_buffer()
            # Do all_gather in sync: FREED -> USING
            if params_buffer.status == BufferState.FREED:
                if params_buffer.is_sharded:
                    fsdp_group.process_group.all_gather(
                        params_buffer.data_buffer, tmp_buffer
                    ).wait()
                    self.buffer_cnt_in_using += 1
                params_buffer.status = BufferState.USING

            # Bind the unsharded param to the real param
            offset = params_buffer.param_offsets[param.name]
            tmp_param = paddle._C_ops.view_slice(
                tmp_buffer,
                offset,
                offset + param._numel(),
            )
            tmp_param.get_tensor()._set_dims(param.shape)
            param.get_tensor()._share_data_with(tmp_param.get_tensor())

    def shard_params(self, params, is_backward=False):
        affected_gids = set()
        for param in params:
            gid = self.buffer_manager.param_to_buffer_id.get(param.name)
            stop_gradient = param.stop_gradient
            _shape = param.shape
            param._clear_data()
            param.stop_gradient = stop_gradient
            param.get_tensor()._set_dims(_shape)

            affected_gids.add(gid)

        for gid in affected_gids:
            group = self.buffer_manager.buffer_groups[gid]
            if group.params_buffer.status == BufferState.USING:
                group.params_buffer.status = BufferState.READY

    def reduce_scatter_grads(self, param):
        if self.need_zero_grads:
            self.need_zero_grads = False
            for group in self.buffer_manager.buffer_groups:
                if group.grads_buffer is not None:
                    group.grads_buffer.data_buffer.zero_()
        gid = self.buffer_manager.param_to_buffer_id.get(param.name)
        group = self.buffer_manager.buffer_groups[gid]
        group.grads_use_cnt += 1
        param.main_grad = None

        if group.is_expert_param:
            return

        if group.grads_use_cnt == group.grads_use_sum:
            self._reduce_group_grads(group)

    def _reduce_group_grads(self, group):
        # Reduce-scatter one group's fused grad buffer over its own group.
        group.grads_use_cnt = 0
        grads_buffer = group.grads_buffer
        if grads_buffer is None:
            return
        if not grads_buffer.is_sharded:
            return
        if grads_buffer.reduce_pending:
            return
        fsdp_group = group.fsdp_group

        # Grad queue mechanism: wait and release completed reduce_scatter async tasks
        self._wait_for_grad_comm()

        tmp_buffer = grads_buffer.get_tmp_buffer()
        shard_size = grads_buffer.data_buffer.shape[0]
        grad_buffer_shard = tmp_buffer._slice(0, shard_size)
        tmp_buffer.scale_(1.0 / fsdp_group.nranks)
        if self.enable_overlap:
            # Comm grads async and check all comm_task before optimizer update
            grads_buffer.comm_task = paddle.distributed.reduce_scatter(
                grad_buffer_shard,
                tmp_buffer,
                op=paddle.distributed.ReduceOp.SUM,
                group=fsdp_group,
                sync_op=False,
            )
            grads_buffer.reduce_pending = True

            # Add async task to queue
            self.grad_reduce_queue.append(grads_buffer)
        else:
            paddle.distributed.reduce_scatter(
                grad_buffer_shard,
                tmp_buffer,
                op=paddle.distributed.ReduceOp.SUM,
                group=fsdp_group,
                sync_op=False,
            ).wait()
            grads_buffer.data_buffer.add_(grad_buffer_shard)
            grads_buffer.clear_tmp_buffer()

    def _finish_grads_buffer_reduce(self, grads_buffer):
        if grads_buffer.comm_task is not None:
            grads_buffer.comm_task.wait()
            grads_buffer.comm_task = None
            shard_size = grads_buffer.data_buffer.shape[0]
            grad_buffer_shard = grads_buffer.tmp_data_buffer._slice(
                0, shard_size
            )
            grads_buffer.data_buffer.add_(grad_buffer_shard)
        grads_buffer.clear_tmp_buffer()
        grads_buffer.reduce_pending = False
        if grads_buffer in self.grad_reduce_queue:
            self.grad_reduce_queue.remove(grads_buffer)

    def wait_grads_buffer_ready(self, param):
        gid = self.buffer_manager.param_to_buffer_id.get(param.name)
        grads_buffer = self.buffer_manager.buffer_groups[gid].grads_buffer
        if grads_buffer is not None and grads_buffer.reduce_pending:
            self._finish_grads_buffer_reduce(grads_buffer)

    def _wait_for_grad_comm(self, queue_limit=2):
        # Wait for async reduce_scatter tasks to complete and release resources
        # queue_limit: max queue size, default use 2, 0 means wait for all
        while len(self.grad_reduce_queue) > queue_limit:
            self._finish_grads_buffer_reduce(self.grad_reduce_queue[0])

    def finish_grads_sync(self):
        # Wait for all async reduce_scatter tasks, call before optimizer.step()
        groups = self.buffer_manager.buffer_groups
        for group in reversed(groups):
            grads_buffer = group.grads_buffer
            if (
                group.is_expert_param
                and grads_buffer is not None
                and grads_buffer.tmp_data_buffer is not None
            ):
                self._reduce_group_grads(group)
        for group in groups:
            if not group.is_expert_param and group.grads_use_cnt > 0:
                self._reduce_group_grads(group)
        self._wait_for_grad_comm(queue_limit=0)

    def reset_params_buffer_status(self):
        self._last_backward_unit_id = None
        for group in self.buffer_manager.buffer_groups:
            group.grads_use_cnt = 0
            params_buffer = group.params_buffer
            if params_buffer.status in (BufferState.READY, BufferState.USING):
                # Clear stale tmp_buffer to force re-all_gather with updated data_buffer
                params_buffer.clear_tmp_buffer()
                params_buffer.status = BufferState.FREED
                if not params_buffer.is_sharded:
                    continue
                if self.buffer_cnt_in_using > 0:
                    self.buffer_cnt_in_using -= 1


class FusionBackwardHook(PyLayer):
    @staticmethod
    def forward(ctx, *inputs, layer, comm_manager, recursive=False):
        ctx.layer = layer
        ctx.comm_manager = comm_manager
        ctx.recursive = recursive
        return inputs if len(inputs) > 1 else inputs[0]

    @staticmethod
    def backward(ctx, *args):
        trainable_params = []

        for param in ctx.layer.parameters(include_sublayers=ctx.recursive):
            if param.trainable:
                trainable_params.append(param)

        ctx.comm_manager.all_gather_params(trainable_params, is_backward=True)
        return args


class FusionForwardHook(PyLayer):
    @staticmethod
    def forward(ctx, *inputs, layer, comm_manager, recursive=False):
        ctx.layer = layer
        ctx.comm_manager = comm_manager
        ctx.recursive = recursive
        return inputs if len(inputs) > 1 else inputs[0]

    @staticmethod
    def backward(ctx, *args):
        ctx.comm_manager.shard_params(
            ctx.layer.parameters(include_sublayers=ctx.recursive),
            is_backward=True,
        )
        return args


class FullyShardFusion:
    def __init__(
        self,
        model,
        fsdp_unit_layers=None,
        moe_layers_name=None,
        enable_tensor_fusion_and_overlap=True,
    ):
        self.model = model
        self.buffer_manager = FSDPBufferManager(
            self.model, fsdp_unit_layers, moe_layers_name
        )
        self.comm_manager = FSDPCommManager(
            self.buffer_manager,
            enable_overlap=enable_tensor_fusion_and_overlap,
        )
        self.register_tensor_fusion_hooks(self.model)
        register_fsdp_context(self)
        # fused buffer name -> {structured param name: slice info in the local shard}
        self._shard_descs = {}
        # Redirect the model's own API; the original stays as our description base.
        self._model_sharded_state_dict = self.model.sharded_state_dict
        self.model.sharded_state_dict = self.sharded_state_dict

    def _base_sharded_state_dict(self, structured_name_prefix):
        """Model descriptions of every param; fused params get a scratch view since their storage is cleared."""
        cleared = []
        scratch = {}
        for group in self.buffer_manager.buffer_groups:
            for param in group.params:
                if param._is_initialized():
                    continue
                numel = int(np.prod(param.shape))
                cleared.append((param, param.shape, numel))
                scratch[param.dtype] = max(scratch.get(param.dtype, 0), numel)

        scratch = {
            dtype: paddle.empty([numel], dtype=dtype)
            for dtype, numel in scratch.items()
        }
        for param, shape, numel in cleared:
            view = paddle._C_ops.view_slice(scratch[param.dtype], 0, numel)
            view.get_tensor()._set_dims(shape)
            param.get_tensor()._share_data_with(view.get_tensor())
        try:
            return self._model_sharded_state_dict(structured_name_prefix)
        finally:
            for param, shape, _ in cleared:
                stop_gradient = param.stop_gradient
                param._clear_data()
                param.stop_gradient = stop_gradient
                param.get_tensor()._set_dims(shape)

    def _create_sharded_weight(self, key, tensor, slice_info):
        """Wrap the ``slice_info`` piece of ``tensor`` as a flattened ShardedWeight."""
        (
            begin,
            end,
            flattened_range,
            local_shape,
            global_shape,
            global_offset,
        ) = slice_info
        return ShardedWeight(
            key=key,
            local_tensor=tensor._slice(begin, end),
            local_shape=local_shape,
            global_shape=global_shape,
            global_offset=global_offset,
            is_flattened=True,
            flattened_range=flattened_range,
        )

    def sharded_state_dict(self, structured_name_prefix=""):
        """Describe every fused param as a flattened piece of this rank's shard, no all_gather."""
        base = self._base_sharded_state_dict(structured_name_prefix)

        static_to_struct = {}
        for key, sharded_weight in sorted(base.items()):
            static_to_struct.setdefault(
                sharded_weight.local_tensor.name, []
            ).append(key)

        result = dict(base)
        self._shard_descs = {}
        hcg = self.buffer_manager.hcg
        for group in self.buffer_manager.buffer_groups:
            params_buffer = group.params_buffer
            param_slice_info = {}
            # (nranks, rank) of the expert parallel group, (1, 0) without EP.
            ep_nranks, ep_rank = 1, 0
            if group.is_expert_param and hasattr(
                hcg, "get_expert_parallel_world_size"
            ):
                nranks = hcg.get_expert_parallel_world_size()
                if nranks > 1:
                    ep_nranks, ep_rank = nranks, hcg.get_expert_parallel_rank()
            for param in group.params:
                struct_names = static_to_struct.get(param.name)
                if not struct_names:
                    continue
                # Tied weights share one shard; only the first key carries optimizer state.
                ref = base[struct_names[0]]
                local_shape = tuple(ref.local_shape)
                global_shape = tuple(ref.global_shape)
                global_offset = tuple(ref.global_offset)
                if ep_nranks > 1 and global_shape == local_shape:
                    # Expert described as replicated: fold EP into axis 0 to keep keys distinct.
                    global_shape = (
                        local_shape[0] * ep_nranks,
                        *local_shape[1:],
                    )
                    global_offset = (
                        ep_rank * local_shape[0],
                        *(0 for _ in local_shape[1:]),
                    )
                shard = params_buffer.param_shard(param)
                if shard is None:
                    # Nothing of this param lives on this rank.
                    for struct_name in struct_names:
                        result.pop(struct_name, None)
                    continue
                begin, end, flattened_range = shard
                slice_info = (
                    begin,
                    end,
                    flattened_range,
                    local_shape,
                    global_shape,
                    global_offset,
                )
                param_slice_info[struct_names[0]] = slice_info
                for struct_name in struct_names:
                    result[struct_name] = self._create_sharded_weight(
                        struct_name, params_buffer.data_buffer, slice_info
                    )
            self._shard_descs[params_buffer.data_buffer.name] = param_slice_info
        return result

    def owns_optimizer_params(self, optimizer):
        """Whether ``optimizer`` steps on the fused buffers of this context."""
        # It still holds the model params; the fused buffers are substituted inside ``step``.
        managed = self.buffer_manager.param_to_buffer_id
        for param in optimizer._parameter_list:
            if getattr(param, "name", None) in managed:
                return True
        return False

    def init_optimizer_state(self, optimizer):
        """Create optimizer accumulators on the fused buffers before load."""
        parameter_list = [
            group.params_buffer.data_buffer
            for group in self.buffer_manager.buffer_groups
            if not group.params_buffer.data_buffer.stop_gradient
        ]
        optimizer._create_accumulators(
            paddle.base.framework.default_main_program().global_block(),
            parameter_list,
        )

    def optimizer_sharded_state_dict(self, optimizer):
        """Split the state keyed by ``fuse_params_<gid>`` into per-param flattened shards."""
        if not self._shard_descs:
            self.sharded_state_dict()

        # Longer tags first so the name split does not stop at a prefix.
        _optimizer_scalar_name = ("beta1_pow_acc", "beta2_pow_acc")
        _optimizer_non_scaler_name = (
            "moment2_max",
            "moment1",
            "moment2",
            "velocity",
        )
        _master_weight_suffix = re.compile(r"^(.*)_fp32_master_\d+$")

        def _split_optimizer_state_name(vname):
            """``fuse_params_0_fp32_master_0_moment1_3`` -> ``("fuse_params_0", "moment1_0")``."""
            for tag in _optimizer_non_scaler_name + _optimizer_scalar_name:
                marker = "_" + tag + "_"
                idx = vname.rfind(marker)
                if idx < 0:
                    continue
                if not vname[idx + len(marker) :].isdigit():
                    continue
                base = vname[:idx]
                matched = _master_weight_suffix.match(base)
                if matched:
                    base = matched.group(1)
                return base, f"{tag}_0"
            return None, None

        def _replicated_scalar(key, tensor):
            return ShardedWeight(
                key=key,
                local_tensor=tensor,
                local_shape=tuple(tensor.shape),
                global_shape=tuple(tensor.shape),
                global_offset=tuple(0 for _ in tensor.shape),
            )

        state_dict = dict(optimizer.state_dict())
        master_weights = state_dict.pop("master_weights", None)

        sharded_state = {}
        for vname, tensor in state_dict.items():
            base_name, tag = _split_optimizer_state_name(vname)
            param_slice_info = self._shard_descs.get(base_name)
            if param_slice_info is None:
                continue
            if tag.startswith(_optimizer_scalar_name):
                for struct_name in param_slice_info:
                    key = f"{struct_name}.{tag}"
                    sharded_state[key] = _replicated_scalar(key, tensor)
            else:
                for struct_name, slice_info in param_slice_info.items():
                    key = f"{struct_name}.{tag}"
                    sharded_state[key] = self._create_sharded_weight(
                        key, tensor, slice_info
                    )

        if master_weights:
            for base_name, tensor in master_weights.items():
                param_slice_info = self._shard_descs.get(base_name)
                if param_slice_info is None:
                    continue
                for struct_name, slice_info in param_slice_info.items():
                    key = f"{struct_name}.w_0"
                    sharded_state[key] = self._create_sharded_weight(
                        key, tensor, slice_info
                    )

        return sharded_state

    def comm_sync_and_reset_status(self):
        self.comm_manager.finish_grads_sync()
        self.comm_manager.reset_params_buffer_status()
        self.comm_manager.need_zero_grads = True
        # Reset main_grad for all trainable parameters
        for param in self.model.parameters():
            if param.trainable:
                param.main_grad = None

    def register_tensor_fusion_hooks(self, model):
        def _pre_forward_hook(sublayers, recursive=False):
            comm_manager = self.comm_manager

            @paddle.autograd.no_grad()
            def all_gather_comm(*_):
                comm_manager.all_gather_params(
                    sublayers.parameters(include_sublayers=recursive)
                )

            return all_gather_comm

        def _post_forward_hook(sublayers, recursive=False):
            comm_manager = self.comm_manager

            @paddle.autograd.no_grad()
            def shard_comm(*_):
                comm_manager.shard_params(
                    sublayers.parameters(include_sublayers=recursive)
                )

            return shard_comm

        def _update_main_grad_hook(param):
            comm_manager = self.comm_manager

            @paddle.autograd.no_grad()
            def comm_hook(grad):
                if grad is not None and grad._is_initialized():
                    # Share mem with grads_tmp_buffer
                    comm_manager.wait_grads_buffer_ready(param)
                    _main_grad = param.get_main_grad()
                    _main_grad.get_tensor()._set_dims(grad.shape)
                    param.main_grad = _main_grad
                    param.main_grad.add_(grad)
                    grad._clear_data()
                comm_manager.shard_params([param], is_backward=True)
                comm_manager.reduce_scatter_grads(param)

            return comm_hook

        def _post_backward_hook(param):
            param.main_grad = None
            if hasattr(param, "get_main_grad"):
                param._register_grad_hook(_update_main_grad_hook(param))

        for param in model.parameters():
            if param.trainable:
                _post_backward_hook(param)

        def _register_recursive(layer):
            is_unit = (
                type(layer).__name__ in self.buffer_manager.fsdp_unit_layers
            )

            if is_unit:
                # For FSDP Unit, register recursive hooks and stop recursion
                layer.register_forward_pre_hook(
                    _pre_forward_hook(layer, recursive=True)
                )
                layer.register_forward_post_hook(
                    _post_forward_hook(layer, recursive=True)
                )
                self._register_fusion_layer_hooks(layer, recursive=True)
                return

            if layer.parameters(include_sublayers=False):
                layer.register_forward_pre_hook(
                    _pre_forward_hook(layer, recursive=False)
                )
                layer.register_forward_post_hook(
                    _post_forward_hook(layer, recursive=False)
                )
                self._register_fusion_layer_hooks(layer, recursive=False)

            for child in layer.children():
                _register_recursive(child)

        _register_recursive(model)

    def _register_fusion_layer_hooks(self, layer, recursive=False):
        def _forward_post_hook(layer, inputs, outputs):
            if isinstance(outputs, dict):
                for key, value in outputs.items():
                    if (
                        isinstance(value, paddle.Tensor)
                        and not value.stop_gradient
                    ):
                        outputs[key] = FusionBackwardHook.apply(
                            value,
                            layer=layer,
                            comm_manager=self.comm_manager,
                            recursive=recursive,
                        )
                return outputs
            elif isinstance(outputs, tuple):
                result = FusionBackwardHook.apply(
                    *outputs,
                    layer=layer,
                    comm_manager=self.comm_manager,
                    recursive=recursive,
                )
                if not isinstance(result, tuple):
                    result = (result,)
                return result
            else:
                return FusionBackwardHook.apply(
                    outputs,
                    layer=layer,
                    comm_manager=self.comm_manager,
                    recursive=recursive,
                )

        def _forward_pre_hook(layer, inputs):
            return FusionForwardHook.apply(
                *inputs,
                layer=layer,
                comm_manager=self.comm_manager,
                recursive=recursive,
            )

        layer.register_forward_post_hook(_forward_post_hook)

        # Register an additional hook for tie_weights shard_params
        for param in layer.parameters(include_sublayers=False):
            if param.name == self.comm_manager.buffer_manager.tie_param_name:
                layer.register_forward_pre_hook(_forward_pre_hook)
