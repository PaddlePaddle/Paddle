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
from collections import OrderedDict
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
    is_expert_param: bool = False
    is_tie: bool = False
    no_decay: bool = False
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

            if self.is_sharded:
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
                    param._main_grad_addr = None

                    def get_grad_from_tmp_buf(param, shape=None):
                        tmp_buffer = param._fusion_buffer.get_tmp_buffer()
                        offset = param._param_offsets[param.name]
                        main_grad = paddle._C_ops.view_slice(
                            tmp_buffer,
                            offset,
                            offset + param._numel(),
                        )
                        main_grad.get_tensor()._set_dims(
                            param.shape if shape is None else shape
                        )
                        param.main_grad = main_grad
                        param._main_grad_addr = main_grad.data_ptr()
                        return main_grad

                    param.get_main_grad = get_grad_from_tmp_buf.__get__(param)

    def rebind_main_grad(self, param):
        stale_grad = getattr(param, "main_grad", None)
        if not (self.is_sharded and self.tmp_data_buffer is None):
            if stale_grad is not None and stale_grad._is_initialized():
                if stale_grad.data_ptr() == getattr(
                    param, "_main_grad_addr", None
                ):
                    return
        main_grad = param.get_main_grad()
        if stale_grad is not None and stale_grad._is_initialized():
            if stale_grad.data_ptr() != main_grad.data_ptr():
                main_grad.add_(stale_grad)
                stale_grad._clear_data()

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
                shape=[self.total_buffer_size],
                dtype=self.dtype if self.is_params else self.main_grad_dtype,
            )
        return self.tmp_data_buffer

    def do_all_gather(self):
        return paddle.distributed.all_gather(
            self.get_tmp_buffer(),
            self.data_buffer,
            group=self.fsdp_group,
            sync_op=False,
        )

    def do_reduce_scatter(self):
        tmp_buffer = self.get_tmp_buffer()
        shard = tmp_buffer._slice(0, self.data_buffer.shape[0])
        tmp_buffer.scale_(1.0 / self.fsdp_degree)
        return paddle.distributed.reduce_scatter(
            shard,
            tmp_buffer,
            op=paddle.distributed.ReduceOp.SUM,
            group=self.fsdp_group,
            sync_op=False,
        )

    def accumulate_reduced_grad(self):
        shard = self.get_tmp_buffer()._slice(0, self.data_buffer.shape[0])
        self.data_buffer.add_(shard)
        self.clear_tmp_buffer()

    def clear_tmp_buffer(self):
        if self.tmp_data_buffer is not None:
            self.tmp_data_buffer._clear_data()
            self.tmp_data_buffer = None


class FSDPBufferManager:
    def __init__(
        self,
        model,
        fsdp_unit_layers=None,
        main_grad_dtype=None,
    ):
        self.model = model
        self.hcg = fleet.get_hybrid_communicate_group()
        self._fsdp_group = self.hcg.get_sharding_parallel_group()
        use_ep = (
            hasattr(self.hcg, "get_expert_parallel_world_size")
            and self.hcg.get_expert_parallel_world_size() > 1
        )
        self._ep_fsdp_group = (
            self.hcg.get_moe_sharding_parallel_group()
            if use_ep
            else self._fsdp_group
        )
        self.main_grad_dtype = (
            main_grad_dtype if main_grad_dtype is not None else paddle.float32
        )

        # Layer types to wrap as FSDP sharding layers
        # Note: 'Qwen3VLTextDecoderLayer' is temporary; fleet models all use 'TransformerLayer'
        self.fsdp_unit_layers = fsdp_unit_layers or [
            'TransformerLayer',
            'Qwen3VLTextDecoderLayer',
            'Qwen3MoeDecoderLayer',
        ]
        # Get tie_param_name if using tie_weights
        self.tie_param_name = None
        if hasattr(self.model, "get_input_embeddings"):
            input_embeddings = self.model.get_input_embeddings()
            if input_embeddings is not None and hasattr(
                input_embeddings, "weight"
            ):
                self.tie_param_name = input_embeddings.weight.name

        param_to_unit_id = {}
        for unit_id, m in enumerate(self.model.modules()):
            if type(m).__name__ in self.fsdp_unit_layers:
                for p in m.parameters():
                    param_to_unit_id[p.name] = unit_id

        keyed_params = OrderedDict()
        for param in self.model.parameters():
            is_expert = getattr(param, "is_moe_param", False) or getattr(
                param, "expert", False
            )
            color = getattr(param, "color", None)
            if isinstance(color, dict) and color.get("group") is not None:
                fsdp_group = color["group"]
            elif is_expert:
                fsdp_group = self._ep_fsdp_group
            else:
                fsdp_group = self._fsdp_group
            key = (
                param.dtype,
                param.trainable,
                param_to_unit_id.get(param.name),
                is_expert,
                fsdp_group,
                param.name == self.tie_param_name,
                # A buffer must not mix params with and without weight decay.
                bool(getattr(param, "no_weight_decay", False)),
            )
            keyed_params.setdefault(key, []).append(param)

        def sort_key(item):
            _, trainable, unit_id, is_expert_param, _, is_tie, _ = item[0]
            return (
                0 if is_tie else (1 if not trainable else 2),
                unit_id if unit_id is not None else float('inf'),
                is_expert_param,
            )

        self.buffer_groups = [
            BufferGroup(
                params=params,
                dtype=dtype,
                trainable=trainable,
                fsdp_unit_id=unit_id,
                is_expert_param=is_expert_param,
                fsdp_group=fsdp_group,
                is_tie=is_tie,
                no_decay=no_decay,
            )
            for (
                dtype,
                trainable,
                unit_id,
                is_expert_param,
                fsdp_group,
                is_tie,
                no_decay,
            ), params in sorted(keyed_params.items(), key=sort_key)
        ]

        self.param_to_buffer_id = {}
        for gid, group in enumerate(self.buffer_groups):
            params = group.params
            group.params_buffer = TensorFusionBuffer(
                gid, params, group.fsdp_group, group.dtype, is_params=True
            )
            if not params[0].stop_gradient:
                group.grads_buffer = TensorFusionBuffer(
                    gid,
                    params,
                    group.fsdp_group,
                    group.dtype,
                    main_grad_dtype=paddle.float32
                    if group.is_expert_param or group.dtype == paddle.float32
                    else self.main_grad_dtype,
                )
            group.grads_use_sum = len(params)
            for param in params:
                self.param_to_buffer_id[param.name] = gid


class FSDPCommManager:
    def __init__(
        self,
        buffer_manager,
        enable_overlap=True,
        double_buffer_limit=None,
    ):
        self.buffer_manager = buffer_manager
        self.enable_overlap = enable_overlap
        self.grad_reduce_queue = []

        # for double buffer mechanism config
        groups_per_unit = 2
        self.double_buffer_limit = double_buffer_limit or (groups_per_unit * 2)
        self.prefetch_units = max(
            self.double_buffer_limit // groups_per_unit - 1, 0
        )
        self.buffer_cnt_in_using = 0
        self.need_zero_grads = True
        self._last_backward_unit_id = None

    # ------------------------------------------------------------------
    # Params all_gather double buffer. Owns buffer_cnt_in_using /
    # double_buffer_limit / prefetch_units and the BufferState machine.
    # ------------------------------------------------------------------

    def _release_one_buffer_if_needed(self, keep=()):
        # Release a buffer with the READY status if needed
        while self.buffer_cnt_in_using >= self.double_buffer_limit:
            found = False
            for gid_idx, group in enumerate(self.buffer_manager.buffer_groups):
                if gid_idx in keep:
                    continue
                if not group.params_buffer.is_sharded:
                    continue
                if group.params_buffer.status == BufferState.READY:
                    group.params_buffer.status = BufferState.FREED
                    group.params_buffer.clear_tmp_buffer()
                    self.buffer_cnt_in_using -= 1
                    found = True
                    break
            if not found:
                break

    def _next_buffer_id(self, req_gids, is_backward):
        # Get next buffer id for prefetch
        groups = self.buffer_manager.buffer_groups
        if self.prefetch_units <= 0:
            return []
        step = -1 if is_backward else 1
        requested = set(req_gids)
        next_gid = (min(req_gids) if is_backward else max(req_gids)) + step
        while next_gid in requested:
            next_gid += step

        gids = []
        seen_units = set()
        while 0 <= next_gid < len(groups):
            group = groups[next_gid]
            if is_backward and not group.params_buffer.trainable:
                next_gid += step
                continue
            if group.fsdp_unit_id not in seen_units:
                if len(seen_units) >= self.prefetch_units:
                    break
                seen_units.add(group.fsdp_unit_id)
            gids.append(next_gid)
            next_gid += step
        return gids

    def _issue_async_gather(self, gid):
        params_buffer = self.buffer_manager.buffer_groups[gid].params_buffer
        params_buffer.status = BufferState.SYNCING
        params_buffer.comm_task = params_buffer.do_all_gather()
        self.buffer_cnt_in_using += 1

    def all_gather_params(self, params, is_backward=False):
        if len(params) == 0:
            return

        req_gids = []
        for param in params:
            gid = self.buffer_manager.param_to_buffer_id[param.name]
            if gid not in req_gids:
                req_gids.append(gid)

        if is_backward:
            unit_id = self.buffer_manager.buffer_groups[
                req_gids[0]
            ].fsdp_unit_id
            if unit_id != self._last_backward_unit_id:
                self._last_backward_unit_id = unit_id
                self._flush_expert_grads_after_unit(unit_id)

        if self.enable_overlap:
            keep = set(req_gids)
            for gid in req_gids:
                group = self.buffer_manager.buffer_groups[gid]
                if not group.params_buffer.is_sharded:
                    continue
                if group.params_buffer.status == BufferState.FREED:
                    self._release_one_buffer_if_needed(keep)
                    self._issue_async_gather(gid)

            for next_gid in self._next_buffer_id(req_gids, is_backward):
                next_params_buffer = self.buffer_manager.buffer_groups[
                    next_gid
                ].params_buffer
                if not next_params_buffer.is_sharded:
                    continue
                if next_params_buffer.status != BufferState.FREED:
                    continue
                self._release_one_buffer_if_needed(keep)
                if self.buffer_cnt_in_using >= self.double_buffer_limit:
                    break
                self._issue_async_gather(next_gid)

        for param in params:
            gid = self.buffer_manager.param_to_buffer_id[param.name]
            group = self.buffer_manager.buffer_groups[gid]
            params_buffer = group.params_buffer

            # Double buffer: reuse buffer if status is READY
            if params_buffer.status == BufferState.READY:
                # Reuse: READY -> USING, no need to all_gather again
                params_buffer.status = BufferState.USING

            # Wait for async comm to complete: SYNCING -> USING
            if params_buffer.status == BufferState.SYNCING:
                params_buffer.status = BufferState.USING
                params_buffer.comm_task.wait()
                params_buffer.comm_task = None

            tmp_buffer = params_buffer.get_tmp_buffer()
            # Do all_gather in sync: FREED -> USING
            if params_buffer.status == BufferState.FREED:
                if params_buffer.is_sharded:
                    params_buffer.fsdp_group.process_group.all_gather(
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
            group = self.buffer_manager.buffer_groups[gid]
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

    def reset_params_buffer_status(self):
        self._last_backward_unit_id = None
        for group in self.buffer_manager.buffer_groups:
            params_buffer = group.params_buffer
            if params_buffer.status in (BufferState.READY, BufferState.USING):
                params_buffer.clear_tmp_buffer()
                params_buffer.status = BufferState.FREED
                if not params_buffer.is_sharded:
                    continue
                if self.buffer_cnt_in_using > 0:
                    self.buffer_cnt_in_using -= 1

    # ------------------------------------------------------------------
    # Grads reduce_scatter queue. Owns grad_reduce_queue / need_zero_grads;
    # must not touch the double buffer state above.
    # ------------------------------------------------------------------

    def _maybe_zero_grads(self):
        if not self.need_zero_grads:
            return
        self.need_zero_grads = False
        for group in self.buffer_manager.buffer_groups:
            if group.grads_buffer is not None:
                group.grads_buffer.data_buffer.zero_()

    def _ensure_grads_writable(self, param):
        gid = self.buffer_manager.param_to_buffer_id.get(param.name)
        if gid is None:
            return
        grads_buffer = self.buffer_manager.buffer_groups[gid].grads_buffer
        if grads_buffer is None:
            return
        while grads_buffer in self.grad_reduce_queue:
            self._wait_for_grad_comm(
                queue_limit=len(self.grad_reduce_queue) - 1
            )

    def reduce_scatter_grads(self, param):
        self._maybe_zero_grads()
        gid = self.buffer_manager.param_to_buffer_id.get(param.name)
        group = self.buffer_manager.buffer_groups[gid]
        group.grads_use_cnt += 1
        param.main_grad = None

        if group.grads_buffer is not None and (
            group.is_expert_param or not group.grads_buffer.is_sharded
        ):
            return

        if group.grads_use_cnt == group.grads_use_sum:
            self._reduce_group_grads(group)

    def _reduce_group_grads(self, group):
        # Reduce-scatter one group's fused grad buffer over its own fsdp_group.
        group.grads_use_cnt = 0
        grads_buffer = group.grads_buffer
        if (
            grads_buffer is None
            or not grads_buffer.is_sharded
            or grads_buffer.tmp_data_buffer is None
            or grads_buffer in self.grad_reduce_queue
        ):
            return
        # Grad queue mechanism: wait and release completed reduce_scatter async tasks
        self._wait_for_grad_comm()
        grads_buffer.comm_task = grads_buffer.do_reduce_scatter()
        self.grad_reduce_queue.append(grads_buffer)
        if not self.enable_overlap:
            self._wait_for_grad_comm(queue_limit=0)

    def _flush_expert_grads_after_unit(self, unit_id):
        # Backward walks the FSDP units from last to first, so every unit after
        # `unit_id` is done and its expert grads are final. Reducing here gives
        # expert params their overlap back on a structural trigger.
        if unit_id is None:
            return
        for group in reversed(self.buffer_manager.buffer_groups):
            if not group.is_expert_param or group.fsdp_unit_id is None:
                continue
            if group.fsdp_unit_id > unit_id:
                self._reduce_group_grads(group)

    def _wait_for_grad_comm(self, queue_limit=2):
        while len(self.grad_reduce_queue) > queue_limit:
            grads_buffer = self.grad_reduce_queue.pop(0)
            if grads_buffer.comm_task is None:
                grads_buffer.clear_tmp_buffer()
                continue
            grads_buffer.comm_task.wait()
            grads_buffer.comm_task = None
            grads_buffer.accumulate_reduced_grad()

    def finish_grads_sync(self):
        # Wait for all async reduce_scatter tasks, call before optimizer.step()
        self._wait_for_grad_comm(queue_limit=0)
        for group in self.buffer_manager.buffer_groups:
            grads_buffer = group.grads_buffer
            if grads_buffer is None:
                continue
            group.grads_use_cnt = 0
            if not grads_buffer.is_sharded:
                continue
            if grads_buffer.tmp_data_buffer is None:
                continue
            grads_buffer.do_reduce_scatter().wait()
            grads_buffer.accumulate_reduced_grad()


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
        return inputs

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
        enable_tensor_fusion_and_overlap=True,
        double_buffer_limit=None,
        mp_policy=None,
    ):
        self.model = model
        self.mp_policy = mp_policy
        self.buffer_manager = FSDPBufferManager(
            self.model,
            fsdp_unit_layers,
            None if mp_policy is None else mp_policy.reduce_dtype,
        )
        self.comm_manager = FSDPCommManager(
            self.buffer_manager,
            enable_overlap=enable_tensor_fusion_and_overlap,
            double_buffer_limit=double_buffer_limit,
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

    def bind_decay_param_fun(self, optimizer):
        """Answer ``apply_decay_param_fun`` for the fused buffers.

        The callback matches model param names, so every ``fuse_params_*`` name
        would answer False and weight decay would be dropped silently.
        """
        fun = getattr(optimizer, "_apply_decay_param_fun", None)
        if fun is None or getattr(optimizer, "_fsdp_decay_fun_bound", False):
            return
        buffer_decay = {}
        for group in self.buffer_manager.buffer_groups:
            buffer_name = group.params_buffer.data_buffer.name
            decayed, exempt = [], []
            for param in group.params:
                (decayed if fun(param.name) else exempt).append(param.name)
            if decayed and exempt:
                # A fused update can only honor one decay answer per buffer.
                raise ValueError(
                    f"FSDP: fused buffer {buffer_name} mixes params with and "
                    "without weight decay; mark the exempt ones with "
                    "`param.no_weight_decay = True` before fully_shard(). "
                    f"Decayed: {decayed}; exempt: {exempt}."
                )
            buffer_decay[buffer_name] = bool(decayed) and not group.no_decay

        def fsdp_apply_decay_param_fun(name):
            if name in buffer_decay:
                return buffer_decay[name]
            return fun(name)

        optimizer._apply_decay_param_fun = fsdp_apply_decay_param_fun
        optimizer._fsdp_decay_fun_bound = True

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

        inner = optimizer
        while hasattr(inner, "_inner_opt"):
            inner = inner._inner_opt
        # Keep the raw value: Adadelta's acc names start with '_' themselves.
        tags = sorted(
            {
                getattr(cls, attr)
                for cls in type(inner).__mro__
                for attr in vars(cls)
                if attr.endswith("_str")
                and "_acc" in attr
                and isinstance(getattr(cls, attr), str)
            },
            key=len,
            reverse=True,
        )
        _master_weight_suffix = re.compile(r"^(.*)_fp32_master_\d+$")
        # ``_add_accumulator`` prefixes the accumulator with the optimizer name.
        opt_name = getattr(inner, "_name", None)

        def _split_optimizer_state_name(vname):
            """``fuse_params_0_fp32_master_0_moment1_3`` -> ``("fuse_params_0", "moment1_0")``."""
            for tag in tags:
                marker = "_" + tag + "_"
                idx = vname.rfind(marker)
                if idx < 0:
                    continue
                if not vname[idx + len(marker) :].isdigit():
                    continue
                base = vname[:idx]
                if opt_name and base.endswith("_" + opt_name):
                    base = base[: -len(opt_name) - 1]
                matched = _master_weight_suffix.match(base)
                if matched:
                    base = matched.group(1)
                return base, f"{tag.lstrip('_')}_0"
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
            owner = next(
                (
                    name
                    for name in self._shard_descs
                    if vname.startswith(name + "_")
                ),
                None,
            )
            if owner is None:
                continue
            base_name, tag = _split_optimizer_state_name(vname)
            if base_name != owner:
                raise NotImplementedError(
                    f"FSDP cannot save optimizer state '{vname}' of "
                    f"{type(inner).__name__} on fused buffer '{owner}': the "
                    "accumulator is not declared through a '_*_acc*_str' "
                    "attribute, and skipping it would silently reset the state "
                    "on load."
                )
            param_slice_info = self._shard_descs[owner]
            if int(tensor.numel()) == 1:
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

    @paddle.autograd.no_grad()
    def _bind_expert_main_grads(self, params):
        self.comm_manager._maybe_zero_grads()
        for param in params:
            if not param.trainable:
                continue
            if not getattr(param, "is_moe_param", False):
                continue
            if not hasattr(param, "get_main_grad"):
                continue
            param._fusion_buffer.rebind_main_grad(param)

    def _register_grad_hook_once(self, param):
        param.main_grad = None
        if not hasattr(param, "get_main_grad"):
            return
        if getattr(param, "_fsdp_grad_hooked", False):
            return
        param._fsdp_grad_hooked = True
        comm_manager = self.comm_manager

        @paddle.autograd.no_grad()
        def comm_hook(grad):
            comm_manager._maybe_zero_grads()
            comm_manager._ensure_grads_writable(param)
            if grad is not None and grad._is_initialized():
                # Share mem with grads_tmp_buffer
                param.get_main_grad(grad.shape)
                param.main_grad.add_(grad)
                grad._clear_data()
            comm_manager.shard_params([param], is_backward=True)
            comm_manager.reduce_scatter_grads(param)

        param._register_grad_hook(comm_hook)

    def register_tensor_fusion_hooks(self, model):
        def _pre_forward_hook(sublayers, recursive=False):
            comm_manager = self.comm_manager

            @paddle.autograd.no_grad()
            def all_gather_comm(*_):
                params = sublayers.parameters(include_sublayers=recursive)
                comm_manager.all_gather_params(params)
                self._bind_expert_main_grads(params)

            return all_gather_comm

        def _post_forward_hook(sublayers, recursive=False):
            comm_manager = self.comm_manager

            @paddle.autograd.no_grad()
            def shard_comm(*_):
                comm_manager.shard_params(
                    sublayers.parameters(include_sublayers=recursive)
                )

            return shard_comm

        for param in model.parameters():
            if param.trainable:
                self._register_grad_hook_once(param)

        @paddle.autograd.no_grad()
        def _bind_experts_pre_forward(layer, inputs):
            self._bind_expert_main_grads(model.parameters())

        model.register_forward_pre_hook(_bind_experts_pre_forward)

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
