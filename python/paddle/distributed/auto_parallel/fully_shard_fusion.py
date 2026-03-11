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

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    align,
    alignment,
    get_current_device_type,
)

# Global registry for fsdp_context
_g_fsdp_context = None


def register_fsdp_context(context):
    global _g_fsdp_context
    _g_fsdp_context = context


def get_fsdp_context():
    return _g_fsdp_context


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
    params_buffer: 'TensorFusionBuffer' = None
    grads_buffer: 'TensorFusionBuffer' = None
    params_use_sum: int = 0
    params_use_cnt: int = 0
    grads_use_sum: int = 0
    grads_use_cnt: int = 0


def _dtensor_from_local(local_tensor, mesh, placements):
    global_dims = list(local_tensor.shape)
    for idx, placement in enumerate(placements):
        if placement.is_shard():
            global_dims[placement.get_dim()] = (
                global_dims[placement.get_dim()] * mesh.shape[idx]
            )
    place = paddle.framework._current_expected_place()
    place = paddle.framework._get_paddle_place(place)

    return paddle.Tensor(
        local_tensor,
        dims=global_dims,
        process_mesh=mesh,
        placements=placements,
        place=place,
    )


class TensorFusionBuffer:
    def __init__(self, unique_key, params, fsdp_degree, dtype, is_params=False):
        # Calculate total buffer size needed (with padding)
        self.unique_key = unique_key
        self.fsdp_degree = fsdp_degree
        self.dtype = dtype
        self.total_buffer_size = 0
        self.param_offsets = {}
        self.tmp_data_buffer = None
        self.comm_task = None
        self.trainable = params[0].trainable

        for param in params:
            self.param_offsets[param.name] = self.total_buffer_size
            self.total_buffer_size += self.get_padded_size(param)

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
                local_shape = param._local_shape
                param.stop_gradient = True
                param._local_value().flatten_()
                paddle.assign(
                    param._local_value(),
                    self.data_buffer._slice(
                        offset,
                        offset + param._numel(),
                    ),
                )

                param._clear_data()
                param.stop_gradient = stop_gradient
                param._local_value().get_tensor()._set_dims(local_shape)
                paddle.device.cuda.empty_cache()

            mesh = dist.auto_parallel.get_mesh()
            curr_global_rank = paddle.distributed.get_rank()
            if curr_global_rank in mesh.process_ids:
                total_nums = self.data_buffer.shape[0]
                num_of_pieces = mesh.shape[0]
                piece_len = (total_nums + num_of_pieces - 1) // num_of_pieces
                rank_relative = mesh.process_ids.index(curr_global_rank)
                start = rank_relative * piece_len
                end = min(start + piece_len, total_nums)
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
                dtype=dtype,
            )

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

    def get_tmp_buffer(self):
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
    def __init__(self, model, mesh, fsdp_unit_layers=None):
        self.model = model
        self._fsdp_group = mesh.get_group("dp")

        # Layer types to wrap as FSDP sharding layers
        # Note: 'Qwen3VLTextDecoderLayer' is temporary; fleet models all use 'TransformerLayer'
        self.fsdp_unit_layers = fsdp_unit_layers or [
            'TransformerLayer',
            'Qwen3VLTextDecoderLayer',
        ]

        # Get tie_param_name if using tie_weights
        self.tie_param_name = None
        if hasattr(self.model, "get_input_embeddings"):
            self.tie_param_name = self.model.get_input_embeddings().weight.name

        # Create buffer_groups
        grouped_params = self._build_groups()
        self.buffer_groups = []
        self.param_to_buffer_id = {}

        # Create params_buffer, grads_buffer with groups
        for gid, params in grouped_params.items():
            params_buffer = TensorFusionBuffer(
                gid,
                params,
                self._fsdp_group.nranks,
                params[0].dtype,
                is_params=True,
            )

            if not params[0].stop_gradient:
                grads_buffer = TensorFusionBuffer(
                    gid,
                    params,
                    self._fsdp_group.nranks,
                    paddle.float32,
                )
            else:
                grads_buffer = None

            self.buffer_groups.append(
                BufferGroup(
                    params=params,
                    dtype=params[0].dtype,
                    trainable=params[0].trainable,
                    params_buffer=params_buffer,
                    grads_buffer=grads_buffer,
                    params_use_sum=len(params),
                    params_use_cnt=0,
                    grads_use_sum=len(params),
                    grads_use_cnt=0,
                )
            )

            for param in params:
                self.param_to_buffer_id[param.name] = gid

    def _build_groups(self):
        parameters = self.model.parameters()
        grouped_params = OrderedDict()
        curr_gid = 0

        param_to_unit_id = {}
        for unit_id, m in enumerate(self.model.modules()):
            if type(m).__name__ in self.fsdp_unit_layers:
                for p in m.parameters():
                    param_to_unit_id[p.name] = unit_id

        param_groups = []
        for param in parameters:
            name = param.name
            is_tie = (
                self.tie_param_name is not None and name == self.tie_param_name
            )

            param_attrs = {
                "dtype": param.dtype,
                "trainable": param.trainable,
                "fsdp_unit_id": param_to_unit_id.get(name),
                "is_tie": is_tie,
            }

            found_group = False
            for param_group in param_groups:
                if (
                    param_group.dtype == param_attrs["dtype"]
                    and param_group.trainable == param_attrs["trainable"]
                    and param_group.fsdp_unit_id == param_attrs["fsdp_unit_id"]
                    and param_group.is_tie == param_attrs["is_tie"]
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
                group.fsdp_unit_id if group.fsdp_unit_id is not None else 999,
            )

        sorted_groups = sorted(param_groups, key=group_sort_key)

        # For each sorted parameter group, buffer them by execution order
        for param_group in sorted_groups:
            cur_params = param_group.params
            if len(cur_params) == 0:
                continue
            for p in cur_params:
                grouped_params.setdefault(curr_gid, []).append(p)
            curr_gid += 1

        return grouped_params


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

    def _release_one_buffer_if_needed(self):
        # Release a buffer with the READY status if needed
        while self.buffer_cnt_in_using >= self.double_buffer_limit:
            for group in self.buffer_manager.buffer_groups:
                if group.params_buffer.status == BufferState.READY:
                    group.params_buffer.status = BufferState.FREED
                    group.params_buffer.clear_tmp_buffer()
                    self.buffer_cnt_in_using -= 1
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
            group.params_use_cnt += 1
            params_buffer = group.params_buffer

            # Double buffer: reuse buffer if status is READY
            if params_buffer.status == BufferState.READY:
                # Reuse: READY -> USING, no need to all_gather again
                params_buffer.status = BufferState.USING

            # Overlap prefetch comm
            if self.enable_overlap:
                next_gid = self._next_buffer_id(gid, is_backward)
                next_params_buffer = self.buffer_manager.buffer_groups[
                    next_gid
                ].params_buffer
                if next_params_buffer.status == BufferState.FREED:
                    # Check double_buffer_limit before prefetch
                    self._release_one_buffer_if_needed()
                    next_params_buffer.status = BufferState.SYNCING
                    tmp_buffer_prefetch = next_params_buffer.get_tmp_buffer()
                    next_params_buffer.comm_task = (
                        paddle.distributed.all_gather(
                            tmp_buffer_prefetch,
                            next_params_buffer.data_buffer,
                            group=self.buffer_manager._fsdp_group,
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
                self.buffer_manager._fsdp_group.process_group.all_gather(
                    params_buffer.data_buffer, tmp_buffer
                ).wait()
                params_buffer.status = BufferState.USING
                self.buffer_cnt_in_using += 1

            # Bind the unsharded param to the real param
            offset = params_buffer.param_offsets[param.name]
            tmp_param = paddle._C_ops.view_slice(
                tmp_buffer,
                offset,
                offset + param._numel(),
            )
            tmp_param.get_tensor()._set_dims(param.shape)
            tmp_param = _dtensor_from_local(
                tmp_param,
                param.process_mesh,
                param.placements,
            )
            param.get_tensor()._share_data_with(tmp_param.get_tensor())

    def shard_params(self, params, is_backward=False):
        for param in params:
            gid = self.buffer_manager.param_to_buffer_id[param.name]
            group = self.buffer_manager.buffer_groups[gid]
            stop_gradient = param.stop_gradient
            local_shape = param._local_shape
            param._clear_data()
            param.stop_gradient = stop_gradient
            param._local_value().get_tensor()._set_dims(local_shape)

            # When all params in buffer_groups are used done
            if group.params_use_cnt == group.params_use_sum:
                group.params_use_cnt = 0
                # for double buffer lazy release, USING -> READY
                group.params_buffer.status = BufferState.READY

    def reduce_scatter_grads(self, param):
        if self.need_zero_grads:
            self.need_zero_grads = False
            for group in self.buffer_manager.buffer_groups:
                if group.grads_buffer is not None:
                    group.grads_buffer.data_buffer.zero_()
        gid = self.buffer_manager.param_to_buffer_id[param.name]
        group = self.buffer_manager.buffer_groups[gid]
        group.grads_use_cnt += 1
        param.main_grad = None

        if group.grads_use_cnt == group.grads_use_sum:
            group.grads_use_cnt = 0

            # reduce_scatter from tmp_grad_buffer into grads_buffer
            grads_buffer = group.grads_buffer

            # Grad queue mechanism: wait and release completed reduce_scatter async tasks
            self._wait_for_grad_comm()

            tmp_buffer = grads_buffer.get_tmp_buffer()
            shard_size = grads_buffer.data_buffer.shape[0]
            grad_buffer_shard = tmp_buffer._slice(0, shard_size)
            if self.enable_overlap:
                # Comm grads async and check all comm_task before optimizer update
                grads_buffer.comm_task = paddle.distributed.reduce_scatter(
                    grad_buffer_shard,
                    tmp_buffer,
                    op=paddle.distributed.ReduceOp.SUM,
                    group=self.buffer_manager._fsdp_group,
                    sync_op=False,
                )

                # Add async task to queue
                self.grad_reduce_queue.append(grads_buffer)
            else:
                paddle.distributed.reduce_scatter(
                    grad_buffer_shard,
                    tmp_buffer,
                    op=paddle.distributed.ReduceOp.SUM,
                    group=self.buffer_manager._fsdp_group,
                    sync_op=False,
                ).wait()
                grads_buffer.data_buffer.add_(grad_buffer_shard)
                grads_buffer.clear_tmp_buffer()

    def _wait_for_grad_comm(self, queue_limit=2):
        # Wait for async reduce_scatter tasks to complete and release resources
        # queue_limit: max queue size, default use 2, 0 means wait for all
        while len(self.grad_reduce_queue) > queue_limit:
            grads_buffer = self.grad_reduce_queue.pop(0)
            if grads_buffer.comm_task is not None:
                grads_buffer.comm_task.wait()
                grads_buffer.comm_task = None
                tmp_buffer = grads_buffer.get_tmp_buffer()
                shard_size = grads_buffer.data_buffer.shape[0]
                grad_buffer_shard = tmp_buffer._slice(0, shard_size)
                grads_buffer.data_buffer.add_(grad_buffer_shard)
            grads_buffer.clear_tmp_buffer()

    def finish_grads_sync(self):
        # Wait for all async reduce_scatter tasks, call before optimizer.step()
        self._wait_for_grad_comm(queue_limit=0)

    def reset_params_buffer_status(self):
        for group in self.buffer_manager.buffer_groups:
            params_buffer = group.params_buffer
            if params_buffer.status in (BufferState.READY, BufferState.USING):
                # Clear stale tmp_buffer to force re-all_gather with updated data_buffer
                params_buffer.clear_tmp_buffer()
                params_buffer.status = BufferState.FREED
                if self.buffer_cnt_in_using > 0:
                    self.buffer_cnt_in_using -= 1


class FusionBackwardHook(PyLayer):
    @staticmethod
    def forward(ctx, inputs, layer, comm_manager):
        ctx.layer = layer
        ctx.comm_manager = comm_manager
        return inputs

    @staticmethod
    def backward(ctx, *args):
        layer = ctx.layer
        trainable_params = []

        for param in layer.parameters(include_sublayers=False):
            if param.trainable:
                trainable_params.append(param)

        ctx.comm_manager.all_gather_params(trainable_params, is_backward=True)
        return args


class FusionForwardHook(PyLayer):
    @staticmethod
    def forward(ctx, *inputs, layer, comm_manager):
        ctx.layer = layer
        ctx.comm_manager = comm_manager
        return inputs

    @staticmethod
    def backward(ctx, *args):
        layer = ctx.layer
        params = list(ctx.layer.parameters(include_sublayers=False))
        ctx.comm_manager.shard_params(params, is_backward=True)
        return args


class FullyShardFusion:
    def __init__(self, model, mesh, fsdp_unit_layers=None):
        self.model = model
        self.mesh = self._check_mesh(mesh)
        self._shard_all_params()
        self.buffer_manager = FSDPBufferManager(
            self.model, self.mesh, fsdp_unit_layers
        )
        self.comm_manager = FSDPCommManager(self.buffer_manager)
        self.register_tensor_fusion_hooks(self.model)
        register_fsdp_context(self)

    def _check_mesh(self, mesh, pp_idx=0):
        if "pp" in mesh.dim_names:
            mesh = mesh.get_mesh_with_dim("pp", pp_idx)
        return mesh

    def _shard_all_params(self):
        def shard_layer_param(layer):
            for param_name in list(layer._parameters.keys()):
                param = getattr(layer, param_name)
                if param is not None:
                    param_placements = [
                        dist.Replicate() for _ in range(len(self.mesh.shape))
                    ]
                    if not param.is_dist():
                        param = dist.shard_tensor(
                            param, self.mesh, param_placements
                        )
                        setattr(layer, param_name, param)

        for name, layer in self.model.named_sublayers(include_self=True):
            shard_layer_param(layer)

    def comm_sync_and_reset_status(self):
        self.comm_manager.finish_grads_sync()
        self.comm_manager.reset_params_buffer_status()
        self.comm_manager.need_zero_grads = True
        # Reset main_grad for all trainable parameters
        for param in self.model.parameters():
            if param.trainable:
                param.main_grad = None

    def register_tensor_fusion_hooks(self, model):
        def _pre_forward_hook(sublayers):
            comm_manager = self.comm_manager

            @paddle.autograd.no_grad()
            def all_gather_comm(*_):
                comm_manager.all_gather_params(
                    sublayers.parameters(include_sublayers=False)
                )

            return all_gather_comm

        def _post_forward_hook(sublayers):
            comm_manager = self.comm_manager

            @paddle.autograd.no_grad()
            def shard_comm(*_):
                comm_manager.shard_params(
                    sublayers.parameters(include_sublayers=False)
                )

            return shard_comm

        def _update_main_grad_hook(param):
            comm_manager = self.comm_manager

            @paddle.autograd.no_grad()
            def comm_hook(grad):
                if grad is not None and grad._is_initialized():
                    # Share mem with grads_tmp_buffer
                    _main_grad = param.get_main_grad()
                    _main_grad.get_tensor()._set_dims(grad._local_shape)
                    param.main_grad = _dtensor_from_local(
                        _main_grad,
                        grad.process_mesh,
                        grad.placements,
                    )
                    param.main_grad._local_value().copy_(grad._local_value())
                    grad._clear_data()
                comm_manager.shard_params([param], is_backward=True)
                comm_manager.reduce_scatter_grads(param)

            return comm_hook

        def _post_backward_hook(param):
            param.main_grad = None
            param._register_grad_hook(_update_main_grad_hook(param))

        # Register pre and post forward hooks
        for name, sublayers in model.named_sublayers(include_self=True):
            sublayers.register_forward_pre_hook(_pre_forward_hook(sublayers))
            sublayers.register_forward_post_hook(_post_forward_hook(sublayers))

        # Register backward layer hooks
        self._register_fusion_layer_hooks(model)

        # Register post backward hooks
        for param in model.parameters():
            if param.trainable:
                _post_backward_hook(param)

    def _register_fusion_layer_hooks(self, layer, name="last_layer"):
        def _forward_post_hook(layer, inputs, outputs):
            return FusionBackwardHook.apply(
                outputs,
                layer=layer,
                comm_manager=self.comm_manager,
            )

        def _forward_pre_hook(layer, inputs):
            return FusionForwardHook.apply(
                *inputs,
                layer=layer,
                comm_manager=self.comm_manager,
            )

        if layer.parameters(include_sublayers=False):
            layer.register_forward_post_hook(_forward_post_hook)

            # Register an additional hook for tie_weights shard_params
            for param in layer.parameters(include_sublayers=False):
                if (
                    param.name
                    == self.comm_manager.buffer_manager.tie_param_name
                ):
                    layer.register_forward_pre_hook(_forward_pre_hook)

        for name, sub_layer in layer.named_children():
            self._register_fusion_layer_hooks(sub_layer, name)
