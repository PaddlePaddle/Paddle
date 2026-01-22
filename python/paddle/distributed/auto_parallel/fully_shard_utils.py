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

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    align,
    alignment,
    get_current_device_type,
)
from paddle.framework import core

from .moe_utils import (
    _dtensor_from_local,
)
from .sharding import (
    get_mesh_comm_list,
)


class TensorFusionBuffer:
    def __init__(self, unique_key, params, fsdp_degree, dtype, is_params=False):
        # Calculate total buffer size needed (with padding)
        self.unique_key = unique_key
        self.fsdp_degree = fsdp_degree  # need fix
        self.total_buffer_size = 0
        self.param2index = {}
        self.dtype = dtype
        self.tmp_data_buffer = None

        for param in params:
            self.param2index[param.name] = self.total_buffer_size
            self.total_buffer_size += self.get_padded_size(param)

        if is_params:
            # Create fused params_buffer
            # TODO(lizhenxing): build full params_buffer on CPU and only move shards to GPU to minimize mem peaks.
            self.data_buffer = paddle.zeros(
                shape=[self.total_buffer_size],
                dtype=dtype,
            )
            self.is_shard = True

            for param in params:
                index = self.param2index[param.name]
                stop_gradient = param.stop_gradient
                local_shape = param._local_shape
                param.stop_gradient = True
                param._local_value().flatten_()

                paddle.assign(
                    param._local_value(),
                    self.data_buffer._slice(
                        index,
                        index + param._numel(),
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

            # init params_buffer attr
            self.data_buffer.name = "fuse_params_" + str(unique_key)
            self.data_buffer.stop_gradient = params[0].stop_gradient
            self.data_buffer.optimize_attr = params[0].optimize_attr
        else:
            # Create fused grads_buffer with shard
            self.data_buffer = paddle.zeros(
                shape=[self.total_buffer_size // self.fsdp_degree],
                dtype=dtype,
            )

            # register get_main_grad method for each param, which returns view_slice of grad_buffer
            for param in params:
                if param.trainable:
                    param._fusion_buffer = self
                    param._param2index = self.param2index  # need fix

                    def main_grad_getter(param):
                        tmp_buffer = param._fusion_buffer.get_tmp_buffer()
                        index = param._param2index[param.name]
                        tmp_grad = paddle._C_ops.view_slice(
                            tmp_buffer,
                            index,
                            index + param._numel(),  # need fix
                        )
                        return tmp_grad

                    param.get_main_grad = main_grad_getter.__get__(param)

    def get_padded_size(self, param):
        size = np.prod(param.shape)
        align_size = (
            alignment[get_current_device_type()]
            // align[param.dtype]
            * self.fsdp_degree
        )
        return ((size + align_size - 1) // align_size) * align_size

    def get_tmp_buffer(self):
        # reuse temp grad_buffer if exists, else create.
        if self.tmp_data_buffer is None:
            self.tmp_data_buffer = paddle.zeros(
                shape=[self.total_buffer_size], dtype=self.dtype
            )
        return self.tmp_data_buffer

    def clear_tmp_buffer(self):
        if self.tmp_data_buffer is not None:
            self.tmp_data_buffer._clear_data()
            self.tmp_data_buffer = None
            paddle.device.cuda.empty_cache()


class FSDPBufferManager:
    def __init__(self, model, mesh):
        self.model = model
        shard_groups = get_mesh_comm_list(mesh, "dp")  # need fix

        for group in shard_groups:
            comm_group = dist.new_group(sorted(group))
            if dist.get_rank() in group:
                self._fsdp_group = comm_group

        # build_param_groups
        vars_groups = self.build_param_groups()

        # create buffer_groups
        self.buffer_groups = []
        self.param_to_buffer_group = {}

        # create params_buffer, grads_buffer with groups
        for gid, params in vars_groups.items():
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
                {
                    "params_buffer": params_buffer,
                    "grads_buffer": grads_buffer,
                    "params_use_sum": len(params),
                    "params_use_cnt": 0,
                    "grads_use_sum": len(params),
                    "grads_use_cnt": 0,
                }
            )

            for param in params:
                self.param_to_buffer_group[param.name] = gid

    def build_param_groups(self):
        parameters = self.model.parameters()
        comm_buffer_size_MB = 256  # need fix
        group_size = comm_buffer_size_MB * 1024 * 1024
        vars_groups = OrderedDict()
        curr_gid = 0
        freeze_parameters, trainable_parameters, tie_parameters = [], [], []

        # get tie_param_name if using tie_weights
        self.tie_param_name = None
        if hasattr(self.model, "get_input_embeddings"):
            self.tie_param_name = self.model.get_input_embeddings().weight.name

        for param in parameters:
            if self.tie_param_name and param.name == self.tie_param_name:
                tie_parameters.append(param)
            elif param.stop_gradient:
                freeze_parameters.append(param)
            else:
                trainable_parameters.append(param)

        for cur_params in [
            freeze_parameters,
            trainable_parameters,
            tie_parameters,
        ]:
            if len(cur_params) == 0:
                continue

            is_sparse_gradient = [False] * len(cur_params)
            dense_params = [param._local_value() for param in cur_params]

            # group params according to comm_buffer_size_MB
            group_indices = core.eager_assign_group_by_size(
                dense_params, is_sparse_gradient, [group_size, group_size]
            )

            for indices in group_indices:
                for i in indices:
                    vars_groups.setdefault(curr_gid, []).append(cur_params[i])
                curr_gid += 1

        return vars_groups


class FSDPCommManager:
    def __init__(self, buffer_manager):
        self.buffer_manager = buffer_manager

    def all_gather_params(self, params):
        for param in params:
            gid = self.buffer_manager.param_to_buffer_group[param.name]
            self.buffer_manager.buffer_groups[gid]["params_use_cnt"] += 1

            params_buffer = self.buffer_manager.buffer_groups[gid][
                "params_buffer"
            ]
            tmp_buffer = params_buffer.get_tmp_buffer()

            if params_buffer.is_shard:
                params_buffer.is_shard = False
                self.buffer_manager._fsdp_group.process_group.all_gather(
                    params_buffer.data_buffer, tmp_buffer
                ).wait()

            index = params_buffer.param2index[param.name]
            tmp_param = paddle._C_ops.view_slice(
                tmp_buffer,
                index,
                index + param._numel(),
            )
            tmp_param.get_tensor()._set_dims(param.shape)
            tmp_param = _dtensor_from_local(
                tmp_param,
                param.process_mesh,
                param.placements,
            )
            param.get_tensor()._share_data_with(tmp_param.get_tensor())

    def shard_params(self, params):
        for param in params:
            gid = self.buffer_manager.param_to_buffer_group[param.name]
            stop_gradient = param.stop_gradient
            local_shape = param._local_shape
            param._clear_data()
            param.stop_gradient = stop_gradient
            param._local_value().get_tensor()._set_dims(local_shape)

            if (
                self.buffer_manager.buffer_groups[gid]["params_use_cnt"]
                == self.buffer_manager.buffer_groups[gid]["params_use_sum"]
            ):
                self.buffer_manager.buffer_groups[gid]["params_use_cnt"] = 0
                params_buffer = self.buffer_manager.buffer_groups[gid][
                    "params_buffer"
                ]
                params_buffer.is_shard = True
                params_buffer.clear_tmp_buffer()

    def reduce_scatter_grad(self, param):
        gid = self.buffer_manager.param_to_buffer_group[param.name]
        self.buffer_manager.buffer_groups[gid]["grads_use_cnt"] += 1

        if (
            self.buffer_manager.buffer_groups[gid]["grads_use_cnt"]
            == self.buffer_manager.buffer_groups[gid]["grads_use_sum"]
        ):
            self.buffer_manager.buffer_groups[gid]["grads_use_cnt"] = 0

            # reduce_scatter from tmp_grad_buffer into grads_buffer.
            grads_buffer = self.buffer_manager.buffer_groups[gid][
                "grads_buffer"
            ]
            tmp_grad_buffer = grads_buffer.get_tmp_buffer()

            paddle.distributed.reduce_scatter(
                grads_buffer.data_buffer,
                tmp_grad_buffer,
                op=paddle.distributed.ReduceOp.SUM,
                group=self.buffer_manager._fsdp_group,
                sync_op=False,
            ).wait()

            grads_buffer.clear_tmp_buffer()


class FusionLayerHook(PyLayer):
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

        ctx.comm_manager.all_gather_params(trainable_params)
        return args


class FusionLayerPostHook(PyLayer):
    @staticmethod
    def forward(ctx, *inputs, layer, comm_manager):
        ctx.layer = layer
        ctx.comm_manager = comm_manager
        return inputs

    @staticmethod
    def backward(ctx, *args):
        layer = ctx.layer
        params = list(ctx.layer.parameters(include_sublayers=False))
        ctx.comm_manager.shard_params(params)
        return args
