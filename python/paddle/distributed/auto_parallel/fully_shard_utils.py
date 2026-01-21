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
    def __init__(
        self, unique_key, params, sharding_degree, dtype, is_params=False
    ):
        # Calculate total buffer size needed (with padding)
        self.unique_key = unique_key
        self.sharding_degree = sharding_degree  # need fix
        self.total_buffer_size = 0
        self.param2index = {}
        self.dtype = dtype
        for param in params:
            self.param2index[param.name] = self.total_buffer_size
            self.total_buffer_size += self.get_padded_size(param)
        # Create fused buffers
        self.data_buffer = paddle.zeros(
            shape=[self.total_buffer_size],
            dtype=dtype,
        )
        self.d_name = "fused_tensor_" + str(unique_key)  # need fix
        self.d_stop_gradient = params[0].stop_gradient
        self.d_optimize_attr = params[0].optimize_attr
        self.tmp_data_buffer = None

        if is_params:
            for param in params:
                optimize_attr = param.optimize_attr
                index = self.param2index[param.name]
                stop_gradient = param.stop_gradient
                param.stop_gradient = True
                param._local_value().flatten_()
                paddle.assign(
                    param._local_value(),
                    self.data_buffer._slice(
                        index,
                        index + param._numel(),
                    ),
                )
                param.stop_gradient = stop_gradient
                tmp_param = paddle._C_ops.view_slice(
                    self.data_buffer,
                    index,
                    index + param._numel(),
                )
                param._local_value().data = tmp_param

                paddle.device.cuda.empty_cache()

        else:
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

        self.data_buffer = _dtensor_from_local(
            self.data_buffer,
            dist.auto_parallel.get_mesh(),
            [dist.Replicate()],  # need fix
        )
        self.data_buffer = dist.reshard(
            self.data_buffer, dist.auto_parallel.get_mesh(), [dist.Shard(0)]
        )

        if is_params:  # need fix
            self.data_buffer.stop_gradient = stop_gradient
            self.data_buffer.optimize_attr = optimize_attr

        self.data_buffer.name = "fused_params_" + str(unique_key)

    def get_padded_size(self, param):
        size = np.prod(param._local_shape)
        align_size = (
            alignment[get_current_device_type()]
            // align[param.dtype]
            * self.sharding_degree
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


class FSDPBufferManager:
    def __init__(self, model, mesh):
        self.model = model
        shard_groups = get_mesh_comm_list(mesh, "dp")  # need fix
        for group in shard_groups:
            comm_group = dist.new_group(sorted(group))
            if dist.get_rank() in group:
                self._sharding_group = comm_group

        parameters = model.parameters()

        # create buffer_groups
        comm_buffer_size_MB = 256  # need fix
        group_size = comm_buffer_size_MB * 1024 * 1024
        is_sparse_gradient = [False] * len(parameters)
        shape_dict = {param.name: param.shape for param in parameters}
        dense_params = [param._local_value() for param in parameters]
        # group params according to comm_buffer_size_MB
        group_indices = core.eager_assign_group_by_size(
            dense_params, is_sparse_gradient, [group_size, group_size]
        )

        vars_groups = OrderedDict()
        for group_idx, indices in enumerate(group_indices):
            for i in indices:
                vars_groups.setdefault(group_idx, []).append(parameters[i])

        self.buffer_groups = []
        self.param_to_buffer_group = {}

        # create params_buffer, grads_buffer with groups
        for group_idx, params in vars_groups.items():
            params_buffer = TensorFusionBuffer(
                group_idx,
                params,
                self._sharding_group.nranks,
                params[0].dtype,
                is_params=True,
            )
            grads_buffer = TensorFusionBuffer(
                group_idx, params, self._sharding_group.nranks, paddle.float32
            )
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
                self.param_to_buffer_group[param.name] = group_idx


class FSDPCommManager:
    def __init__(self, buffer_manager):
        self.buffer_manager = buffer_manager

    def all_gather_params(self, params):
        if len(params) == 0:
            return
        for param in params:
            group_idx = self.buffer_manager.param_to_buffer_group[param.name]
            self.buffer_manager.buffer_groups[group_idx]["params_use_cnt"] += 1
            buffer = self.buffer_manager.buffer_groups[group_idx][
                "params_buffer"
            ].data_buffer
            if buffer.placements[0] == dist.Shard(0):
                name = buffer.name  # need fix with hand comm
                stop_grad = param.stop_gradient
                buffer = dist.reshard(
                    buffer,
                    buffer.process_mesh,
                    [dist.Replicate(), dist.Replicate(), dist.Replicate()],
                )
                buffer.name = name
                buffer.stop_gradient = stop_grad
                self.buffer_manager.buffer_groups[group_idx][
                    "params_buffer"
                ].data_buffer = buffer

    def shard_params(self, params):
        for param in params:
            group_idx = self.buffer_manager.param_to_buffer_group[param.name]
            if (
                self.buffer_manager.buffer_groups[group_idx]["params_use_cnt"]
                == self.buffer_manager.buffer_groups[group_idx][
                    "params_use_sum"
                ]
            ):
                self.buffer_manager.buffer_groups[group_idx][
                    "params_use_cnt"
                ] = 0
                buffer = self.buffer_manager.buffer_groups[group_idx][
                    "params_buffer"
                ].data_buffer
                name = buffer.name  # need fix with hand comm
                stop_grad = param.stop_gradient
                buffer = dist.reshard(
                    buffer, buffer.process_mesh, [dist.Shard(0)]
                )
                buffer.name = name
                buffer.stop_gradient = stop_grad
                self.buffer_manager.buffer_groups[group_idx][
                    "params_buffer"
                ].data_buffer = buffer

    def reduce_scatter_grad(self, param):
        group_idx = self.buffer_manager.param_to_buffer_group[param.name]
        self.buffer_manager.buffer_groups[group_idx]["grads_use_cnt"] += 1
        if (
            self.buffer_manager.buffer_groups[group_idx]["grads_use_cnt"]
            == self.buffer_manager.buffer_groups[group_idx]["grads_use_sum"]
        ):
            self.buffer_manager.buffer_groups[group_idx]["grads_use_cnt"] = 0
            # reduce-scatter from tmp_grad_buffer into grads_buffer.
            tmp_grad_buffer = self.buffer_manager.buffer_groups[group_idx][
                "grads_buffer"
            ].get_tmp_buffer()
            grads_buffer = self.buffer_manager.buffer_groups[group_idx][
                "grads_buffer"
            ].data_buffer

            tmp_grad_buffer = _dtensor_from_local(
                tmp_grad_buffer,
                dist.auto_parallel.get_mesh(),
                [
                    dist.Partial(dist.ReduceType.kRedSum),
                    dist.Replicate(),
                    dist.Replicate(),
                ],  # 这里最好改成全Replicate
            )
            tmp_grad_buffer = dist.reshard(
                tmp_grad_buffer,
                tmp_grad_buffer.process_mesh,
                [dist.Shard(0), dist.Replicate(), dist.Replicate()],
            )
            grads_buffer.get_tensor()._share_data_with(
                tmp_grad_buffer.get_tensor()
            )  # need fix

            # clear tmp_grad_buffer
            self.buffer_manager.buffer_groups[group_idx][
                "grads_buffer"
            ].clear_tmp_buffer()


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
