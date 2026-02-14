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

import copy
import os
from types import MethodType

import paddle
import paddle.distributed as dist
from paddle.autograd import PyLayer

from .auto_dp_utils import in_auto_dp_mode
from .fully_shard_fusion import FullyShardFusion


def shard_accumulators(parameters_and_grads, optimizer, target_block):
    if getattr(optimizer, "_has_sharded_accumulators", False):
        return
    optimizer._has_sharded_accumulators = True
    for param, _ in parameters_and_grads:
        optimizer._create_accumulators(
            target_block,
            [param],
        )
        target_name = param.name
        if param.name in optimizer._master_weights.keys():
            master_weight = optimizer._master_weights[param.name]
            target_name = master_weight.name
        for key in optimizer._accumulators.keys():
            accumulator = optimizer._accumulators[key][target_name]
            if accumulator.is_dist():
                continue
            origin_accumulator_name = accumulator.name

            if 'beta' not in key:
                placements = copy.deepcopy(param.placements)
            else:
                placements = [
                    dist.Replicate()
                    for _ in range(len(param.process_mesh.shape))
                ]
            optimizer._accumulators[key][target_name] = dist.shard_tensor(
                accumulator,
                mesh=param.process_mesh,
                placements=placements,
            )
            optimizer._accumulators[key][
                target_name
            ].name = origin_accumulator_name

    def _finish_update_impl(self, block, parameters_and_grads):
        if not isinstance(parameters_and_grads, list):
            parameters_and_grads = parameters_and_grads['params']
        for param, _ in parameters_and_grads:
            param.main_grad = None

    optimizer._finish_update = MethodType(_finish_update_impl, optimizer)


class FullyShardAuto:
    def __init__(self, model, mesh, enable_tensor_fusion_and_overlap=True):
        if enable_tensor_fusion_and_overlap:
            FullyShardFusion(model, mesh)
        else:
            self.model = model
            self.mesh = mesh
            # use first dims as sharding axis
            self._shard_fn = dist.ShardingStage3(0, self.mesh)
            for param in self.model.parameters():
                param._need_shard_auto = True
                self._shard_fn._shard_parameter(param)
                if not in_auto_dp_mode():
                    self._shard_fn._register_hook_for_param_grad(param)
            if in_auto_dp_mode():
                self._register_comm_hook(self.model)
            os.environ["skip_sharding3_output_reshard"] = "1"

    def _register_comm_hook(self, model):
        def _pre_forward_hook(sublayers):
            @paddle.autograd.no_grad()
            def gather_comm(*_):
                dp_axis = dist.auto_parallel.get_mesh().dim_names.index('dp')
                for key, param in sublayers._parameters.items():
                    if param.placements[dp_axis] != dist.Replicate():
                        new_placements = copy.deepcopy(param.placements)
                        new_placements[dp_axis] = dist.Replicate()
                        replicate_param = dist.reshard(
                            param, param.process_mesh, new_placements
                        )
                        param.get_tensor()._share_data_with(
                            replicate_param.get_tensor()
                        )

            return gather_comm

        def _post_forward_hook(sublayers):
            @paddle.autograd.no_grad()
            def shard_comm(*_):
                dp_axis = dist.auto_parallel.get_mesh().dim_names.index('dp')
                for key, param in sublayers._parameters.items():
                    if (
                        param.trainable
                        and param.placements[dp_axis] == dist.Replicate()
                    ):
                        new_placements = copy.deepcopy(param.placements)
                        new_placements[dp_axis] = dist.Shard(dp_axis)
                        shard_param = dist.reshard(
                            param, param.process_mesh, new_placements
                        )
                        param.get_tensor()._share_data_with(
                            shard_param.get_tensor()
                        )

            return shard_comm

        def _post_backward_hook(param):
            def shard_comm(grad):
                dp_axis = dist.auto_parallel.get_mesh().dim_names.index('dp')
                if param.placements[dp_axis] == dist.Replicate():
                    new_placements = copy.deepcopy(param.placements)
                    new_placements[dp_axis] = dist.Shard(dp_axis)
                    shard_param = dist.reshard(
                        param, param.process_mesh, new_placements
                    )
                    param.get_tensor()._share_data_with(
                        shard_param.get_tensor()
                    )
                return grad

            param.register_hook(shard_comm)

        # register forward hooks
        for name, sublayers in model.named_sublayers(include_self=True):
            sublayers.register_forward_pre_hook(_pre_forward_hook(sublayers))
            sublayers.register_forward_post_hook(_post_forward_hook(sublayers))

        # register backward hooks
        for param in model.parameters():
            if param.trainable:
                _post_backward_hook(param)

        # register layer hooks for param sync in tie weights
        self._register_layer_hooks(model)

    def _register_layer_hooks(self, layer, name="last_layer"):
        def _forward_post_hook(layer, inputs, outputs):
            return LayerHook.apply(
                outputs,
                layer=layer,
            )

        if layer.parameters(include_sublayers=False):
            layer.register_forward_post_hook(_forward_post_hook)
        for name, sub_layer in layer.named_children():
            self._register_layer_hooks(sub_layer, name)


class LayerHook(PyLayer):
    @staticmethod
    def forward(ctx, inputs, layer):
        ctx.layer = layer
        return inputs

    @staticmethod
    def backward(ctx, *args):
        layer = ctx.layer
        dp_axis = dist.auto_parallel.get_mesh().dim_names.index('dp')
        for param in layer.parameters(include_sublayers=False):
            if (
                param.trainable
                and param.placements[dp_axis] != dist.Replicate()
            ):
                new_placements = copy.deepcopy(param.placements)
                new_placements[dp_axis] = dist.Replicate()
                replicate_param = dist.reshard(
                    param, param.process_mesh, new_placements
                )
                param.get_tensor()._share_data_with(
                    replicate_param.get_tensor()
                )
        return args
