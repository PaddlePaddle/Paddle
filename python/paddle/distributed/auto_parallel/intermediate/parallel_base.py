#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import logging

import paddle
import paddle.distributed as dist
from paddle import pir
from paddle.base.framework import (
    in_dygraph_mode,
    in_pir_mode,
)
from paddle.distributed import fleet
from paddle.nn import Layer
from paddle.optimizer import Optimizer


def is_tensor(tensor):
    if in_dygraph_mode():
        return isinstance(tensor, paddle.Tensor)
    elif in_pir_mode():
        return isinstance(tensor, pir.Value)
    else:
        raise RuntimeError(
            "PipelineParallel are only supported in dynamic or pir mode."
        )


class ParallelOptimizer:
    def __init__(
        self,
        optimizer,
        level=None,
        sharding_mesh_dim=None,
    ):
        self.level = None
        self.sharding_mesh_dim = None
        self.optimizer = None

        if isinstance(optimizer, ParallelOptimizer):
            self.optimizer = optimizer.optimizer
            if level is None:
                self.level = optimizer.level
                self.sharding_mesh_dim = optimizer.sharding_mesh_dim
            else:
                if isinstance(level, int):
                    level = str(level)
                assert level in ("0", "1", "2", "3", None)
                if optimizer.level is not None:
                    assert (
                        level == optimizer.level
                    ), f"The level passed in is not identical with previous level. Current level is {level}, previous level is {optimizer.level}"
                self.level = level
                self.sharding_mesh_dim = sharding_mesh_dim
        else:
            assert isinstance(optimizer, Optimizer)
            self.optimizer = optimizer
            if isinstance(level, int):
                level = str(level)
            assert level in ("0", "1", "2", "3", None)
            # level=0 and level=None are all mean pure dp
            self.level = level
            self.sharding_mesh_dim = sharding_mesh_dim

        self.is_initialized = False

    def parallelize(self):
        assert self.optimizer is not None
        if self.is_initialized:
            return self.optimizer

        mesh = fleet.auto.get_mesh()
        if self.level == "1":
            self.optimizer = dist.shard_optimizer(
                self.optimizer,
                dist.ShardingStage1(self.sharding_mesh_dim, mesh),
            )
        elif self.level == "2":
            self.optimizer = dist.shard_optimizer(
                self.optimizer,
                dist.ShardingStage2(self.sharding_mesh_dim, mesh),
            )
        elif self.level == "3":
            self.optimizer = dist.shard_optimizer(
                self.optimizer,
                dist.ShardingStage3(self.sharding_mesh_dim, mesh),
            )
        else:
            self.optimizer = dist.shard_optimizer(self.optimizer, None)
        self.is_initialized = True

        return self.optimizer

    def update_param_list(self, parallelized_parameters):
        self.optimizer._parameter_list = parallelized_parameters
        if isinstance(parallelized_parameters[0], dict):
            self.optimizer._param_groups = []
            for param_group in self.parallelized_parameters:
                self.optimizer._add_param_group(param_group.copy())
        else:
            self.optimizer._param_groups = self.optimizer._parameter_list


class ParallelModel:
    def __init__(self, model):
        super().__init__()
        self.pp_parallelizer = None
        self.tp_parallelizer = None
        self.sharding_parallelizer = None
        self.model = None
        self.share_param_list = {}
        self.layer_param_placements = {}
        if isinstance(model, ParallelModel):
            self.pp_parallelizer = model.pp_parallelizer
            self.tp_parallelizer = model.tp_parallelizer
            self.sharding_parallelizer = model.sharding_parallelizer
            self.model = model.model
        else:
            assert isinstance(model, Layer)
            self.model = model

        self.is_parallelized = False

    def get_mesh(self, pp_idx=0):
        mesh = fleet.auto.get_mesh()
        if "pp" in mesh.dim_names:
            mesh = mesh.get_mesh_with_dim("pp", pp_idx)
        return mesh

    def parallelize_model(self):
        assert self.model is not None
        if self.is_parallelized:
            return self.model

        if self.pp_parallelizer is not None:
            assert callable(self.pp_parallelizer)
            self.model = self.pp_parallelizer(self.model)

        if self.tp_parallelizer is not None:
            assert callable(self.tp_parallelizer)
            self.model, self.layer_param_placements = self.tp_parallelizer(
                self.model
            )
        if self.sharding_parallelizer is not None:
            assert callable(self.sharding_parallelizer)
            self.model = self.sharding_parallelizer(self.model)
        self._shard_all_param(self.model)
        self.is_parallelized = True

        return self.model

    def _process_share_weight_layer(
        self, layer, origin_weight, param_name, param_placements
    ):
        ipp = (
            layer.pipeline_stage_index
            if hasattr(layer, "pipeline_stage_index")
            else 0
        )

        def create_pre_hook(origin_weight, param_name):
            def forward_pre_hook(layer, input):
                setattr(
                    layer,
                    param_name,
                    None,
                )
                delattr(layer, param_name)
                mesh = self.get_mesh(ipp)
                share_weight = dist.reshard(
                    origin_weight,
                    mesh,
                    param_placements,
                )
                setattr(
                    layer,
                    param_name,
                    share_weight,
                )

            return forward_pre_hook

        def create_post_hook(origin_weight, param_name):
            def forward_post_hook(layer, input, output):
                setattr(
                    layer,
                    param_name,
                    origin_weight,
                )

            return forward_post_hook

        layer.register_forward_pre_hook(
            create_pre_hook(origin_weight, param_name)
        )
        layer.register_forward_post_hook(
            create_post_hook(origin_weight, param_name)
        )

    def _shard_all_param(self, model):
        param_name_to_shard_param = {}
        param_name_to_pp_stage = {}

        def shard_layer_param(layer):
            if self.pp_parallelizer is not None:
                assert hasattr(layer, "pipeline_stage_index")
            for param_name in list(layer._parameters.keys()):
                param = getattr(layer, param_name)
                if param is not None:
                    param_full_name = param.name
                    ipp = (
                        layer.pipeline_stage_index
                        if hasattr(layer, "pipeline_stage_index")
                        else 0
                    )
                    mesh = self.get_mesh(ipp)
                    param_placements = [
                        dist.Replicate() for _ in range(len(mesh._shape))
                    ]
                    if layer in self.layer_param_placements:
                        if param_name in self.layer_param_placements[layer]:
                            param_placements = (
                                self.layer_param_placements[layer][param_name]
                                if self.layer_param_placements[layer][
                                    param_name
                                ]
                                is not None
                                else param_placements
                            )
                    if not param.is_dist():
                        if param_full_name in param_name_to_shard_param:
                            setattr(
                                layer,
                                param_name,
                                param_name_to_shard_param[param_full_name],
                            )
                            if ipp != param_name_to_pp_stage[param_full_name]:
                                self._process_share_weight_layer(
                                    layer,
                                    param_name_to_shard_param[param_full_name],
                                    param_name,
                                    param_placements,
                                )
                        else:
                            param = dist.shard_tensor(
                                param, mesh, param_placements
                            )
                            param_name_to_shard_param[param_full_name] = param
                            param_name_to_pp_stage[param_full_name] = ipp
                            setattr(layer, param_name, param)
                    else:
                        if (
                            param_full_name in param_name_to_shard_param
                            and ipp != param_name_to_pp_stage[param_full_name]
                        ):
                            self._process_share_weight_layer(
                                layer,
                                param_name_to_shard_param[param_full_name],
                                param_name,
                                param_placements,
                            )
                        elif param_full_name not in param_name_to_shard_param:
                            param_name_to_shard_param[param_full_name] = param
                            param_name_to_pp_stage[param_full_name] = ipp

        for name, layer in model.named_sublayers():
            shard_layer_param(layer)


def parallelize_model_and_optimizer(model, optimizer=None):
    if not isinstance(model, ParallelModel):
        assert not isinstance(optimizer, ParallelOptimizer)
        logging.warning(
            "The method `parallelize_model_and_optimizer` won't do anything since the model is not parallelized."
        )
        return model, optimizer
    parallelized_model = model.parallelize_model()
    parallelized_optimizer = None
    if optimizer is not None:
        assert isinstance(optimizer, ParallelOptimizer)
        optimizer.update_param_list(parallelized_model.parameters())
        parallelized_optimizer = optimizer.parallelize()

    return parallelized_model, parallelized_optimizer
