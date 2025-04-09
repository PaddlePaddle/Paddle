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
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypedDict

from typing_extensions import NotRequired

from paddle.distributed import fleet
from paddle.framework import core

from .parallel_base import ParallelOptimizer, parallelize_model_and_optimizer
from .pipeline_parallel import pipeline_parallel
from .sharded_data_parallel import sharded_data_parallel
from .tensor_parallel import tensor_parallel

if TYPE_CHECKING:
    import paddle

    from .pipeline_parallel import SplitPoint
    from .tensor_parallel import PlanBase

    class _DPConfig(TypedDict):
        sharding_level: str | int

    class _MPConfig(TypedDict):
        parallelize_plan: dict[str, PlanBase | list[PlanBase]]

    class _PPConfig(TypedDict):
        split_spec: str | dict[str, SplitPoint]
        global_spec: NotRequired[str]

    class _ParallelizeConfig(TypedDict):
        dp_config: NotRequired[_DPConfig]
        mp_config: NotRequired[_MPConfig]
        pp_config: NotRequired[_PPConfig]


def parallelize(
    model: paddle.nn.Layer,
    optimizer: paddle.optimizer.Optimizer | None = None,
    mesh: paddle.distributed.ProcessMesh | None = None,
    config: _ParallelizeConfig | None = None,
) -> tuple[paddle.nn.Layer, paddle.optimizer.Optimizer]:
    """

    Parallelize the model and optimizer from a single card version to a distributed version.

    Args:
        model (paddle.nn.Layer): the model to be parallelized.
        optimizer (paddle.optimizer.Optimizer, optional): the optimizer to be parallelized.
            Could be `None` if no optimizer to be parallelized.
        mesh (paddle.distributed.ProcessMesh, optional): the process mesh for parallelize the model and the optimizer.
            Best practice: calling `dist.auto_parallel.set_mesh` to set the global mesh ahead of calling `parallelize`
            and keep the `mesh` parameter as `None.
            If the `mesh` is not None, the mesh passed to `parallelize` will overwrite the mesh set by `set_mesh`.
        config (dict, optional): a dict contains the parallel config.
            The keys of the dict can be chosen from `dp_config`, `mp_config` and `pp_config` which will be used to
            determine the parallel method for data parallel, tensor parallel and pipeline parallel separately.
            A valid config can be like this: {"dp_config": for more information refer the `dp_config` section of
            this doc, "mp_config": for more information refer the `mp_config` section of this doc, "pp_config":
            for more information refer the `pp_config` section of this doc}.

            dp_config (dict): a dict specifying the data parallel config. The keys of `dp_config` is `sharding_level`.
                The value of `sharding_level` can be chosen from 0/1/2/3, which means pure data parallel, sharding
                parallel stage 1, sharding parallel stage 2 and sharding parallel stage 3  separately. A valid
                dp_config can be like this: {"sharding_level": 2}.

            mp_config (dict): a dict specifying the tensor parallel config. The keys of `mp_config` is
                `parallelize_plan`. The value of `parallelize_plan` is another dict, mapping a layer name or a param
                name to a specific parallel plan. Note that the layer name could be written in regular format. If
                mapping a param name to a specific plan, the name of the param must be ended with `weight` or `bias`.
                And all valid parallel plan is `ColWiseParallel`, `RowWiseParallel`, `SequenceParallelBegin,
                `SequenceParallelDisable`, `SequenceParallelEnable`, `SequenceParallelEnd`, `PrepareLayerInput` and
                `PrepareLayerOutput`. A valid mp_config can be like this: {"llama.embed_tokens": dist.ColWiseParallel(),
                "llama.norm": dist.SequenceParallelEnable(), "lm_head.weight": dist.ColWiseParallel()}.

            pp_config (dict): a dict specifying the pipeline parallel config. The keys of `pp_config` is `split_spec`
                and `global_spec`. The `split_spec` can be a dict or a string. If the `split_spec` is a dict, it maps
                a layer name to a `SplitPoint`, note that the layer name could be written in regular format. The
                pipeline parallel will exactly split the model at the point indicated by the map. If the `split_spec`
                is a string, it contains the prefix of a set of layers. The pipeline parallel will automatically split
                the model evenly at target layer. The `global_spec` is a string indicating a layer that contains global
                tensors, which will be duplicated through all stages of the pipeline parallel. Some valid pp_config
                can be list these:  {"split_spec": "llama.layers", "global_spec": "llama.global_layer"}
                or {"split_spec": {"llama.layers.1": SplitPoint.END}}.

    Note:
        If the mesh is `None` or neither of `dp_config`, `mp_config` and `pp_config` is in the config, this
        api will do nothing but return the model and optimizer passed in.

    Returns:
        model, optimizer: the model and the optimizer after parallelize

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import paddle.distributed as dist

            >>> class ModelConfig:
            ...     def __init__(self):
            ...         self.vocab_size = 10
            ...         self.hidden_size = 20
            ...         self.intermediate_size = 20
            ...         self.num_layers = 2

            >>> model_config = ModelConfig()

            >>> class LlamaRMSNorm(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.weight = paddle.create_parameter(
            ...             shape=[model_config.hidden_size],
            ...             dtype=paddle.get_default_dtype(),
            ...         )
            ...
            ...     def forward(self, input):
            ...         pass

            >>> class LlamaAttention(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...
            ...         self.qkv_proj = paddle.nn.Linear(
            ...             model_config.hidden_size,
            ...             model_config.hidden_size * 3,
            ...             bias_attr=False,
            ...         )
            ...
            ...         self.o_proj = paddle.nn.Linear(
            ...             model_config.hidden_size,
            ...             model_config.hidden_size,
            ...             bias_attr=False,
            ...         )
            ...
            ...     def forward(self, input):
            ...         pass

            >>> class LlamaMLP(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.gate_up_proj = paddle.nn.Linear(
            ...             model_config.hidden_size,
            ...             model_config.intermediate_size * 2,
            ...             bias_attr=False
            ...         )
            ...
            ...         self.down_proj = paddle.nn.Linear(
            ...             model_config.intermediate_size, model_config.hidden_size, bias_attr=False
            ...         )
            ...
            ...     def forward(self, input):
            ...         pass

            >>> class LlamaDecoderLayer(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.self_attn = LlamaAttention()
            ...         self.mlp = LlamaMLP()
            ...         self.input_layernorm = LlamaRMSNorm()
            ...         self.post_attention_layernorm = LlamaRMSNorm()
            ...
            ...     def forward(self, input):
            ...         pass

            >>> class LlamaModel(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.embedding = paddle.nn.Embedding(model_config.vocab_size, model_config.hidden_size)
            ...         decoder_layers = []
            ...         for _ in range(model_config.num_layers):
            ...             decoder_layers.append(LlamaDecoderLayer())
            ...
            ...         self.layers = paddle.nn.LayerList(decoder_layers)
            ...         self.norm = LlamaRMSNorm()
            ...
            ...     def forward(self, input):
            ...         pass

            >>> class LlamaLMHead(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.weight = self.create_parameter(
            ...             shape=[model_config.hidden_size, model_config.vocab_size],
            ...             dtype=paddle.get_default_dtype(),
            ...         )
            ...
            ...     def forward(self, input):
            ...         pass

            >>> class LlamaForCausalLM(paddle.nn.Layer):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.llama = LlamaModel()
            ...         self.lm_head = LlamaLMHead()
            ...
            ...     def forward(self, input):
            ...         pass

            >>> mesh = dist.ProcessMesh([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dim_names=["dp", "mp", "pp"])
            >>> dist.auto_parallel.set_mesh(mesh)
            >>> parallel_config = {
            ...     "dp_config": {'sharding_level': 1},
            ...     "mp_config": {
            ...         "parallelize_plan": {
            ...             "llama.embed_tokens": [
            ...                 dist.ColWiseParallel(),
            ...                 dist.SequenceParallelBegin(),
            ...             ],
            ...             "llama.position_embedding": [
            ...                 dist.ColWiseParallel(),
            ...                 dist.SequenceParallelBegin(),
            ...             ],
            ...             "llama.layers.*.self_attn.qkv_proj": dist.ColWiseParallel(),
            ...             "llama.layers.*.self_attn.o_proj": dist.RowWiseParallel(),
            ...             "llama.layers.*.self_attn": dist.SequenceParallelDisable(),
            ...             "llama.layers.*.mlp.gate_up_proj": dist.ColWiseParallel(),
            ...             "llama.layers.*.mlp.down_proj": dist.RowWiseParallel(),
            ...             "llama.layers.*.mlp": dist.SequenceParallelDisable(
            ...                 need_transpose=False
            ...             ),
            ...             "lm_head.weight": dist.ColWiseParallel(),
            ...             "lm_head": dist.SequenceParallelEnd(),
            ...         }
            ...     },
            ...     "pp_config": {'split_spec': "llama.layers"}
            ... }

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> model = LlamaForCausalLM()
            >>> optimizer = paddle.optimizer.AdamW(parameters=model.parameters())
            >>> dist_model, dist_optimizer = dist.parallelize(model, optimizer, config=parallel_config) # type: ignore[arg-type]
            >>> # This case need to be executed in multi-card environment
            >>> # python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 {test_case}.py

    """
    if config is None:
        warnings.warn(
            "The `parallelize will do nothing since the config is `None`."
        )
        return model, optimizer
    assert isinstance(config, dict)
    if mesh is not None:
        assert isinstance(
            mesh, core.ProcessMesh
        ), "The mesh must be an instance of paddle.distributed.ProcessMesh."
        g_mesh = fleet.auto.get_mesh()
        if g_mesh is not None and g_mesh != mesh:
            warnings.warn(
                "The mesh set by `fleet.auto.set_mesh` is different with the mesh pass to "
                "`parallelize`. Will overwrite the previous mesh"
            )
        fleet.auto.set_mesh(mesh)
    pp_config = config.get('pp_config')
    mp_config = config.get('mp_config')
    dp_config = config.get('dp_config')
    if pp_config is not None:
        assert isinstance(pp_config, dict)
        model, optimizer = pipeline_parallel(
            model,
            optimizer,
            pp_config,
        )
    if mp_config is not None:
        assert isinstance(mp_config, dict)
        model, optimizer = tensor_parallel(model, optimizer, mp_config)
    if dp_config is not None:
        assert isinstance(dp_config, dict)
        if 'sharding_level' not in dp_config.keys():
            warnings.warn(
                "The dp_config doesn't contain sharding_level, will run under dp."
            )
        model, optimizer = sharded_data_parallel(
            model,
            optimizer,
            config=dp_config,
        )
    model, optimizer = parallelize_model_and_optimizer(model, optimizer)
    return model, optimizer


has_parallelized_model = False


def parallelize_model(model, mesh=None, config=None):
    if config is None:
        warnings.warn(
            "The `parallelize_model will do nothing since the config is `None`."
        )
        return model
    assert isinstance(config, dict)
    if mesh is not None:
        assert isinstance(
            mesh, core.ProcessMesh
        ), "The mesh must be an instance of paddle.distributed.ProcessMesh."
        g_mesh = fleet.auto.get_mesh()
        if g_mesh is not None and g_mesh != mesh:
            warnings.warn(
                "The mesh set by `fleet.auto.set_mesh` is different with the mesh pass to "
                "`parallelize_model`. Will overwrite the previous mesh"
            )
        fleet.auto.set_mesh(mesh)
    global has_parallelized_model
    has_parallelized_model = True
    model, _ = parallelize(model, None, mesh, config)
    return model


def parallelize_optimizer(optimizer, mesh=None, config=None):
    if config is None:
        warnings.warn(
            "The `parallelize_optimizer will do nothing since the config is `None`."
        )
        return optimizer
    assert isinstance(config, dict)
    if mesh is not None:
        assert isinstance(
            mesh, core.ProcessMesh
        ), "The mesh must be an instance of paddle.distributed.ProcessMesh."
        g_mesh = fleet.auto.get_mesh()
        if g_mesh is not None and g_mesh != mesh:
            warnings.warn(
                "The mesh set by `fleet.auto.set_mesh` is different with the mesh pass to "
                "`parallelize_optimizer`. Will overwrite the previous mesh"
            )
        fleet.auto.set_mesh(mesh)

    global has_parallelized_model
    assert (
        has_parallelized_model
    ), "Please parallelize the model before parallelize optimizer."
    param_list = optimizer._parameter_list
    if isinstance(param_list[0], dict):
        for param_group in param_list:
            for param in param_group['params']:
                assert (
                    param.is_dist()
                ), "Please use model after parallelize to create optimizer."
    else:
        for param in param_list:
            assert (
                param.is_dist()
            ), "Please use model after parallelize to create optimizer."

    dp_config = config.get('dp_config')
    level = None
    sharding_mesh_dim = None
    if dp_config is not None:
        if 'sharding_level' not in dp_config.keys():
            warnings.warn(
                "The dp_config doesn't contain sharding_level, will run under dp."
            )
        level = dp_config.get('sharding_level')
        sharding_mesh_dim = dp_config.get('sharding_mesh_dim', "dp")
    optimizer = ParallelOptimizer(optimizer, level, sharding_mesh_dim)
    optimizer = optimizer.parallelize()
    return optimizer
