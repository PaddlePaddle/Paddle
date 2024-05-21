#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# The file has been adapted from the file:
#     https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/module.py
#     Git commit hash: fafc827d643b3eed611e282d909025f16be36601
# We retain the following license from the original files:
# MIT License

# Copyright (c) Microsoft Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

import glob
import math
import os
import re
from functools import partial

import paddle
from paddle import framework, nn
from paddle.device.cuda.cuda_graphed_layer import CUDAGraphedLayer
from paddle.distributed.fleet.utils.log_util import layer_to_str, logger
from paddle.incubate.distributed.fleet import recompute_hybrid

__all__ = []


class LayerDesc:
    def __init__(self, layer_func, *inputs, **kwargs):
        self.layer_func = layer_func
        self.inputs = inputs
        self.kwargs = kwargs

        if not issubclass(layer_func, nn.Layer):
            raise TypeError(
                "The input(layer_func) should be a derived class of Layer."
            )

    def build_layer(self):
        return self.layer_func(*self.inputs, **self.kwargs)

    def __repr__(self):
        return layer_to_str(
            self.layer_func.__name__, *self.inputs, **self.kwargs
        )


class SharedLayerDesc(LayerDesc):
    def __init__(
        self,
        key,
        layer_func,
        forward_func=None,
        shared_weight_attr='weight',
        *inputs,
        **kwargs,
    ):
        super().__init__(layer_func, *inputs, **kwargs)
        self.layer_name = key
        self.forward_func = forward_func
        self.shared_weight_attr = shared_weight_attr


class SegmentLayers:
    def __init__(
        self,
        layers_desc,
        num_parts,
        method="uniform",
        num_virtual_pipeline_stage=None,
    ):
        self._layers_desc = layers_desc
        self.method = method
        self.num_parts = num_parts
        self.num_items = len(layers_desc)
        self.num_virtual_pipeline_stage = num_virtual_pipeline_stage
        if self.num_virtual_pipeline_stage is not None:
            self.total_parts = num_parts * self.num_virtual_pipeline_stage
        assert (
            self.num_items >= self.num_parts
        ), "layer number should be greater than number of segments"

    def do_segment(self):
        if isinstance(self.method, list):
            seg_method = self.method[:]
            source_num_parts = len(seg_method) - 1

            def check_sanity():
                assert seg_method[0] == 0, "seg_method[0] should be 0"
                for part in seg_method:
                    assert isinstance(part, int), "part should be int"
                    assert part >= 0, f"part[{part}] should be greater than 0"
                    assert (
                        part <= self.num_items
                    ), f"part[{part}] should be less than num_items[{self.num_items}]"

            check_sanity()

            if self.num_parts == source_num_parts + 1:
                seg_method.append(self.num_items)
                return seg_method
            elif self.num_parts == source_num_parts:
                return seg_method
            else:
                raise ValueError(
                    f"We set seg_method as {seg_method}, this length is {len(seg_method)}, but the number of stages is {self.num_parts}"
                )

        elif self.method == "uniform":
            return self.uniform(self.num_items, self.num_parts)

        elif self.method.startswith('layer:'):
            # Divide equally according to the specified layer
            layername = self.method.split(':')[1]
            weights = [0] * len(self._layers_desc)
            weight_idxs = self._gen_layer_weight(layername)
            for idx in weight_idxs:
                weights[idx] = 1

            actual_num_parts = (
                self.num_parts
                if self.num_virtual_pipeline_stage is None
                else self.total_parts
            )

            assert (
                sum(weights) % actual_num_parts == 0
            ), f"number of layers ({sum(weights)}) should be divided by part number({actual_num_parts})"
            part_size = sum(weights) // actual_num_parts
            result = [0 for _ in range(actual_num_parts + 1)]

            memory_counter = 0
            result_idx = 1
            for idx, weight in enumerate(weights):
                memory_counter += weight
                if memory_counter == part_size:
                    result[result_idx] = idx + 1
                    result_idx += 1
                    memory_counter = 0
            result[actual_num_parts] = len(weights)
            return result
        else:
            raise ValueError(f"method {self.method} is not supported")

    def _gen_layer_weight(self, layername):
        weight_idxs = []
        regex = re.compile(layername, re.IGNORECASE)
        for idx, layer in enumerate(self._layers_desc):
            name = None
            if isinstance(layer, nn.Layer):
                name = layer.__class__.__name__
            elif isinstance(layer, LayerDesc):
                name = layer.layer_func.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    # it is not error
                    continue
            if regex.search(name):
                weight_idxs.append(idx)

        assert (
            len(weight_idxs) > 0
        ), "weight_idxs' length should be greater than 0"
        return weight_idxs

    def uniform(self, num_items, num_parts):
        result = [0 for _ in range(num_parts + 1)]
        part_size = math.floor(num_items / num_parts)
        extra_layers = num_items % num_parts
        for i in range(1, num_parts):
            offset = 1 if i > (num_parts - extra_layers) else 0
            result[i] = int(min(result[i - 1] + part_size + offset, num_items))
        result[num_parts] = num_items
        return result


class PipelineLayerChunk(nn.Layer):
    def __init__(self):
        super().__init__()
        self.run_function = []

    def append(self, sublayer):
        # This method is used to unify codes in _build_layer_impl.
        # For 1f1b scheduler, it will call append method of a List.
        # For interleave scheduler, it will call append method of this class.
        if isinstance(sublayer, nn.Layer):
            self.add_sublayer(str(len(self.run_function)), sublayer)
        self.run_function.append(sublayer)

    def extend(self, layer_list):
        for layer in layer_list:
            self.append(layer)

    def get_run_function(self):
        return self.run_function

    def forward(self, *args, **kwargs):
        # Users shouldn't call PipelineLayerChunk directly, since all logics relating with recompute
        # are in the forward function of PipelineLayer. Any directly call will bring unexpected
        # behavior under recompute circumstance.
        raise PermissionError(
            "The forward function of PipelineLayerChunk cannot be called directly. "
            "Please call forward function of PipelineLayer."
        )

    def __iter__(self):
        return iter(self.run_function)


class PipelineSublayers(nn.Layer):
    def __init__(self, run_function):
        super().__init__()
        self.run_function = run_function
        for idx, sublayer in enumerate(self.run_function):
            if isinstance(sublayer, nn.Layer):
                self.add_sublayer(str(idx), sublayer)

    def forward(self, x):
        for layer in self.run_function:
            x = layer(x)
        return x

    def __iter__(self):
        return iter(self.run_function)


class PipelineLayer(nn.Layer):
    """PipelineLayer
    Args:
        layers(Iterable): A sequence of layers description to define the structure for pipeline.
        num_stages(int, optional): pp degree, if not specified, 'topology' parameter must be given.
        topology(CommunicateTopology, optional): topo of hybrid parallel, if it is None, 'num_stages' parameters must be given.
        loss_fn(callable, optional): Loss function.
        seg_method(str, optional): the method of splitting pp layer, default 'uniform', or use specific layer to split, method's name must be start with 'layer:'.
        recompute_interval(int, optional): the number of layers to be used recompute, the value of 0 represents no recompute. default 0.
        recompute_ctx(dict,optional): the context of recompute, when 'recompute_interval' > 0, the context must be given.
        num_virtual_pipeline_stages(int, optional): the num of virtual pipeline stages for interleave pp.
        use_cudagraph(bool, optional): enable CUDAGraphedLayer in pp layers.
    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:DISTRIBUTED)
            >>> import paddle.nn as nn
            >>> import paddle.nn.functional as F
            >>> from paddle.distributed import fleet
            >>> from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer

            >>> pipeline_parallel_size = 2
            >>> strategy = fleet.DistributedStrategy()
            >>> strategy.hybrid_configs = {
            ...     "dp_degree": 1,
            ...     "mp_degree": 1,
            ...     "pp_degree": pipeline_parallel_size
            >>> }
            >>> strategy.pipeline_configs = {
            ...     "accumulate_steps": 4,
            ...     "micro_batch_size": 2
            >>> }

            >>> fleet.init(is_collective=True, strategy=strategy)

            >>> hcg = fleet.get_hybrid_communicate_group()

            >>> class ReshapeHelp(nn.Layer):
            ...     def __init__(self, shape):
            ...         super().__init__()
            ...         self.shape = shape
            ...     def forward(self, x):
            ...         return x.reshape(shape=self.shape)

            >>> class AlexNetPipeDesc(PipelineLayer):
            ...     def __init__(self, num_classes=10, **kwargs):
            ...         self.num_classes = num_classes
            ...         decs = [
            ...             LayerDesc(
            ...                 nn.Conv2D, 1, 64, kernel_size=11, stride=4, padding=5),
            ...             LayerDesc(nn.ReLU),
            ...             LayerDesc(
            ...                 nn.MaxPool2D, kernel_size=2, stride=2),
            ...             LayerDesc(
            ...                 nn.Conv2D, 64, 192, kernel_size=5, padding=2),
            ...             F.relu,
            ...             LayerDesc(
            ...                 nn.MaxPool2D, kernel_size=2, stride=2),
            ...             LayerDesc(
            ...                 nn.Conv2D, 192, 384, kernel_size=3, padding=1),
            ...             F.relu,
            ...             LayerDesc(
            ...                 nn.Conv2D, 384, 256, kernel_size=3, padding=1),
            ...             F.relu,
            ...             LayerDesc(
            ...                 nn.Conv2D, 256, 256, kernel_size=3, padding=1),
            ...             F.relu,
            ...             LayerDesc(
            ...                 nn.MaxPool2D, kernel_size=2, stride=2),
            ...             LayerDesc(
            ...                 ReshapeHelp, shape=[-1, 256]),
            ...             LayerDesc(nn.Linear, 256, self.num_classes),  # classifier
            ...         ]
            ...         super().__init__(
            ...             layers=decs, loss_fn=nn.CrossEntropyLoss(), **kwargs)

            >>> model = AlexNetPipeDesc(num_stages=pipeline_parallel_size, topology=hcg._topo)

    """

    def __init__(
        self,
        layers,
        num_stages=None,
        topology=None,
        loss_fn=None,
        seg_method="uniform",
        recompute_interval=0,
        recompute_ctx=None,
        num_virtual_pipeline_stages=None,
        use_cudagraph=False,
    ):
        super().__init__()
        if num_stages is None and topology is None:
            raise ValueError("should provide num_stages or topology")

        if num_virtual_pipeline_stages:
            assert isinstance(
                num_virtual_pipeline_stages, int
            ), "virtual_pipeline_stage should be None or an int"
            if num_virtual_pipeline_stages > 1:
                logger.info(
                    "set num_virtual_pipeline_stages > 1 means using interleave scheduler instead of 1f1b scheduler"
                )
                assert isinstance(
                    seg_method, str
                ), "seg_method should be a str for interleave scheduler"
                assert seg_method.startswith(
                    'layer:'
                ), "seg_method should be start with layer: for interleave scheduler"

        self._num_virtual_pipeline_stages = (
            1
            if num_virtual_pipeline_stages is None
            else num_virtual_pipeline_stages
        )

        # lazy import
        import paddle.distributed as dist
        from paddle.distributed import fleet

        self.device_id = dist.ParallelEnv().device_id
        self.layers = layers
        self._loss_fn = loss_fn if isinstance(loss_fn, list) else [loss_fn]
        self._topo = topology
        self._recompute_interval = recompute_interval
        self.recompute_ctx = recompute_ctx
        self.use_cudagraph = use_cudagraph

        # Defaults to 1234 to initialize layer parameters
        self._base_seed = 1234

        if recompute_interval > 0:
            assert (
                recompute_ctx is not None
            ), "recompute_ctx must be not None for recompute."

            offload = recompute_ctx.get('offload', False)
            partition = recompute_ctx.get('partition', False)
            logger.info(
                f"Start Recompute for PipeLineParallel. recompute_offload: {offload}, recompute_partition: {partition}"
            )

        world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()

        if self._topo:
            self._stage_id = self._topo.get_coord(self.global_rank).pipe
            self._num_stages = self._topo.get_dim_size("pipe")
            if num_stages:
                assert (
                    self._num_stages == num_stages
                ), "num_stages should be equal to be %d" % (self._num_stages)
        else:
            # construct default topology
            if world_size % num_stages != 0:
                raise ValueError(
                    f"should provide correct num_stages({num_stages}) "
                    f"which can be divided by world_size({world_size})"
                )
            dp_num = world_size // num_stages
            self._topo = fleet.CommunicateTopology(
                ["data", "pipe", "model"], [dp_num, num_stages, 1]
            )
            self._stage_id = self._topo.get_coord(self.global_rank).pipe
            self._num_stages = self._topo.get_dim_size("pipe")

        self._total_stages_with_virtual_stages = (
            self._num_stages * self._num_virtual_pipeline_stages
        )

        # initialize segment
        self._layers_desc = list(self.layers)
        self._num_layers = len(self._layers_desc)
        self.shared_layers = paddle.nn.LayerDict()
        self.shared_weight_attrs = {}

        if self._num_virtual_pipeline_stages > 1:
            # interleaving pipeline segmentation
            self._start_poss = []
            self._end_poss = []
            self._segment_network_for_interleave(seg_method)
            # The _model_chunks is a list of PipelineLayerChunk,
            # while PipelineLayerChunk is a list of Layers relating with one model chunk.
            # Therefore, the _model_chunks is something like 'list of a list of layers'.
            self._model_chunks = []
            self._build_layer_with_interleave()
        else:
            # 1f1b pipeline segmentation
            self._start_pos = 0
            self._end_pos = self._num_layers - 1
            self._segment_network(seg_method)
            # construct layer
            self.run_function = []
            self._build_layer()

        self.shared_comm = self._construct_shared_comm()
        self._synchronize_shared_weights()

    def get_stage_from_index(self, layer_idx):
        assert 0 <= layer_idx < self._num_layers, "layer_idx is out of bound"
        for virtual_pp_rank in range(self._num_virtual_pipeline_stages):
            # Mapping the virtual pipeline stage to the real pipeline stage.
            # start_idx marks the start of a new virtual pp stage.
            start_idx = virtual_pp_rank * self._num_stages
            for stage in range(self._num_stages):
                # stage mark the real pp stage
                if (
                    self.segment_parts[start_idx + stage]
                    <= layer_idx
                    < self.segment_parts[start_idx + stage + 1]
                ):
                    return stage

    def get_num_virtual_stages(self):
        return self._num_virtual_pipeline_stages

    def get_model_chunks(self):
        return (
            None
            if self._num_virtual_pipeline_stages == 1
            else self._model_chunks
        )

    def _construct_shared_comm(self):
        shared_comm = {}
        if self._topo.get_dim("pipe") == 1:
            return

        layers_desc = self._layers_desc
        shared_layer_names = {
            s.layer_name for s in layers_desc if isinstance(s, SharedLayerDesc)
        }
        for key in shared_layer_names:
            shared_layers = []
            for idx, layer in enumerate(layers_desc):
                if (
                    isinstance(layer, SharedLayerDesc)
                    and layer.layer_name == key
                ):
                    shared_layers.append(idx)

            shared_stages = {
                self.get_stage_from_index(idx) for idx in shared_layers
            }
            self._dp_degree = self._topo.get_dim('data')
            self._mp_degree = self._topo.get_dim('model')
            self._sharding_degree = self._topo.get_dim('sharding')

            shared_ranks = []
            for dp in range(self._dp_degree):
                for sharding in range(self._sharding_degree):
                    for mp in range(self._mp_degree):
                        shared_ranks = []
                        for s in sorted(shared_stages):
                            shared_ranks.append(
                                self._topo.get_rank_from_stage(
                                    self.global_rank,
                                    pipe=s,
                                    data=dp,
                                    sharding=sharding,
                                    model=mp,
                                )
                            )

                        group = paddle.distributed.new_group(ranks=shared_ranks)
                        if self.global_rank in shared_ranks:
                            assert key in self.shared_layers
                            if key in self.shared_layers:
                                shared_comm[key] = {
                                    'ranks': shared_ranks,
                                    'group': group,
                                    'weight_attr': self.shared_weight_attrs[
                                        key
                                    ],
                                    'layer': self.shared_layers[key],
                                }
        return shared_comm

    def _synchronize_shared_weights(self):
        for key, comm in self.shared_comm.items():
            with paddle.framework.no_grad():
                paddle.distributed.broadcast(
                    getattr(comm['layer'], comm['weight_attr']),
                    src=min(comm['ranks']),
                    group=comm['group'],
                )

            for param in comm['layer'].parameters():
                if self.global_rank != min(comm['ranks']):
                    param.is_firstly_shared = False

    def allreduce_shared_weight_gradients(self):
        for key, comm in self.shared_comm.items():
            param = getattr(self.shared_layers[key], comm['weight_attr'])
            # need use trace_op to allreduce weight
            if framework.in_dynamic_mode():
                with paddle.framework.no_grad():
                    paddle.distributed.all_reduce(
                        param.grad
                        if not hasattr(param, "main_grad")
                        else param.main_grad,
                        group=comm['group'],
                    )
            else:
                with paddle.framework.no_grad():
                    framework._dygraph_tracer().trace_op(
                        type="c_allreduce_sum",
                        inputs={'X': param._grad_ivar()},
                        outputs={'Out': param._grad_ivar()},
                        attrs={
                            'ring_id': comm['group'].id,
                            'use_calc_stream': True,
                        },
                    )

    def _segment_network_for_interleave(self, seg_method):
        logger.info("start segment network for interleave scheduler")
        seg = SegmentLayers(
            self._layers_desc,
            num_parts=self._num_stages,
            method=seg_method,
            num_virtual_pipeline_stage=self._num_virtual_pipeline_stages,
        )
        self.segment_parts = seg.do_segment()

        logger.info(
            f"segment with method: {seg_method}; result: "
            + ", ".join(str(arg) for arg in self.segment_parts)
        )

        for i in range(
            self._stage_id,
            self._total_stages_with_virtual_stages,
            self._num_stages,
        ):
            # If there are 2 real pp stages and 2 virtual pp stages, and the model has 8 layers.
            # Layers [0, 1], [4, 5] will be assigned to the first real pp stage.
            # Layers [2, 3], [6, 7] will be assigned to the second real pp stage.
            # Layers [0, 1] and [2, 3] are the first virtual pp stage in each real pp stage.
            # Layers [4, 5] and [6, 7] are the second virtual pp stage in each real pp stage.
            assert self.segment_parts[i] <= self.segment_parts[i + 1]
            self._start_poss.append(self.segment_parts[i])
            self._end_poss.append(self.segment_parts[i + 1])

        assert len(self._start_poss) == len(self._end_poss)

        self._print_segmentation_for_debug()

    def _segment_network(self, seg_method):
        logger.info("start segment network..")
        seg = SegmentLayers(
            self._layers_desc, num_parts=self._num_stages, method=seg_method
        )
        self.segment_parts = seg.do_segment()

        logger.info(
            f"segment with method: {seg_method}; result: "
            + ", ".join(str(arg) for arg in self.segment_parts)
        )

        self._start_pos = self.segment_parts[self._stage_id]
        self._end_pos = self.segment_parts[self._stage_id + 1]
        self._print_segmentation_for_debug()

    def _print_segmentation_for_debug(self):
        # print information for debug
        for stage in range(
            self._num_stages * self._num_virtual_pipeline_stages
        ):
            start = self.segment_parts[stage]
            end = self.segment_parts[stage + 1]
            logger.info(
                f"stage={stage}, global_rank={self.global_rank} ,layer_number={end - start}"
            )

            for index, layer in enumerate(self._layers_desc[start:end]):
                logger.info(f"{index + start}: {str(layer)}")

        if self._num_virtual_pipeline_stages > 1:
            for stage in range(self._num_stages):
                stage_to_virtual_stage_info = (
                    f"stage {stage} contains virtual stages: "
                )
                for i in range(
                    stage,
                    self._total_stages_with_virtual_stages,
                    self._num_stages,
                ):
                    stage_to_virtual_stage_info += f" {i},"
                logger.info(stage_to_virtual_stage_info)

        if self._loss_fn[0]:
            loss_fn_names = []
            for idx in range(len(self._loss_fn)):
                try:
                    loss_fn_names.append(self._loss_fn[idx].__name__)
                except AttributeError:
                    loss_fn_names.append(self._loss_fn[idx].__class__.__name__)
            logger.info(f"loss: {', '.join(loss_fn_names)}")

    def _build_layer_with_interleave(self):
        from paddle.distributed.fleet.meta_parallel.parallel_layers.random import (
            get_rng_state_tracker,
        )

        orig_rng_state = paddle.get_rng_state()
        orig_rng_tracker = get_rng_state_tracker().get_states_tracker()

        for i in range(len(self._start_poss)):
            start = self._start_poss[i]
            end = self._end_poss[i]
            # Get a model chunk
            chunk = self._build_layer_impl(start, end)
            assert isinstance(chunk, PipelineLayerChunk)
            # Add the chunk to all chunks and add this chunk to the sublayer
            self._model_chunks.append(chunk)
            self.add_sublayer(str(start), chunk)

        paddle.set_rng_state(orig_rng_state)
        get_rng_state_tracker().set_states_tracker(orig_rng_tracker)

    def _build_layer(self):
        from paddle.distributed.fleet.meta_parallel.parallel_layers.random import (
            get_rng_state_tracker,
        )

        orig_rng_state = paddle.get_rng_state()
        orig_rng_tracker = get_rng_state_tracker().get_states_tracker()

        start = self._start_pos
        end = self._end_pos
        self.run_function = self._build_layer_impl(start, end)

        paddle.set_rng_state(orig_rng_state)
        get_rng_state_tracker().set_states_tracker(orig_rng_tracker)

    def _build_layer_impl(self, start, end):
        if self._num_virtual_pipeline_stages > 1:
            # For interleave scheduler, all layers relating with one model chunk will be saved in PipelineLayerChunk
            run_function = PipelineLayerChunk()
        else:
            # For 1f1b scheduler, just use run_function list
            run_function = self.run_function

        self.groupable_layers = []

        def flush_into_run_function():
            if len(self.groupable_layers) > 0:
                logger.info(
                    f"flush {len(self.groupable_layers)} of layers into run_function"
                )
                if self.use_cudagraph:
                    pipeline_sublayer = PipelineSublayers(
                        self.groupable_layers.copy()
                    )
                    pipeline_sublayer = CUDAGraphedLayer(pipeline_sublayer)
                    run_function.append(pipeline_sublayer)
                else:
                    run_function.extend(self.groupable_layers)
                self.groupable_layers = []

        for index, layer in enumerate(self._layers_desc[start:end]):
            layer_index = start + index

            # NOTE(shenliang03): need set different seeds for pipeline parameters initialization.
            # Since the parameters of model_parallel are controlled by its own RNG_STATE_TRACKER,
            # only non-mp parameters in pp are controlled here.
            paddle.seed(self._base_seed + layer_index)

            if isinstance(layer, nn.Layer):
                self.groupable_layers.append(layer)
                if self._num_virtual_pipeline_stages == 1:
                    # Only add sublayer for 1f1b scheduler,
                    # for interleave, PipelineLayerChunk will do this
                    self.add_sublayer(str(layer_index), layer)
            elif isinstance(layer, SharedLayerDesc):
                flush_into_run_function()
                if layer.layer_name not in self.shared_layers:
                    self.shared_layers[layer.layer_name] = layer.build_layer()
                    self.shared_weight_attrs[
                        layer.layer_name
                    ] = layer.shared_weight_attr
                    for param in self.shared_layers[
                        layer.layer_name
                    ].parameters():
                        param.is_firstly_shared = True

                if layer.forward_func is None:
                    run_function.append(self.shared_layers[layer.layer_name])

                else:
                    run_function.append(
                        partial(
                            layer.forward_func,
                            self.shared_layers[layer.layer_name],
                        )
                    )
                    # Note: the PipelineLayerChunk won't add the partial function to the sub layer,
                    # will introduce error when calling chunk.parameters(). Have to manually add
                    # this layer to the chunk's sub layer.
                    if self._num_virtual_pipeline_stages > 1:
                        run_function.add_sublayer(
                            layer.layer_name,
                            self.shared_layers[layer.layer_name],
                        )

            elif isinstance(layer, LayerDesc):
                model = layer.build_layer()
                self.groupable_layers.append(model)
                if self._num_virtual_pipeline_stages == 1:
                    # Only add sublayer for 1f1b scheduler,
                    # for interleave, PipelineLayerChunk will do this
                    self.add_sublayer(str(layer_index), model)
            else:
                flush_into_run_function()
                run_function.append(layer)

        flush_into_run_function()
        return run_function

    def forward_function(self, start, end):
        run_function = self.run_function

        def execute_func(*x):
            if len(x) == 1:
                x = x[0]
            for idx, layer in enumerate(run_function[start:end]):
                x = layer(x)
            return x

        return execute_func

    def forward(self, input, chunk_id=None):
        if chunk_id is not None:
            assert isinstance(chunk_id, int), "chunk_id should be an int"
            assert (
                self._num_virtual_pipeline_stages > 1
            ), "chunk_id is only valid when using virtual pipeline stage"
            assert chunk_id < len(self._model_chunks), (
                f"The virtual pipeline only has {len(self._model_chunks)} chunks, "
                f"but received chunk_id {chunk_id}."
            )
            # Get the target model chunk.
            model_chunk = self._model_chunks[chunk_id]
            # Update the self.run_function to the target run functions.
            # Runs for 1f1b and interleave are similar, just handle all functions in self.run_function.
            # The only different is that, for 1f1b, self.run_function has already been inited during build_layer.
            # But for interleave, self.run_function will keep updating to the target functions at every run.
            self.run_function = model_chunk.get_run_function()

        if self._recompute_interval == 0:
            input = self.forward_function(0, len(self.run_function))(input)
        else:
            num_layers = len(self.run_function)
            for start_idx in range(0, num_layers, self._recompute_interval):
                end_idx = min(start_idx + self._recompute_interval, num_layers)
                funcs = self.run_function[start_idx:end_idx]

                if not isinstance(input, tuple):
                    input = (input,)

                if self._need_recompute(funcs, input):
                    input = recompute_hybrid(
                        self.recompute_ctx,
                        self.forward_function(start_idx, end_idx),
                        *input,
                    )
                else:
                    input = self.forward_function(start_idx, end_idx)(*input)

        return input

    def _need_recompute(self, funcs, inputs):
        if not any(
            not input_.stop_gradient
            for input_ in inputs
            if isinstance(input_, paddle.Tensor)
        ):
            return False

        params = [f.parameters() for f in funcs if isinstance(f, nn.Layer)]
        return any(len(list(p)) > 0 for p in params)

    def save_state_dict(self, path):
        if self._topo.get_coord(self.global_rank).data != 0:
            return

        def _offset_dirname(ckpt_dir, local_layer_idx, local_chunk_id=None):
            if self._num_virtual_pipeline_stages == 1:
                pos_offset = self._start_pos
            else:
                assert hasattr(self, '_start_poss')
                assert local_chunk_id < len(self._start_poss)
                pos_offset = self._start_poss[local_chunk_id]
            idx = local_layer_idx + pos_offset
            model_rank = self._topo.get_coord(self.global_rank).model
            rank_message = "-tensor_" + f"{model_rank:0>2d}"
            virtual_pipeline_stage_message = ""
            if self._num_virtual_pipeline_stages > 1:
                # add virtual pipeline info to the save path
                assert local_chunk_id is not None
                virtual_pipeline_stage_message = (
                    f"-virtual_pp_stage_{local_chunk_id:0>2d}"
                )
            layer_save_path = os.path.join(ckpt_dir, f'layer_{idx:0>2d}')
            layer_save_path = (
                layer_save_path
                + virtual_pipeline_stage_message
                + rank_message
                + '-model_states.pdparams'
            )
            return layer_save_path

        def _save_model(run_functions, local_chunk_id=None):
            for idx, layer in enumerate(run_functions):
                model_save_path = _offset_dirname(path, idx, local_chunk_id)
                if not hasattr(layer, 'state_dict'):
                    continue
                paddle.save(layer.state_dict(), model_save_path)

        os.makedirs(path, exist_ok=True)
        if self._num_virtual_pipeline_stages > 1:
            logger.info("save model state for virtual pipeline stage...")
            for chunk_id in range(len(self._model_chunks)):
                run_function = self._model_chunks[chunk_id].get_run_function()
                _save_model(run_function, chunk_id)
        else:
            _save_model(self.run_function)

        logger.info("save model state successfully...")

    def set_state_dir(self, path):
        assert os.path.exists(path), f"{path} not found, please check the path"

        def _load_model(run_functions, local_chunk_id=None):
            for idx, layer in enumerate(run_functions):
                if not hasattr(layer, 'set_state_dict'):
                    continue
                if self._num_virtual_pipeline_stages == 1:
                    pos_offset = self._start_pos
                else:
                    assert hasattr(self, '_start_poss')
                    assert local_chunk_id < len(self._start_poss)
                    pos_offset = self._start_poss[local_chunk_id]
                layer_idx = idx + pos_offset
                layer_save_path = os.path.join(path, f'layer_{layer_idx:0>2d}')
                if self._num_virtual_pipeline_stages > 1:
                    # add virtual pipeline info to the path
                    assert local_chunk_id is not None
                    layer_save_path = (
                        layer_save_path
                        + f"-virtual_pp_stage_{local_chunk_id:0>2d}"
                    )
                model_files = glob.glob(
                    layer_save_path + "*model_states.pdparams"
                )
                model_files.sort()
                mp_rank = self._topo.get_coord(self.global_rank).model
                mp_world_size = self._topo.get_dim('model')
                num_files = len(model_files)

                load_param_path = model_files[
                    mp_rank * num_files // mp_world_size
                ]
                model_state_dict = paddle.load(load_param_path)
                layer.set_state_dict(model_state_dict)

        if self._num_virtual_pipeline_stages > 1:
            logger.info("load model state for virtual pipeline stage...")
            for chunk_id in range(len(self._model_chunks)):
                run_function = self._model_chunks[chunk_id].get_run_function()
                _load_model(run_function, chunk_id)
        else:
            _load_model(self.run_function)

        self._synchronize_shared_weights()
        logger.info("load model state successfully...")
