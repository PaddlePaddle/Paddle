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
import math
import paddle
from paddle.fluid.dygraph.layers import Layer
from ...utils.log_util import logger, layer_to_str

__all__ = []


class SegmentLayers(object):
    def __init__(self, layers_desc, num_parts, method="uniform"):
        self._layers_desc = layers_desc
        self.method = method
        self.num_parts = num_parts
        self.num_items = len(layers_desc)
        assert self.num_items >= self.num_parts, "layer number should be greater than number of segments"

    def do_segment(self):
        if self.method == "uniform":
            return self.uniform(self.num_items, self.num_parts)

    def uniform(self, num_items, num_parts):
        result = [0 for _ in range(num_parts + 1)]
        part_size = math.floor(num_items / num_parts)
        for i in range(num_parts):
            result[i] = int(min(part_size * i, num_items))
        result[num_parts] = num_items
        return result


class LayerDesc(object):
    def __init__(self, layer_func, *inputs, **kwargs):
        self.layer_func = layer_func
        self.inputs = inputs
        self.kwargs = kwargs

        if not issubclass(layer_func, Layer):
            raise TypeError(
                "The input(layer_func) should be a derived class of Layer.")

    def build_layer(self):
        return self.layer_func(*self.inputs, **self.kwargs)

    def __repr__(self):
        return layer_to_str(self.layer_func.__name__, *self.inputs,
                            **self.kwargs)


class PipelineLayer(Layer):
    def __init__(self,
                 layers,
                 num_stages=None,
                 topology=None,
                 loss_fn=None,
                 seg_method="uniform"):
        super(PipelineLayer, self).__init__()
        if num_stages is None and topology is None:
            raise ValueError("should provide num_stages or topology")

        # lazy import
        import paddle.distributed as dist
        from paddle.distributed import fleet

        self.device_id = dist.ParallelEnv().device_id
        self.layers = layers
        self._loss_fn = loss_fn
        self._topo = topology
        world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()

        if self._topo:
            self._stage_id = self._topo.get_coord(self.global_rank).pipe
            self._num_stages = self._topo.get_dim_size("pipe")
            if num_stages:
                assert self._num_stages == num_stages, "num_stages should be equal to be %d" % (
                    self._num_stages)
        else:
            # construct default topology
            if world_size % num_stages != 0:
                raise ValueError("should provide correct num_stages({}) "
                                 "which can be divided by world_size({})".
                                 format(num_stages, world_size))
            dp_num = world_size // num_stages
            self._topo = fleet.CommunicateTopology(["data", "pipe", "model"],
                                                   [dp_num, num_stages, 1])
            self._stage_id = self._topo.get_coord(self.global_rank).pipe
            self._num_stages = self._topo.get_dim_size("pipe")

        # initialize segment
        self._layers_desc = list(self.layers)
        self._num_layers = len(self._layers_desc)
        self._start_pos = 0
        self._end_pos = self._num_layers - 1
        self._segment_network(seg_method)

        # construct layer
        self.run_function = []
        self._build_layer()

    def _segment_network(self, seg_method):
        logger.info("start segment network..")
        seg = SegmentLayers(
            self._layers_desc, num_parts=self._num_stages, method=seg_method)
        self.segment_parts = seg.do_segment()

        self._start_pos = self.segment_parts[self._stage_id]
        self._end_pos = self.segment_parts[self._stage_id + 1]

        # print information for debug
        for stage in range(self._num_stages):
            start = self.segment_parts[stage]
            end = self.segment_parts[stage + 1]
            logger.info("stage={}, global_rank={} ,layer_number={}".format(
                stage, self.global_rank, end - start))

            for index, layer in enumerate(self._layers_desc[start:end]):
                logger.info("{}: {}".format(index + start, str(layer)))

        if self._loss_fn:
            try:
                logger.info("loss: {}".format(self._loss_fn.__name__))
            except AttributeError:
                logger.info("loss: {}".format(self._loss_fn.__class__.__name__))

    def _build_layer(self):
        start = self._start_pos
        end = self._end_pos
        for index, layer in enumerate(self._layers_desc[start:end]):
            layer_index = start + index
            if isinstance(layer, Layer):
                self.run_function.append(layer)
                self.add_sublayer(str(layer_index), layer)
            elif isinstance(layer, LayerDesc):
                model = layer.build_layer()
                self.run_function.append(model)
                self.add_sublayer(str(layer_index), model)
            else:
                self.run_function.append(layer)

    def forward(self, input):
        for layer in self.run_function:
            input = layer(input)
        return input
