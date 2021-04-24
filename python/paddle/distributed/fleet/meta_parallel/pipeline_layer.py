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

from paddle.fluid.dygraph.layers import Layer
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
import math
from ..utils import hybrid_parallel_util as hp_util


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
        return hp_util.call_to_str(self.layer_func.__name__, *self.inputs,
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
        self.device_id = dist.ParallelEnv().device_id

        self.layers = layers
        self._loss_fn = loss_fn
        self._topo = topology
        word_size = dist.get_world_size()
        self.global_rank = dist.get_rank()

        if self._topo:
            self._stage_id = self._topo.get_coord(self.global_rank).pipe
            self._num_stages = self._topo.get_dim_size("pipe")
        else:
            # construct topology
            if word_size % num_stages != 0:
                raise ValueError("should provide correct num_stages({}) "
                                 "which can be divided by word_size({})".format(
                                     num_stages, word_size))
            dp_num = word_size // num_stages
            self._topo = fleet.CommunicateTopology(["data", "model", "pipe"],
                                                   [dp_num, 1, num_stages])
            self._stage_id = self._topo.get_coord(self.global_rank).pipe
            self._num_stages = self._topo.get_dim_size("pipe")

        # initialize segment
        self._layers_desc = list(self.layers)
        self._num_layers = len(self._layers_desc)
        self._start_pos = 0
        self._end_pos = self._num_layers - 1
        self._segment_network(seg_method)

        # construct layer
        self.forward_funcs = []
        self._build()
        self.to(paddle.CUDAPlace(self.device_id))

    def _segment_network(self, seg_method):
        print("start segment network..")
        seg = hp_util.SegmentLayers(
            self._layers_desc, num_parts=self._num_stages, method=seg_method)
        self.segment_parts = seg.do_segment()

        self._start_pos = self.segment_parts[self._stage_id]
        self._end_pos = self.segment_parts[self._stage_id + 1]

        # print information for debug
        for stage in range(self._num_stages):
            start = self.segment_parts[stage]
            end = self.segment_parts[stage + 1]
            print("stage={}, global_rank={} ,layer_number={}".format(
                stage, self.global_rank, end - start))
            for index, layer in enumerate(self._layers_desc[start:end]):
                print("{}: {}".format(index + start, str(layer)))

        if self._loss_fn:
            try:
                print("loss: {}".format(self._loss_fn.__name__))
            except AttributeError:
                print("loss: {}".format(self._loss_fn.__class__.__name__))

    def _build(self):
        start = self._start_pos
        end = self._end_pos
        for index, layer in enumerate(self._layers_desc[start:end]):
            layer_index = start + index
            if isinstance(layer, Layer):
                self.forward_funcs.append(layer)
                self.add_sublayer(str(layer_index), layer)
            elif isinstance(layer, LayerDesc):
                model = layer.build_layer()
                self.forward_funcs.append(model)
                self.add_sublayer(str(layer_index), model)
            else:
                self.forward_funcs.append(layer)

    def forward(self, input):
        for layer in self.forward_funcs:
            input = paddle.cast(input, "float32")
            input = layer(input)
        return input
