# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import math
import sys
import time
from abc import abstractmethod
from collections import OrderedDict
from functools import reduce

import numpy as np

import paddle
from paddle.distributed.auto_parallel.cluster_v2 import DeviceMesh
from paddle.distributed.auto_parallel.completion import Completer
from paddle.distributed.auto_parallel.cost import CostEstimator
from paddle.distributed.auto_parallel.dist_attribute import (
    OperatorDistributedAttribute,
    TensorDistributedAttribute,
)
from paddle.distributed.auto_parallel.dist_context import DistributedContext
from paddle.distributed.auto_parallel.dist_tensor import DistributedTensor
from paddle.distributed.auto_parallel.process_mesh import ProcessMesh
from paddle.distributed.auto_parallel.utils import (
    is_gradient_clip_op,
    print_program_with_dist_attr,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole
from paddle.fluid import program_guard
from paddle.fluid.backward import append_backward
from paddle.fluid.framework import Parameter, unique_name

from ..graph import Graph

_PATTERNS = []
_PATTERN_MAP = {}


def register_pattern(cls):
    """Register pattern for rule-based tuner."""

    def register():
        global _PATTERNS
        global _PATTERN_MAP
        pattern = cls()
        _PATTERNS.append(pattern)
        _PATTERNS.sort(key=lambda x: -x.attrs["weights"])
        _PATTERN_MAP[pattern.name] = pattern

    register()

    return cls


class BasePattern(Graph):
    _name = "base"

    def __init__(self):
        super().__init__()
        self.build()

    @property
    def name(self):
        return self.__class__._name

    @abstractmethod
    def build(self):
        pass


@register_pattern
class QKVPattern(BasePattern):
    name = "qkv"

    def __init__(self):
        super().__init__()

    def build(self):
        query = self.add_node(0, **{"type": "var"})

        # define q, k, v weight
        q_weight = self.add_node(1, **{"dim": 2, "type": "param"})
        k_weight = self.add_node(2, **{"dim": 2, "type": "param"})
        v_weight = self.add_node(3, **{"dim": 2, "type": "param"})
        # define q, k, v matmul_v2
        q_matmul_v2 = self.add_node(4, **{"type": "matmul_v2"})
        k_matmul_v2 = self.add_node(5, **{"type": "matmul_v2"})
        v_matmul_v2 = self.add_node(6, **{"type": "matmul_v2"})
        # define input edge
        q_x_edge = self.add_edge(
            query.id, q_matmul_v2.id, **{"input_name": "X"}
        )
        k_x_edge = self.add_edge(
            query.id, k_matmul_v2.id, **{"input_name": "X"}
        )
        v_x_edge = self.add_edge(
            query.id, v_matmul_v2.id, **{"input_name": "X"}
        )
        q_y_edge = self.add_edge(
            q_weight.id, q_matmul_v2.id, **{"input_name": "Y"}
        )
        k_y_edge = self.add_edge(
            k_weight.id, k_matmul_v2.id, **{"input_name": "Y"}
        )
        v_y_edge = self.add_edge(
            v_weight.id, v_matmul_v2.id, **{"input_name": "Y"}
        )
        # define q, k, v matmul_v2 output
        q = self.add_node(7, **{"type": "var"})
        k = self.add_node(8, **{"type": "var"})
        v = self.add_node(9, **{"type": "var"})
        # define output edge
        q_out_edge = self.add_edge(
            q_matmul_v2.id, q.id, **{"output_name": "Out"}
        )
        k_out_edge = self.add_edge(
            k_matmul_v2.id, k.id, **{"output_name": "Out"}
        )
        v_out_edge = self.add_edge(
            v_matmul_v2.id, v.id, **{"output_name": "Out"}
        )

        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {
                0: [0, -1, -1],
                1: [-1, 1],
                2: [-1, 1],
                3: [-1, 1],
            },
            "mp_dp": {
                0: [1, -1, -1],
                1: [-1, 0],
                2: [-1, 0],
                3: [-1, 0],
            },
            "mp": {0: [-1, -1, -1], 1: [-1, 0], 2: [-1, 0], 3: [-1, 0]},
            "dp": {
                0: [0, -1, -1],
                1: [-1, -1],
                2: [-1, -1],
                3: [-1, -1],
            },
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 3


@register_pattern
class RowMatmulPattern(BasePattern):
    name = "row_matmul"

    def __init__(self):
        super().__init__()

    def build(self):
        # define reshape input
        input = self.add_node(0, **{"type": "var"})
        # define reshape
        reshape = self.add_node(1, **{"type": "reshape2"})
        # define reshape input egde
        x_edge = self.add_edge(input.id, reshape.id, **{"input_name": "X"})
        # define reshape out
        output = self.add_node(2, **{"type": "var"})
        # define reshape output edge
        out_edge = self.add_edge(
            reshape.id, output.id, **{"output_name": "Out"}
        )

        # define matmul_v2 weight
        weight = self.add_node(3, **{"dim": 2, "type": "param"})
        # define matmul_v2
        matmul_v2 = self.add_node(4, **{"type": "matmul_v2"})

        # define input edge
        x_edge = self.add_edge(output.id, matmul_v2.id, **{"input_name": "X"})

        y_edge = self.add_edge(weight.id, matmul_v2.id, **{"input_name": "Y"})

        # define q, k, v matmul_v2 output
        output = self.add_node(5, **{"type": "var"})

        # define output edge
        out_edge = self.add_edge(
            matmul_v2.id, output.id, **{"output_name": "Out"}
        )

        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {
                3: [1, -1],
            },
            "mp_dp": {
                3: [0, -1],
            },
            "mp": {3: [0, -1]},
            "dp": {
                3: [-1, -1],
            },
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 1


@register_pattern
class SelfAttentionPattern(BasePattern):
    name = "self_attention"

    def __init__(self):
        super().__init__()

    def build(self):
        query = self.add_node(0, **{"type": "var"})

        # define q, k, v weight
        q_weight = self.add_node(1, **{"dim": 2, "type": "param"})
        k_weight = self.add_node(2, **{"dim": 2, "type": "param"})
        v_weight = self.add_node(3, **{"dim": 2, "type": "param"})
        # define q, k, v matmul_v2
        q_matmul_v2 = self.add_node(4, **{"type": "matmul_v2"})
        k_matmul_v2 = self.add_node(5, **{"type": "matmul_v2"})
        v_matmul_v2 = self.add_node(6, **{"type": "matmul_v2"})
        # define input edge
        q_x_edge = self.add_edge(
            query.id, q_matmul_v2.id, **{"input_name": "X"}
        )
        k_x_edge = self.add_edge(
            query.id, k_matmul_v2.id, **{"input_name": "X"}
        )
        v_x_edge = self.add_edge(
            query.id, v_matmul_v2.id, **{"input_name": "X"}
        )
        q_y_edge = self.add_edge(
            q_weight.id, q_matmul_v2.id, **{"input_name": "Y"}
        )
        k_y_edge = self.add_edge(
            k_weight.id, k_matmul_v2.id, **{"input_name": "Y"}
        )
        v_y_edge = self.add_edge(
            v_weight.id, v_matmul_v2.id, **{"input_name": "Y"}
        )
        # define q, k, v matmul_v2 output
        q = self.add_node(7, **{"type": "var"})
        k = self.add_node(8, **{"type": "var"})
        v = self.add_node(9, **{"type": "var"})
        # define output edge
        q_out_edge = self.add_edge(
            q_matmul_v2.id, q.id, **{"output_name": "Out"}
        )
        k_out_edge = self.add_edge(
            k_matmul_v2.id, k.id, **{"output_name": "Out"}
        )
        v_out_edge = self.add_edge(
            v_matmul_v2.id, v.id, **{"output_name": "Out"}
        )

        # define add weight
        q_bias = self.add_node(10, **{"dim": 1, "type": "param"})
        k_bias = self.add_node(11, **{"dim": 1, "type": "param"})
        v_bias = self.add_node(12, **{"dim": 1, "type": "param"})
        # define add
        q_add = self.add_node(13, **{"type": "elementwise_add"})
        k_add = self.add_node(14, **{"type": "elementwise_add"})
        v_add = self.add_node(15, **{"type": "elementwise_add"})
        # define add input edge
        q_x_edge = self.add_edge(q.id, q_add.id, **{"input_name": "X"})
        k_x_edge = self.add_edge(k.id, k_add.id, **{"input_name": "X"})
        v_x_edge = self.add_edge(v.id, v_add.id, **{"input_name": "X"})
        q_y_edge = self.add_edge(q_bias.id, q_add.id, **{"input_name": "Y"})
        k_y_edge = self.add_edge(k_bias.id, k_add.id, **{"input_name": "Y"})
        v_y_edge = self.add_edge(v_bias.id, v_add.id, **{"input_name": "Y"})
        # define add output
        q = self.add_node(16, **{"type": "var"})
        k = self.add_node(17, **{"type": "var"})
        v = self.add_node(18, **{"type": "var"})
        # define add output egde
        q_out_edge = self.add_edge(q_add.id, q.id, **{"output_name": "Out"})
        k_out_edge = self.add_edge(k_add.id, k.id, **{"output_name": "Out"})
        v_out_edge = self.add_edge(v_add.id, v.id, **{"output_name": "Out"})

        # define reshape
        q_reshape = self.add_node(19, **{"type": "reshape2"})
        k_reshape = self.add_node(20, **{"type": "reshape2"})
        v_reshape = self.add_node(21, **{"type": "reshape2"})
        # define reshape input egde
        q_x_edge = self.add_edge(q.id, q_reshape.id, **{"input_name": "X"})
        k_x_edge = self.add_edge(k.id, k_reshape.id, **{"input_name": "X"})
        v_x_edge = self.add_edge(v.id, v_reshape.id, **{"input_name": "X"})
        # define reshape out
        q = self.add_node(22, **{"type": "var"})
        k = self.add_node(23, **{"type": "var"})
        v = self.add_node(24, **{"type": "var"})
        # define reshape output edge
        q_out_edge = self.add_edge(q_reshape.id, q.id, **{"output_name": "Out"})
        k_out_edge = self.add_edge(k_reshape.id, k.id, **{"output_name": "Out"})
        v_out_edge = self.add_edge(v_reshape.id, v.id, **{"output_name": "Out"})

        # define transpose
        q_transpose = self.add_node(25, **{"type": "transpose2"})
        k_transpose = self.add_node(26, **{"type": "transpose2"})
        v_transpose = self.add_node(27, **{"type": "transpose2"})
        # define transpose input edge
        q_x_edge = self.add_edge(q.id, q_transpose.id, **{"input_name": "X"})
        k_x_edge = self.add_edge(k.id, k_transpose.id, **{"input_name": "X"})
        v_x_edge = self.add_edge(v.id, v_transpose.id, **{"input_name": "X"})
        # define transpose output
        q = self.add_node(28, **{"type": "var"})
        k = self.add_node(29, **{"type": "var"})
        v = self.add_node(30, **{"type": "var"})
        # define transpose output edege
        q_out_edge = self.add_edge(
            q_transpose.id, q.id, **{"output_name": "Out"}
        )
        k_out_edge = self.add_edge(
            k_transpose.id, k.id, **{"output_name": "Out"}
        )
        v_out_edge = self.add_edge(
            v_transpose.id, v.id, **{"output_name": "Out"}
        )

        # define matmul
        matmul = self.add_node(31, **{"type": "matmul_v2"})
        # define matmul input edge
        x_edge = self.add_edge(q.id, matmul.id, **{"input_name": "X"})
        y_edge = self.add_edge(k.id, matmul.id, **{"input_name": "Y"})
        # define matmul output
        out = self.add_node(32, **{"type": "var"})
        # define matmul output edge
        out_edge = self.add_edge(matmul.id, out.id, **{"output_name": "Out"})

        # define add y
        attention_mask = self.add_node(33, **{"type": "data"})
        # define add
        add = self.add_node(34, **{"type": "elementwise_add"})
        # define add input edge
        x_edge = self.add_edge(out.id, add.id, **{"input_name": "X"})
        y_edge = self.add_edge(attention_mask.id, add.id, **{"input_name": "Y"})
        # define add output
        out = self.add_node(35, **{"type": "var"})
        # define add output egde
        out_edge = self.add_edge(add.id, out.id, **{"output_name": "Out"})

        # define softmax
        softmax = self.add_node(36, **{"type": "softmax"})
        # define input edge
        input_edge = self.add_edge(out.id, softmax.id, **{"input_name": "X"})
        # define softmax output
        out = self.add_node(37, **{"type": "var"})
        # define softmax output edge
        output_edge = self.add_edge(
            softmax.id, out.id, **{"output_name": "Out"}
        )

        # define matmul_v2
        matmul_v2 = self.add_node(38, **{"type": "matmul_v2"})
        # define input edge
        x_edge = self.add_edge(out.id, matmul_v2.id, **{"input_name": "X"})
        y_edge = self.add_edge(v.id, matmul_v2.id, **{"input_name": "Y"})
        # define output
        out = self.add_node(39, **{"type": "var"})
        # define output edge
        out_edge = self.add_edge(matmul_v2.id, out.id, **{"output_name": "Out"})

        # define transpose
        transpose = self.add_node(40, **{"type": "transpose2"})
        # define transpose input edge
        x_edge = self.add_edge(out.id, transpose.id, **{"input_name": "X"})
        # define transpose output
        out = self.add_node(41, **{"type": "var"})
        # define transpose output edege
        out_edge = self.add_edge(transpose.id, out.id, **{"output_name": "Out"})

        # define reshape
        reshape = self.add_node(42, **{"type": "reshape2"})
        # define reshape input egde
        x_edge = self.add_edge(out.id, reshape.id, **{"input_name": "X"})
        # define reshape out
        out = self.add_node(43, **{"type": "var"})
        # define reshape output edge
        out_edge = self.add_edge(reshape.id, out.id, **{"output_name": "Out"})

        # define matmul_v2 weight
        y = self.add_node(44, **{"type": "param", "dim": 2})
        # define matmul_v2
        matmul_v2 = self.add_node(45, **{"type": "matmul_v2"})
        # define input edge
        x_edge = self.add_edge(out.id, matmul_v2.id, **{"input_name": "X"})
        y_edge = self.add_edge(y.id, matmul_v2.id, **{"input_name": "Y"})
        # define output
        out = self.add_node(46, **{"type": "var"})
        # define output edge
        out_edge = self.add_edge(matmul_v2.id, out.id, **{"output_name": "Out"})

        # define add weight
        bias = self.add_node(47, **{"dim": 1, "type": "param"})
        # define add
        add = self.add_node(48, **{"type": "elementwise_add"})
        # define add input edge
        x_edge = self.add_edge(out.id, add.id, **{"input_name": "X"})
        y_edge = self.add_edge(bias.id, add.id, **{"input_name": "Y"})
        # define add output
        out = self.add_node(49, **{"type": "var"})
        # define add output egde
        out_edge = self.add_edge(add.id, out.id, **{"output_name": "Out"})

        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {
                0: [0, -1, -1],
                1: [-1, 1],
                2: [-1, 1],
                3: [-1, 1],
                44: [1, -1],
            },
            "mp_dp": {
                0: [1, -1, -1],
                1: [-1, 0],
                2: [-1, 0],
                3: [-1, 0],
                44: [0, -1],
            },
            "mp": {1: [-1, 0], 2: [-1, 0], 3: [-1, 0], 44: [0, -1]},
            "dp": {
                0: [0, -1, -1],
                1: [-1, -1],
                2: [-1, -1],
                3: [-1, -1],
                44: [-1, -1],
            },
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 4


@register_pattern
class FFNPattrern(BasePattern):
    name = "ffn"

    def __init__(self):
        super().__init__()

    def build(self):
        x = self.add_node(0, **{"type": "var"})

        w1_weight = self.add_node(1, **{"dim": 2, "type": "param"})
        w1_matmul = self.add_node(2, **{"type": "matmul_v2"})

        w1_x = self.add_edge(0, 2, **{"input_name": "X"})
        w1_y = self.add_edge(1, 2, **{"input_name": "Y"})

        out1 = self.add_node(3, **{"type": "var"})
        w1_out = self.add_edge(2, 3, **{"output_name": "Out"})

        w1_b = self.add_node(4, **{"dim": 1, "type": "param"})
        add1 = self.add_node(5, **{"type": "elementwise_add"})

        add1_x = self.add_edge(3, 5, **{"input_name": "X"})
        add1_y = self.add_edge(4, 5, **{"input_name": "Y"})

        out2 = self.add_node(6, **{"type": "var"})
        add1_out = self.add_edge(5, 6, **{"output_name": "Out"})

        gelu = self.add_node(7, **{"type": "gelu"})

        gelu_x = self.add_edge(6, 7, **{"input_name": "X"})
        out3 = self.add_node(8, **{"type": "var"})
        gelu_out = self.add_edge(7, 8, **{"output_name": "Out"})

        w2_weight = self.add_node(9, **{"dim": 2, "type": "param"})
        w2_matmul = self.add_node(10, **{"type": "matmul_v2"})

        w1_x = self.add_edge(8, 10, **{"input_name": "X"})
        w1_y = self.add_edge(9, 10, **{"input_name": "Y"})

        out4 = self.add_node(11, **{"type": "var"})
        w2_out = self.add_edge(10, 11, **{"output_name": "Out"})

        w2_b = self.add_node(12, **{"dim": 1, "type": "param"})
        add2 = self.add_node(13, **{"type": "elementwise_add"})

        add2_x = self.add_edge(11, 13, **{"input_name": "X"})
        add2_y = self.add_edge(12, 13, **{"input_name": "Y"})

        out5 = self.add_node(14, **{"type": "var"})
        add2_out = self.add_edge(13, 14, **{"output_name": "Out"})

        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {0: [0, -1, -1], 1: [-1, 1], 9: [1, -1]},
            "mp_dp": {0: [1, -1, -1], 1: [-1, 0], 9: [0, -1]},
            "mp": {1: [-1, 0], 9: [0, -1]},
            "dp": {0: [0, -1, -1], 1: [-1, -1], 9: [-1, -1]},
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 2


@register_pattern
class SharedWordEmbeddingPattern(BasePattern):
    name = "shared_word_embedding"

    def __init__(self):
        super().__init__()

    def build(self):
        # define embedding input
        tokens = self.add_node(0, **{"type": "data"})
        word_embeddings = self.add_node(1, **{"dim": 2, "type": "param"})
        # define embedding
        embedding = self.add_node(2, **{"type": "lookup_table_v2"})
        # define embedding input edge
        ids = self.add_edge(0, 2, **{"input_name": "Ids"})
        w = self.add_edge(1, 2, **{"input_name": "W"})
        # define embedding output
        out = self.add_node(3, **{"type": "var"})
        # define embedding output edge
        out_edge = self.add_edge(2, 3, **{"output_name": "Out"})

        # define matmul_v2 input
        x = self.add_node(4, **{"type": "var"})
        # define matmul_v2
        matmul = self.add_node(5, **{"type": "matmul_v2"})
        # define matmul_v2 input edge
        x_edge = self.add_edge(4, 5, **{"input_name": "X"})
        y_edge = self.add_edge(1, 5, **{"input_name": "Y"})
        # define matmul_v2 output
        out = self.add_node(6, **{"type": "var"})
        # define matmul_v2 output edge
        out_edge = self.add_edge(5, 6, **{"output_name": "Out"})

        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {0: [0, -1], 1: [1, -1], 4: [0, -1, -1]},
            "mp_dp": {0: [1, -1], 1: [0, -1], 4: [1, -1, -1]},
            "mp": {0: [-1, -1], 1: [0, -1], 4: [-1, -1, -1]},
            "dp": {0: [0, -1], 1: [-1, -1], 4: [0, -1, -1]},
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 3


@register_pattern
class PositionEmbeddingPattern(BasePattern):
    name = "position_embedding"

    def __init__(self):
        super().__init__()

    def build(self):
        # define embedding input
        tokens = self.add_node(0, **{"type": "data"})
        word_embeddings = self.add_node(1, **{"dim": 2, "type": "param"})
        # define embedding
        embedding = self.add_node(2, **{"type": "lookup_table_v2"})
        # define embedding input edge
        ids = self.add_edge(0, 2, **{"input_name": "Ids"})
        w = self.add_edge(1, 2, **{"input_name": "W"})
        # define embedding output
        out = self.add_node(3, **{"type": "var"})
        # define embedding output edge
        out_edge = self.add_edge(2, 3, **{"output_name": "Out"})

        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {0: [0, -1], 1: [-1, -1], 3: [-1, -1, -1]},
            "mp_dp": {0: [1, -1], 1: [-1, -1], 3: [1, -1, -1]},
            "mp": {0: [-1, -1], 1: [-1, -1], 3: [-1, -1, -1]},
            "dp": {0: [0, -1], 1: [-1, -1], 3: [0, -1, -1]},
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 1


@register_pattern
class UnsqueezeDataPattern(BasePattern):
    name = "unsqueeze_data"

    def __init__(self):
        super().__init__()

    def build(self):
        # define unsequeeze input
        tokens = self.add_node(0, **{"type": "data"})
        # define unsequeeze
        unsqueeze = self.add_node(1, **{"type": "unsqueeze2"})
        # define unsequeeze input edge
        x_edge = self.add_edge(0, 1, **{"input_name": "X"})
        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {0: [0, -1]},
            "mp_dp": {0: [1, -1]},
            "mp": {0: [-1, -1]},
            "dp": {0: [0, -1]},
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 1


@register_pattern
class ReshapeDataPattern(BasePattern):
    name = "reshape_data"

    def __init__(self):
        super().__init__()

    def build(self):
        # define unsequeeze input
        data = self.add_node(0, **{"type": "data"})
        # define unsequeeze
        reshape = self.add_node(1, **{"type": "reshape2"})
        # define unsequeeze input edge
        x_edge = self.add_edge(0, 1, **{"input_name": "X"})
        # pattern: pure mp or hybrid dp+mp
        shard_spec = {
            "dp_mp": {0: [0, -1]},
            "mp_dp": {0: [1, -1]},
            "mp": {0: [-1, -1]},
            "dp": {0: [0, -1]},
        }
        self.attrs["shard_spec"] = shard_spec
        self.attrs["weights"] = 1


class GraphUtil:
    @staticmethod
    def convert_to_graph(block):
        """Convert ops to graph."""
        graph = Graph()
        graph.attrs["var_to_id"] = {}  # {var_name: node_id}
        graph.attrs["id_to_var_desc_id"] = {}  # {node_id: var_desc_id}
        graph.attrs["id_to_var_name"] = {}
        graph.attrs["op_to_id"] = {}  # {op_id: node_id}
        graph.attrs["id_to_op"] = {}  # {node_id: op}

        ops = block.ops
        node_id = -1
        for op in ops:
            attrs = op.all_attrs()
            attrs["type"] = op.type
            node_id += 1

            # create op node
            op_node = graph.add_node(node_id, **attrs)
            graph.attrs["op_to_id"][op.desc.id()] = op_node.id
            graph.attrs["id_to_op"][op_node.id] = op
            graph._attr_to_nodes[op_node.id] = {}
            for input_name in op.input_names:
                graph._attr_to_nodes[op_node.id][input_name] = []
                for var_name in op.input(input_name):
                    if var_name not in graph.attrs["var_to_id"]:
                        # create var node
                        node_id += 1
                        var_node = graph.add_node(node_id)
                        var = block._var_recursive(var_name)
                        if var.is_parameter:
                            var_node.attrs["type"] = "param"
                            var_node.attrs["dim"] = len(var.shape)
                        elif var.is_data:
                            var_node.attrs["type"] = "data"
                            var_node.attrs["dim"] = len(var.shape)
                        else:
                            var_node.attrs["type"] = "var"
                        graph.attrs["var_to_id"][var_name] = var_node.id
                        graph.attrs["id_to_var_desc_id"][
                            var_node.id
                        ] = var.desc.original_id()
                        graph.attrs["id_to_var_name"][var_node.id] = var_name
                    else:
                        var_node_id = graph.attrs["var_to_id"][var_name]
                        var_node = graph._nodes[var_node_id]

                    # create edge that input -> op
                    input_edge = graph.add_edge(var_node.id, op_node.id)
                    input_edge.attrs["input_name"] = input_name
                    graph._attr_to_nodes[op_node.id][input_name].append(
                        var_node
                    )

                for output_name in op.output_names:
                    graph._attr_to_nodes[op_node.id][output_name] = []
                    for var_name in op.output(output_name):
                        if var_name not in graph.attrs["var_to_id"]:
                            # create var node
                            node_id += 1
                            var_node = graph.add_node(node_id)
                            var = block._var_recursive(var_name)
                            if var.is_parameter:
                                var_node.attrs["type"] = "param"
                            else:
                                var_node.attrs["type"] = "var"
                            graph.attrs["var_to_id"][var_name] = var_node.id
                            graph.attrs["id_to_var_desc_id"][
                                var_node.id
                            ] = var.desc.original_id()
                            graph.attrs["id_to_var_name"][
                                var_node.id
                            ] = var_name
                        else:
                            var_node_id = graph.attrs["var_to_id"][var_name]
                            var_node = graph._nodes[var_node_id]

                        # create edge that op -> output
                        output_edge = graph.add_edge(op_node.id, var_node.id)
                        output_edge.attrs["output_name"] = output_name

                        graph._attr_to_nodes[op_node.id][output_name].append(
                            var_node
                        )

        return graph

    @staticmethod
    def match_pattern(pattern, graph):
        def _is_op_node(node):
            """Judge whether node is op node"""
            if node.attrs["type"] not in ["var", "param", "data"]:
                return True

            return False

        def _compare_op_node(src, tgt):
            """Compare whether two op nodes are equal"""
            if src.attrs["type"] != tgt.attrs["type"]:
                return False

            return True

        def _compare_var_node(src, tgt):
            """Compare whether two var nodes are equal"""
            for key in src.attrs:
                if key not in tgt.attrs:
                    return False
                if src.attrs[key] != tgt.attrs[key]:
                    return False

            return True

        def _match_core(src_node, tgt_node):
            nonlocal not_matched
            # do not support one input name or output name corresponding to multiple vars
            if not_matched:
                return

            if _is_op_node(src_node):
                # compare op node whether equal
                if not _compare_op_node(src_node, tgt_node):
                    not_matched = True
                    return

                result[src_node.id] = tgt_node.id

                # input var nodes
                src_input_nodes = src_reverse_adjs[src_node.id]
                for node in src_input_nodes:
                    # has visited
                    if node.id in result:
                        continue
                    edge = src_edges[node.id][src_node.id]
                    input_name = edge.attrs["input_name"]

                    # NOTE: do not support one input name or output name corresponding to multiple vars
                    compare_nodes = tgt_attr_to_nodes[tgt_node.id].get(
                        input_name, None
                    )
                    if not compare_nodes:
                        not_matched = True
                        return
                    _match_core(node, compare_nodes[0])

                # output var nodes
                src_output_node_ids = src_edges[src_node.id].keys()
                for node_id in src_output_node_ids:
                    # has visited
                    if node_id in result:
                        continue
                    node = src_nodes[node_id]
                    edge = src_edges[src_node.id][node_id]
                    output_name = edge.attrs["output_name"]

                    # NOTE: do not support one input name or output name corresponding to multiple vars
                    compare_nodes = tgt_attr_to_nodes[tgt_node.id].get(
                        output_name, None
                    )
                    if not compare_nodes:
                        not_matched = True
                        return
                    _match_core(node, compare_nodes[0])

            else:
                # compare var nodes whether equal
                if not _compare_var_node(src_node, tgt_node):
                    not_matched = True
                    return

                result[src_node.id] = tgt_node.id

                # as input for op node
                src_as_input_node_ids = src_edges[src_node.id].keys()
                for node_id in src_as_input_node_ids:
                    if node_id in result:
                        continue

                    src_edge = src_edges[src_node.id][node_id]
                    input_name = src_edge.attrs["input_name"]
                    compare_node_ids = tgt_edges[tgt_node.id].keys()

                    compare_node = None
                    for compare_node_id in compare_node_ids:
                        edge = tgt_edges[tgt_node.id][compare_node_id]
                        if (
                            edge.attrs["input_name"] == input_name
                            and compare_node_id not in result.values()
                        ):
                            compare_node = tgt_nodes[compare_node_id]
                            break

                    if not compare_node:
                        not_matched = True
                        return
                    _match_core(src_nodes[node_id], compare_node)

                # as output for op node
                src_as_output_nodes = src_reverse_adjs[src_node.id]
                for node in src_as_output_nodes:
                    if node.id in result:
                        continue

                    src_edge = src_edges[node.id][src_node.id]
                    output_name = src_edge.attrs["output_name"]

                    compare_nodes = tgt_reverse_adjs[tgt_node.id]

                    compare_node = None
                    for item in compare_nodes:
                        node_id = item.id
                        edge = tgt_edges[node_id][tgt_node.id]
                        if edge.attrs["output_name"] == output_name:
                            compare_node = tgt_nodes[node_id]
                            break
                    if not compare_node:
                        not_matched = True
                        return
                    _match_core(src_nodes[node.id], compare_node)

        results = []
        matched_ids = set()
        matched_op_node_ids = set()
        result = {}
        src_nodes = pattern.nodes
        src_edges = pattern._adjs
        src_reverse_adjs = pattern._reverse_adjs

        tgt_nodes = graph.nodes
        tgt_edges = graph._adjs
        tgt_reverse_adjs = graph._reverse_adjs
        tgt_attr_to_nodes = graph._attr_to_nodes

        # starts with a op node
        src_start_node = None
        for node_id in src_nodes:
            node = src_nodes[node_id]
            if node.attrs["type"] not in ["var", "param", "data"]:
                src_start_node = node
                break
        assert src_start_node is not None

        for node_id in tgt_nodes:
            node = tgt_nodes[node_id]
            if node.attrs["type"] == src_start_node.attrs["type"]:
                not_matched = False
                _match_core(src_start_node, node)
                if not not_matched:
                    need_to_append = True
                    for value in result.values():
                        if value in matched_op_node_ids:
                            result = {}
                            need_to_append = False
                            break
                    if need_to_append:
                        results.append(result)
                        for value in result.values():
                            matched_ids.add(value)
                            if value in graph.attrs["id_to_op"].keys():
                                matched_op_node_ids.add(value)
                        result = {}
                else:
                    not_matched = False
                    result = {}
        return results, matched_ids

    @staticmethod
    def match_all_patterns(graph):
        matched_results = (
            {}
        )  # {"pattern_name": [{0: graph_node}, {0: graph_node}]}
        matched_ids = set()
        for pattern in _PATTERNS:
            results, matched = GraphUtil.match_pattern(pattern, graph)
            print("pattern.name: ", pattern.name, results, matched, matched_ids)
            for result in results:
                has_matched = False
                for id in result:
                    if result[id] in matched_ids:
                        print("has_matched: ", result[id])
                        has_matched = True
                        break
                if not has_matched:
                    for item in result:
                        matched_ids.add(result[id])
                    if pattern.name not in matched_results:
                        matched_results[pattern.name] = []
                    matched_results[pattern.name].append(result)

            # for id in matched:
            #     if id in matched_ids:
            #         has_matched = True
            #         break
            # if not has_matched:
            #     for item in matched:
            #         matched_ids.add(item)
            #     matched_results[pattern.name] = results

        return matched_results


class OperatorClusteringUtil:
    common_starts = ["layer_norm", "matmul_v2", "matmul"]

    @staticmethod
    def get_ranks(seq):
        """Get rank array of the given seq by doubled algorithm."""
        ordered_seq = sorted(list(set(seq)))
        item_to_rank = {item: idx for idx, item in enumerate(ordered_seq)}
        inter_ranks = [item_to_rank[item] for item in seq]

        length = len(inter_ranks)
        power = 0
        interval = 2**power
        while interval < length:
            for idx, item in enumerate(inter_ranks):
                if idx + interval >= length:
                    inter_ranks[idx] = [item, -1]
                else:
                    inter_ranks[idx] = [item, inter_ranks[idx + interval]]

            tmp = []
            for item in inter_ranks:
                if item not in tmp:
                    tmp.append(item)
            tmp.sort(key=lambda x: (x[0], x[1]))
            item_to_rank = {}
            for idx, val in enumerate(tmp):
                key = ",".join(str(item) for item in val)
                item_to_rank[key] = idx

            inter_ranks = [
                item_to_rank[",".join(str(val) for val in item)]
                for item in inter_ranks
            ]
            power += 1
            interval = 2**power

        return inter_ranks

    @staticmethod
    def get_suffixes(ranks):
        """Get suffix array by the given rank array."""
        suffixes = [0 for idx in range(len(ranks))]
        for idx, item in enumerate(ranks):
            suffixes[item] = idx
        return suffixes

    @staticmethod
    def get_heights(suffixes, seq):
        """Get height array by the suffix array and seq"""
        heights = [-1 for i in range(len(suffixes))]
        for i in range(1, len(seq)):
            x = seq[suffixes[i - 1] :]
            y = seq[suffixes[i] :]
            max_len = len(x) if len(x) > len(y) else len(y)
            same_count = 0
            for j in range(max_len):
                if j >= len(x) or j >= len(y):
                    break
                else:
                    if x[j] == y[j]:
                        same_count += 1
                    else:
                        break
            heights[i] = same_count

        return heights

    @staticmethod
    def get_longest_repeated_sub_seq(suffixes, heights, seq):
        """Get longest repeated sub sequence by suffix array algorithm."""
        length = len(seq)
        if length <= 1:
            return None
        k = length // 2
        height_groups = []
        longest_sub_seq = None
        longest_sub_seqs = []

        while k >= 2:
            height_group = []
            for i in range(1, len(heights)):
                if heights[i] >= k:
                    if i == 1:
                        height_group.append(0)
                    height_group.append(i)
                else:
                    if i == 1:
                        height_groups.append([0])
                        height_group = [i]
                    else:
                        height_groups.append(height_group)
                        height_group = [i]

            if height_group:
                height_groups.append(height_group)

            for height_group in height_groups:
                suffix_group = []
                index_group = []
                for idx in height_group:
                    suffix_group.append(idx)
                    index_group.append(suffixes[idx])

                max_index = max(index_group)
                min_index = min(index_group)
                if max_index - min_index >= k:
                    longest_sub_seq = seq[min_index : min_index + k]
                    if (
                        longest_sub_seq[0]
                        in OperatorClusteringUtil.common_starts
                    ):
                        return longest_sub_seq
            if longest_sub_seq is not None:
                return longest_sub_seq

            k -= 1
            height_groups = []

        return longest_sub_seq

    @staticmethod
    def get_decomposed_sub_seq(seq):
        """Get decomposed sub seq s by seq S such as s * R = S."""
        if not seq:
            return seq

        decomposed_sub_seq = seq
        seq_len = len(seq)
        if seq_len == 1:
            return decomposed_sub_seq
        else:
            for interval in range(2, seq_len + 1):
                if seq_len % interval == 0:
                    repeated_times = seq_len // interval
                    decomposed_sub_seq = seq[0:interval]
                    decomposed = True
                    for j in range(1, repeated_times + 1):
                        sub_seq = seq[interval * (j - 1) : interval * j]
                        if sub_seq != decomposed_sub_seq:
                            decomposed = False
                            break
                    if decomposed:
                        return decomposed_sub_seq

        return decomposed_sub_seq

    @staticmethod
    def replace_by_decomposed_seq(sub_seq, seq):
        """Replace seq by sub seq."""
        if not sub_seq:
            return seq

        result = []
        sub_seq_len = len(sub_seq)
        i = 0
        while i < len(seq):
            if seq[i : i + sub_seq_len] == sub_seq:
                result.append(seq[i : i + sub_seq_len])
                i += sub_seq_len
            else:
                result.append(seq[i])
                i += 1

        return result

    @staticmethod
    def stop_replace(seq):
        for item in seq:
            if not isinstance(item, list):
                return False
        return True


class ClusterPartitionUtil:
    @staticmethod
    def factorization(num):
        factors = []
        for i in range(1, int(math.floor(math.sqrt(num))) + 1):
            if num % i == 0:
                factors.append([i, int(num / i)])
        return factors

    @staticmethod
    def complete_meshes(partitions: list, num: int):
        if len(partitions) == 1:
            partitions = ClusterPartitionUtil.factorization(num - 1)
            partitions.append([1])
        return partitions

    @staticmethod
    def partition_cluster(
        n: int,
        m: int,
        filter=[
            complete_meshes.__func__,
        ],
    ) -> list:
        """
        Partiton cluster into possible device meshes.
        Args:
            n (int): The number of nodes.
            m (int): The number of single devices on each node.
            filter (list): Functions for filtering useful meshes
        Returns:
            device_meshed (list) : The possible device meshes.
        """
        partition_result = ClusterPartitionUtil.factorization(n)
        for func in filter:
            partition_result = func(partition_result, n)
        device_meshes = []
        if n == 1:
            partition_result = ClusterPartitionUtil.factorization(m)
            for partition in partition_result:
                device_mesh = []
                for i in range(partition[0]):
                    device_mesh.append([1, partition[1]])
                device_meshes.append(device_mesh)
        else:
            incerement = 1 if partition_result[-1] == [1] else 0
            for partition in partition_result:
                if len(partition) < 2:
                    continue
                device_mesh = []
                for i in range(partition[0]):
                    device_mesh.append([partition[1], m])
                device_mesh[-1][0] += incerement
                device_meshes.append(device_mesh)

        return device_meshes


def convert_to_process_meshes(device_mesh: list) -> list:
    """
    Transfer device_meshes into possible process meshes.
    Args:
        device meshes (list): [n,m], one device mesh.
    Returns:
        process_meshes (list): Possible process_meshes
    """
    print("convert_to_process_meshes device_mesh: ", device_mesh)
    n, m = device_mesh[0], device_mesh[1]
    factors = (
        ClusterPartitionUtil.factorization(m)
        if n == 1
        else ClusterPartitionUtil.factorization(n)
    )
    process_meshes = []
    if n == 1:
        for factor in factors:
            if factor[0] == 1:
                process_meshes.append([factor[1]])
                continue
            process_meshes.append(factor)
    else:
        for factor in factors:
            mul1, mul2 = factor[0], factor[1]
            if mul1 == 1:
                process_meshes.append([m * mul2])
            elif mul1 != mul2:
                process_meshes.append([int(n / mul2), m * mul2])
            process_meshes.append([int(n / mul1), m * mul1])
    return process_meshes


class RuleBasedTuner:
    def __init__(self, dist_context, mode="train", level="o1"):
        self._dist_context = dist_context
        self._cluster = self.dist_context.cluster
        self._mode = mode
        assert level in ["o1", "o2"]
        self._level = level

        # forward sub program
        self.fwd_sub_programs = OrderedDict()

        # dist_context of sub program
        self.sub_programs_dist_context = OrderedDict()

        # graph of forward sub program
        self.fwd_sub_program_graphs = OrderedDict()

        # full main program
        self.full_main_program = None

        # full startup program
        self.full_startup_program = None

        # full main program dist context
        self.full_main_program_dist_context = None

        # tensor dist attribute from pattern setting
        self.tensor_dist_attrs = {}

        # single sub program cost
        self.sub_program_cost = (
            {}
        )  # {idx: {process_mesh: {parallelism: {"time: ", "memory": }}}}}

        # the cost of some layers with parallelism at process_mesh
        self.stage_cost = (
            {}
        )  # {start: {end: {process_mesh: {idx: parallelism}}}

        # op original id to op mapping
        self.op_original_id_to_op = {}

        # op original id to op idx in program
        self.op_original_id_to_idx = {}

        # op original id to grad op original id mapping
        self.op_original_id_to_grad_op_original_id = {}

        # all process meshes that the cluster can express
        self.process_meshes = []

        # all device meshes that the cluster can be partitioned
        self.device_meshes_list = []

        self.stage_best_cost_of_dm = {}
        self.stage_best_cost_of_pm = {}

        self.layers = None

    @property
    def dist_context(self):
        return self._dist_context

    @property
    def cluster(self):
        return self._cluster

    @property
    def mode(self):
        return self._mode

    @property
    def level(self):
        return self._level

    @property
    def fast(self):
        return self._fast

    def gen_full_program(self):
        self.full_main_program = self.dist_context.serial_main_program.clone()
        self.full_startup_program = (
            self.dist_context.serial_startup_program.clone()
        )
        loss = self.full_main_program.global_block().vars[
            self.dist_context.serial_loss.name
        ]
        serial_optimizer = self._dist_context.serial_optimizer
        optimizer = copy.deepcopy(serial_optimizer)
        self.full_main_program_dist_context = DistributedContext(
            serial_main_prog=self.full_main_program,
            serial_startup_prog=self.full_startup_program,
            serial_loss=loss,
        )
        # if in train mode, generate backward and update program.
        with program_guard(self.full_main_program, self.full_startup_program):
            params_grads = append_backward(
                loss,
                distop_context=self.full_main_program_dist_context.dist_op_context,
            )

        # optimizer = copy.deepcopy(serial_optimizer)
        # self._dist_context._serial_optimizer = optimizer
        with program_guard(self.full_main_program, self.full_startup_program):
            with unique_name.guard("opt_"):
                optimizer_ops = optimizer.apply_gradients(params_grads)

        # op original id to grad op id
        for idx, op in enumerate(self.full_main_program.global_block().ops):
            self.op_original_id_to_op[op.desc.original_id()] = op
            self.op_original_id_to_idx[op.desc.original_id()] = idx

        grad_op_id_to_op_id = (
            self.full_main_program_dist_context.dist_op_context.grad_op_id_to_op_id
        )

        for grad_op_original_id in grad_op_id_to_op_id:
            op_id = grad_op_id_to_op_id[grad_op_original_id]
            self.op_original_id_to_grad_op_original_id[
                op_id
            ] = grad_op_original_id

    def cluster_operators(self):
        """
        Group operators to layers.

        Args:
            ops (list): A operator list.

        Returns:
            List: The list contains the list of operators which belong to the same layer.
        """
        ops = self._dist_context._serial_main_program.global_block().ops
        seq = [op.type for op in ops]

        while not OperatorClusteringUtil.stop_replace(seq):
            to_replace_seq = []
            to_replace_idxes = []
            has_append = False
            for idx, item in enumerate(seq):
                if not isinstance(item, list):
                    has_append = True
                    to_replace_seq.append(item)
                    to_replace_idxes.append(idx)
                elif isinstance(seq, list) and not has_append:
                    continue
                elif isinstance(seq, list) and has_append:
                    break

            ranks = OperatorClusteringUtil.get_ranks(to_replace_seq)
            suffixes = OperatorClusteringUtil.get_suffixes(ranks)
            heights = OperatorClusteringUtil.get_heights(
                suffixes, to_replace_seq
            )
            longest_sub_seq = (
                OperatorClusteringUtil.get_longest_repeated_sub_seq(
                    suffixes, heights, to_replace_seq
                )
            )
            has_merged = False
            if longest_sub_seq is None:
                for i in range(to_replace_idxes[-1] + 1, len(seq)):
                    if isinstance(seq[i], list):
                        seq[i] = to_replace_seq + seq[i]
                        has_merged = True
                        break
                if not has_merged:
                    for i in range(to_replace_idxes[0] - 1, -1, -1):
                        if isinstance(seq[i], list):
                            seq[i].extend(to_replace_seq)
                            has_merged = True
                            break
                if not has_merged:
                    seq = [to_replace_seq]
                    break

            decomposed_sub_seq = OperatorClusteringUtil.get_decomposed_sub_seq(
                longest_sub_seq
            )
            to_replace_seq = OperatorClusteringUtil.replace_by_decomposed_seq(
                decomposed_sub_seq, to_replace_seq
            )
            result = seq[: to_replace_idxes[0]]
            if not has_merged:
                result.extend(to_replace_seq)
            result.extend(seq[to_replace_idxes[-1] + 1 :])
            seq = result

        layers = []
        idx = 0
        for groups in seq:
            layer = []
            for op in groups:
                layer.append(ops[idx])
                idx += 1
            layers.append(layer)

        return layers

    def match_program(self, program):
        graph = GraphUtil.convert_to_graph(program.global_block())
        results = GraphUtil.match_all_patterns(graph)
        if results:
            for pattern_name in results.keys():
                print("pattern name: ", pattern_name)
                pattern = _PATTERN_MAP[pattern_name]
                for parallelism in pattern.attrs["shard_spec"].keys():
                    print("parallelism: ", parallelism)
                    shard_spec = pattern.attrs["shard_spec"][parallelism]
                    print("shard_spec: ", shard_spec, results[pattern_name])
                    for pattern_node_id in shard_spec.keys():
                        for item in results[pattern_name]:
                            var_id = item[pattern_node_id]
                            var_desc_id = graph.attrs["id_to_var_desc_id"][
                                var_id
                            ]
                            if var_desc_id not in self.tensor_dist_attrs:
                                self.tensor_dist_attrs[var_desc_id] = {}
                            self.tensor_dist_attrs[var_desc_id][
                                parallelism
                            ] = shard_spec[pattern_node_id]
                            tensor_name = graph.attrs["id_to_var_name"][var_id]
                            print(
                                "{}'s shard_spec is {}".format(
                                    tensor_name, shard_spec[pattern_node_id]
                                )
                            )

    def gen_fwd_sub_programs_by_removed(self):
        serial_main_program = self._dist_context._serial_main_program
        op_len = len(serial_main_program.global_block().ops)
        start_idx = 0
        for idx, layer in enumerate(self.layers):
            layer_len = len(layer)
            remove_op_idxs = [i for i in range(0, start_idx)] + [
                i for i in range(start_idx + layer_len, op_len)
            ]
            sub_fwd_program = self._gen_fwd_sub_program_by_removed(
                remove_op_idxs
            )
            self.fwd_sub_programs[idx] = sub_fwd_program
            start_idx += layer_len

    def _gen_fwd_sub_program_by_removed(self, remove_op_idxs):
        """Get fwd sub program by removing ops and vars which not belong to layer"""
        sub_fwd_program = self._dist_context._serial_main_program.clone()
        block = sub_fwd_program.global_block()
        for idx in remove_op_idxs[::-1]:
            block._remove_op(idx)
        remove_vars = set()
        ops = block.ops
        vars = block.vars
        need_vars = set()
        for op in ops:
            for var_name in op.input_arg_names:
                if var_name in vars:
                    need_vars.add(var_name)
            for var_name in op.output_arg_names:
                if var_name in vars:
                    need_vars.add(var_name)
        for var in vars:
            if var not in need_vars:
                remove_vars.add(var)
        for var in remove_vars:
            block._remove_var(var)

        return sub_fwd_program

    def gen_fwd_sub_programs_by_clone(self):
        for idx, layer in enumerate(self.layers):
            sub_fwd_program = self._gen_fwd_sub_program_by_clone(layer)
            self.fwd_sub_programs[idx] = sub_fwd_program

    def _gen_fwd_sub_program_by_clone(self, ops):
        program = paddle.static.Program()
        block = ops[0].block
        vars = block.vars
        target_block = program.global_block()
        with paddle.static.program_guard(program):
            has_cloned_vars = set()
            for op in ops:
                new_op_desc = target_block.desc.append_op()
                new_op_desc.copy_from(op.desc)
                for var_name in op.input_arg_names:
                    if var_name not in has_cloned_vars:
                        if vars[var_name].is_parameter:
                            src_var = vars[var_name]
                            copied_kwargs = {}
                            copied_kwargs['trainable'] = src_var.trainable
                            copied_kwargs[
                                'optimize_attr'
                            ] = src_var.optimize_attr
                            copied_kwargs['regularizer'] = src_var.regularizer
                            copied_kwargs[
                                'do_model_average'
                            ] = src_var.do_model_average
                            copied_kwargs['need_clip'] = src_var.need_clip

                            param = Parameter(
                                block=target_block,
                                type=src_var.type,
                                name=src_var.name,
                                shape=src_var.shape,
                                dtype=src_var.dtype,
                                lod_level=src_var.lod_level,
                                error_clip=src_var.error_clip,
                                stop_gradient=src_var.stop_gradient,
                                is_data=src_var.is_data,
                                belong_to_optimizer=src_var.belong_to_optimizer,
                                **copied_kwargs
                            )
                        else:
                            target_block._clone_variable(vars[var_name])
                        target_block.vars[var_name].desc.set_original_id(
                            vars[var_name].desc.original_id()
                        )
                        has_cloned_vars.add(var_name)
                for var_name in op.output_arg_names:
                    if var_name not in has_cloned_vars:
                        target_block._clone_variable(vars[var_name])
                        target_block.vars[var_name].desc.set_original_id(
                            vars[var_name].desc.original_id()
                        )
                        has_cloned_vars.add(var_name)
        target_block._sync_with_cpp()
        return program

    def complete_sub_fwd_programs_old(self, process_mesh):
        selective_parallelisms = (
            ["dp", "mp"] if len(process_mesh.shape) == 1 else ["dp_mp", "mp_dp"]
        )
        # clone for not changing program user defined
        for idx in self.fwd_sub_programs.keys():
            sub_fwd_program = self.fwd_sub_programs[idx]
            self.sub_programs_dist_context[idx] = {}
            start_time = time.time()
            graph = GraphUtil.convert_to_graph(sub_fwd_program.global_block())
            self.fwd_sub_program_graphs[idx] = graph
            end_time = time.time()
            start_time = time.time()
            results = GraphUtil.match_all_patterns(graph)
            end_time = time.time()

            if results:
                for parallelism in selective_parallelisms:
                    dist_context = DistributedContext(sub_fwd_program)
                    dist_context.add_process_mesh(process_mesh)
                    for pattern_name in results.keys():
                        # print("pattern name: ", pattern_name)
                        pattern = _PATTERN_MAP[pattern_name]
                        shard_spec = pattern.attrs["shard_spec"][parallelism]
                        for pattern_node_id in shard_spec.keys():
                            for item in results[pattern_name]:
                                tensor_id = item[pattern_node_id]
                                tensor_name = graph.attrs["id_to_var"][
                                    tensor_id
                                ]
                                # print("tensor {}'s shard_spec is {}".format(tensor_name, shard_spec[pattern_node_id]))
                                # set dist tensor
                                tensor = sub_fwd_program.global_block()._var_recursive(
                                    tensor_name
                                )
                                dist_tensor = DistributedTensor(tensor)
                                dist_tensor.dist_attr.process_mesh = (
                                    process_mesh
                                )
                                dist_tensor.dist_attr.dims_mapping = shard_spec[
                                    pattern_node_id
                                ]
                                dist_tensor.dist_attr.mark_annotated(
                                    "dims_mapping"
                                )
                                dist_tensor.dist_attr.mark_annotated(
                                    "process_mesh"
                                )
                                dist_context.add_dist_tensor_for_program(
                                    dist_tensor
                                )

                    dist_context.initialize()
                    completer = Completer(dist_context)
                    completer.complete_forward_annotation()
                    self.sub_programs_dist_context[idx][
                        parallelism
                    ] = dist_context
            else:
                print("Failed to match any pattern in this program.")

    def _compelte_sub_fwd_program(self, idx, sub_fwd_program, process_mesh):
        selective_parallelisms = (
            ["dp", "mp"] if len(process_mesh.shape) == 1 else ["dp_mp", "mp_dp"]
        )
        # print("_compelte_sub_fwd_program process_mesh: ", process_mesh)
        for parallelism in selective_parallelisms:
            has_set_tensor_count = 0
            dist_context = DistributedContext(sub_fwd_program)
            dist_context.process_meshes = []
            dist_context.add_process_mesh(process_mesh)
            vars = sub_fwd_program.global_block().vars
            for var_name in vars:
                var_id = vars[var_name].desc.original_id()
                if var_id in self.tensor_dist_attrs:
                    if parallelism in self.tensor_dist_attrs[var_id]:
                        dims_mapping = self.tensor_dist_attrs[var_id][
                            parallelism
                        ]
                        # print("var_id: ", var_name, dims_mapping, process_mesh)
                        dist_tensor = DistributedTensor(vars[var_name])
                        dist_tensor.dist_attr.process_mesh = process_mesh
                        dist_tensor.dist_attr.dims_mapping = dims_mapping
                        dist_tensor.dist_attr.mark_annotated("dims_mapping")
                        dist_tensor.dist_attr.mark_annotated("process_mesh")
                        # print("dist_tensor: ", dist_tensor)
                        dist_context.add_dist_tensor_for_program(dist_tensor)
                        has_set_tensor_count += 1

            # check whether no dist attr in dist context
            if has_set_tensor_count > 0:
                # print("rule_based_tuner.py dist_context0****", len(dist_context.process_meshes), dist_context.process_meshes[0])
                dist_context.initialize(no_default=True)
                # print("rule_based_tuner.py dist_context1****", len(dist_context.process_meshes), dist_context.process_meshes[0])
                completer = Completer(dist_context)
                completer.complete_forward_annotation()
                # print("rule_based_tuner.py parallelism", parallelism)
                # print_program_with_dist_attr(dist_context.serial_main_program, dist_context)
                if parallelism not in self.sub_programs_dist_context[idx]:
                    self.sub_programs_dist_context[idx][parallelism] = {}
                key = self.convert_process_mesh_to_key(process_mesh)
                self.sub_programs_dist_context[idx][parallelism][
                    key
                ] = dist_context
                # print("rule_based_tuner.py dist_context****", len(dist_context.process_meshes), dist_context.process_meshes[0])
            else:
                print(
                    "no pattern matched in this sub program with parallel mode {}".format(
                        parallelism
                    )
                )

    def complete_sub_fwd_programs(self, process_mesh):
        for idx in self.fwd_sub_programs.keys():
            sub_fwd_program = self.fwd_sub_programs[idx]
            if idx not in self.sub_programs_dist_context:
                self.sub_programs_dist_context[idx] = {}
            self._compelte_sub_fwd_program(idx, sub_fwd_program, process_mesh)

    def _complete_sub_bwd_program(self, sub_program_dist_context):
        def _is_grad_var_name(name):
            if "@GRAD" in name:
                return True
            return False

        sub_fwd_program = sub_program_dist_context.serial_main_program
        block = sub_fwd_program.global_block()
        vars = self.full_main_program.global_block().vars
        ops = self.full_main_program.global_block().ops
        grad_var_to_var = (
            self.full_main_program_dist_context.dist_op_context.grad_var_to_var[
                1
            ]
        )
        for forward_op in block.ops:
            grad_op_id = self.op_original_id_to_grad_op_original_id[
                forward_op.desc.original_id()
            ]
            # for unsqueeze2 op in gpt, it has no grad op
            # or for no need to bwd
            if grad_op_id not in self.op_original_id_to_op:
                continue
            grad_op = self.op_original_id_to_op[grad_op_id]
            # print("op: ", forward_op)
            # print("grad_op: ", grad_op)
            if grad_op.type == "concat" and forward_op.type == "split":
                forward_op_dist_attr = (
                    sub_program_dist_context.get_op_dist_attr_for_program(
                        forward_op
                    )
                )
                output_var = vars[grad_op.desc.output('Out')[0]]
                split_input_var_name = forward_op.input("X")[0]
                ref_dims_mapping = forward_op_dist_attr.get_input_dims_mapping(
                    split_input_var_name
                )
                ref_mesh = forward_op_dist_attr.process_mesh

                grad_op_dist_attr = OperatorDistributedAttribute()
                for input_name in grad_op.input_arg_names:
                    grad_op_dist_attr.set_input_dims_mapping(
                        input_name, ref_dims_mapping
                    )

                output_var_dist_attr = TensorDistributedAttribute()
                output_var_dist_attr.dims_mapping = ref_dims_mapping
                output_var_dist_attr.process_mesh = ref_mesh
                sub_program_dist_context.set_tensor_dist_attr_for_program(
                    output_var, output_var_dist_attr
                )

                grad_op_dist_attr.set_output_dims_mapping(
                    output_var.name, ref_dims_mapping
                )
                grad_op_dist_attr.process_mesh = ref_mesh
                sub_program_dist_context.set_op_dist_attr_for_program(
                    grad_op, grad_op_dist_attr
                )
                grad_op_dist_attr.impl_type = (
                    fwd_op_dist_attr.impl_type  # noqa: F821
                )
                grad_op_dist_attr.impl_idx = (
                    fwd_op_dist_attr.impl_idx  # noqa: F821
                )
                continue

            fwd_op_dist_attr = (
                sub_program_dist_context.get_op_dist_attr_for_program(
                    forward_op
                )
            )
            fwd_op_process_mesh = fwd_op_dist_attr.process_mesh
            grad_op_dist_attr = OperatorDistributedAttribute()
            grad_op_dist_attr.process_mesh = fwd_op_process_mesh

            for input_name in grad_op.input_arg_names:
                if (
                    input_name not in forward_op.input_arg_names
                    and input_name not in forward_op.output_arg_names
                ):
                    if input_name in grad_var_to_var.keys():
                        fwd_name = grad_var_to_var[input_name]
                        ref_dims_mapping = (
                            fwd_op_dist_attr.get_output_dims_mapping(fwd_name)
                        )
                    else:
                        input_var = vars[input_name]
                        ref_dims_mapping = sub_program_dist_context.get_tensor_dist_attr_for_program(
                            input_var
                        ).dims_mapping
                else:
                    if input_name in forward_op.input_arg_names:
                        ref_dims_mapping = (
                            fwd_op_dist_attr.get_input_dims_mapping(input_name)
                        )
                    else:
                        ref_dims_mapping = (
                            fwd_op_dist_attr.get_output_dims_mapping(input_name)
                        )
                assert (
                    ref_dims_mapping is not None
                ), "[{}] 's dims mapping is NONE".format(input_name)
                grad_op_dist_attr.set_input_dims_mapping(
                    input_name, ref_dims_mapping
                )

            for output_name in grad_op.output_arg_names:
                assert output_name in grad_var_to_var
                fwd_name = grad_var_to_var[output_name]
                ref_dims_mapping = fwd_op_dist_attr.get_input_dims_mapping(
                    fwd_name
                )
                # var
                output_var = vars[output_name]
                tensor_dist_attr = TensorDistributedAttribute()
                tensor_dist_attr.dims_mapping = ref_dims_mapping
                tensor_dist_attr.process_mesh = fwd_op_process_mesh
                sub_program_dist_context.set_tensor_dist_attr_for_program(
                    output_var, tensor_dist_attr
                )
                # op
                grad_op_dist_attr.set_output_dims_mapping(
                    output_name, ref_dims_mapping
                )

            grad_op_dist_attr.impl_type = fwd_op_dist_attr.impl_type
            grad_op_dist_attr.impl_idx = fwd_op_dist_attr.impl_idx
            sub_program_dist_context.set_op_dist_attr_for_program(
                grad_op, grad_op_dist_attr
            )

            grad_op_idx = self.op_original_id_to_idx[grad_op_id]
            if grad_op_idx + 1 < len(ops):
                grad_op_next_op = ops[grad_op_idx + 1]
                if grad_op_next_op.type == "sum":
                    # print("grad_op_next_op: ", grad_op_next_op)
                    assert all(
                        map(_is_grad_var_name, grad_op_next_op.input_arg_names)
                    )
                    output_name = grad_op_next_op.output_arg_names[0]
                    assert (
                        output_name in grad_var_to_var
                    ), "sum op's output '{}' has no corresponding var".format(
                        output_name
                    )
                    ref_fwd_var_name = grad_var_to_var[output_name]
                    ref_fwd_var = vars[ref_fwd_var_name]
                    ref_fwd_dist_attr = sub_program_dist_context.get_tensor_dist_attr_for_program(
                        ref_fwd_var
                    )
                    ref_fwd_dims_mapping = ref_fwd_dist_attr.dims_mapping
                    ref_fwd_process_mesh = ref_fwd_dist_attr.process_mesh

                    # output
                    tensor_dist_attr = TensorDistributedAttribute()
                    tensor_dist_attr.dims_mapping = ref_fwd_dims_mapping
                    tensor_dist_attr.process_mesh = ref_fwd_process_mesh
                    output_var = vars[output_name]
                    sub_program_dist_context.set_tensor_dist_attr_for_program(
                        output_var, tensor_dist_attr
                    )

                    # op
                    grad_op_dist_attr = OperatorDistributedAttribute()
                    grad_op_dist_attr.process_mesh = ref_fwd_process_mesh
                    for var_name in grad_op_next_op.input_arg_names:
                        grad_op_dist_attr.set_input_dims_mapping(
                            var_name, ref_fwd_dims_mapping
                        )
                    grad_op_dist_attr.set_output_dims_mapping(
                        output_name, ref_fwd_dims_mapping
                    )
                    grad_op_dist_attr.impl_type = "default"
                    grad_op_dist_attr.impl_idx = 0

                    sub_program_dist_context.set_op_dist_attr_for_program(
                        grad_op_next_op, grad_op_dist_attr
                    )

    def complete_sub_bwd_programs(self):
        for idx in self.sub_programs_dist_context:
            for parallelism in self.sub_programs_dist_context[idx]:
                for key in self.sub_programs_dist_context[idx][parallelism]:
                    sub_program_dist_context = self.sub_programs_dist_context[
                        idx
                    ][parallelism][key]
                    # print("completed fwd sub program: ", idx, parallelism, key)
                    # print_program_with_dist_attr(self.full_main_program, sub_program_dist_context)
                    self._complete_sub_bwd_program(sub_program_dist_context)
                    # print("complete sub bwd program: ", idx, parallelism, key)
                    # print_program_with_dist_attr(self.full_main_program, sub_program_dist_context)

    def _complete_sub_update_programs(self, sub_program_dist_context):
        world_ranks = ProcessMesh(
            [
                i
                for i in range(
                    self._cluster.get_num_machines()
                    * self._cluster._num_devices_per_machine
                )
            ]
        )
        dist_tensors = sub_program_dist_context._dist_tensors_for_program

        vars = self.full_main_program.global_block().vars
        ops = self.full_main_program.global_block().ops
        learning_rate_completed = False
        for idx in range(len(ops)):
            op = ops[idx]
            if int(op.attr('op_role')) == int(OpRole.Optimize):
                if is_gradient_clip_op(op):
                    if op.type in [
                        "sum",
                        "sqrt",
                        "fill_constant",
                        "elementwise_max",
                        "elementwise_div",
                    ]:
                        op_dist_attr = OperatorDistributedAttribute()
                        op_dist_attr.process_mesh = world_ranks
                        for in_name in op.input_arg_names:
                            in_var = vars[in_name]
                            if in_var.desc.original_id() in dist_tensors:
                                in_dist_attr = sub_program_dist_context.get_tensor_dist_attr_for_program(
                                    in_var
                                )
                                op_dist_attr.set_input_dist_attr(
                                    in_name, in_dist_attr
                                )
                            else:
                                continue
                        for out_name in op.output_arg_names:
                            out_var = vars[out_name]
                            if out_var.desc.original_id() in dist_tensors:
                                out_dist_attr = TensorDistributedAttribute()
                                out_dist_attr.process_mesh = world_ranks
                                out_dist_attr.dims_mapping = [
                                    -1 for _ in range(len(out_var.shape))
                                ]
                                sub_program_dist_context.set_tensor_dist_attr_for_program(
                                    out_var, out_dist_attr
                                )
                                op_dist_attr.set_output_dist_attr(
                                    out_name, out_dist_attr
                                )
                            else:
                                continue
                    else:
                        in_var = vars[op.input("X")[0]]
                        if in_var.desc.original_id() in dist_tensors:
                            in_dist_attr = sub_program_dist_context.get_tensor_dist_attr_for_program(
                                in_var
                            )
                            assert in_dist_attr is not None
                            ref_process_mesh = in_dist_attr.process_mesh
                            ref_dims_mapping = in_dist_attr.dims_mapping

                            if (
                                op.type == "cast"
                                and ops[idx + 1].type == "elementwise_mul"
                            ):
                                ref_var = vars[ops[idx + 1].input("X")[0]]
                                if (
                                    ref_var.desc.original_id()
                                    not in dist_tensors
                                ):
                                    raise ValueError()
                                ref_dist_attr = sub_program_dist_context.get_tensor_dist_attr_for_program(
                                    ref_var
                                )
                                assert ref_dist_attr is not None
                                ref_process_mesh = ref_dist_attr.process_mesh

                            out_var = vars[op.output("Out")[0]]
                            out_dist_attr = TensorDistributedAttribute()
                            out_dist_attr.process_mesh = ref_process_mesh
                            if out_var.shape == in_var.shape:
                                out_dist_attr.dims_mapping = ref_dims_mapping
                            else:
                                assert (
                                    len(out_var.shape) == 1
                                    and out_var.shape[0] == 1
                                )
                                out_dist_attr.dims_mapping = [-1]
                            sub_program_dist_context.set_tensor_dist_attr_for_program(
                                out_var, out_dist_attr
                            )

                            op_dist_attr = OperatorDistributedAttribute()
                            op_dist_attr.process_mesh = ref_process_mesh
                            op_dist_attr.set_input_dist_attr(
                                in_var.name, in_dist_attr
                            )
                            op_dist_attr.set_output_dist_attr(
                                out_var.name, out_dist_attr
                            )

                            sub_program_dist_context.set_op_dist_attr_for_program(
                                op, op_dist_attr
                            )
                        else:
                            continue

                if "Grad" in op.input_names and "Param" in ops[idx].input_names:
                    assert (
                        len(op.input("Param")) == 1
                    ), "Only support one-to-one now."
                    assert (
                        len(op.input("Grad")) == 1
                    ), "Only support one-to-one now."
                    param = vars[op.input("Param")[0]]
                    grad_var = vars[op.input("Grad")[0]]
                    if param.desc.original_id() in dist_tensors:
                        param_dist_attr = sub_program_dist_context.get_tensor_dist_attr_for_program(
                            param
                        )
                        assert param_dist_attr is not None
                        ref_process_mesh = sub_program_dist_context.get_tensor_dist_attr_for_program(
                            param
                        ).process_mesh
                        assert ref_process_mesh is not None
                        ref_dims_mapping = sub_program_dist_context.get_tensor_dist_attr_for_program(
                            param
                        ).dims_mapping
                        assert ref_dims_mapping is not None
                        op_dist_attr = OperatorDistributedAttribute()
                        op_dist_attr.process_mesh = ref_process_mesh
                        op_dist_attr.set_input_dims_mapping(
                            grad_var.name, ref_dims_mapping
                        )
                        op_dist_attr.set_input_dims_mapping(
                            param.name, ref_dims_mapping
                        )
                        op_dist_attr.set_output_dims_mapping(
                            param.name, ref_dims_mapping
                        )
                        learning_var = vars[op.input("LearningRate")[0]]
                        op_dist_attr.set_input_dims_mapping(
                            learning_var.name, [-1]
                        )
                        op_dist_attr.set_output_dims_mapping(
                            learning_var.name, [-1]
                        )

                        if not learning_rate_completed:
                            learning_rate_completed = True
                            var_dist_attr = TensorDistributedAttribute()
                            var_dist_attr.process_mesh = world_ranks
                            var_dist_attr.dims_mapping = [-1]
                            sub_program_dist_context.set_tensor_dist_attr_for_program(
                                learning_var, var_dist_attr
                            )

                        for input_name in op.desc.input_names():

                            if input_name in [
                                'Param',
                                'Grad',
                                'LearningRate',
                                "SkipUpdate",
                                "Beta1Tensor",
                                "Beta2Tensor",
                                "EpsilonTensor",
                            ]:
                                continue
                            if len(op.desc.input(input_name)) == 0:
                                continue

                            assert len(op.desc.input(input_name)) == 1
                            input_var = vars[op.desc.input(input_name)[0]]
                            input_var_attr = TensorDistributedAttribute()

                            if (
                                "Beta1Pow" in input_name
                                or "Beta2Pow" in input_name
                            ):
                                input_var_attr.dims_mapping = [-1]
                                op_dist_attr.set_input_dims_mapping(
                                    input_var.name, [-1]
                                )
                                op_dist_attr.set_output_dims_mapping(
                                    input_var.name, [-1]
                                )
                            else:
                                input_var_attr.dims_mapping = ref_dims_mapping
                                op_dist_attr.set_input_dims_mapping(
                                    input_var.name, ref_dims_mapping
                                )
                                op_dist_attr.set_output_dims_mapping(
                                    input_var.name, ref_dims_mapping
                                )

                            input_var_attr.process_mesh = ref_process_mesh
                            sub_program_dist_context.set_tensor_dist_attr_for_program(
                                input_var, input_var_attr
                            )

                        sub_program_dist_context.set_op_dist_attr_for_program(
                            op, op_dist_attr
                        )
                        continue
                    else:
                        continue

    def complete_sub_update_programs(self):
        for idx in self.sub_programs_dist_context:
            for parallelism in self.sub_programs_dist_context[idx]:
                for key in self.sub_programs_dist_context[idx][parallelism]:
                    sub_program_dist_context = self.sub_programs_dist_context[
                        idx
                    ][parallelism][key]
                    # print("completed fwd sub program: ", idx, parallelism, key)
                    # print_program_with_dist_attr(self.full_main_program, sub_program_dist_context)
                    self._complete_sub_update_programs(sub_program_dist_context)
                    # print("complete sub bwd program: ", idx, parallelism, key)
                    # print_program_with_dist_attr(self.full_main_program, sub_program_dist_context)

    def convert_process_mesh_to_key(self, process_mesh):
        processes = ",".join([str(x) for x in process_mesh.processes])
        topology = ",".join([str(x) for x in process_mesh.topology])
        key = processes + ";" + topology
        return key

    def convert_process_mesh_to_key_v2(self, process_mesh):
        topology = ",".join([str(x) for x in process_mesh.topology])
        key = topology
        return key

    def convert_device_mesh_to_key(self, device_mesh):
        processes = ",".join([str(x) for x in device_mesh.device_ids])
        topology = ",".join([str(x) for x in device_mesh.shape])
        key = processes + ";" + topology
        return key

    def convert_device_mesh_to_key_v2(self, device_mesh):
        topology = ",".join([str(x) for x in device_mesh.shape])
        key = topology
        return key

    def combine_sub_program(self, sub_dist_context, clear=False):
        if clear:
            self.full_main_program_dist_context._dist_tensors_for_program = {}
            self.full_main_program_dist_context._dist_ops_for_program = {}
            self.full_main_program_dist_context.process_meshes = []
        full_dist_context = self.full_main_program_dist_context
        # set dist tensor, pay attention to shared param or var as input for multi op
        for tensor_id in sub_dist_context._dist_tensors_for_program:
            dist_tensor = sub_dist_context._dist_tensors_for_program[tensor_id]
            if tensor_id not in full_dist_context._dist_tensors_for_program:
                full_dist_context.add_dist_tensor_for_program(dist_tensor)

        # set dist op
        for op_id in sub_dist_context._dist_ops_for_program:
            dist_op = sub_dist_context._dist_ops_for_program[op_id]
            full_dist_context.add_dist_op_for_program(dist_op)

        for process_mesh in sub_dist_context.process_meshes:
            full_dist_context.add_process_mesh(process_mesh)

    def _get_sub_program_cost(self, dist_context):
        cost_estimator = CostEstimator(self.full_main_program, self._cluster)
        global_cost = cost_estimator.estimate(dist_context)
        max_memory = cost_estimator._estimate_max_memory_by_dist_op(
            dist_context
        )

        # Print the cost
        # cost_estimator.pretty_print_cost()
        return global_cost.time, max_memory

    def get_sub_programs_cost(self):
        for idx in self.sub_programs_dist_context:
            if idx not in self.sub_program_cost:
                self.sub_program_cost[idx] = {}
            for parallelism in self.sub_programs_dist_context[idx]:
                if parallelism not in self.sub_program_cost[idx]:
                    self.sub_program_cost[idx][parallelism] = {}
                for key in self.sub_programs_dist_context[idx][parallelism]:
                    if key not in self.sub_program_cost[idx][parallelism]:
                        self.sub_program_cost[idx][parallelism][key] = {}
                    sub_program_dist_context = self.sub_programs_dist_context[
                        idx
                    ][parallelism][key]

                    time, max_memory = self._get_sub_program_cost(
                        sub_program_dist_context
                    )
                    self.sub_program_cost[idx][parallelism][key]["cost"] = time
                    self.sub_program_cost[idx][parallelism][key][
                        "memory"
                    ] = max_memory
                    print(
                        "idx, parallelism, key, time, memory",
                        idx,
                        parallelism,
                        key,
                        time,
                        max_memory,
                    )

    def _local_stage_cost(self, start, end, process_mesh, fast):
        assert start >= 0
        assert end < len(self.layers)
        # if fast is true, it means that the layers in this process mesh have same structure and parallism.
        # if not, it means that we use the more accurate calculation method.
        selective_parallelisms = (
            ["dp", "mp"] if len(process_mesh.shape) == 1 else ["dp_mp", "mp_dp"]
        )
        key = self.convert_process_mesh_to_key(process_mesh)
        if start not in self.stage_cost:
            self.stage_cost[start] = {}
        best_cost = sys.maxsize
        best_strategy = None

        if fast:
            for parallelism in selective_parallelisms:
                local_stage_time = 0
                local_stage_memory = 0
                for i in range(start, end + 1):
                    if i not in self.stage_cost[start]:
                        self.stage_cost[start][i] = {}
                    if key not in self.stage_cost[start][i]:
                        self.stage_cost[start][i][key] = {}

                    # if partition mode not in sub program, the cost is inf
                    if parallelism in self.sub_program_cost[i]:
                        local_stage_time += self.sub_program_cost[i][
                            parallelism
                        ][key]["time"]
                        local_stage_memory += self.sub_program_cost[i][
                            parallelism
                        ][key]["memory"]
                        if local_stage_memory > 32 * 1024 * 1024 * 1024:
                            self.stage_cost[start][i][key][
                                parallelism
                            ] = sys.maxsize
                            continue
                        else:
                            self.stage_cost[start][i][key][
                                parallelism
                            ] = local_stage_time
                    else:
                        self.stage_cost[start][i][key][
                            parallelism
                        ] = sys.maxsize

                if self.stage_cost[start][end][key][parallelism] < best_cost:
                    best_cost = self.stage_cost[start][end][key][parallelism]
                    best_strategy = parallelism
                    print("best_cost: ", best_cost)
            self.stage_cost[start][end][key]["best_strategy"] = best_strategy
        else:
            raise NotImplementedError()

        return best_cost

    def _local_stage_pass(self, start, end, process_mesh):
        # convert process mesh to dict key
        key = self.convert_process_mesh_to_key(process_mesh)

        if start in self.stage_best_cost_of_pm:
            if end in self.stage_best_cost_of_pm[start]:
                if key in self.stage_best_cost_of_pm[start][end]:
                    return self.stage_best_cost_of_pm[start][end][key]["cost"]

        assert end >= start
        selective_parallelisms = (
            ["dp", "mp"] if len(process_mesh.shape) == 1 else ["dp_mp", "mp_dp"]
        )
        if start not in self.stage_best_cost_of_pm:
            self.stage_best_cost_of_pm[start] = {}
        if end not in self.stage_best_cost_of_pm[start]:
            self.stage_best_cost_of_pm[start][end] = {}

        if key not in self.stage_best_cost_of_pm[start][end]:
            self.stage_best_cost_of_pm[start][end][key] = {}

        if end == start:
            dist_contexts_x = [DistributedContext(), DistributedContext()]
        else:
            dist_contexts_x = self.stage_best_cost_of_pm[start][end - 1][key][
                "dist_context"
            ]

        # beam search 2
        count = 0
        for dist_context_x in dist_contexts_x:
            if end == start and count == 1:
                break
            for parallelism in selective_parallelisms:
                # print("rule_based_tuner.py self.sub_programs_dist_context[end]: ", self.sub_programs_dist_context[end])
                dist_context_y = self.sub_programs_dist_context[end][
                    parallelism
                ][key]
                dist_context = self.combine_dist_contexts(
                    [dist_context_x, dist_context_y]
                )
                if (
                    "dist_context"
                    not in self.stage_best_cost_of_pm[start][end][key]
                ):
                    self.stage_best_cost_of_pm[start][end][key][
                        "dist_context"
                    ] = [None, None]
                    self.stage_best_cost_of_pm[start][end][key]["cost"] = [
                        sys.maxsize,
                        sys.maxsize,
                    ]

                # estimate cost and memory
                cost, local_stage_memory = self._get_sub_program_cost(
                    dist_context
                )
                # print_program_with_dist_attr(self.full_main_program, dist_context)
                print(
                    "parallelism cost: ",
                    parallelism,
                    start,
                    end,
                    cost,
                    process_mesh,
                )
                if local_stage_memory > 32 * 1024 * 1024 * 1024:
                    cost = sys.maxsize
                    # self.stage_best_cost_of_pm[start][end][key]["cost"] = sys.maxsize
                index = -1
                for idx, item in enumerate(
                    self.stage_best_cost_of_pm[start][end][key]["cost"]
                ):
                    if cost <= item:
                        index = idx
                        break
                if index == 0:
                    self.stage_best_cost_of_pm[start][end][key]["cost"][
                        1
                    ] = self.stage_best_cost_of_pm[start][end][key]["cost"][0]
                    self.stage_best_cost_of_pm[start][end][key]["dist_context"][
                        1
                    ] = self.stage_best_cost_of_pm[start][end][key][
                        "dist_context"
                    ][
                        0
                    ]
                    self.stage_best_cost_of_pm[start][end][key]["cost"][
                        0
                    ] = cost
                    self.stage_best_cost_of_pm[start][end][key]["dist_context"][
                        0
                    ] = dist_context

                elif index == 1:
                    self.stage_best_cost_of_pm[start][end][key]["cost"][
                        1
                    ] = cost
                    self.stage_best_cost_of_pm[start][end][key]["dist_context"][
                        1
                    ] = dist_context
            count += 1

        if (
            self.stage_best_cost_of_pm[start][end][key]["cost"][1]
            < self.stage_best_cost_of_pm[start][end][key]["cost"][0]
        ):
            self.stage_best_cost_of_pm[start][end][key][
                "best_cost"
            ] = self.stage_best_cost_of_pm[start][end][key]["cost"][1]
            self.stage_best_cost_of_pm[start][end][key][
                "best_dist_context"
            ] = self.stage_best_cost_of_pm[start][end][key]["dist_context"][1]
        else:
            self.stage_best_cost_of_pm[start][end][key][
                "best_cost"
            ] = self.stage_best_cost_of_pm[start][end][key]["cost"][0]
            self.stage_best_cost_of_pm[start][end][key][
                "best_dist_context"
            ] = self.stage_best_cost_of_pm[start][end][key]["dist_context"][0]

        return self.stage_best_cost_of_pm[start][end][key]["best_cost"]

    def local_stage_pass(self, start, end, device_mesh):
        dm_key = self.convert_device_mesh_to_key(device_mesh)
        # if start in self.stage_best_cost_of_dm:
        #     if end in self.stage_best_cost_of_dm[start]:
        #         if dm_key in self.stage_best_cost_of_dm[start][end]:
        #             return self.stage_best_cost_of_dm[start][end][dm_key]["cost"]

        # device_mesh to process_mesh
        device_mesh_shape = device_mesh.shape
        if len(device_mesh_shape) == 1:
            device_mesh_shape.insert(0, 1)
        process_mesh_shapes = convert_to_process_meshes(device_mesh_shape)
        best_cost = sys.maxsize

        if start not in self.stage_best_cost_of_dm:
            self.stage_best_cost_of_dm[start] = {}
        if end not in self.stage_best_cost_of_dm[start]:
            self.stage_best_cost_of_dm[start][end] = {}
        if dm_key not in self.stage_best_cost_of_dm[start][end]:
            self.stage_best_cost_of_dm[start][end][dm_key] = {}

        for process_mesh_shape in process_mesh_shapes:
            process_mesh = ProcessMesh(
                np.array(device_mesh.device_ids)
                .reshape(process_mesh_shape)
                .tolist()
            )
            key = self.convert_process_mesh_to_key(process_mesh)
            for i in range(start, end + 1):
                self._local_stage_pass(start, i, process_mesh)
            print("cost: ", self.stage_best_cost_of_pm[start][end][key]["cost"])
            if (
                self.stage_best_cost_of_pm[start][end][key]["best_cost"]
                <= best_cost
            ):
                best_cost = self.stage_best_cost_of_pm[start][end][key][
                    "best_cost"
                ]
                self.stage_best_cost_of_dm[start][end][dm_key][
                    "cost"
                ] = best_cost
                self.stage_best_cost_of_dm[start][end][dm_key][
                    "dist_context"
                ] = self.stage_best_cost_of_pm[start][end][key][
                    "best_dist_context"
                ]

        print("device_mesh: ", device_mesh)
        print(
            "best cost: ",
            self.stage_best_cost_of_dm[start][end][dm_key]["cost"],
        )
        # print("program: ")
        # print_program_with_dist_attr(self.full_main_program, self.stage_best_cost_of_dm[start][end][dm_key]["dist_context"])
        return best_cost

    def combine_dist_contexts(self, dist_contexts):
        combined_dist_context = DistributedContext()
        # set dist_context_x
        # set dist tensor, pay attention to shared param or var as input for multi op
        for dist_context in dist_contexts:
            for tensor_id in dist_context._dist_tensors_for_program:
                dist_tensor = dist_context._dist_tensors_for_program[tensor_id]
                if (
                    tensor_id
                    not in combined_dist_context._dist_tensors_for_program
                ):
                    combined_dist_context.add_dist_tensor_for_program(
                        dist_tensor
                    )

            # set dist op
            for op_id in dist_context._dist_ops_for_program:
                dist_op = dist_context._dist_ops_for_program[op_id]
                combined_dist_context.add_dist_op_for_program(dist_op)

            for process_mesh in dist_context.process_meshes:
                combined_dist_context.add_process_mesh(process_mesh)

        #         for tensor_id in dist_context_x._dist_tensors_for_program:
        #             dist_tensor = dist_context_x._dist_tensors_for_program[tensor_id]
        #             if tensor_id not in combined_dist_context._dist_tensors_for_program:
        #                 combined_dist_context.add_dist_tensor_for_program(dist_tensor)

        #         # set dist op
        #         for op_id in dist_context_x._dist_ops_for_program:
        #             dist_op = dist_context_x._dist_ops_for_program[op_id]
        #             combined_dist_context.add_dist_op_for_program(dist_op)

        #         for process_mesh in dist_context_x.process_meshes:
        #             combined_dist_context.add_process_mesh(process_mesh)

        #         # set dist_context_y
        #         # set dist tensor, pay attention to shared param or var as input for multi op
        #         for tensor_id in dist_context_y._dist_tensors_for_program:
        #             dist_tensor = dist_context_y._dist_tensors_for_program[tensor_id]
        #             if tensor_id not in combined_dist_context._dist_tensors_for_program:
        #                 combined_dist_context.add_dist_tensor_for_program(dist_tensor)

        #         # set dist op
        #         for op_id in dist_context_y._dist_ops_for_program:
        #             dist_op = dist_context_y._dist_ops_for_program[op_id]
        #             combined_dist_context.add_dist_op_for_program(dist_op)

        #         for process_mesh in dist_context_y.process_meshes:
        #             combined_dist_context.add_process_mesh(process_mesh)

        return combined_dist_context

    def prepare(self):
        """Prepare sub program and corresponding cost"""

        # step1: cluster operators to layers
        self.layers = self.cluster_operators()

        # print("rule_based_tuner.py self.layers0: ", self.layers[0])
        # print("rule_based_tuner.py program: ", self._dist_context.serial_main_program)
        # print("rule_based_tuner.py self.layers1: ", self.layers[1])
        # print("rule_based_tuner.py self.layers2: ", self.layers[2])
        # step2: generate sub program of each layer
        self.gen_fwd_sub_programs_by_clone()

        # step3: match patterns to get tensor dist attr setting
        self.match_program(self._dist_context.serial_main_program)
        # now, we have dist attr setting of tensor in each layer when in different parallelism

        # step4: partition devices to device meshes
        n, m = (
            self._cluster.get_num_machines(),
            self._cluster._num_devices_per_machine,
        )
        device_meshes_list = ClusterPartitionUtil.partition_cluster(n, m)
        print("device_meshes_list: ", device_meshes_list)

        # step5: transform device mesh to process meshes
        dm_idx = 0
        for device_meshes in device_meshes_list:
            has_used_devices = 0
            self.device_meshes_list.append([])
            for device_mesh in device_meshes:
                devices = reduce(lambda x, y: x * y, device_mesh)
                processes = [
                    i
                    for i in range(has_used_devices, has_used_devices + devices)
                ]
                device_mesh_shape = (
                    device_mesh
                    if device_mesh[0] != 1
                    else [device_mesh[i] for i in range(1, len(device_mesh))]
                )
                self.device_meshes_list[-1].append(
                    DeviceMesh(
                        mesh=np.array(processes)
                        .reshape(device_mesh_shape)
                        .tolist(),
                        name="device_mesh_" + str(dm_idx),
                    )
                )
                dm_idx += 1
                print("processes: ", processes)
                has_used_devices += devices
                process_mesh_shapes = convert_to_process_meshes(device_mesh)
                print("process_mesh_shapes: ", process_mesh_shapes)
                for process_mesh_shape in process_mesh_shapes:
                    process_mesh = ProcessMesh(
                        np.array(processes).reshape(process_mesh_shape).tolist()
                    )
                    if process_mesh not in self.process_meshes:
                        self.process_meshes.append(process_mesh)
        print("self.process_meshes: ")
        for process_mesh in self.process_meshes:
            print(process_mesh)

        print("self.device_meshes_list: ")
        for device_meshes in self.device_meshes_list:
            for device_mesh in device_meshes:
                print(device_mesh)

        # step6: generate full program
        self.gen_full_program()
        # print("self.full_program: ", self.full_main_program)

        # step7: fwd complete sub program
        for process_mesh in self.process_meshes:
            self.complete_sub_fwd_programs(process_mesh)

        # step8: bwd complete sub program
        self.complete_sub_bwd_programs()

        # step8: update compete sub program
        self.complete_sub_update_programs()

    def layer_placement_pass(self, stages, layers, device_meshes):
        stage_layer_cost = [
            [sys.maxsize for i in range(layers)] for j in range(stages)
        ]
        min_max_stage_costs = [
            [None for i in range(layers)] for j in range(stages)
        ]
        best_strategies = [[None for i in range(layers)] for j in range(stages)]
        for s in range(len(device_meshes)):
            for i in range(0, layers):
                if s == 0:
                    stage_layer_cost[s][i] = self.local_stage_pass(
                        0, i, device_meshes[s]
                    )
                    min_max_stage_costs[s][i] = stage_layer_cost[s][i]
                    key = self.convert_device_mesh_to_key(device_meshes[s])
                    best_strategies[s][i] = self.stage_best_cost_of_dm[0][i][
                        key
                    ]["dist_context"]
                else:
                    min_cost = sys.maxsize
                    min_max_stage_cost = sys.maxsize
                    for j in range(0, i):
                        key = self.convert_device_mesh_to_key(device_meshes[s])
                        local_stage_cost = self.local_stage_pass(
                            j + 1, i, device_meshes[s]
                        )
                        # cost = stage_layer_cost[s-1][j] + local_stage_cost
                        dist_context = self.combine_dist_contexts(
                            [
                                best_strategies[s - 1][j],
                                self.stage_best_cost_of_dm[j + 1][i][key][
                                    "dist_context"
                                ],
                            ]
                        )
                        cost, _ = self._get_sub_program_cost(dist_context)
                        max_stage_cost = (
                            min_max_stage_costs[s - 1][j]
                            if local_stage_cost < min_max_stage_costs[s - 1][j]
                            else local_stage_cost
                        )

                        if cost <= min_cost:
                            if cost == min_cost:
                                if max_stage_cost < min_max_stage_cost:
                                    min_max_stage_cost = max_stage_cost
                                    best_strategies[s][i] = dist_context
                                    # print("min_max_stage_cost cost: ", s, i, j, min_cost, min_max_stage_cost)
                                else:
                                    break
                            min_cost = cost
                    stage_layer_cost[s][i] = min_cost
                    min_max_stage_costs[s][i] = min_max_stage_cost
        #             print("stage_layer_cost cost: ", s, i, stage_layer_cost[s][i])
        # print("stage_layer_cost: ", stage_layer_cost)
        # print("min_max_stage_cost: ", min_max_stage_costs)
        # print("最终cost: ", stage_layer_cost[stages - 1][layers - 1])
        # print("最终program: ")
        # print_program_with_dist_attr(self.full_main_program, best_strategies[stages-1][layers-1])
        return (
            stage_layer_cost[stages - 1][layers - 1],
            best_strategies[stages - 1][layers - 1],
        )

    def tune_o2(self):
        best_dist_context = None
        best_cost = sys.maxsize
        for device_meshes in self.device_meshes_list:
            cost, dist_context = self.layer_placement_pass(
                len(device_meshes), len(self.layers), device_meshes
            )
            print("device_meshes and cost: ", device_meshes, cost)
            if cost <= best_cost:
                print("cost, best_cost: ", cost, best_cost)
                best_cost = cost
                best_dist_context = dist_context
        return best_dist_context

    def tune_o1(self):
        # self.get_sub_programs_cost()
        best_cost = sys.maxsize
        best_dist_context = None

        for device_meshes in self.device_meshes_list:
            pp_stages = len(device_meshes)
            average_layers = len(self.layers) // pp_stages
            device_mesh_shape = device_meshes[0].shape
            if len(device_mesh_shape) == 1:
                device_mesh_shape.insert(0, 1)
            process_mesh_shapes = convert_to_process_meshes(device_mesh_shape)

            # [8], [2, 4] device_meshes里为[[1, 8]]
            # [4] [2, 2] device_meshes里为[[1, 4], [1, 4]]
            # dp模式下的[8]的cost, dp模式下[4], [4]的cost
            for parallelism in ["dp", "mp", "dp_mp", "mp_dp"]:
                for process_mesh_shape in process_mesh_shapes:
                    # total_cost_of_device_meshes = 0
                    # total_memory_of_device_meshes = 0
                    dist_context_of_device_meshes = None
                    for idx, device_mesh in enumerate(device_meshes):
                        device_mesh_shape = device_mesh.shape
                        process_mesh = ProcessMesh(
                            np.array(device_mesh.device_ids)
                            .reshape(process_mesh_shape)
                            .tolist()
                        )
                        selective_parallelisms = (
                            ["dp", "mp"]
                            if len(process_mesh.shape) == 1
                            else ["dp_mp", "mp_dp"]
                        )
                        if parallelism not in selective_parallelisms:
                            total_cost_of_device_meshes = sys.maxsize
                            continue

                        key = self.convert_process_mesh_to_key(process_mesh)

                        if idx == len(device_meshes) - 1:
                            start = idx * average_layers
                            end = len(self.layers)
                        else:
                            start = idx * average_layers
                            end = (idx + 1) * average_layers

                        # total_cost = 0
                        # total_memory = 0

                        # for i in range(start, end):
                        #     memory = self.sub_program_cost[i][parallelism][key]["memory"]
                        #     total_memory += memory
                        #     if total_memory > 32 * 1024 ** 3:
                        #         total_cost = sys.maxsize
                        #         break
                        #     cost = self.sub_program_cost[i][parallelism][key]["cost"]
                        #     total_cost += cost

                        dist_context = self.combine_dist_contexts(
                            [
                                self.sub_programs_dist_context[j][parallelism][
                                    key
                                ]
                                for j in range(start, end)
                            ]
                        )
                        # total_cost_of_device_meshes = total_cost_of_device_meshes + total_cost if total_cost_of_device_meshes + total_cost < sys.maxsize else sys.maxsize
                        # total_memory_of_device_meshes += total_memory
                        dist_context_of_device_meshes = (
                            dist_context
                            if dist_context_of_device_meshes is None
                            else self.combine_dist_contexts(
                                [dist_context_of_device_meshes, dist_context]
                            )
                        )
                    if dist_context_of_device_meshes is not None:
                        cost, memory = self._get_sub_program_cost(
                            dist_context_of_device_meshes
                        )
                        if memory > 32 * 1024**3:
                            cost = sys.maxsize
                        if cost < best_cost:
                            best_cost = cost
                            best_dist_context = dist_context_of_device_meshes
                            # print("o1 program with dist_attr: ")
                            # print_program_with_dist_attr(self.full_main_program, best_dist_context)
                            # print("best_cost o1 change: ", best_cost, parallelism, process_mesh_shape, device_meshes)
                        # else:
                        #     print("best_cost o1: ", best_cost, parallelism, process_mesh_shape, device_meshes)

        return best_dist_context

    def tune(self):
        # prepare
        self.prepare()
        best_dist_context = None
        if self.level == "o2":
            best_dist_context = self.tune_o2()

            # best_dist_context = None
            # best_cost = sys.maxsize
            # for device_meshes in self.device_meshes_list:
            #     cost, dist_context = self.layer_placement_pass(
            #         len(device_meshes), len(self.layers), device_meshes
            #     )
            #     print("device_meshes and cost: ", device_meshes, cost)
            #     if cost <= best_cost:
            #         print("cost, best_cost: ", cost, best_cost)
            #         best_cost = cost
            #         best_dist_context = dist_context

        elif self.level == "o1":
            # If lebel is o1, it means all layers within same parallelism. when in pipeline parallism, it means that place layers evenly.
            use_o2_level = False
            for device_meshes in self.device_meshes_list:
                if len(device_meshes) > 1:
                    shape = None
                    for device_mesh in device_meshes:
                        if shape is None:
                            shape = device_mesh.shape
                            continue
                        else:
                            if shape != device_mesh.shape:
                                print(
                                    "Warning: The o1 level is not be supported when the number of machines is prime numer which greaters than 1. We will use o2 level to tune."
                                )
                                use_o2_level = True
                                break
            if use_o2_level:
                best_dist_context = self.tune_o2()
            else:

                best_dist_context = self.tune_o1()

        assert best_dist_context is not None

        for key in best_dist_context._dist_tensors_for_program:
            if key in self._dist_context._dist_tensors_for_program:
                self._dist_context._dist_tensors_for_program[
                    key
                ] = best_dist_context._dist_tensors_for_program[key]
            # else:
            #     print("rule_based_tuner.py tensor key: ", key, best_dist_context._dist_tensors_for_program[key])

        for key in best_dist_context._dist_ops_for_program:
            if key in self._dist_context._dist_ops_for_program:
                self._dist_context._dist_ops_for_program[
                    key
                ] = best_dist_context._dist_ops_for_program[key]
            # else:
            #     print("rule_based_tuner.py op key: ", key, best_dist_context._dist_ops_for_program[key])

        self._dist_context._process_meshes = best_dist_context._process_meshes
        print(
            "best_dist_context._process_meshes: ",
            best_dist_context._process_meshes,
        )
        print_program_with_dist_attr(self.full_main_program, best_dist_context)
