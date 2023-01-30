#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
=======
from __future__ import print_function

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import argparse
import json
import logging
from collections import defaultdict

import paddle.fluid.core as core
import paddle.fluid.proto.framework_pb2 as framework_pb2
from paddle.fluid.log_helper import get_logger

logger = get_logger(__name__, logging.INFO)

try:
    from .graphviz import Graph
except ImportError:
    logger.info(
        'Cannot import graphviz, which is required for drawing a network. This '
        'can usually be installed in python with "pip install graphviz". Also, '
        'pydot requires graphviz to convert dot files to pdf: in ubuntu, this '
<<<<<<< HEAD
        'can usually be installed with "sudo apt-get install graphviz".'
    )
    print(
        'net_drawer will not run correctly. Please install the correct '
        'dependencies.'
    )
=======
        'can usually be installed with "sudo apt-get install graphviz".')
    print('net_drawer will not run correctly. Please install the correct '
          'dependencies.')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    exit(0)

OP_STYLE = {
    'shape': 'oval',
    'color': '#0F9D58',
    'style': 'filled',
<<<<<<< HEAD
    'fontcolor': '#FFFFFF',
=======
    'fontcolor': '#FFFFFF'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}

VAR_STYLE = {}

GRAPH_STYLE = {
    "rankdir": "TB",
}

GRAPH_ID = 0


def unique_id():
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def generator():
        GRAPH_ID += 1
        return GRAPH_ID

    return generator


def draw_node(op):
    node = OP_STYLE
    node["name"] = op.type
    node["label"] = op.type
    return node


def draw_edge(var_parent, op, var, arg):
    edge = VAR_STYLE
    edge["label"] = "%s(%s)" % (var.parameter, arg)
    edge["head_name"] = op.type
    edge["tail_name"] = var_parent[arg]
    return edge


def parse_graph(program, graph, var_dict, **kwargs):

    # fill the known variables
    for block in program.blocks:
        for var in block.vars:
            if var not in var_dict:
                var_dict[var] = "Feed"

    temp_id = 0
    proto = framework_pb2.ProgramDesc.FromString(
<<<<<<< HEAD
        program.desc.serialize_to_string()
    )
=======
        program.desc.serialize_to_string())
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    for block in proto.blocks:
        for op in block.ops:
            op.type = op.type + "_" + str(temp_id)
            temp_id += 1
            graph.node(**draw_node(op))
            for o in op.outputs:
                for arg in o.arguments:
                    var_dict[arg] = op.type
            for e in op.inputs:
                for arg in e.arguments:
                    if arg in var_dict:
                        graph.edge(**draw_edge(var_dict, op, e, arg))
        break  # only plot the first block


def draw_graph(startup_program, main_program, **kwargs):
    if "graph_attr" in kwargs:
        GRAPH_STYLE.update(kwargs["graph_attr"])
    if "node_attr" in kwargs:
        OP_STYLE.update(kwargs["node_attr"])
    if "edge_attr" in kwargs:
        VAR_STYLE.update(kwargs["edge_attr"])

    graph_id = unique_id()
    filename = kwargs.get("filename")
<<<<<<< HEAD
    if filename is None:
        filename = str(graph_id) + ".gv"
    g = Graph(
        name=str(graph_id),
        filename=filename,
        graph_attr=GRAPH_STYLE,
        node_attr=OP_STYLE,
        edge_attr=VAR_STYLE,
        **kwargs
    )
=======
    if filename == None:
        filename = str(graph_id) + ".gv"
    g = Graph(name=str(graph_id),
              filename=filename,
              graph_attr=GRAPH_STYLE,
              node_attr=OP_STYLE,
              edge_attr=VAR_STYLE,
              **kwargs)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    var_dict = {}
    parse_graph(startup_program, g, var_dict)
    parse_graph(main_program, g, var_dict)

<<<<<<< HEAD
    if filename is not None:
=======
    if filename != None:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        g.save()
    return g
