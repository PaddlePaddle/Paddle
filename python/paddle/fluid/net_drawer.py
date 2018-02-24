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

import argparse
import json
import logging
from collections import defaultdict

import paddle.fluid.core as core
import paddle.fluid.proto.framework_pb2 as framework_pb2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from graphviz import Digraph
except ImportError:
    logger.info(
        'Cannot import graphviz, which is required for drawing a network. This '
        'can usually be installed in python with "pip install graphviz". Also, '
        'pydot requires graphviz to convert dot files to pdf: in ubuntu, this '
        'can usually be installed with "sudo apt-get install graphviz".')
    print('net_drawer will not run correctly. Please install the correct '
          'dependencies.')
    exit(0)

OP_STYLE = {
    'shape': 'oval',
    'color': '#0F9D58',
    'style': 'filled',
    'fontcolor': '#FFFFFF'
}

VAR_STYLE = {}

GRAPH_STYLE = {"rankdir": "TB", }

GRAPH_ID = 0


def unique_id():
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
            if not var_dict.has_key(var):
                var_dict[var] = "Feed"

    temp_id = 0
    proto = framework_pb2.ProgramDesc.FromString(
        program.desc.serialize_to_string())
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
                    if var_dict.has_key(arg):
                        graph.edge(**draw_edge(var_dict, op, e, arg))
        break  # only plot the first block


def draw_graph(startup_program, main_program, **kwargs):
    if kwargs.has_key("graph_attr"):
        GRAPH_STYLE.update(kwargs[graph_attr])
    if kwargs.has_key("node_attr"):
        OP_STYLE.update(kwargs[node_attr])
    if kwargs.has_key("edge_attr"):
        VAR_STYLE.update(kwargs[edge_attr])

    graph_id = unique_id()
    filename = kwargs.get("filename")
    if filename == None:
        filename = str(graph_id) + ".gv"
    g = Digraph(
        name=str(graph_id),
        filename=filename,
        graph_attr=GRAPH_STYLE,
        node_attr=OP_STYLE,
        edge_attr=VAR_STYLE,
        **kwargs)

    var_dict = {}
    parse_graph(startup_program, g, var_dict)
    parse_graph(main_program, g, var_dict)

    if filename != None:
        g.save()
    return g
