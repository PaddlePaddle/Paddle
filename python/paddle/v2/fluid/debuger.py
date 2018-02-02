#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import re
from graphviz import GraphPreviewGenerator
import proto.framework_pb2 as framework_pb2


def draw_block_graphviz(block, highlights=None, path="./temp.dot"):
    '''
    Generate a debug graph for block.
    Args:
        block(Block): a block.
    '''
    graph = GraphPreviewGenerator("some graph")
    # collect parameters and args
    protostr = block.desc.serialize_to_string()
    desc = framework_pb2.BlockDesc.FromString(str(protostr))

    def need_highlight(name):
        if highlights is None: return False
        for pattern in highlights:
            assert type(pattern) is str
            if re.match(pattern, name):
                return True
        return False

    # draw parameters and args
    vars = {}
    for var in desc.vars:
        shape = [str(i) for i in var.lod_tensor.tensor.dims]
        if not shape:
            shape = ['null']
        # create var
        if var.persistable:
            varn = graph.add_param(
                var.name, var.type, shape, highlight=need_highlight(var.name))
        else:
            varn = graph.add_arg(var.name, highlight=need_highlight(var.name))
        vars[var.name] = varn

    def add_op_link_var(op, var, op2var=False):
        for arg in var.arguments:
            if arg not in vars:
                # add missing variables as argument
                vars[arg] = graph.add_arg(arg, highlight=need_highlight(arg))
            varn = vars[arg]
            highlight = need_highlight(op.description) or need_highlight(
                varn.description)
            if op2var:
                graph.add_edge(op, varn, highlight=highlight)
            else:
                graph.add_edge(varn, op, highlight=highlight)

    for op in desc.ops:
        opn = graph.add_op(op.type, highlight=need_highlight(op.type))
        for var in op.inputs:
            add_op_link_var(opn, var, False)
        for var in op.outputs:
            add_op_link_var(opn, var, True)

    graph(path, show=True)
