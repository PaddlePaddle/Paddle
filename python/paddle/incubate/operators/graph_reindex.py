#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid import core
from paddle import _C_ops


def graph_reindex(x, neighbors, count, name=None):
    """
    Graph Reindex API.

    This API is mainly used in Graph Learning domain, which should be used
    in conjunction with `graph_sample_neighbors` API. And the main purpose
    is to reindex the ids information of the input nodes, and return the 
    corresponding graph edges after reindex.

    Take input nodes x = [0, 1, 2] as an example. 
    If we have neighbors = [8, 9, 0, 4, 7, 6, 7], and count = [2, 3, 2], 
    then we know that the neighbors of 0 is [8, 9], the neighbors of 1
    is [0, 4, 7], and the neighbors of 2 is [6, 7].

    Args:
        x (Tensor): The input nodes which we sample neighbors for. The available
                    data type is int32, int64.
        neighbors (Tensor): The neighbors of the input nodes `x`. The data type
                            should be the same with `x`.
        count (Tensor): The neighbor count of the input nodes `x`. And the 
                        data type should be int32.
    
    Returns:
        reindex_src (Tensor): The source node index of graph edges after reindex.
        reindex_dst (Tensor): The destination node index of graph edges after reindex.
        out_nodes (Tensor): The index of unique input nodes and neighbors before reindex,
                            where we put the input nodes `x` in the front, and put neighbor
                            nodes in the back.

    Examples:
        
        .. code-block:: python

        import paddle

        x = [0, 1, 2]
        neighbors = [8, 9, 0, 4, 7, 6, 7]
        count = [2, 3, 2]
        x = paddle.to_tensor(x, dtype="int64")
        neighbors = paddle.to_tensor(neighbors, dtype="int64")
        count = paddle.to_tensor(count, dtype="int32")

        reindex_src, reindex_dst, out_nodes = \
             paddle.incubate.graph_reindex(x, neighbors, count)
        # reindex_src: [3, 4, 0, 5, 6, 7, 6]
        # reindex_dst: [0, 0, 1, 1, 1, 2, 2]
        # out_nodes: [0, 1, 2, 8, 9, 4, 7, 6]
    """

    if in_dygraph_mode():
        print("Enter python funcs: graph_reindex")
        reindex_src, reindex_dst, out_nodes, _, _ = \
            _C_ops.graph_reindex(x, neighbors, count,
                                 paddle.to_tensor([]),
                                 paddle.to_tensor([]),
                                 "flag_buffer_hashtable", False)
        return reindex_src, reindex_dst, out_nodes

    check_variable_and_dtype(x, "X", ("int32", "int64"), "graph_reindex")
    check_variable_and_dtype(neighbors, "Neighbors", ("int32", "int64"),
                             "graph_reindex")
    check_variable_and_dtype(count, "Count", ("int32"), "graph_reindex")

    helper = LayerHelper("graph_reindex", **locals())
    reindex_src = helper.create_variable_for_type_inference(dtype=x.dtype)
    reindex_dst = helper.create_variable_for_type_inference(dtype=x.dtype)
    out_nodes = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="graph_reindex",
        inputs={
            "X": x,
            "Neighbors": neighbors,
            "Count": count,
            "HashTable_Value": None,
            "HashTable_Index": None
        },
        outputs={
            "Reindex_Src": reindex_src,
            "Reindex_Dst": reindex_dst,
            "Out_Nodes": out_nodes
        },
        attrs={"flag_buffer_hashtable": False})
    return reindex_src, reindex_dst, out_nodes
