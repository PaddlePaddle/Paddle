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
from paddle.fluid.framework import _non_static_mode, Variable
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid import core
from paddle import _C_ops, _legacy_C_ops

__all__ = []


def reindex_graph(x,
                  neighbors,
                  count,
                  value_buffer=None,
                  index_buffer=None,
                  name=None):
    """
    Reindex Graph API.

    This API is mainly used in Graph Learning domain, which should be used
    in conjunction with `graph_sample_neighbors` API. And the main purpose
    is to reindex the ids information of the input nodes, and return the
    corresponding graph edges after reindex.

    **Notes**:
        The number in x should be unique, otherwise it would cause potential errors.
    We will reindex all the nodes from 0.

    Take input nodes x = [0, 1, 2] as an example.
    If we have neighbors = [8, 9, 0, 4, 7, 6, 7], and count = [2, 3, 2],
    then we know that the neighbors of 0 is [8, 9], the neighbors of 1
    is [0, 4, 7], and the neighbors of 2 is [6, 7].
    Then after graph_reindex, we will have 3 different outputs:
        1. reindex_src: [3, 4, 0, 5, 6, 7, 6]
        2. reindex_dst: [0, 0, 1, 1, 1, 2, 2]
        3. out_nodes: [0, 1, 2, 8, 9, 4, 7, 6]
    We can see that the numbers in `reindex_src` and `reindex_dst` is the corresponding index
    of nodes in `out_nodes`.

    Args:
        x (Tensor): The input nodes which we sample neighbors for. The available
                    data type is int32, int64.
        neighbors (Tensor): The neighbors of the input nodes `x`. The data type
                            should be the same with `x`.
        count (Tensor): The neighbor count of the input nodes `x`. And the
                        data type should be int32.
        value_buffer (Tensor|None): Value buffer for hashtable. The data type should be int32,
                                    and should be filled with -1. Only useful for gpu version.
        index_buffer (Tensor|None): Index buffer for hashtable. The data type should be int32,
                                    and should be filled with -1. Only useful for gpu version.
                                    `value_buffer` and `index_buffer` should be both not None
                                    if you want to speed up by using hashtable buffer.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

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
             paddle.geometric.reindex_graph(x, neighbors, count)
        # reindex_src: [3, 4, 0, 5, 6, 7, 6]
        # reindex_dst: [0, 0, 1, 1, 1, 2, 2]
        # out_nodes: [0, 1, 2, 8, 9, 4, 7, 6]

    """
    use_buffer_hashtable = True if value_buffer is not None \
                                and index_buffer is not None else False

    if _non_static_mode():
        reindex_src, reindex_dst, out_nodes = \
            _legacy_C_ops.graph_reindex(x, neighbors, count, value_buffer, index_buffer,
                                 "flag_buffer_hashtable", use_buffer_hashtable)
        return reindex_src, reindex_dst, out_nodes

    check_variable_and_dtype(x, "X", ("int32", "int64"), "graph_reindex")
    check_variable_and_dtype(neighbors, "Neighbors", ("int32", "int64"),
                             "graph_reindex")
    check_variable_and_dtype(count, "Count", ("int32"), "graph_reindex")

    if use_buffer_hashtable:
        check_variable_and_dtype(value_buffer, "HashTable_Value", ("int32"),
                                 "graph_reindex")
        check_variable_and_dtype(index_buffer, "HashTable_Index", ("int32"),
                                 "graph_reindex")

    helper = LayerHelper("reindex_graph", **locals())
    reindex_src = helper.create_variable_for_type_inference(dtype=x.dtype)
    reindex_dst = helper.create_variable_for_type_inference(dtype=x.dtype)
    out_nodes = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type="graph_reindex",
                     inputs={
                         "X":
                         x,
                         "Neighbors":
                         neighbors,
                         "Count":
                         count,
                         "HashTable_Value":
                         value_buffer if use_buffer_hashtable else None,
                         "HashTable_Index":
                         index_buffer if use_buffer_hashtable else None,
                     },
                     outputs={
                         "Reindex_Src": reindex_src,
                         "Reindex_Dst": reindex_dst,
                         "Out_Nodes": out_nodes
                     },
                     attrs={"flag_buffer_hashtable": use_buffer_hashtable})
    return reindex_src, reindex_dst, out_nodes


def reindex_heter_graph(x,
                        neighbors,
                        count,
                        value_buffer=None,
                        index_buffer=None,
                        name=None):
    """
    Reindex HeterGraph API.

    This API is mainly used in Graph Learning domain, which should be used
    in conjunction with `graph_sample_neighbors` API. And the main purpose
    is to reindex the ids information of the input nodes, and return the
    corresponding graph edges after reindex.

    **Notes**:
        The number in x should be unique, otherwise it would cause potential errors.
    We support multi-edge-types neighbors reindexing in reindex_heter_graph api.
    We will reindex all the nodes from 0.

    Take input nodes x = [0, 1, 2] as an example.
    For graph A, suppose we have neighbors = [8, 9, 0, 4, 7, 6, 7], and count = [2, 3, 2],
    then we know that the neighbors of 0 is [8, 9], the neighbors of 1
    is [0, 4, 7], and the neighbors of 2 is [6, 7].
    For graph B, suppose we have neighbors = [0, 2, 3, 5, 1], and count = [1, 3, 1],
    then we know that the neighbors of 0 is [0], the neighbors of 1 is [2, 3, 5],
    and the neighbors of 3 is [1].
    We will get following outputs:
        1. reindex_src: [3, 4, 0, 5, 6, 7, 6, 0, 2, 8, 9, 1]
        2. reindex_dst: [0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2]
        3. out_nodes: [0, 1, 2, 8, 9, 4, 7, 6, 3, 5]

    Args:
        x (Tensor): The input nodes which we sample neighbors for. The available
                    data type is int32, int64.
        neighbors (list|tuple): The neighbors of the input nodes `x` from different graphs.
                                The data type should be the same with `x`.
        count (list|tuple): The neighbor counts of the input nodes `x` from different graphs.
                            And the data type should be int32.
        value_buffer (Tensor|None): Value buffer for hashtable. The data type should be int32,
                                    and should be filled with -1. Only useful for gpu version.
        index_buffer (Tensor|None): Index buffer for hashtable. The data type should be int32,
                                    and should be filled with -1. Only useful for gpu version.
                                    `value_buffer` and `index_buffer` should be both not None
                                    if you want to speed up by using hashtable buffer.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

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
        neighbors_a = [8, 9, 0, 4, 7, 6, 7]
        count_a = [2, 3, 2]
        x = paddle.to_tensor(x, dtype="int64")
        neighbors_a = paddle.to_tensor(neighbors_a, dtype="int64")
        count_a = paddle.to_tensor(count_a, dtype="int32")

        neighbors_b = [0, 2, 3, 5, 1]
        count_b = [1, 3, 1]
        neighbors_b = paddle.to_tensor(neighbors_b, dtype="int64")
        count_b = paddle.to_tensor(count_b, dtype="int32")

        neighbors = [neighbors_a, neighbors_b]
        count = [count_a, count_b]
        reindex_src, reindex_dst, out_nodes = \
             paddle.geometric.reindex_heter_graph(x, neighbors, count)
        # reindex_src: [3, 4, 0, 5, 6, 7, 6, 0, 2, 8, 9, 1]
        # reindex_dst: [0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2]
        # out_nodes: [0, 1, 2, 8, 9, 4, 7, 6, 3, 5]

    """
    use_buffer_hashtable = True if value_buffer is not None \
                                and index_buffer is not None else False

    if _non_static_mode():
        neighbors = paddle.concat(neighbors, axis=0)
        count = paddle.concat(count, axis=0)
        reindex_src, reindex_dst, out_nodes = \
            _legacy_C_ops.graph_reindex(x, neighbors, count, value_buffer, index_buffer,
                                 "flag_buffer_hashtable", use_buffer_hashtable)
        return reindex_src, reindex_dst, out_nodes

    if isinstance(neighbors, Variable):
        neighbors = [neighbors]
    if isinstance(count, Variable):
        count = [count]

    neighbors = paddle.concat(neighbors, axis=0)
    count = paddle.concat(count, axis=0)

    check_variable_and_dtype(x, "X", ("int32", "int64"), "heter_graph_reindex")
    check_variable_and_dtype(neighbors, "Neighbors", ("int32", "int64"),
                             "graph_reindex")
    check_variable_and_dtype(count, "Count", ("int32"), "graph_reindex")

    if use_buffer_hashtable:
        check_variable_and_dtype(value_buffer, "HashTable_Value", ("int32"),
                                 "graph_reindex")
        check_variable_and_dtype(index_buffer, "HashTable_Index", ("int32"),
                                 "graph_reindex")

    helper = LayerHelper("reindex_heter_graph", **locals())
    reindex_src = helper.create_variable_for_type_inference(dtype=x.dtype)
    reindex_dst = helper.create_variable_for_type_inference(dtype=x.dtype)
    out_nodes = helper.create_variable_for_type_inference(dtype=x.dtype)
    neighbors = paddle.concat(neighbors, axis=0)
    count = paddle.concat(count, axis=0)
    helper.append_op(type="graph_reindex",
                     inputs={
                         "X":
                         x,
                         "Neighbors":
                         neighbors,
                         "Count":
                         count,
                         "HashTable_Value":
                         value_buffer if use_buffer_hashtable else None,
                         "HashTable_Index":
                         index_buffer if use_buffer_hashtable else None,
                     },
                     outputs={
                         "Reindex_Src": reindex_src,
                         "Reindex_Dst": reindex_dst,
                         "Out_Nodes": out_nodes
                     },
                     attrs={"flag_buffer_hashtable": use_buffer_hashtable})
    return reindex_src, reindex_dst, out_nodes
