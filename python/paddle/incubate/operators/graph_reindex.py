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

from paddle import _C_ops
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode
from paddle.utils import deprecated


@deprecated(
    since="2.4.0",
    update_to="paddle.geometric.reindex_graph",
    level=1,
    reason="paddle.incubate.graph_reindex will be removed in future",
)
def graph_reindex(
    x,
    neighbors,
    count,
    value_buffer=None,
    index_buffer=None,
    flag_buffer_hashtable=False,
    name=None,
):
    """

    Graph Reindex API.

    This API is mainly used in Graph Learning domain, which should be used
    in conjunction with `graph_sample_neighbors` API. And the main purpose
    is to reindex the ids information of the input nodes, and return the
    corresponding graph edges after reindex.

    Notes:
        The number in x should be unique, otherwise it would cause potential errors.
        Besides, we also support multi-edge-types neighbors reindexing. If we have different
        edge_type neighbors for x, we should concatenate all the neighbors and count of x.
        We will reindex all the nodes from 0.

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
        value_buffer (Tensor, optional): Value buffer for hashtable. The data type should
                                    be int32, and should be filled with -1. Default is None.
        index_buffer (Tensor, optional): Index buffer for hashtable. The data type should
                                    be int32, and should be filled with -1. Default is None.
        flag_buffer_hashtable (bool, optional): Whether to use buffer for hashtable to speed up.
                                      Default is False. Only useful for gpu version currently.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        - reindex_src (Tensor), The source node index of graph edges after reindex.
        - reindex_dst (Tensor), The destination node index of graph edges after reindex.
        - out_nodes (Tensor), The index of unique input nodes and neighbors before reindex,
          where we put the input nodes `x` in the front, and put neighbor
          nodes in the back.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> x = [0, 1, 2]
            >>> neighbors_e1 = [8, 9, 0, 4, 7, 6, 7]
            >>> count_e1 = [2, 3, 2]
            >>> x = paddle.to_tensor(x, dtype="int64")
            >>> neighbors_e1 = paddle.to_tensor(neighbors_e1, dtype="int64")
            >>> count_e1 = paddle.to_tensor(count_e1, dtype="int32")

            >>> reindex_src, reindex_dst, out_nodes = paddle.incubate.graph_reindex(
            ...     x,
            ...     neighbors_e1,
            ...     count_e1,
            ... )
            >>> print(reindex_src)
            Tensor(shape=[7], dtype=int64, place=Place(cpu), stop_gradient=True,
            [3, 4, 0, 5, 6, 7, 6])
            >>> print(reindex_dst)
            Tensor(shape=[7], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 0, 1, 1, 1, 2, 2])
            >>> print(out_nodes)
            Tensor(shape=[8], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 1, 2, 8, 9, 4, 7, 6])

            >>> neighbors_e2 = [0, 2, 3, 5, 1]
            >>> count_e2 = [1, 3, 1]
            >>> neighbors_e2 = paddle.to_tensor(neighbors_e2, dtype="int64")
            >>> count_e2 = paddle.to_tensor(count_e2, dtype="int32")

            >>> neighbors = paddle.concat([neighbors_e1, neighbors_e2])
            >>> count = paddle.concat([count_e1, count_e2])
            >>> reindex_src, reindex_dst, out_nodes = paddle.incubate.graph_reindex(x, neighbors, count)
            >>> print(reindex_src)
            Tensor(shape=[12], dtype=int64, place=Place(cpu), stop_gradient=True,
            [3, 4, 0, 5, 6, 7, 6, 0, 2, 8, 9, 1])
            >>> print(reindex_dst)
            Tensor(shape=[12], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 0, 1, 1, 1, 2, 2, 0, 1, 1, 1, 2])
            >>> print(out_nodes)
            Tensor(shape=[10], dtype=int64, place=Place(cpu), stop_gradient=True,
            [0, 1, 2, 8, 9, 4, 7, 6, 3, 5])

    """
    if flag_buffer_hashtable:
        if value_buffer is None or index_buffer is None:
            raise ValueError(
                "`value_buffer` and `index_buffer` should not"
                "be None if `flag_buffer_hashtable` is True."
            )

    if in_dynamic_or_pir_mode():
        reindex_src, reindex_dst, out_nodes = _C_ops.reindex_graph(
            x,
            neighbors,
            count,
            value_buffer,
            index_buffer,
        )
        return reindex_src, reindex_dst, out_nodes

    check_variable_and_dtype(x, "X", ("int32", "int64"), "graph_reindex")
    check_variable_and_dtype(
        neighbors, "Neighbors", ("int32", "int64"), "graph_reindex"
    )
    check_variable_and_dtype(count, "Count", ("int32"), "graph_reindex")

    if flag_buffer_hashtable:
        check_variable_and_dtype(
            value_buffer, "HashTable_Value", ("int32"), "graph_reindex"
        )
        check_variable_and_dtype(
            index_buffer, "HashTable_Index", ("int32"), "graph_reindex"
        )

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
            "HashTable_Value": value_buffer if flag_buffer_hashtable else None,
            "HashTable_Index": index_buffer if flag_buffer_hashtable else None,
        },
        outputs={
            "Reindex_Src": reindex_src,
            "Reindex_Dst": reindex_dst,
            "Out_Nodes": out_nodes,
        },
    )
    return reindex_src, reindex_dst, out_nodes
