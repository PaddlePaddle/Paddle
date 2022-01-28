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


def graph_khop_sampler(row,
                       colptr,
                       input_nodes,
                       sample_sizes,
                       sorted_eids=None,
                       return_eids=False,
                       name=None):
    """
    Graph Khop Sampler API.

    This API is mainly used in Graph Learning domain, and the main purpose is to 
    provide high performance graph khop sampling method with subgraph reindex step.
    For example, we get the CSC(Compressed Sparse Column) format of the input graph
    edges as `row` and `colptr`, so as to covert graph data into a suitable format 
    for sampling. And the `input_nodes` means the nodes we need to sample neighbors,
    and `sample_sizes` means the number of neighbors and number of layers we want
    to sample. 

    **Note**: 
        Currently the API will reindex the output edges after finishing sampling. We
    will add a choice or a new API for whether to reindex the edges in the near future.

    Args:
        row (Tensor): One of the components of the CSC format of the input graph, and 
                      the shape should be [num_edges, 1] or [num_edges]. The available
                      data type is int32, int64.
        colptr (Tensor): One of the components of the CSC format of the input graph,
                         and the shape should be [num_nodes + 1, 1] or [num_nodes]. 
                         The data type should be the same with `row`.
        input_nodes (Tensor): The input nodes we need to sample neighbors for, and the 
                              data type should be the same with `row`.
        sample_sizes (list|tuple): The number of neighbors and number of layers we want
                                   to sample. The data type should be int, and the shape
                                   should only have one dimension.
        sorted_eids (Tensor): The sorted edge ids, should not be None when `return_eids`
                              is True. The shape should be [num_edges, 1], and the data
                              type should be the same with `row`.
        return_eids (bool): Whether to return the id of the sample edges. Default is False.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        edge_src (Tensor): The src index of the output edges, also means the first column of 
                           the edges. The shape is [num_sample_edges, 1] currently.
        edge_dst (Tensor): The dst index of the output edges, also means the second column
                           of the edges. The shape is [num_sample_edges, 1] currently.
        sample_index (Tensor): The original id of the input nodes and sampled neighbor nodes.
        reindex_nodes (Tensor): The reindex id of the input nodes.
        edge_eids (Tensor): Return the id of the sample edges if `return_eids` is True.

    Examples:
        
        .. code-block:: python

        import paddle

        row = [3, 7, 0, 9, 1, 4, 2, 9, 3, 9, 1, 9, 7]
        colptr = [0, 2, 4, 5, 6, 7, 9, 11, 11, 13, 13]
        nodes = [0, 8, 1, 2]
        sample_sizes = [2, 2]
        row = paddle.to_tensor(row, dtype="int64")
        colptr = paddle.to_tensor(colptr, dtype="int64")
        nodes = paddle.to_tensor(nodes, dtype="int64")
        
        edge_src, edge_dst, sample_index, reindex_nodes = \
            paddle.incubate.graph_khop_sampler(row, colptr, nodes, sample_sizes, False)

    """

    if in_dygraph_mode():
        if return_eids:
            if sorted_eids is None:
                raise ValueError(f"`sorted_eid` should not be None "
                                 f"if return_eids is True.")
            edge_src, edge_dst, sample_index, reindex_nodes, edge_eids = \
                _C_ops.graph_khop_sampler(row, sorted_eids,
                                              colptr, input_nodes,
                                              "sample_sizes", sample_sizes,
                                              "return_eids", True)
            return edge_src, edge_dst, sample_index, reindex_nodes, edge_eids
        else:
            edge_src, edge_dst, sample_index, reindex_nodes, _ = \
                _C_ops.graph_khop_sampler(row, None,
                                              colptr, input_nodes,
                                              "sample_sizes", sample_sizes,
                                              "return_eids", False)
            return edge_src, edge_dst, sample_index, reindex_nodes

    check_variable_and_dtype(row, "Row", ("int32", "int64"),
                             "graph_khop_sampler")

    if return_eids:
        if sorted_eids is None:
            raise ValueError(f"`sorted_eid` should not be None "
                             f"if return_eids is True.")
        check_variable_and_dtype(sorted_eids, "Eids", ("int32", "int64"),
                                 "graph_khop_sampler")

    check_variable_and_dtype(colptr, "Col_Ptr", ("int32", "int64"),
                             "graph_khop_sampler")
    check_variable_and_dtype(input_nodes, "X", ("int32", "int64"),
                             "graph_khop_sampler")

    helper = LayerHelper("graph_khop_sampler", **locals())
    edge_src = helper.create_variable_for_type_inference(dtype=row.dtype)
    edge_dst = helper.create_variable_for_type_inference(dtype=row.dtype)
    sample_index = helper.create_variable_for_type_inference(dtype=row.dtype)
    reindex_nodes = helper.create_variable_for_type_inference(dtype=row.dtype)
    edge_eids = helper.create_variable_for_type_inference(dtype=row.dtype)
    helper.append_op(
        type="graph_khop_sampler",
        inputs={
            "Row": row,
            "Eids": sorted_eids,
            "Col_Ptr": colptr,
            "X": input_nodes
        },
        outputs={
            "Out_Src": edge_src,
            "Out_Dst": edge_dst,
            "Sample_Index": sample_index,
            "Reindex_X": reindex_nodes,
            "Out_Eids": edge_eids
        },
        attrs={"sample_sizes": sample_sizes,
               "return_eids": return_eids})
    if return_eids:
        return edge_src, edge_dst, sample_index, reindex_nodes, edge_eids
    else:
        return edge_src, edge_dst, sample_index, reindex_nodes
