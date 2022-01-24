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


def graph_sample_neighbors(sorted_src,
                           dst_cumsum_counts,
                           nodes,
                           sample_sizes,
                           sorted_eids=None,
                           return_eids=False,
                           name=None):
    """
    Graph Sample Neighbors operator.

    This operator is mainly used Graph Learning domain. 
    """

    if in_dygraph_mode():
        if return_eids:
            if sorted_eids is None:
                raise ValueError(f"`sorted_eid` should not be None "
                                 f"if return_eids is True.")
            edge_src, edge_dst, sample_index, reindex_nodes, edge_eids = \
                _C_ops.graph_sample_neighbors(sorted_src, sorted_eids,
                                              dst_cumsum_counts, nodes,
                                              "sample_sizes", sample_sizes,
                                              "return_eids", True)
            return edge_src, edge_dst, sample_index, reindex_nodes, edge_eids
        else:
            edge_src, edge_dst, sample_index, reindex_nodes, _ = \
                _C_ops.graph_sample_neighbors(sorted_src, None,
                                              dst_cumsum_counts, nodes,
                                              "sample_sizes", sample_sizes,
                                              "return_eids", False)
            return edge_src, edge_dst, sample_index, reindex_nodes

    check_variable_and_dtype(sorted_src, "Src", ("int32", "int64"),
                             "graph_sample_neighbors")

    if return_eids:
        if sorted_eids is None:
            raise ValueError(f"`sorted_eid` should not be None "
                             f"if return_eids is True.")
        check_variable_and_dtype(sorted_eids, "Src_Eids", ("int32", "int64"),
                                 "graph_sample_neighbors")

    check_variable_and_dtype(dst_cumsum_counts, "Dst_Count", ("int32", "int64"),
                             "graph_sample_neighbors")
    check_variable_and_dtype(nodes, "X", ("int32", "int64"),
                             "graph_sample_neighbors")

    helper = LayerHelper("graph_sample_neighbors", **locals())
    edge_src = helper.create_variable_for_type_inference(dtype=sorted_src.dtype)
    edge_dst = helper.create_variable_for_type_inference(dtype=sorted_src.dtype)
    sample_index = helper.create_variable_for_type_inference(
        dtype=sorted_src.dtype)
    reindex_nodes = helper.create_variable_for_type_inference(
        dtype=sorted_src.dtype)
    edge_eids = helper.create_variable_for_type_inference(
        dtype=sorted_src.dtype)
    helper.append_op(
        type="graph_sample_neighbors",
        inputs={
            "Src": sorted_src,
            "Src_Eids": sorted_eids,
            "Dst_Count": dst_cumsum_counts,
            "X": nodes
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
