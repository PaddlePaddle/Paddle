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

import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _in_legacy_dygraph, in_dygraph_mode
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle import _C_ops

from .utils import reshape_lhs_rhs


def send_uv(x, y, src_index, dst_index, compute_type="add", name=None):
    """

    Graph Learning message passing api.

    This api is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory 
    consumption in the process of message passing. Take `x` as the source node feature tensor, take `y` as 
    the destination node feature tensor. Then we use `src_index` and `dst_index` to gather the corresponding data,
    and then compute the edge features in different compute_types like `add`, `sub`, `mul`, `div`.

    .. code-block:: text

           Given:

           X = [[0, 2, 3],
                [1, 4, 5],
                [2, 6, 7]]

           Y = [[0, 1, 2],
                [2, 3, 4],
                [4, 5, 6]]

           src_index = [0, 1, 2, 0]

           dst_index = [1, 2, 1, 0]

           compute_type = "add"

           Then:

           Out = [[2, 5, 7],
                  [5, 9, 11],
                  [4, 9, 11],
                  [0, 3, 5]]

    Args:
        x (Tensor): The source node feature tensor, and the available data type is float32, float64, int32, int64. And we support float16 in gpu version.
        y (Tensor): The destination node feature tensor, and the available data type is float32, float64, int32, int64. And we support float16 in gpu version.
        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.
        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`. 
                            The available data type is int32, int64. 
        compute_type (Tensor): Different compute types for x and y, including `add`, `sub`, `mul` and `div`.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The output tensor.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            y = paddle.to_tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
            src_index = indexes[:, 0]
            dst_index = indexes[:, 1]
            out = paddle.geometric.send_uv(x, y, src_index, dst_index, compute_type="add")
            # Outputs: [[2., 5., 7.], [5., 9., 11.], [4., 9., 11.], [0., 3., 5.]]

    """

    if compute_type not in ['add', 'sub', 'mul', 'div']:
        raise ValueError(
            "compute_type should be `add`, `sub`, `mul`, `div`, but received %s"
            % compute_type)

    x, y = reshape_lhs_rhs(x, y)

    if compute_type == 'sub':
        compute_type = 'add'
        y = -y
    if compute_type == 'div':
        compute_type = 'mul'
        y = 1. / y

    if in_dygraph_mode():
        return _C_ops.final_state_graph_send_uv(x, y, src_index, dst_index,
                                                compute_type.upper())
    else:
        if _in_legacy_dygraph():
            return _C_ops.graph_send_uv(x, y, src_index, dst_index,
                                        "compute_type", compute_type.upper())
        else:
            helper = LayerHelper("send_uv", **locals())
            check_variable_and_dtype(x, 'x',
                                     ['int32', 'int64', 'float32', 'float64'],
                                     'graph_send_uv')
            check_variable_and_dtype(y, 'y',
                                     ['int32', 'int64', 'float32', 'float64'],
                                     'graph_send_uv')
            check_variable_and_dtype(src_index, 'src_index', ['int32', 'int64'],
                                     'graph_send_uv')
            check_variable_and_dtype(dst_index, 'dst_index', ['int32', 'int64'],
                                     'graph_send_uv')
            out = helper.create_variable_for_type_inference(dtype=x.dtype)

            inputs = {
                'x': x,
                'y': y,
                'src_index': src_index,
                'dst_index': dst_index
            }
            attrs = {'compute_type': compute_type.upper()}
            helper.append_op(type="graph_send_uv",
                             inputs=inputs,
                             attrs=attrs,
                             outputs={"out": out})
            return out
