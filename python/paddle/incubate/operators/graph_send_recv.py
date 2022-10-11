#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.framework import _non_static_mode, _in_legacy_dygraph, in_dygraph_mode
from paddle.fluid.framework import Variable
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from paddle.fluid.layers.tensor import cast
from paddle import _C_ops, _legacy_C_ops
import paddle.utils.deprecated as deprecated


@deprecated(
    since="2.4.0",
    update_to="paddle.geometric.send_u_recv",
    level=1,
    reason="graph_send_recv in paddle.incubate will be removed in future")
def graph_send_recv(x,
                    src_index,
                    dst_index,
                    pool_type="sum",
                    out_size=None,
                    name=None):
    r"""

    Graph Learning Send_Recv combine operator.

    This operator is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory
    consumption in the process of message passing. Take `x` as the input tensor, we first use `src_index`
    to gather the corresponding data, and then use `dst_index` to update the corresponding position of output tensor
    in different pooling types, like sum, mean, max, or min. Besides, we can set `out_size` to get necessary output shape.

    .. code-block:: text

           Given:

           X = [[0, 2, 3],
                [1, 4, 5],
                [2, 6, 7]]

           src_index = [0, 1, 2, 0]

           dst_index = [1, 2, 1, 0]

           pool_type = "sum"

           out_size = None

           Then:

           Out = [[0, 2, 3],
                  [2, 8, 10],
                  [1, 4, 5]]

    Args:
        x (Tensor): The input tensor, and the available data type is float32, float64, int32, int64.
        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.
        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`.
                            The available data type is int32, int64.
        pool_type (str): The pooling types of graph_send_recv, including `sum`, `mean`, `max`, `min`.
                         Default value is `sum`.
        out_size (int|Tensor|None): We can set `out_size` to get necessary output shape. If not set or
                                    out_size is smaller or equal to 0, then this input will not be used.
                                    Otherwise, `out_size` should be equal with or larger than
                                    max(dst_index) + 1.
        name (str, optional): Name for the operation (optional, default is None).
                              For more information, please refer to :ref:`api_guide_Name`.

    Returns:
        out (Tensor): The output tensor, should have the same shape and same dtype as input tensor `x`.
                      If `out_size` is set correctly, then it should have the same shape as `x` except
                      the 0th dimension.

    Examples:

        .. code-block:: python

            import paddle

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
            src_index = indexes[:, 0]
            dst_index = indexes[:, 1]
            out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum")
            # Outputs: [[0., 2., 3.], [2., 8., 10.], [1., 4., 5.]]

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
            src_index = indexes[:, 0]
            dst_index = indexes[:, 1]
            out_size = paddle.max(dst_index) + 1
            out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum", out_size=out_size)
            # Outputs: [[0., 2., 3.], [[2., 8., 10.]]]

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
            src_index = indexes[:, 0]
            dst_index = indexes[:, 1]
            out = paddle.incubate.graph_send_recv(x, src_index, dst_index, pool_type="sum")
            # Outputs: [[0., 2., 3.], [2., 8., 10.], [0., 0., 0.]]

    """

    if pool_type not in ["sum", "mean", "max", "min"]:
        raise ValueError(
            "pool_type should be `sum`, `mean`, `max` or `min`, but received %s"
            % pool_type)

    # TODO(daisiming): Should we add judgement for out_size: max(dst_index) + 1.

    if _in_legacy_dygraph():
        out_size = convert_out_size_to_list(out_size)
        out, tmp = _legacy_C_ops.graph_send_recv(x, src_index, dst_index,
                                                 None, 'reduce_op',
                                                 pool_type.upper(), 'out_size',
                                                 out_size)
        return out
    if in_dygraph_mode():
        out_size = convert_out_size_to_list(out_size)
        return _C_ops.graph_send_recv(x, src_index, dst_index,
                                      pool_type.upper(), out_size)

    check_variable_and_dtype(x, "X", ("float32", "float64", "int32", "int64"),
                             "graph_send_recv")
    check_variable_and_dtype(src_index, "Src_index", ("int32", "int64"),
                             "graph_send_recv")
    check_variable_and_dtype(dst_index, "Dst_index", ("int32", "int64"),
                             "graph_send_recv")
    if out_size:
        check_type(out_size, 'out_size', (int, np.int32, np.int64, Variable),
                   'graph_send_recv')
    if isinstance(out_size, Variable):
        check_dtype(out_size.dtype, 'out_size', ['int32', 'int64'],
                    'graph_send_recv')

    helper = LayerHelper("graph_send_recv", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    dst_count = helper.create_variable_for_type_inference(dtype="int32",
                                                          stop_gradient=True)

    inputs = {"X": x, "Src_index": src_index, "Dst_index": dst_index}
    attrs = {"reduce_op": pool_type.upper()}
    get_out_size_tensor_inputs(inputs=inputs,
                               attrs=attrs,
                               out_size=out_size,
                               op_type='graph_send_recv')

    helper.append_op(type="graph_send_recv",
                     inputs=inputs,
                     outputs={
                         "Out": out,
                         "Dst_count": dst_count
                     },
                     attrs=attrs)
    return out


def convert_out_size_to_list(out_size):
    """
    Convert out_size(int, np.int32, np.int64, Variable) to list
    in imperative mode.
    """
    if out_size is None:
        out_size = [0]
    elif isinstance(out_size, (int, np.int32, np.int64)):
        out_size = [out_size]
    else:
        out_size = [out_size.numpy().astype(int)[0]]
    return out_size


def get_out_size_tensor_inputs(inputs, attrs, out_size, op_type):
    """
    Convert out_size(int, np.int32, np.int64, Variable) to inputs
    and attrs in static mode.
    """
    if out_size is None:
        attrs['out_size'] = [0]
    elif isinstance(out_size, (int, np.int32, np.int64)):
        attrs['out_size'] = [out_size]
    elif isinstance(out_size, Variable):
        out_size.stop_gradient = True
        check_dtype(out_size.dtype, 'out_size', ['int32', 'int64'], op_type,
                    '(When type of out_size in' + op_type + ' is Variable.)')
        if (convert_dtype(out_size.dtype) == 'int64'):
            out_size = cast(out_size, 'int32')
        inputs["Out_size"] = out_size
    else:
        raise TypeError("Out_size only supports Variable or int.")
