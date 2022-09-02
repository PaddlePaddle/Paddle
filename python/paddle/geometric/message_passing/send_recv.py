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
from paddle.fluid.framework import _non_static_mode, _in_legacy_dygraph, in_dygraph_mode
from paddle.fluid.framework import Variable
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype, convert_dtype
from paddle import _C_ops, _legacy_C_ops

from .utils import convert_out_size_to_list, get_out_size_tensor_inputs, reshape_lhs_rhs

__all__ = []


def send_u_recv(x,
                src_index,
                dst_index,
                reduce_op="sum",
                out_size=None,
                name=None):
    """

    Graph Learning message passing api.

    This api is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory 
    consumption in the process of message passing. Take `x` as the input tensor, we first use `src_index`
    to gather the corresponding data, and then use `dst_index` to update the corresponding position of output tensor 
    in different reduce ops, like sum, mean, max, or min. Besides, we can use `out_size` to set necessary output shape.

    .. code-block:: text

           Given:

           x = [[0, 2, 3],
                [1, 4, 5],
                [2, 6, 7]]

           src_index = [0, 1, 2, 0]

           dst_index = [1, 2, 1, 0]

           reduce_op = "sum"

           out_size = None

           Then:

           out = [[0, 2, 3],
                  [2, 8, 10],
                  [1, 4, 5]]

    Args:
        x (Tensor): The input tensor, and the available data type is float32, float64, int32, int64.
                    And we support float16 in gpu version.
        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.
        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`. 
                            The available data type is int32, int64. 
        reduce_op (str): Different reduce ops, including `sum`, `mean`, `max`, `min`.
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
            src_index, dst_index = indexes[:, 0], indexes[:, 1]
            out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum")
            # Outputs: [[0., 2., 3.], [2., 8., 10.], [1., 4., 5.]]

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
            src_index, dst_index = indexes[:, 0], indexes[:, 1]
            out_size = paddle.max(dst_index) + 1
            out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum", out_size=out_size)
            # Outputs: [[0., 2., 3.], [[2., 8., 10.]]]

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
            src_index, dst_index = indexes[:, 0], indexes[:, 1]
            out = paddle.geometric.send_u_recv(x, src_index, dst_index, reduce_op="sum")
            # Outputs: [[0., 2., 3.], [2., 8., 10.], [0., 0., 0.]]

    """

    if reduce_op not in ["sum", "mean", "max", "min"]:
        raise ValueError(
            "reduce_op should be `sum`, `mean`, `max` or `min`, but received %s"
            % reduce_op)

    # TODO(daisiming): Should we add judgement for out_size: max(dst_index) + 1.

    if _in_legacy_dygraph():
        out_size = convert_out_size_to_list(out_size)
        out, tmp = _legacy_C_ops.graph_send_recv(x, src_index, dst_index,
                                                 None, 'reduce_op',
                                                 reduce_op.upper(), 'out_size',
                                                 out_size)
        return out
    if in_dygraph_mode():
        out_size = convert_out_size_to_list(out_size)
        return _C_ops.graph_send_recv(x, src_index, dst_index,
                                      reduce_op.upper(), out_size)

    check_variable_and_dtype(
        x, "X", ("float32", "float64", "int32", "int64", "float16"),
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

    helper = LayerHelper("send_u_recv", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    dst_count = helper.create_variable_for_type_inference(dtype="int32",
                                                          stop_gradient=True)

    inputs = {"X": x, "Src_index": src_index, "Dst_index": dst_index}
    attrs = {"reduce_op": reduce_op.upper()}
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


def send_ue_recv(x,
                 y,
                 src_index,
                 dst_index,
                 message_op="add",
                 reduce_op="sum",
                 out_size=None,
                 name=None):
    """

    Graph Learning message passing api.

    This api is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory 
    consumption in the process of message passing. Take `x` as the input tensor, we first use `src_index`
    to gather the corresponding data, after computing with `y` in different message ops like add/sub/mul/div, then use `dst_index` to 
    update the corresponding position of output tensor in different reduce ops, like sum, mean, max, or min. 
    Besides, we can use `out_size` to set necessary output shape.

    .. code-block:: text

           Given:

           x = [[0, 2, 3],
                [1, 4, 5],
                [2, 6, 7]]

           y = [1, 1, 1]

           src_index = [0, 1, 2, 0]

           dst_index = [1, 2, 1, 0]

           message_op = "add"

           reduce_op = "sum"

           out_size = None

           Then:

           out = [[1, 3, 4],
                  [4, 10, 12],
                  [2, 5, 6]]
    Args:
        x (Tensor): The input node feature tensor, and the available data type is float32, float64, int32, int64.
                    And we support float16 in gpu version.
        y (Tensor): The input edge feature tensor, and the available data type is float32, float64, int32, int64.
                    And we support float16 in gpu version.
        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.
        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`. 
                            The available data type is int32, int64.
        message_op (str): Different message ops for x and e, including `add`, `sub`, `mul`, `div`.
        reduce_op (str): Different reduce ops, including `sum`, `mean`, `max`, `min`.
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
            y = paddle.to_tensor([1, 1, 1, 1], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [1, 2], [2, 1], [0, 0]], dtype="int32")
            src_index, dst_index = indexes[:, 0], indexes[:, 1]
            out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum")
            # Outputs: [[1., 3., 4.], [4., 10., 12.], [2., 5., 6.]]

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            y = paddle.to_tensor([1, 1, 1], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
            src_index, dst_index = indexes[:, 0], indexes[:, 1]
            out_size = paddle.max(dst_index) + 1
            out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum", out_size=out_size)
            # Outputs: [[1., 3., 4.], [[4., 10., 12.]]]

            x = paddle.to_tensor([[0, 2, 3], [1, 4, 5], [2, 6, 7]], dtype="float32")
            y = paddle.to_tensor([1, 1, 1], dtype="float32")
            indexes = paddle.to_tensor([[0, 1], [2, 1], [0, 0]], dtype="int32")
            src_index, dst_index = indexes[:, 0], indexes[:, 1]
            out = paddle.geometric.send_ue_recv(x, y, src_index, dst_index, message_op="add", reduce_op="sum")
            # Outputs: [[1., 3., 4.], [4., 10., 12.], [0., 0., 0.]]

    """

    if message_op not in ["add", "sub", "mul", "div"]:
        raise ValueError(
            "message_op should be `add`, `sub`, `mul`, `div`, but received %s" %
            message_op)

    if reduce_op not in ["sum", "mean", "max", "min"]:
        raise ValueError(
            "reduce_op should be `sum`, `mean`, `max` or `min`, but received %s"
            % reduce_op)

    x, y = reshape_lhs_rhs(x, y)

    if message_op == 'sub':
        message_op = 'add'
        y = -y
    if message_op == "div":
        message_op = 'mul'
        y = 1. / (y + 1e-12)

    # TODO(daisiming): Should we add judgement for out_size: max(dst_index) + 1.

    if _in_legacy_dygraph():
        out_size = convert_out_size_to_list(out_size)
        out, tmp = _legacy_C_ops.graph_send_ue_recv(x, y, src_index, dst_index,
                                                    None, 'message_op',
                                                    message_op.upper(),
                                                    'reduce_op',
                                                    reduce_op.upper(),
                                                    'out_size', out_size)
        return out
    if in_dygraph_mode():
        out_size = convert_out_size_to_list(out_size)
        return _C_ops.graph_send_ue_recv(x, y, src_index, dst_index,
                                         message_op.upper(), reduce_op.upper(),
                                         out_size)

    check_variable_and_dtype(
        x, "X", ("float32", "float64", "int32", "int64", "float16"),
        "graph_send_ue_recv")
    check_variable_and_dtype(
        y, "Y", ("float32", "float64", "int32", "int64", "float16"),
        "graph_send_ue_recv")
    check_variable_and_dtype(src_index, "Src_index", ("int32", "int64"),
                             "graph_send_ue_recv")
    check_variable_and_dtype(dst_index, "Dst_index", ("int32", "int64"),
                             "graph_send_ue_recv")
    if out_size:
        check_type(out_size, 'out_size', (int, np.int32, np.int64, Variable),
                   'graph_send_ue_recv')
    if isinstance(out_size, Variable):
        check_dtype(out_size.dtype, 'out_size', ['int32', 'int64'],
                    'graph_send_ue_recv')

    helper = LayerHelper("send_ue_recv", **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    dst_count = helper.create_variable_for_type_inference(dtype="int32",
                                                          stop_gradient=True)

    inputs = {"X": x, "Y": y, "Src_index": src_index, "Dst_index": dst_index}
    attrs = {"message_op": message_op.upper(), "reduce_op": reduce_op.upper()}
    get_out_size_tensor_inputs(inputs=inputs,
                               attrs=attrs,
                               out_size=out_size,
                               op_type='graph_send_ue_recv')

    helper.append_op(type="graph_send_ue_recv",
                     inputs=inputs,
                     outputs={
                         "Out": out,
                         "Dst_count": dst_count
                     },
                     attrs=attrs)
    return out


def send_uv(x, y, src_index, dst_index, message_op="add", name=None):
    """

    Graph Learning message passing api.

    This api is mainly used in Graph Learning domain, and the main purpose is to reduce intermediate memory 
    consumption in the process of message passing. Take `x` as the source node feature tensor, take `y` as 
    the destination node feature tensor. Then we use `src_index` and `dst_index` to gather the corresponding data,
    and then compute the edge features in different message_ops like `add`, `sub`, `mul`, `div`.

    .. code-block:: text

           Given:

           x = [[0, 2, 3],
                [1, 4, 5],
                [2, 6, 7]]

           y = [[0, 1, 2],
                [2, 3, 4],
                [4, 5, 6]]

           src_index = [0, 1, 2, 0]

           dst_index = [1, 2, 1, 0]

           message_op = "add"

           Then:

           out = [[2, 5, 7],
                  [5, 9, 11],
                  [4, 9, 11],
                  [0, 3, 5]]

    Args:
        x (Tensor): The source node feature tensor, and the available data type is float32, float64, int32, int64. And we support float16 in gpu version.
        y (Tensor): The destination node feature tensor, and the available data type is float32, float64, int32, int64. And we support float16 in gpu version.
        src_index (Tensor): An 1-D tensor, and the available data type is int32, int64.
        dst_index (Tensor): An 1-D tensor, and should have the same shape as `src_index`. 
                            The available data type is int32, int64. 
        message_op (str): Different message ops for x and y, including `add`, `sub`, `mul` and `div`.
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
            out = paddle.geometric.send_uv(x, y, src_index, dst_index, message_op="add")
            # Outputs: [[2., 5., 7.], [5., 9., 11.], [4., 9., 11.], [0., 3., 5.]]

    """

    if message_op not in ['add', 'sub', 'mul', 'div']:
        raise ValueError(
            "message_op should be `add`, `sub`, `mul`, `div`, but received %s" %
            message_op)

    x, y = reshape_lhs_rhs(x, y)

    if message_op == 'sub':
        message_op = 'add'
        y = -y
    if message_op == 'div':
        message_op = 'mul'
        y = 1. / (y + 1e-12)

    if in_dygraph_mode():
        return _C_ops.graph_send_uv(x, y, src_index, dst_index,
                                    message_op.upper())
    else:
        if _in_legacy_dygraph():
            return _legacy_C_ops.graph_send_uv(x, y, src_index, dst_index,
                                               "message_op", message_op.upper())
        else:
            helper = LayerHelper("send_uv", **locals())
            check_variable_and_dtype(
                x, 'x', ['int32', 'int64', 'float32', 'float64', 'float16'],
                'graph_send_uv')
            check_variable_and_dtype(
                y, 'y', ['int32', 'int64', 'float32', 'float64', 'float16'],
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
            attrs = {'message_op': message_op.upper()}
            helper.append_op(type="graph_send_uv",
                             inputs=inputs,
                             attrs=attrs,
                             outputs={"out": out})
            return out
