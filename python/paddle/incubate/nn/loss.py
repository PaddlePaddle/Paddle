# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING

from paddle import _C_ops
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_or_pir_mode

if TYPE_CHECKING:
    from typing import Literal, TypeAlias, Union

    from paddle import Tensor

    _ReduceModeStringLiteral: TypeAlias = Literal['mean', 'sum', 'none']
    _ReduceModeNumberLiteral: TypeAlias = Literal[0, 1, 2]
    _ReduceMode: TypeAlias = Union[
        _ReduceModeStringLiteral, _ReduceModeNumberLiteral
    ]


def identity_loss(x: Tensor, reduction: _ReduceMode = "none") -> Tensor:
    r"""Marks a tensor as being part of the loss calculation for IPU.

    This operator is used to handle on the (final) loss of a model so that
    it is used as the start of backpropagation.

    When `reduction` is `none`, return raw `Out`.

    When `reduction` is `mean`, return

    .. math::
        Out = MEAN(Out)

    When `reduction` is `sum`, return

    .. math::
        Out = SUM(Out)

    Parameters:
        x (Variable): The input tensor. The shapes is [N, *], where N is batch size and `*` means any number of
             additional dimensions. It's data type should be float32, float64 on CPU and float16, float32 on IPU.
        reduction(str|int, optional): Reduce the loss output. Supported string values are: 'sum', 'mean', 'none'
                            the corresponding int values are 0, 1, 2 respectively. The default value is "none".

    Returns:
        Variable: The loss ``Tensor`` with the specified reduction applied.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> paddle.enable_static()
            >>> loss = paddle.static.data(name="loss", shape=[-1, 1], dtype="float32")
            >>> out = paddle.incubate.identity_loss(loss, reduction=1)
    """
    if isinstance(reduction, str):
        reduction = {"sum": 0, "mean": 1, "none": 2}.get(reduction.lower())
        if reduction is None:
            raise Exception("Unsupported reduction type.")

    if in_dynamic_or_pir_mode():
        return _C_ops.identity_loss(x, reduction)

    check_variable_and_dtype(x, 'x', ['float32', 'float64'], "identity_loss")
    attrs = {'reduction': reduction}
    helper = LayerHelper('identity_loss', **locals())
    dtype = helper.input_dtype(input_param_name='x')
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="identity_loss", inputs={"X": x}, outputs={"Out": out}, attrs=attrs
    )
    return out
