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

import numpy as np
from functools import partial, reduce
import paddle
from paddle.utils import deprecated
from . import nn
from .layer_function_generator import templatedoc
from ..layer_helper import LayerHelper
from ..framework import (
    Variable,
    _non_static_mode,
    static_only,
    _in_legacy_dygraph,
    in_dygraph_mode,
)
from .. import core
from ..data_feeder import check_variable_and_dtype, check_type
from ..param_attr import ParamAttr
from ..initializer import NumpyArrayInitializer, Constant
from .. import core
import warnings
from paddle import _C_ops, _legacy_C_ops

__all__ = [
    'square_error_cost',
    'softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
]

kIgnoreIndex = -100


def square_error_cost(input, label):
    r"""

    Accept input predictions and target label and returns the
    squared error cost.

    For predictions label, and target label, the equation is:

    .. math::

        Out = (input - label)^2

    Parameters:
        input (Tensor): Input tensor, the data type should be float32.
        label (Tensor): Label tensor, the data type should be float32.

    Returns:
        Tensor, The tensor storing the element-wise squared
        error difference between input and label.

    Examples:

        .. code-block:: python

            import paddle
            input = paddle.to_tensor([1.1, 1.9])
            label = paddle.to_tensor([1.0, 2.0])
            output = paddle.nn.functional.square_error_cost(input, label)
            print(output)
            # [0.01, 0.01]

    """
    return paddle.nn.functional.square_error_cost(input, label)


def softmax_with_cross_entropy(
    logits,
    label,
    soft_label=False,
    ignore_index=kIgnoreIndex,
    numeric_stable_mode=True,
    return_softmax=False,
    axis=-1,
):
    r"""

    This operator implements the cross entropy loss function with softmax. This function
    combines the calculation of the softmax operation and the cross entropy loss function
    to provide a more numerically stable gradient.

    Because this operator performs a softmax on logits internally, it expects
    unscaled logits. This operator should not be used with the output of
    softmax operator since that would produce incorrect results.

    When the attribute :attr:`soft_label` is set :attr:`False`, this operators
    expects mutually exclusive hard labels, each sample in a batch is in exactly
    one class with a probability of 1.0. Each sample in the batch will have a
    single label.

    The equation is as follows:

    1) Hard label (one-hot label, so every sample has exactly one class)

    .. math::

        loss_j =  -\\text{logits}_{label_j} +
        \\log\\left(\\sum_{i=0}^{K}\\exp(\\text{logits}_i)\\right), j = 1,..., K

    2) Soft label (each sample can have a distribution over all classes)

    .. math::

        loss_j =  -\\sum_{i=0}^{K}\\text{label}_i
        \\left(\\text{logits}_i - \\log\\left(\\sum_{i=0}^{K}
        \\exp(\\text{logits}_i)\\right)\\right), j = 1,...,K

    3) If :attr:`numeric_stable_mode` is :attr:`True`, softmax is calculated first by:

    .. math::

        max_j &= \\max_{i=0}^{K}{\\text{logits}_i}

        log\\_max\\_sum_j &= \\log\\sum_{i=0}^{K}\\exp(logits_i - max_j)

        softmax_j &= \\exp(logits_j - max_j - {log\\_max\\_sum}_j)

    and then cross entropy loss is calculated by softmax and label.

    Args:
        logits (Tensor): A multi-dimension ``Tensor`` , and the data type is float32 or float64. The input tensor of unscaled log probabilities.
        label (Tensor): The ground truth  ``Tensor`` , data type is the same
            as the ``logits`` . If :attr:`soft_label` is set to :attr:`True`,
            Label is a ``Tensor``  in the same shape with :attr:`logits`.
            If :attr:`soft_label` is set to :attr:`True`, Label is a ``Tensor``
            in the same shape with :attr:`logits` expect shape in dimension :attr:`axis` as 1.
        soft_label (bool, optional): A flag to indicate whether to interpretant the given
            labels as soft labels. Default False.
        ignore_index (int, optional): Specifies a target value that is ignored and does
                                      not contribute to the input gradient. Only valid
                                      if :attr:`soft_label` is set to :attr:`False`.
                                      Default: kIgnoreIndex(-100).
        numeric_stable_mode (bool, optional): A flag to indicate whether to use a more
                                              numerically stable algorithm. Only valid
                                              when :attr:`soft_label` is :attr:`False`
                                              and GPU is used. When :attr:`soft_label`
                                              is :attr:`True` or CPU is used, the
                                              algorithm is always numerically stable.
                                              Note that the speed may be slower when use
                                              stable algorithm. Default: True.
        return_softmax (bool, optional): A flag indicating whether to return the softmax
                                         along with the cross entropy loss. Default: False.
        axis (int, optional): The index of dimension to perform softmax calculations. It
                              should be in range :math:`[-1, rank - 1]`, while :math:`rank`
                              is the rank of input :attr:`logits`. Default: -1.

    Returns:
        ``Tensor`` or Tuple of two ``Tensor`` : Return the cross entropy loss if \
                                                    `return_softmax` is False, otherwise the tuple \
                                                    (loss, softmax), softmax is in the same shape \
                                                    with input logits and cross entropy loss is in \
                                                    the same shape with input logits except shape \
                                                    in dimension :attr:`axis` as 1.

    Examples:
        .. code-block:: python

            import paddle
            import numpy as np

            data = np.random.rand(128).astype("float32")
            label = np.random.rand(1).astype("int64")
            data = paddle.to_tensor(data)
            label = paddle.to_tensor(label)
            linear = paddle.nn.Linear(128, 100)
            x = linear(data)
            out = paddle.nn.functional.softmax_with_cross_entropy(logits=x, label=label)
            print(out)
    """
    return paddle.nn.functional.loss.fluid_softmax_with_cross_entropy(
        logits,
        label,
        soft_label,
        ignore_index,
        numeric_stable_mode,
        return_softmax,
        axis,
    )


@templatedoc()
def sigmoid_cross_entropy_with_logits(
    x, label, ignore_index=kIgnoreIndex, name=None, normalize=False
):
    """

    ${comment}

    Args:
        x(Tensor): a 2-D tensor with shape N x D, where N is the batch size and
                D is the number of classes. This input is a tensor of logits computed
                by the previous operator. Logits are unscaled log probabilities given
                as log(p/(1-p)) The data type should be float32 or float64.
        label (Tensor): a 2-D tensor of the same type and shape as X.
                This input is a tensor of probabalistic labels for each logit.
        ignore_index(int): Specifies a target value that is ignored and
                does not contribute to the input gradient.
        name(str|None): The default value is None.  Normally there is
            no need for user to set this property.  For more information,
            please refer to :ref:`api_guide_Name`
        normalize(bool): If true, divide the output by the number of
            targets != ignore_index.

    Returns:
        out(Tensor): ${out_comment}

    Examples:
        .. code-block:: python


            import paddle

            input = paddle.rand(shape=[10], dtype='float32')
            label = paddle.rand(shape=[10], dtype='float32')
            loss = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(input, label,
                                                            ignore_index=-1, normalize=True)
            print(loss)
    """

    if in_dygraph_mode():
        return _C_ops.sigmoid_cross_entropy_with_logits(
            x, label, normalize, int(ignore_index)
        )
    check_variable_and_dtype(
        x,
        'input',
        ['float16', 'float32', 'float64'],
        'sigmoid_cross_entropy_with_logits',
    )

    helper = LayerHelper("sigmoid_cross_entropy_with_logits", **locals())

    out = helper.create_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type="sigmoid_cross_entropy_with_logits",
        inputs={"X": x, "Label": label},
        attrs={"ignore_index": ignore_index, 'normalize': normalize},
        outputs={"Out": out},
    )
    return out
