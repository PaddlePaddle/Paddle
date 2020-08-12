#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: define loss functions of neural network  
import paddle
import paddle.fluid as fluid
from ...fluid.framework import core, in_dygraph_mode
from ...fluid.layers.nn import _elementwise_op_in_dygraph
from ...fluid.layers import bpr_loss  #DEFINE_ALIAS
from ...fluid.layers import center_loss  #DEFINE_ALIAS
from ...fluid.layers import cross_entropy  #DEFINE_ALIAS
from ...fluid.layers import dice_loss  #DEFINE_ALIAS
from ...fluid.layers import iou_similarity  #DEFINE_ALIAS
from ...fluid.layers import kldiv_loss  #DEFINE_ALIAS
from ...fluid.layers import log_loss  #DEFINE_ALIAS
from ...fluid.layers import mse_loss  #DEFINE_ALIAS
from ...fluid.layers import npair_loss  #DEFINE_ALIAS
from ...fluid.layers import rank_loss  #DEFINE_ALIAS
from ...fluid.layers import sigmoid_cross_entropy_with_logits  #DEFINE_ALIAS
from ...fluid.layers import sigmoid_focal_loss  #DEFINE_ALIAS
from ...fluid.layers import smooth_l1  #DEFINE_ALIAS
from ...fluid.layers import softmax_with_cross_entropy  #DEFINE_ALIAS
from ...fluid.layers import square_error_cost  #DEFINE_ALIAS
from ...fluid.layers import ssd_loss  #DEFINE_ALIAS
from ...fluid.layers import teacher_student_sigmoid_loss  #DEFINE_ALIAS

from ...fluid.layers import edit_distance  #DEFINE_ALIAS
from ...fluid.layers import huber_loss  #DEFINE_ALIAS
from ...fluid.layers import sampled_softmax_with_cross_entropy  #DEFINE_ALIAS
from ...fluid.layer_helper import LayerHelper

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

__all__ = [
    'bpr_loss',
    'center_loss',
    'cross_entropy',
    'dice_loss',
    'edit_distance',
    'huber_loss',
    'iou_similarity',
    'kldiv_loss',
    'l1_loss',
    'log_loss',
    'mse_loss',
    'margin_ranking_loss',
    #       'nce',
    'npair_loss',
    'rank_loss',
    'sampled_softmax_with_cross_entropy',
    'sigmoid_cross_entropy_with_logits',
    'sigmoid_focal_loss',
    'smooth_l1',
    'softmax_with_cross_entropy',
    'square_error_cost',
    'ssd_loss',
    'teacher_student_sigmoid_loss'
]


def margin_ranking_loss(input,
                        other,
                        target,
                        margin=0.0,
                        reduction='mean',
                        name=None):
    """

    This op the calcluate the the margin rank loss between the input x, y and target, use the math function as follows. 

    .. math:: 
        margin\_rank\_loss = max(0, -target * (input - other) + margin)

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(margin\_rank\_loss)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(margin\_rank\_loss)

    If :attr:`reduction` set to ``'none'``, just return the origin ``margin_rank_loss``.

    Parameters:
        input(Tensor): the first input tensor, it's data type should be float32, float64.
        other(Tensor): the second input tensor, it's data type should be float32, float64.
        target(Tensor): the target value corresponding to input, it's data type should be float32, float64. 
        margin (float, optional): The margin value to add, default value is 0;
        reduction (str, optional): Indicate the reduction to apply to the loss, the candicates are ``'none'``, ``'mean'``, ``'sum'``.If :attr:`reduction` is ``'none'``, the unreduced loss is returned; If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns: Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[1]`, otherwise the shape is the same as `input` .The same dtype as input tensor.

    Examples:

        .. code-block:: python

            import numpy as np 
            import paddle 
            
            paddle.disable_static()
             
            x = paddle.to_variable(np.array([[1, 2], [3, 4]]).astype('float32'))
            y = paddle.to_variable(np.array([[2, 1], [2, 4]]).astype('float32'))
            target = paddle.to_variable(np.array([[1, -1], [-1, -1]]).astype('float32'))
            loss = paddle.nn.functional.margin_ranking_loss(x, y, target) 
            print(loss.numpy()) # [0.75]
    """
    if fluid.framework.in_dygraph_mode():
        out = core.ops.elementwise_sub(other, input)
        out = core.ops.elementwise_mul(out, target)
        if margin != 0.0:
            margin = fluid.dygraph.base.to_variable([margin], dtype=out.dtype)
            out = core.ops.elementwise_add(out, margin)
        out = core.ops.relu(out)
        if reduction == 'sum':
            return core.ops.reduce_sum(out, 'reduce_all', True)
        elif reduction == 'mean':
            return core.ops.mean(out)
        return out

    helper = LayerHelper("margin_ranking_loss", **locals())
    fluid.data_feeder.check_variable_and_dtype(
        input, 'input', ['float32', 'float64'], 'margin_rank_loss')
    fluid.data_feeder.check_variable_and_dtype(
        other, 'other', ['float32', 'float64'], 'margin_rank_loss')
    fluid.data_feeder.check_variable_and_dtype(
        target, 'target', ['float32', 'float64'], 'margin_rank_loss')

    out = paddle.elementwise_sub(other, input)
    out = paddle.multiply(out, target)

    if margin != 0.0:
        margin_var = out.block.create_var(dtype=out.dtype)
        paddle.fill_constant([1], out.dtype, margin, out=margin_var)
        out = paddle.add(out, margin_var)

    result_out = helper.create_variable_for_type_inference(input.dtype)

    if reduction == 'none':
        helper.append_op(
            type="relu", inputs={"X": out}, outputs={"Out": result_out})
        return result_out
    elif reduction == 'sum':
        out = paddle.nn.functional.relu(out)
        attrs = {"dim": [0], "keep_dim": False, "reduce_all": True}
        helper.append_op(
            type="reduce_sum",
            inputs={"X": out},
            outputs={"Out": result_out},
            attrs=attrs)
        return result_out
    elif reduction == 'mean':
        out = paddle.nn.functional.relu(out)
        helper.append_op(
            type="mean",
            inputs={"X": out},
            outputs={"Out": result_out},
            attrs={})
        return result_out


def l1_loss(x, label, reduction='mean', name=None):
    """
    This operator computes the L1 Loss of Tensor ``x`` and ``label`` as follows.

    If :attr:`reduction` set to ``'none'``, the loss is:

    .. math::
        Out = \lvert x - label\rvert

    If :attr:`reduction` set to ``'mean'``, the loss is:

    .. math::
        Out = MEAN(\lvert x - label\rvert)

    If :attr:`reduction` set to ``'sum'``, the loss is:

    .. math::
        Out = SUM(\lvert x - label\rvert)

    
    Parameters:
        x (Tensor): The input tensor. The shapes is [N, *], where N is batch size and `*` means any number of additional dimensions. It's data type should be float32, float64, int32, int64.
        label (Tensor): label. The shapes is [N, *], same shape as ``x`` . It's data type should be float32, float64, int32, int64.
        reduction (str, optional): Indicate the reduction to apply to the loss, 
            the candicates are ``'none'`` | ``'mean'`` | ``'sum'``.
            If :attr:`reduction` is ``'none'``, the unreduced loss is returned; 
            If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. 
            If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. 
            Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.
    Returns:
        Tensor, the L1 Loss of Tensor ``x`` and ``label``.
            If :attr:`reduction` is ``'none'``, the shape of output loss is [N, *], the same as ``x`` .
            If :attr:`reduction` is ``'mean'`` or ``'sum'``, the shape of output loss is [1], which means the output is a scalar.
    Examples:
        .. code-block:: python
            import paddle
            import numpy as np
            
            paddle.disable_static()
            x_data = np.array([[1.5, 0.8], [0.2, 1.3]]).astype("float32")
            label_data = np.array([[1.7, 1], [0.4, 0.5]]).astype("float32")
            x = paddle.to_variable(x_data)
            label = paddle.to_variable(label_data)

            l1_loss = paddle.nn.functional.l1_loss(x, label)
            print(l1_loss.numpy())  
            # [0.35]

            l1_loss = paddle.nn.functional.l1_loss(x, label, reduction='none')
            print(l1_loss.numpy())  
            # [[0.20000005 0.19999999]
            # [0.2        0.79999995]]

            l1_loss = paddle.nn.functional.l1_loss(x, label, reduction='sum')
            print(l1_loss.numpy())  
            # [1.4]
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise ValueError(
            "The value of 'reduction' in L1Loss should be 'sum', 'mean' or 'none', but "
            "received %s, which is not allowed." % reduction)

    if in_dygraph_mode():
        unreduced = _elementwise_op_in_dygraph(
            x, label, axis=-1, act='abs', op_name='elementwise_sub')
        if reduction == 'mean':
            return core.ops.mean(unreduced)
        elif reduction == 'sum':
            return core.ops.reduce_sum(unreduced, 'dim', [0], 'keep_dim', False,
                                       'reduce_all', True)
        else:
            return unreduced

    fluid.data_feeder.check_variable_and_dtype(
        x, 'x', ['float32', 'float64', 'int32', 'int64'], 'l1_loss')
    fluid.data_feeder.check_variable_and_dtype(
        label, 'label', ['float32', 'float64', 'int32', 'int64'], 'l1_loss')

    if reduction == 'sum':
        unreduced = paddle.elementwise_sub(x, label, act='abs')
        return paddle.sum(unreduced, name=name)
    elif reduction == 'mean':
        unreduced = paddle.elementwise_sub(x, label, act='abs')
        return paddle.mean(unreduced, name=name)
    else:
        return paddle.elementwise_sub(x, label, act='abs', name=name)
