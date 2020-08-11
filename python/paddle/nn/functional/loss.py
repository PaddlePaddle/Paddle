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


def margin_ranking_loss(x1, x2, target, margin=0.0, reduction='mean',
                        name=None):
    """

    This op the calcluate the the margin rank loss between the input x, y and target, use the math function as follows. 

    .. math:: 
        margin\_rank\_loss = max(0, -target * (x1- x2) + margin)

    If :attr:`reduction` set to ``'mean'``, the reduced mean loss is:

    .. math::
        Out = MEAN(margin\_rank\_loss)

    If :attr:`reduction` set to ``'sum'``, the reduced sum loss is:

    .. math::
        Out = SUM(margin\_rank\_loss)

    If :attr:`reduction` set to ``'none'``, just return the origin ``margin_rank_loss``.

    Parameters:
        x1(Tensor): the first input tensor, it's data type should be float32, float64.
        x2(Tensor): the second input tensor, it's data type should be float32, float64.
        target(Tensor): the target value corresponding to input, it's data type should be float32, float64. 
        margin (float, optional): The margin value to add, default value is 0;
        reduction (str, optional): Indicate the reduction to apply to the loss, the candicates are ``'none'``, ``'mean'``, ``'sum'``.If :attr:`reduction` is ``'none'``, the unreduced loss is returned; If :attr:`reduction` is ``'mean'``, the reduced mean loss is returned. If :attr:`reduction` is ``'sum'``, the reduced sum loss is returned. Default is ``'mean'``.
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Returns: Tensor, if :attr:`reduction` is ``'mean'`` or ``'sum'``, the out shape is :math:`[1]`, otherwise the shape is the same as input `x1` .The same dtype as input tensor.

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
        out = core.ops.elementwise_sub(x2, x1)
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
        x1, 'x1', ['float32', 'float64'], 'margin_rank_loss')
    fluid.data_feeder.check_variable_and_dtype(
        x2, 'x2', ['float32', 'float64'], 'margin_rank_loss')
    fluid.data_feeder.check_variable_and_dtype(
        target, 'target', ['float32', 'float64'], 'margin_rank_loss')

    out = paddle.elementwise_sub(x2, x1)
    out = paddle.multiply(out, target)

    if margin != 0.0:
        margin_var = out.block.create_var(dtype=out.dtype)
        paddle.fill_constant([1], out.dtype, margin, out=margin_var)
        out = paddle.add(out, margin_var)

    result_out = helper.create_variable_for_type_inference(x1.dtype)

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
